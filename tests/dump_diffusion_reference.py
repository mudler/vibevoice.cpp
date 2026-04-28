#!/usr/bin/env python3
"""Reference fixture for VibeVoice's DiffusionHead (single-step v-prediction
output) and a 20-step DPM-Solver trajectory (cosine schedule).

Implements the architecture inline (no upstream import needed):
  TimestepEmbedder = sinusoidal(t, dim=256) -> Linear(256,h) -> SiLU -> Linear(h,h)
  HeadLayer:
    cond_mod = SiLU + Linear(h, 3h)            (shift, scale, gate)
    x' = x + gate * SwiGLU_FFN(modulate(RMSNorm(x), shift, scale))
  FinalLayer:
    cond_mod = SiLU + Linear(h, 2h)            (shift, scale)
    x' = Linear(modulate(RMSNorm_no_scale(x), shift, scale))
  DiffusionHead:
    x = noisy_proj(z)
    c = cond_proj(c_in) + t_embedder(t)
    x = head_layers(x, c) -> final_layer(x, c)

Configuration kept tiny for fast tests:
  hidden=64  latent=16  head_layers=2  ffn_ratio=3.0  T_frames=8
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- modules ----------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)
        if self.affine:
            n = n * self.weight
        return n


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_size=256):
        super().__init__()
        self.freq_size = freq_size
        self.lin1 = nn.Linear(freq_size, hidden, bias=False)
        self.lin2 = nn.Linear(hidden, hidden,    bias=False)

    @staticmethod
    def sinusoidal(t, dim, max_period=10_000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        e = self.sinusoidal(t, self.freq_size)
        return self.lin2(F.silu(self.lin1(e)))


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden, ffn_dim):
        super().__init__()
        self.gate = nn.Linear(hidden, ffn_dim, bias=False)
        self.up   = nn.Linear(hidden, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class HeadLayer(nn.Module):
    def __init__(self, hidden, ffn_dim, eps=1e-5):
        super().__init__()
        self.norm = RMSNorm(hidden, eps=eps, affine=True)
        self.ffn  = SwiGLUFFN(hidden, ffn_dim)
        self.adaln = nn.Linear(hidden, 3 * hidden, bias=False)

    def forward(self, x, c):
        m = self.adaln(F.silu(c))
        shift, scale, gate = m.chunk(3, dim=-1)
        return x + gate * self.ffn(modulate(self.norm(x), shift, scale))


class FinalLayer(nn.Module):
    def __init__(self, hidden, latent, eps=1e-5):
        super().__init__()
        self.norm  = RMSNorm(hidden, eps=eps, affine=False)
        self.proj  = nn.Linear(hidden, latent, bias=False)
        self.adaln = nn.Linear(hidden, 2 * hidden, bias=False)

    def forward(self, x, c):
        m = self.adaln(F.silu(c))
        shift, scale = m.chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


class DiffusionHead(nn.Module):
    def __init__(self, hidden, latent, head_layers, ffn_ratio, eps=1e-5):
        super().__init__()
        ffn_dim = int(hidden * ffn_ratio)
        self.noisy_proj = nn.Linear(latent, hidden, bias=False)
        self.cond_proj  = nn.Linear(hidden, hidden, bias=False)
        self.t_embed    = TimestepEmbedder(hidden)
        self.layers     = nn.ModuleList([HeadLayer(hidden, ffn_dim, eps) for _ in range(head_layers)])
        self.final      = FinalLayer(hidden, latent, eps)

    def forward(self, noisy, cond, t):
        x = self.noisy_proj(noisy)
        c = self.cond_proj(cond) + self.t_embed(t)[:, None, :]
        for layer in self.layers:
            x = layer(x, c)
        return self.final(x, c)


# ---------- DPM-Solver++ (cosine, v_prediction) ----------

def cosine_betas(n=1000, max_beta=0.999):
    betas = []
    for i in range(n):
        t1 = i / n; t2 = (i + 1) / n
        a1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
        a2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
        betas.append(min(1 - a2 / a1, max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def dpm_solver_trajectory(head: DiffusionHead, cond: torch.Tensor, x_T: torch.Tensor,
                          *, num_train_timesteps=1000, num_inference_steps=20):
    """Run a DPM-Solver multistep (order-2) sample with v-prediction.

    Implementation follows diffusers DPMSolverMultistepScheduler defaults
    (algorithm_type="dpmsolver++", solver_order=2, lower_order_final=True).
    Returns the per-step `x_t` trajectory as a tensor [steps+1, *x.shape].
    """
    betas = cosine_betas(num_train_timesteps)
    alphas = 1.0 - betas
    ac     = torch.cumprod(alphas, dim=0)
    alpha_t = torch.sqrt(ac)
    sigma_t = torch.sqrt(1 - ac)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)

    # uniform timestep selection — diffusers default
    timesteps = torch.linspace(num_train_timesteps - 1, 0, num_inference_steps + 1).round().long()
    # diffusers uses one extra sigma at the end (zero), so we have N+1 sample points
    # and N integration steps.

    x = x_T.clone()
    traj = [x.clone()]
    prev_v = None
    prev_x0 = None

    for i in range(num_inference_steps):
        t = timesteps[i]
        s = timesteps[i + 1]

        a_t = alpha_t[t]; sg_t = sigma_t[t]; l_t = lambda_t[t]
        if s == timesteps[0] and i == 0 and False:
            pass
        if s == 0:
            # final step: alpha_s = 1, sigma_s = 0
            a_s = torch.tensor(1.0); sg_s = torch.tensor(0.0); l_s = torch.tensor(float("inf"))
        else:
            a_s = alpha_t[s]; sg_s = sigma_t[s]; l_s = lambda_t[s]

        v_pred = head(x, cond, torch.tensor([t], dtype=torch.float32))
        # v_prediction: x0 = alpha_t * x - sigma_t * v
        x0 = a_t * x - sg_t * v_pred

        # Choose order: 1 for first step OR last step (lower_order_final)
        is_first = i == 0
        is_last  = i == num_inference_steps - 1
        order = 1 if (is_first or is_last) else 2

        if order == 1:
            # DPM-Solver++ first-order step
            h = l_s - l_t
            if torch.isinf(l_s):
                # final step toward sigma=0 is just x0
                x = x0
            else:
                x = (sg_s / sg_t) * x - a_s * (torch.exp(-h) - 1.0) * x0
        else:
            # second-order multistep
            h = l_s - l_t
            l_prev = lambda_t[timesteps[i - 1]]
            h_0 = l_t - l_prev
            r0 = h_0 / h
            D0 = x0
            D1 = (1.0 / r0) * (x0 - prev_x0)
            x = (sg_s / sg_t) * x \
                - a_s * (torch.exp(-h) - 1.0) * D0 \
                - 0.5 * a_s * (torch.exp(-h) - 1.0) * D1

        prev_v  = v_pred
        prev_x0 = x0
        traj.append(x.clone())

    return torch.stack(traj, dim=0)


# ---------- dump ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",            required=True)
    ap.add_argument("--hidden",         type=int, default=64)
    ap.add_argument("--latent",         type=int, default=16)
    ap.add_argument("--head-layers",    type=int, default=2)
    ap.add_argument("--ffn-ratio",      type=float, default=3.0)
    ap.add_argument("--frames",         type=int, default=8)
    ap.add_argument("--seed",           type=int, default=303)
    ap.add_argument("--eps",            type=float, default=1e-5)
    ap.add_argument("--inference-steps",type=int, default=20)
    args = ap.parse_args()

    try:
        import gguf
    except ImportError:
        sys.stderr.write("error: pip install gguf\n")
        return 1

    torch.manual_seed(args.seed)
    head = DiffusionHead(args.hidden, args.latent, args.head_layers,
                         args.ffn_ratio, eps=args.eps).eval()

    # --- single-step inputs ---
    noisy = torch.randn(1, args.frames, args.latent)
    cond  = torch.randn(1, args.frames, args.hidden)
    t     = torch.tensor([500.0])
    with torch.no_grad():
        v_out = head(noisy, cond, t)         # [1, frames, latent]

    # --- DPM-Solver trajectory ---
    torch.manual_seed(args.seed + 7)
    x_T   = torch.randn(1, args.frames, args.latent)
    cond2 = torch.randn(1, args.frames, args.hidden)
    with torch.no_grad():
        traj = dpm_solver_trajectory(head, cond2, x_T,
                                     num_inference_steps=args.inference_steps)
    final_x = traj[-1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g = gguf.GGUFWriter(str(out), arch="diffusion-test")

    g.add_uint32 ("dh.hidden",          args.hidden)
    g.add_uint32 ("dh.latent",          args.latent)
    g.add_uint32 ("dh.head_layers",     args.head_layers)
    g.add_float32("dh.ffn_ratio",       args.ffn_ratio)
    g.add_uint32 ("dh.frames",          args.frames)
    g.add_float32("dh.eps",             args.eps)
    g.add_uint32 ("dh.inference_steps", args.inference_steps)

    # I/O
    g.add_tensor("test.noisy",            noisy.numpy())
    g.add_tensor("test.cond",             cond.numpy())
    g.add_tensor("test.t",                t.numpy())
    g.add_tensor("test.v_out",            v_out.numpy())
    g.add_tensor("test.x_T",              x_T.numpy())
    g.add_tensor("test.cond2",            cond2.numpy())
    g.add_tensor("test.traj_final",       final_x.numpy())

    # Weights
    sd = head.state_dict()
    g.add_tensor("dh.noisy_proj",  sd["noisy_proj.weight"].numpy())
    g.add_tensor("dh.cond_proj",   sd["cond_proj.weight"].numpy())
    g.add_tensor("dh.t_embed_lin1",sd["t_embed.lin1.weight"].numpy())
    g.add_tensor("dh.t_embed_lin2",sd["t_embed.lin2.weight"].numpy())
    for i in range(args.head_layers):
        p = f"dh.layer_{i}"
        g.add_tensor(p + ".norm",      sd[f"layers.{i}.norm.weight"].numpy())
        g.add_tensor(p + ".ffn_gate",  sd[f"layers.{i}.ffn.gate.weight"].numpy())
        g.add_tensor(p + ".ffn_up",    sd[f"layers.{i}.ffn.up.weight"].numpy())
        g.add_tensor(p + ".ffn_down",  sd[f"layers.{i}.ffn.down.weight"].numpy())
        g.add_tensor(p + ".adaln",     sd[f"layers.{i}.adaln.weight"].numpy())
    g.add_tensor("dh.final.proj",  sd["final.proj.weight"].numpy())
    g.add_tensor("dh.final.adaln", sd["final.adaln.weight"].numpy())

    g.write_header_to_file()
    g.write_kv_data_to_file()
    g.write_tensors_to_file()
    g.close()
    sys.stderr.write(
        f"wrote {out}: hidden={args.hidden} latent={args.latent} "
        f"layers={args.head_layers} frames={args.frames}\n"
        f"  v_out shape={tuple(v_out.shape)} "
        f"traj_final shape={tuple(final_x.shape)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
