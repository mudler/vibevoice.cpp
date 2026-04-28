#ifndef VIBEVOICE_DPM_SOLVER_HPP
#define VIBEVOICE_DPM_SOLVER_HPP

// DPM-Solver++ multistep sampler with cosine beta schedule and v-prediction.
// Mirrors vibevoice/schedule/dpm_solver.py (which itself follows diffusers'
// DPMSolverMultistepScheduler with algorithm_type="dpmsolver++",
// solver_order=2, lower_order_final=True).
//
// Each call to dpm_solver_sample() builds N small graphs (one per inference
// step) that invoke the diffusion head; the order-1 / order-2 update rule
// runs on the CPU between graph executions.

#include "diffusion_head.hpp"

#include <vector>

namespace vv {

struct DPMSolverConfig {
    int   num_train_timesteps = 1000;
    int   num_inference_steps = 20;
    int   solver_order        = 2;
    bool  lower_order_final   = true;
};

struct DPMSolverState {
    std::vector<float> alpha_t;     // [num_train_timesteps]
    std::vector<float> sigma_t;
    std::vector<float> lambda_t;
    std::vector<int>   timesteps;   // length num_inference_steps + 1; last = -1 (sigma=0)
};

// Initialize state with cosine schedule.
void dpm_solver_init(const DPMSolverConfig& cfg, DPMSolverState* state);

// Run the sampler. On entry, `x` holds x_T (initial Gaussian noise);
// on return, `x` holds x_0 (denoised latent).
//
//   x:        [B * frames * latent], updated in place
//   cond:     [B * frames * hidden], positive condition
//   cond_neg: optional negative condition with the same shape (CFG). Pass
//             empty to disable CFG (cfg_scale ignored).
//   cfg_scale: > 1.0 sharpens text conditioning. 1.3 is a sane default.
//
// Returns 0 on success, non-zero on failure.
int dpm_solver_sample(std::vector<float>&         x,
                      int                         latent,
                      int                         frames,
                      int                         batch,
                      const std::vector<float>&   cond,
                      int                         hidden,
                      const DiffusionHeadWeights& w,
                      const DiffusionHeadConfig&  head_cfg,
                      const DPMSolverConfig&      solver_cfg,
                      const DPMSolverState&       state,
                      const std::vector<float>&   cond_neg = {},
                      float                       cfg_scale = 1.0f);

}  // namespace vv

#endif  // VIBEVOICE_DPM_SOLVER_HPP
