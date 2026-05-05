#include "tokenizer.hpp"
#include "common.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

namespace vv {

namespace {

// ----- UTF-8 helpers -----

inline int utf8_len(uint8_t b0) {
    if (b0 < 0x80) return 1;
    if ((b0 & 0xE0) == 0xC0) return 2;
    if ((b0 & 0xF0) == 0xE0) return 3;
    if ((b0 & 0xF8) == 0xF0) return 4;
    return 1;  // malformed; treat as 1
}

inline uint32_t utf8_to_cp(const char* s, size_t n, int& consumed) {
    if (n == 0) { consumed = 0; return 0; }
    auto b0 = static_cast<uint8_t>(s[0]);
    int  k  = utf8_len(b0);
    if (static_cast<size_t>(k) > n) k = 1;
    uint32_t cp = 0;
    if (k == 1) cp = b0;
    else if (k == 2 && n >= 2) cp = (uint32_t(b0 & 0x1F) << 6)
                                   | (uint8_t(s[1]) & 0x3F);
    else if (k == 3 && n >= 3) cp = (uint32_t(b0 & 0x0F) << 12)
                                   | (uint32_t(uint8_t(s[1]) & 0x3F) << 6)
                                   | (uint8_t(s[2]) & 0x3F);
    else if (k == 4 && n >= 4) cp = (uint32_t(b0 & 0x07) << 18)
                                   | (uint32_t(uint8_t(s[1]) & 0x3F) << 12)
                                   | (uint32_t(uint8_t(s[2]) & 0x3F) << 6)
                                   | (uint8_t(s[3]) & 0x3F);
    consumed = k;
    return cp;
}

inline void cp_to_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ----- Unicode classification (M1: ASCII-correct, non-ASCII heuristic) -----

inline bool is_ascii_letter(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}
inline bool is_ascii_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
inline bool is_newline_cp(uint32_t cp)   { return cp == '\n' || cp == '\r'; }
inline bool is_ascii_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\v' || cp == '\f';
}
// Common Unicode whitespace (subset). Adequate for typical text.
inline bool is_unicode_space(uint32_t cp) {
    if (is_ascii_space(cp)) return true;
    if (cp == 0x00A0) return true;          // NBSP
    if (cp == 0x1680) return true;          // Ogham
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    if (cp == 0x2028 || cp == 0x2029) return true;
    if (cp == 0x202F || cp == 0x205F || cp == 0x3000) return true;
    return false;
}
inline bool is_letter_cp(uint32_t cp) {
    if (is_ascii_letter(cp)) return true;
    if (cp < 0x80)           return false;
    if (is_unicode_space(cp)) return false;
    if (cp < 0xA0)           return false;  // C1 controls

    // Latin-1 supplement: U+00A0..U+00BF is mostly punct/symbols. The few real
    // letters in this range:
    if (cp >= 0x00A0 && cp < 0x00C0) {
        return cp == 0x00AA || cp == 0x00B5 || cp == 0x00BA;
    }

    // U+2000..U+27BF — general punctuation, currency, arrows, math, misc
    // symbols, dingbats. Mark as non-letter. (Letter-like symbols at
    // U+2100..U+214F lose some accuracy but it's a small minority of text.)
    if (cp >= 0x2000 && cp <= 0x27BF) return false;

    // CJK Symbols and Punctuation
    if (cp >= 0x3000 && cp <= 0x303F) return false;

    // Common emoji blocks
    if (cp >= 0x1F300 && cp <= 0x1F9FF) return false;
    if (cp >= 0x1FA00 && cp <= 0x1FAFF) return false;

    return true;
}
inline bool is_number_cp(uint32_t cp) {
    if (is_ascii_digit(cp)) return true;
    // Only ASCII digits classified as Number for now (Qwen2 splits each digit
    // into its own pre-token regardless).
    return false;
}

// ----- GPT-2 byte ↔ unicode map -----

void make_byte_to_unicode(uint32_t out[256]) {
    bool   used[256] = {};
    int    n         = 0;
    auto   add       = [&](int b) { out[b] = static_cast<uint32_t>(b); used[b] = true; };
    for (int b = 33; b <= 126; ++b)  add(b);
    for (int b = 161; b <= 172; ++b) add(b);
    for (int b = 174; b <= 255; ++b) add(b);
    for (int b = 0; b < 256; ++b) {
        if (!used[b]) { out[b] = 256u + static_cast<uint32_t>(n); ++n; }
    }
}

}  // namespace

Tokenizer::Tokenizer() { build_byte_maps(); }

void Tokenizer::build_byte_maps() {
    uint32_t cps[256];
    make_byte_to_unicode(cps);
    for (int b = 0; b < 256; ++b) {
        std::string s;
        cp_to_utf8(cps[b], s);
        byte_to_chstr_[b] = s;
        chstr_to_byte_[s] = static_cast<uint8_t>(b);
    }
}

bool Tokenizer::load(const ModelLoader& m) {
    tokens_      = m.get_str_array("tokenizer.tokens");
    auto types   = m.get_i32_array("tokenizer.token_type");
    auto merges  = m.get_str_array("tokenizer.merges");

    if (tokens_.empty()) {
        VV_LOG_ERROR("Tokenizer: tokenizer.tokens missing/empty");
        return false;
    }
    token_type_.assign(types.begin(), types.end());
    token_type_.resize(tokens_.size(), 0);

    token_id_.reserve(tokens_.size());
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (!tokens_[i].empty()) token_id_[tokens_[i]] = static_cast<int32_t>(i);
    }

    merge_rank_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        const auto& m_str = merges[i];
        auto sp = m_str.find(' ');
        if (sp == std::string::npos) continue;
        merge_rank_.emplace(std::make_pair(m_str.substr(0, sp), m_str.substr(sp + 1)),
                            static_cast<int32_t>(i));
    }

    auto sp_ids  = m.get_i32_array("tokenizer.special_tokens_ids");
    auto sp_text = m.get_str_array("tokenizer.special_tokens_text");
    const size_t k = std::min(sp_ids.size(), sp_text.size());
    special_.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        special_.emplace_back(sp_text[i], sp_ids[i]);
    }
    std::sort(special_.begin(), special_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    bos_id_ = m.get_i32("tokenizer.bos_id", -1);
    eos_id_ = m.get_i32("tokenizer.eos_id", -1);
    pad_id_ = m.get_i32("tokenizer.pad_id", -1);

    VV_LOG_INFO("Tokenizer: loaded %zu tokens, %zu merges, %zu special",
                tokens_.size(), merge_rank_.size(), special_.size());
    return true;
}

bool Tokenizer::load_from_file(const std::string& path) {
    // Local loader: load() copies tokens_/merge_rank_/special_ out of the
    // ggml context, so the loader doesn't need to outlive this call. A
    // previously-static thread_local instance prevented re-loading the
    // tokenizer in the same process (e.g. the C-API engine wants both
    // a TTS and an ASR Tokenizer instance).
    ModelLoader loader;
    return loader.load(path) && load(loader);
}

const std::string& Tokenizer::token(int32_t id) const {
    static const std::string empty;
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return empty;
    return tokens_[id];
}

// ----- pre-tokenizer -----

namespace {

bool starts_with_ci(const std::string& text, size_t pos, const char* lit) {
    size_t n = std::strlen(lit);
    if (pos + n > text.size()) return false;
    for (size_t i = 0; i < n; ++i) {
        char c = text[pos + i];
        char l = lit[i];
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
        if (l >= 'A' && l <= 'Z') l = static_cast<char>(l + 32);
        if (c != l) return false;
    }
    return true;
}

// Try to match a contraction at `pos`. Returns its byte length, or 0.
// Mirrors HF's (?i:'s|'t|'re|'ve|'m|'ll|'d).
size_t match_contraction(const std::string& text, size_t pos) {
    if (pos >= text.size() || text[pos] != '\'') return 0;
    // Order: longer first so 're/'ve/'ll win over single-letter alternatives.
    static const char* tries[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
    for (const char* t : tries) {
        if (starts_with_ci(text, pos, t)) return std::strlen(t);
    }
    return 0;
}

// Scan back to the start of the previous codepoint within [floor, end).
size_t prev_cp(const std::string& text, size_t end, size_t floor) {
    if (end <= floor) return floor;
    size_t p = end - 1;
    while (p > floor && (static_cast<uint8_t>(text[p]) & 0xC0) == 0x80) --p;
    return p;
}

// Pattern 1: [^\r\n\p{L}\p{N}]?\p{L}+
size_t try_letters(const std::string& text, size_t i) {
    const size_t n = text.size();
    if (i >= n) return i;

    size_t j = i;
    int    k0 = 1;
    uint32_t c0 = utf8_to_cp(text.data() + j, n - j, k0);
    // Try the optional prefix: anything except \r, \n, Letter, Number.
    bool   tried_prefix = false;
    size_t after_pref   = j;
    if (!is_newline_cp(c0) && !is_letter_cp(c0) && !is_number_cp(c0)) {
        after_pref   = j + k0;
        tried_prefix = true;
    }

    // Now require ≥1 letter at after_pref.
    if (after_pref >= n) return i;
    int k1 = 1;
    uint32_t c1 = utf8_to_cp(text.data() + after_pref, n - after_pref, k1);
    if (!is_letter_cp(c1)) {
        if (!tried_prefix) return i;
        return i;  // optional prefix didn't lead to a letter run
    }

    j = after_pref + k1;
    while (j < n) {
        int kk = 1; uint32_t cc = utf8_to_cp(text.data() + j, n - j, kk);
        if (!is_letter_cp(cc)) break;
        j += kk;
    }
    return j;
}

// Pattern 2: \p{N} — one digit codepoint.
size_t try_number(const std::string& text, size_t i) {
    const size_t n = text.size();
    if (i >= n) return i;
    int k = 1;
    uint32_t c = utf8_to_cp(text.data() + i, n - i, k);
    if (!is_number_cp(c)) return i;
    return i + k;
}

// Pattern 3: " ?[^\s\p{L}\p{N}]+[\r\n]*"  — leading space is the LITERAL ' '.
size_t try_punct(const std::string& text, size_t i) {
    const size_t n = text.size();
    size_t       j = i;
    if (j < n && text[j] == ' ') j++;

    size_t sym_start = j;
    while (j < n) {
        int kk = 1; uint32_t cc = utf8_to_cp(text.data() + j, n - j, kk);
        if (is_unicode_space(cc) || is_letter_cp(cc) || is_number_cp(cc)) break;
        j += kk;
    }
    if (j == sym_start) return i;  // ≥1 symbol char required

    while (j < n) {
        int kk = 1; uint32_t cc = utf8_to_cp(text.data() + j, n - j, kk);
        if (!is_newline_cp(cc)) break;
        j += kk;
    }
    return j;
}

// Pattern 4: \s*[\r\n]+
size_t try_newlines(const std::string& text, size_t i) {
    const size_t n = text.size();
    if (i >= n) return i;

    // Scan whitespace run; remember if any newline was present.
    size_t end           = i;
    bool   had_newline   = false;
    while (end < n) {
        int kk = 1; uint32_t cc = utf8_to_cp(text.data() + end, n - end, kk);
        if (!is_unicode_space(cc)) break;
        if (is_newline_cp(cc)) had_newline = true;
        end += kk;
    }
    if (!had_newline) return i;

    // Cut back any trailing non-newline whitespace so the match ends with \r\n.
    size_t cut = end;
    while (cut > i) {
        size_t p = prev_cp(text, cut, i);
        int kk = 1; uint32_t cc = utf8_to_cp(text.data() + p, cut - p, kk);
        if (is_newline_cp(cc)) break;
        cut = p;
    }
    return cut;
}

// Pattern 5: \s+(?!\S)  — whitespace run that ends at EOS or before more \s.
size_t try_trailing_ws(const std::string& text, size_t i) {
    const size_t n = text.size();
    if (i >= n) return i;
    int k0 = 1; uint32_t c0 = utf8_to_cp(text.data() + i, n - i, k0);
    if (!is_unicode_space(c0)) return i;

    // Find end of \s run.
    size_t end = i;
    while (end < n) {
        int kk = 1; uint32_t cc = utf8_to_cp(text.data() + end, n - end, kk);
        if (!is_unicode_space(cc)) break;
        end += kk;
    }
    if (end == n) return end;  // ends at EOS — match all
    // Otherwise back off one codepoint so the lookahead sees \s, not \S.
    size_t prev = prev_cp(text, end, i);
    if (prev <= i) return i;  // would yield empty match
    return prev;
}

// Pattern 6: \s+
size_t try_ws(const std::string& text, size_t i) {
    const size_t n = text.size();
    size_t       j = i;
    while (j < n) {
        int kk = 1; uint32_t cc = utf8_to_cp(text.data() + j, n - j, kk);
        if (!is_unicode_space(cc)) break;
        j += kk;
    }
    return j;
}

}  // namespace

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> out;
    const size_t n = text.size();
    size_t i = 0;

    auto take = [&](size_t start, size_t end) {
        if (end > start) out.emplace_back(text.substr(start, end - start));
    };

    while (i < n) {
        // Special tokens first (longest first).
        bool matched_special = false;
        for (const auto& sp : special_) {
            const auto& s = sp.first;
            if (s.empty()) continue;
            if (i + s.size() <= n &&
                std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                take(i, i + s.size());
                i += s.size();
                matched_special = true;
                break;
            }
        }
        if (matched_special) continue;

        size_t end;

        // 0: contraction
        size_t cm = match_contraction(text, i);
        if (cm) { take(i, i + cm); i += cm; continue; }

        // 1: letters (with optional non-letter/digit/newline prefix)
        end = try_letters(text, i);
        if (end > i) { take(i, end); i = end; continue; }

        // 2: single digit
        end = try_number(text, i);
        if (end > i) { take(i, end); i = end; continue; }

        // 3: punctuation/symbols (with optional leading literal space)
        end = try_punct(text, i);
        if (end > i) { take(i, end); i = end; continue; }

        // 4: \s*[\r\n]+
        end = try_newlines(text, i);
        if (end > i) { take(i, end); i = end; continue; }

        // 5: \s+(?!\S)
        end = try_trailing_ws(text, i);
        if (end > i) { take(i, end); i = end; continue; }

        // 6: \s+
        end = try_ws(text, i);
        if (end > i) { take(i, end); i = end; continue; }

        // Nothing matched — emit one byte to make progress (defensive).
        take(i, i + 1);
        i += 1;
    }

    return out;
}

// ----- byte-level BPE -----

int Tokenizer::bpe_pretoken_to_ids(const std::string& pre_tok,
                                   std::vector<int32_t>& out) const {
    // Step 1: encode bytes through byte_to_unicode and build symbol list.
    std::vector<std::string> symbols;
    symbols.reserve(pre_tok.size());
    for (unsigned char b : pre_tok) symbols.emplace_back(byte_to_chstr_[b]);

    // Step 2: greedy lowest-rank-pair merging (standard GPT-2 BPE).
    while (symbols.size() >= 2) {
        int32_t  best_rank = std::numeric_limits<int32_t>::max();
        size_t   best_idx  = std::numeric_limits<size_t>::max();
        for (size_t k = 0; k + 1 < symbols.size(); ++k) {
            auto it = merge_rank_.find({symbols[k], symbols[k + 1]});
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = k;
            }
        }
        if (best_idx == std::numeric_limits<size_t>::max()) break;
        symbols[best_idx] += symbols[best_idx + 1];
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    // Step 3: lookup ids.
    int unknown = 0;
    for (const auto& s : symbols) {
        auto it = token_id_.find(s);
        if (it == token_id_.end()) { ++unknown; continue; }
        out.push_back(it->second);
    }
    return unknown;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;

    // First pass: split `text` into a list of fragments — either a literal
    // special-token (emitted as a single id) or a plain-text run that the
    // pre-tokenizer + BPE handles. Without this top-level split the punct
    // pattern in pre_tokenize() can greedily swallow the `<|` prefix of a
    // bracketed special (e.g. "0:<|vision_start|>" is one punct run if no
    // earlier rule peels off the special), which then masks the special-token
    // identification at the BPE level.
    const size_t n = text.size();
    size_t i = 0;
    auto flush_text = [&](size_t start, size_t end) {
        if (end <= start) return;
        std::string sub = text.substr(start, end - start);
        auto pre = pre_tokenize(sub);
        for (const auto& pt : pre) {
            bool is_special = false;
            for (const auto& sp : special_) {
                if (sp.first == pt) {
                    out.push_back(sp.second);
                    is_special = true;
                    break;
                }
            }
            if (is_special) continue;
            bpe_pretoken_to_ids(pt, out);
        }
    };

    size_t text_start = 0;
    while (i < n) {
        bool matched = false;
        for (const auto& sp : special_) {
            const auto& s = sp.first;
            if (s.empty()) continue;
            if (i + s.size() <= n &&
                std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                flush_text(text_start, i);
                out.push_back(sp.second);
                i += s.size();
                text_start = i;
                matched = true;
                break;
            }
        }
        if (!matched) ++i;
    }
    flush_text(text_start, n);
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string bytes;
    bytes.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) continue;
        const std::string& tok = tokens_[id];
        // Special tokens: emit text verbatim (don't byte-decode).
        if (id < static_cast<int32_t>(token_type_.size()) &&
            (token_type_[id] == 1 || token_type_[id] == 3)) {
            bytes.append(tok);
            continue;
        }
        // Normal: walk Unicode codepoints and reverse byte_to_unicode map.
        const char* p = tok.data();
        size_t      n = tok.size();
        while (n > 0) {
            int k = utf8_len(static_cast<uint8_t>(p[0]));
            if (static_cast<size_t>(k) > n) k = 1;
            std::string chstr(p, k);
            auto it = chstr_to_byte_.find(chstr);
            if (it != chstr_to_byte_.end()) bytes.push_back(static_cast<char>(it->second));
            p += k; n -= static_cast<size_t>(k);
        }
    }
    return bytes;
}

}  // namespace vv
