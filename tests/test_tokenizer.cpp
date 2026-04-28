// Tokenizer parity test.
//
// Loads:
//   ${VV_FIXTURES_DIR}/tokenizer.gguf            — produced by
//                                                  scripts/convert_tokenizer.py
//   ${VV_FIXTURES_DIR}/tokenizer_reference.jsonl — produced by
//                                                  tests/dump_tokenizer_reference.py
// and asserts our C++ tokenizer reproduces every reference id stream exactly.
//
// The test is skipped (returns 77, ctest "skip" code) if either file is
// missing — this keeps it green on a fresh checkout where the user hasn't
// yet downloaded the tokenizer.

#include "tokenizer.hpp"
#include "model_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool file_ok(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Tiny ad-hoc JSON parser sufficient for our reference jsonl:
//   {"text": "...", "ids": [int, ...]}
struct Record {
    std::string text;
    std::vector<int32_t> ids;
};

bool parse_record(const std::string& line, Record* r) {
    auto t_pos = line.find("\"text\"");
    if (t_pos == std::string::npos) return false;
    auto colon = line.find(':', t_pos);
    auto q1    = line.find('"', colon);
    if (q1 == std::string::npos) return false;
    // Walk until matching unescaped quote.
    std::string s;
    size_t p = q1 + 1;
    while (p < line.size()) {
        char c = line[p];
        if (c == '\\' && p + 1 < line.size()) {
            char e = line[p + 1];
            switch (e) {
                case 'n':  s.push_back('\n'); p += 2; continue;
                case 't':  s.push_back('\t'); p += 2; continue;
                case 'r':  s.push_back('\r'); p += 2; continue;
                case '"':  s.push_back('"');  p += 2; continue;
                case '\\': s.push_back('\\'); p += 2; continue;
                case '/':  s.push_back('/');  p += 2; continue;
                case 'u':
                    if (p + 6 <= line.size()) {
                        unsigned cp = std::stoul(line.substr(p + 2, 4), nullptr, 16);
                        // emit UTF-8
                        if (cp < 0x80) s.push_back(static_cast<char>(cp));
                        else if (cp < 0x800) {
                            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        p += 6; continue;
                    }
                    s.push_back(e); p += 2; continue;
                default: s.push_back(e); p += 2; continue;
            }
        }
        if (c == '"') break;
        s.push_back(c);
        p++;
    }
    r->text = std::move(s);

    auto i_pos = line.find("\"ids\"", p);
    if (i_pos == std::string::npos) return false;
    auto lb = line.find('[', i_pos);
    auto rb = line.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;

    std::string body = line.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // strip whitespace
        size_t a = 0, b = tok.size();
        while (a < b && std::isspace(static_cast<unsigned char>(tok[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(tok[b - 1]))) --b;
        if (a == b) continue;
        r->ids.push_back(std::stoi(tok.substr(a, b - a)));
    }
    return true;
}

}  // namespace

int main() {
#ifndef VV_FIXTURES_DIR
#  define VV_FIXTURES_DIR "tests/fixtures"
#endif
    const std::string fix    = VV_FIXTURES_DIR;
    const std::string gguf   = fix + "/tokenizer.gguf";
    const std::string jsonl  = fix + "/tokenizer_reference.jsonl";

    if (!file_ok(gguf) || !file_ok(jsonl)) {
        std::fprintf(stderr,
                     "test_tokenizer: skipping — missing fixtures.\n"
                     "  expected: %s\n"
                     "  expected: %s\n"
                     "  run scripts/convert_tokenizer.py and "
                     "tests/dump_tokenizer_reference.py to generate them.\n",
                     gguf.c_str(), jsonl.c_str());
        return 77;  // ctest skip
    }

    vv::ModelLoader loader;
    if (!loader.load(gguf)) {
        std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
        return 1;
    }
    vv::Tokenizer tok;
    if (!tok.load(loader)) {
        std::fprintf(stderr, "tokenizer load failed\n");
        return 2;
    }

    std::ifstream in(jsonl);
    std::string line;
    int n_total = 0, n_pass = 0, n_fail = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        Record r;
        if (!parse_record(line, &r)) {
            std::fprintf(stderr, "parse error on line: %s\n", line.c_str());
            return 3;
        }
        ++n_total;
        auto got = tok.encode(r.text);
        if (got == r.ids) {
            ++n_pass;
            continue;
        }
        ++n_fail;
        if (n_fail <= 5) {
            std::fprintf(stderr, "MISMATCH on text=%.80s\n", r.text.c_str());
            std::fprintf(stderr, "  expected (%zu): ", r.ids.size());
            for (size_t k = 0; k < r.ids.size() && k < 30; ++k) std::fprintf(stderr, "%d ", r.ids[k]);
            std::fprintf(stderr, "\n  got      (%zu): ", got.size());
            for (size_t k = 0; k < got.size() && k < 30; ++k) std::fprintf(stderr, "%d ", got[k]);
            std::fprintf(stderr, "\n");
        }
    }

    std::fprintf(stderr, "tokenizer parity: %d/%d pass, %d fail\n", n_pass, n_total, n_fail);
    return n_fail == 0 ? 0 : 4;
}
