#include "common.hpp"

#include <atomic>
#include <fstream>
#include <mutex>
#include <sys/stat.h>

namespace vv {

namespace {
struct LogState {
    std::mutex   mu;
    vv_log_cb    cb       = nullptr;
    void*        user     = nullptr;
    vv_log_level min_lvl  = VV_LOG_INFO;
};
LogState& state() {
    static LogState s;
    return s;
}

const char* lvl_str(vv_log_level l) {
    switch (l) {
        case VV_LOG_ERROR: return "E";
        case VV_LOG_WARN:  return "W";
        case VV_LOG_INFO:  return "I";
        case VV_LOG_DEBUG: return "D";
    }
    return "?";
}
}  // namespace

void log(vv_log_level lvl, const char* fmt, ...) {
    auto& s = state();
    if (lvl > s.min_lvl) return;

    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    std::lock_guard<std::mutex> lk(s.mu);
    if (s.cb) {
        s.cb(lvl, buf, s.user);
    } else {
        std::fprintf(lvl <= VV_LOG_WARN ? stderr : stdout,
                     "[vv %s] %s\n", lvl_str(lvl), buf);
    }
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

size_t file_size(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return 0;
    return static_cast<size_t>(st.st_size);
}

bool read_file(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    auto n = f.tellg();
    if (n < 0) return false;
    f.seekg(0, std::ios::beg);
    out->resize(static_cast<size_t>(n));
    if (n > 0) f.read(out->data(), n);
    return f.good() || f.eof();
}

}  // namespace vv

extern "C" {

void vv_set_log_callback(vv_log_cb cb, void* user_data) {
    auto& s = vv::state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.cb   = cb;
    s.user = user_data;
}

const char* vv_version(void) {
    return "vibevoice.cpp 0.0.1 (M1-foundations)";
}

}  // extern "C"
