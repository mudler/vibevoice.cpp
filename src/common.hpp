#ifndef VIBEVOICE_COMMON_HPP
#define VIBEVOICE_COMMON_HPP

#include "vibevoice.h"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace vv {

void log(vv_log_level lvl, const char* fmt, ...);

#define VV_LOG_ERROR(...) ::vv::log(VV_LOG_ERROR, __VA_ARGS__)
#define VV_LOG_WARN(...)  ::vv::log(VV_LOG_WARN,  __VA_ARGS__)
#define VV_LOG_INFO(...)  ::vv::log(VV_LOG_INFO,  __VA_ARGS__)
#define VV_LOG_DEBUG(...) ::vv::log(VV_LOG_DEBUG, __VA_ARGS__)

bool   read_file(const std::string& path, std::string* out);
size_t file_size(const std::string& path);
bool   file_exists(const std::string& path);

}  // namespace vv

#endif  // VIBEVOICE_COMMON_HPP
