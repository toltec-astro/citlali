#pragma once

#include "filesystem.h"
#include <fmt/format.h>
#include <regex>
#include <spdlog/spdlog.h>

namespace filename_utils {

namespace fs = std::filesystem;

template <typename... Args>
inline std::string parse_pattern(const std::string &pattern,
                                 const std::string &filename, Args &&... args) {
    if (pattern.empty()) {
        return "";
    }
    fs::path p(filename);
    std::string stem(p.stem().string());
    SPDLOG_TRACE("filename components: {stem}", fmt::arg("stem", stem));
    auto parsed =
        fmt::format(pattern, fmt::arg("stem", stem), FWD(args)...);
    return fs::absolute(parsed).string();
}

inline std::string create_dir_if_not_exist(
    const std::string& dirname)  {
    auto p = fs::absolute(dirname);
    if (fs::is_directory(p)) {
        SPDLOG_TRACE("use existing dir {}", p);
    } else if (fs::exists(p)) {
        throw std::runtime_error(fmt::format(
            "path {} exist and is not dir", p));
    } else {
        SPDLOG_TRACE("create dir {}", p);
        fs::create_directories(p);
    }
    return p.string();
}

inline std::vector<std::string> find_regex(const std::string &dirname,
                                           const std::string &pattern) {

    std::regex re_filename{pattern};
    auto checkfile = [&](auto &&fp) {
        // parse filepath
        std::string filename{fp.filename().string()};
        SPDLOG_TRACE("checking file {} with pattern {}", filename, pattern);
        std::smatch match;
        if (std::regex_match(filename, match, std::regex(re_filename))) {
            return true;
        }
        return false;
    };

    std::vector<std::string> result;
    for (auto &p_ : fs::directory_iterator(dirname)) {
        decltype(auto) p = p_.path();
        if (checkfile(p)) {
            result.push_back(p.string());
        }
    }
    return result;
}

} // namespace filename_utils
