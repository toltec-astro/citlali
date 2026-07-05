#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace citlali::engine_detail {

inline std::string normalized_pointing_axis_name(std::string axis_name) {
    std::transform(axis_name.begin(), axis_name.end(), axis_name.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return axis_name;
}

}  // namespace citlali::engine_detail
