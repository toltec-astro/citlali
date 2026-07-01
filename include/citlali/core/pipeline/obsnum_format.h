#pragma once

#include <iomanip>
#include <sstream>
#include <string>

namespace citlali::pipeline {

inline std::string format_obsnum(int obsnum) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << obsnum;
    return ss.str();
}

}  // namespace citlali::pipeline
