#pragma once

#include <map>
#include <string>

namespace citlali::pipeline {

inline std::string stats_unit_or_empty(
    const std::map<std::string, std::string> &units,
    const std::string &stat) {
    const auto it = units.find(stat);
    return it == units.end() ? "" : it->second;
}

}  // namespace citlali::pipeline
