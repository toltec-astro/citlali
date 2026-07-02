#pragma once

#include <cstddef>
#include <string>

namespace citlali::pipeline {

template <class Values>
std::string join_numeric_values(const Values &values,
                                const std::string &separator = ",") {
    std::string joined;
    for (std::size_t i=0; i<values.size(); ++i) {
        if (i > 0) {
            joined += separator;
        }
        joined += std::to_string(values[i]);
    }
    return joined;
}

template <class Values>
std::string join_string_values(const Values &values,
                               const std::string &separator = ",") {
    std::string joined;
    for (std::size_t i=0; i<values.size(); ++i) {
        if (i > 0) {
            joined += separator;
        }
        joined += values[i];
    }
    return joined;
}

}  // namespace citlali::pipeline
