#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

template <class Config, class InvalidKeys>
std::vector<double> beammap_fixed_double_vector(
    Config &config, const std::vector<std::string> &path,
    std::size_t expected_size, InvalidKeys &invalid_keys) {
    std::vector<double> values;
    if (path.size() == 2) {
        values = config.template get_typed<std::vector<double>>(
            std::make_tuple(path[0], path[1]));
    }
    else {
        values = config.template get_typed<std::vector<double>>(
            std::make_tuple(path[0], path[1], path[2]));
    }
    if (values.size() != expected_size) {
        invalid_keys.push_back(path);
        values.resize(expected_size, 0.0);
    }
    return values;
}

}  // namespace citlali::pipeline
