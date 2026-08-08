#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <Eigen/Core>

#include <array>
#include <string>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

template <class Config, class ArrayNameMap, class Diagnostics>
void read_jinc_filter_request_config(
    Config &config, const ArrayNameMap &array_name_map,
    citlali::config::MapmakingJincFilterConfig &target,
    Diagnostics &diagnostics) {
    read_config_value(
        config, target.r_max, diagnostics,
        std::tuple{"mapmaking", "jinc_filter", "r_max"});
    if (config.template has_typed<int>(
            std::tuple{"mapmaking", "jinc_filter", "subpixel_n"})) {
        read_config_value(
            config, target.subpixel_n, diagnostics,
            std::tuple{"mapmaking", "jinc_filter", "subpixel_n"}, {},
            {1});
    }
    for (const auto &[array_index, array_name] : array_name_map) {
        (void)array_index;
        const auto key = std::tuple{
            "mapmaking", "jinc_filter", "shape_params", array_name};
        const auto current = target.shape_params.find(array_name);
        std::vector<double> shape;
        if (current != target.shape_params.end()) {
            shape.assign(current->second.begin(), current->second.end());
        }
        const auto missing_before = diagnostics.missing_key_paths().size();
        const auto invalid_before = diagnostics.invalid_key_paths().size();
        read_config_value(config, shape, diagnostics, key);
        if (!config_parse_clean(
                diagnostics.missing_key_paths(),
                diagnostics.invalid_key_paths(), missing_before,
                invalid_before)) {
            continue;
        }
        if (shape.size() != 3) {
            add_invalid_config_key(key, diagnostics.invalid_key_paths());
            continue;
        }
        target.shape_params[array_name] = {
            shape[0], shape[1], shape[2]};
    }
}

template <class Config, class Diagnostics>
void read_maximum_likelihood_request_config(
    Config &config,
    citlali::config::MapmakingMaximumLikelihoodConfig &target,
    Diagnostics &diagnostics) {
    read_config_value(
        config, target.tolerance, diagnostics,
        std::tuple{
            "mapmaking", "maximum_likelihood", "tolerance"});
    read_config_value(
        config, target.max_iterations, diagnostics,
        std::tuple{
            "mapmaking", "maximum_likelihood", "max_iterations"});
}

template <class Config, class ArrayNameMap, class Diagnostics>
void read_mapmaking_method_request_config(
    Config &config, citlali::config::MapMethod method,
    const ArrayNameMap &array_name_map,
    citlali::config::MapmakingConfig &target, Diagnostics &diagnostics) {
    if (citlali::config::is_jinc_map_method(method)) {
        read_jinc_filter_request_config(
            config, array_name_map, target.jinc_filter, diagnostics);
    } else if (citlali::config::is_maximum_likelihood_map_method(method)) {
        read_maximum_likelihood_request_config(
            config, target.maximum_likelihood, diagnostics);
    }
}

template <class JincMapmaker, class ArrayNameMap>
void adapt_jinc_filter_config_one_way(
    const citlali::config::MapmakingJincFilterConfig &source,
    const ArrayNameMap &array_name_map, JincMapmaker &target) {
    target.r_max = source.r_max;
    target.subpixel_n = source.subpixel_n;
    target.shape_params.clear();
    if constexpr (requires { target.array_names.clear(); }) {
        target.array_names.clear();
    }
    for (const auto &[array_index, array_name] : array_name_map) {
        const auto &shape = source.shape_params.at(array_name);
        target.shape_params[array_index] = Eigen::Map<const Eigen::VectorXd>(
            shape.data(), static_cast<Eigen::Index>(shape.size()));
        if constexpr (requires { target.array_names[array_index] = array_name; }) {
            target.array_names[array_index] = array_name;
        }
    }
}

template <class MaximumLikelihoodMapmaker>
void adapt_maximum_likelihood_config_one_way(
    const citlali::config::MapmakingMaximumLikelihoodConfig &source,
    MaximumLikelihoodMapmaker &target) {
    target.tolerance = source.tolerance;
    target.max_iterations = source.max_iterations;
}

}  // namespace citlali::pipeline
