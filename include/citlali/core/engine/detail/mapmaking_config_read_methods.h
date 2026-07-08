#pragma once

// Included by mapmaking_config_read.h inside namespace citlali::engine_detail {

template <class Config, class JincMapmaker, class ArrayNameMap,
          class MissingKeys, class InvalidKeys>
void read_jinc_filter_config(Config &config, JincMapmaker &jinc_mm,
                             const ArrayNameMap &array_name_map,
                             MissingKeys &missing_keys,
                             InvalidKeys &invalid_keys) {
    ::get_config_value(config, jinc_mm.r_max, missing_keys, invalid_keys,
                       std::tuple{"mapmaking", "jinc_filter", "r_max"});
    for (auto const& [arr_index, arr_name] : array_name_map) {
        auto shape = config.template get_typed<std::vector<double>>(
            std::tuple{"mapmaking", "jinc_filter", "shape_params",
                       arr_name});
        if (shape.size() != 3) {
            invalid_keys.push_back(
                {"mapmaking", "jinc_filter", "shape_params", arr_name});
            shape.resize(3, 0.0);
        }
        jinc_mm.shape_params[arr_index] =
            Eigen::Map<Eigen::VectorXd>(shape.data(), shape.size());
    }
    if (config.template has_typed<int>(
            std::tuple{"mapmaking", "jinc_filter", "subpixel_n"})) {
        ::get_config_value(
            config, jinc_mm.subpixel_n, missing_keys, invalid_keys,
            std::tuple{"mapmaking", "jinc_filter", "subpixel_n"}, {}, {1});
    }
}

template <class Config, class MaximumLikelihoodMapmaker, class MissingKeys,
          class InvalidKeys>
void read_maximum_likelihood_mapmaker_config(
    Config &config, MaximumLikelihoodMapmaker &ml_mm,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    ::get_config_value(
        config, ml_mm.tolerance, missing_keys, invalid_keys,
        std::tuple{"mapmaking", "maximum_likelihood", "tolerance"});
    ::get_config_value(
        config, ml_mm.max_iterations, missing_keys, invalid_keys,
        std::tuple{"mapmaking", "maximum_likelihood", "max_iterations"});
}

template <class Config, class JincMapmaker, class MaximumLikelihoodMapmaker,
          class ArrayNameMap, class PtcProc, class MissingKeys,
          class InvalidKeys>
void read_method_specific_mapmaker_config(
    Config &config, citlali::config::MapMethod map_method,
    JincMapmaker &jinc_mm,
    MaximumLikelihoodMapmaker &ml_mm, const ArrayNameMap &array_name_map,
    PtcProc &ptcproc, double pixel_size_rad, MissingKeys &missing_keys,
    InvalidKeys &invalid_keys) {
    if (citlali::config::is_jinc_map_method(map_method)) {
        read_jinc_filter_config(
            config, jinc_mm, array_name_map, missing_keys, invalid_keys);
        citlali::pipeline::finalize_jinc_filter_config(
            jinc_mm, ptcproc, pixel_size_rad);
    }
    else if (citlali::config::is_maximum_likelihood_map_method(map_method)) {
        read_maximum_likelihood_mapmaker_config(
            config, ml_mm, missing_keys, invalid_keys);
    }
}
