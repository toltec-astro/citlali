#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

namespace citlali::engine_detail {

template <class Config, class MissingKeys, class InvalidKeys,
          class MapmakingConfig>
void read_mapmaking_enabled_config(Config &config, bool &enabled,
                                   MapmakingConfig &typed_config,
                                   MissingKeys &missing_keys,
                                   InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, enabled, missing_keys, invalid_keys,
                       std::tuple{"mapmaking", "enabled"});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.enabled = enabled;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class MapmakingConfig>
void read_map_grouping_config(Config &config, std::string &grouping,
                              MapmakingConfig &typed_config,
                              MissingKeys &missing_keys,
                              InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(
        config, grouping, missing_keys, invalid_keys,
        std::tuple{"mapmaking", "grouping"},
        {"auto", "array", "nw", "detector", "fg"});
    if (!config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        return;
    }
    if (auto parsed = citlali::config::parse_map_grouping(grouping)) {
        typed_config.grouping = *parsed;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class MapmakingConfig>
void read_map_method_config(Config &config, std::string &method,
                            MapmakingConfig &typed_config,
                            MissingKeys &missing_keys,
                            InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(
        config, method, missing_keys, invalid_keys,
        std::tuple{"mapmaking", "method"},
        {"naive", "jinc", "maximum_likelihood"});
    if (!config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        return;
    }
    if (auto parsed = citlali::config::parse_map_method(method)) {
        typed_config.method = *parsed;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class PixelAxes, class MapmakingConfig>
void read_map_pixel_axes_config(Config &config, PixelAxes &pixel_axes,
                                MapmakingConfig &typed_config,
                                MissingKeys &missing_keys,
                                InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, pixel_axes, missing_keys, invalid_keys,
                       std::tuple{"mapmaking", "pixel_axes"},
                       {"radec", "altaz", "galactic"});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.pixel_axes = pixel_axes;
    }
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_map_regime_config(Config &config, std::string &map_regime,
                            MissingKeys &missing_keys,
                            InvalidKeys &invalid_keys) {
    map_regime = "unknown";
    const auto key = std::tuple{"source", "map_regime"};
    if (!config.template has_typed<std::string>(key)) {
        return;
    }
    map_regime = config.template get_typed<std::string>(key);
    ::check_allowed(map_regime, missing_keys, invalid_keys,
                    citlali::pipeline::allowed_map_regimes(), key);
}

template <class Config, class MissingKeys, class InvalidKeys,
          class CoaddConfig>
void read_coadd_enabled_config(Config &config, bool &enabled,
                               CoaddConfig &typed_config,
                               MissingKeys &missing_keys,
                               InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, enabled, missing_keys, invalid_keys,
                       std::tuple{"coadd", "enabled"});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.enabled = enabled;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseConfig>
void read_noise_maps_enabled_config(Config &config, bool &enabled,
                                    NoiseConfig &typed_config,
                                    MissingKeys &missing_keys,
                                    InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, enabled, missing_keys, invalid_keys,
                       std::tuple{"noise_maps", "enabled"});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.enabled = enabled;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseCount, class NoiseConfig>
void read_noise_map_count_config(Config &config, NoiseCount &n_noise,
                                 NoiseConfig &typed_config,
                                 MissingKeys &missing_keys,
                                 InvalidKeys &invalid_keys) {
    using value_type = std::decay_t<NoiseCount>;
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(
        config, n_noise, missing_keys, invalid_keys,
        std::tuple{"noise_maps", "n_noise_maps"},
        std::vector<value_type>{}, std::vector<value_type>{0},
        std::vector<value_type>{});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.n_noise_maps = static_cast<int>(n_noise);
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseConfig>
void read_noise_randomize_dets_config(Config &config, bool &randomize_dets,
                                      NoiseConfig &typed_config,
                                      MissingKeys &missing_keys,
                                      InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, randomize_dets, missing_keys, invalid_keys,
                       std::tuple{"noise_maps", "randomize_dets"});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.randomize_dets = randomize_dets;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseConfig>
void read_noise_write_realizations_config(Config &config,
                                          bool &write_realizations,
                                          NoiseConfig &typed_config,
                                          MissingKeys &missing_keys,
                                          InvalidKeys &invalid_keys) {
    write_realizations = false;
    const auto key = std::tuple{"noise_maps", "write_realizations"};
    if (!config.template has_typed<bool>(key)) {
        return;
    }
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, write_realizations, missing_keys,
                       invalid_keys, key);
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.write_realizations = write_realizations;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseConfig>
void read_noise_products_enabled_config(Config &config,
                                        bool &products_enabled,
                                        bool default_enabled,
                                        NoiseConfig &typed_config,
                                        MissingKeys &missing_keys,
                                        InvalidKeys &invalid_keys) {
    products_enabled = default_enabled;
    typed_config.products_enabled = products_enabled;
    const auto key = std::tuple{"noise_maps", "products", "enabled"};
    if (!config.template has_typed<bool>(key)) {
        return;
    }
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, products_enabled, missing_keys, invalid_keys,
                       key);
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.products_enabled = products_enabled;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseConfig>
void read_noise_empirical_weights_config(Config &config,
                                         bool &apply_weights,
                                         bool default_enabled,
                                         NoiseConfig &typed_config,
                                         MissingKeys &missing_keys,
                                         InvalidKeys &invalid_keys) {
    apply_weights = default_enabled;
    typed_config.apply_empirical_weights = apply_weights;
    const auto key =
        std::tuple{"noise_maps", "products", "apply_empirical_weights"};
    if (!config.template has_typed<bool>(key)) {
        return;
    }
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, apply_weights, missing_keys, invalid_keys,
                       key);
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.apply_empirical_weights = apply_weights;
    }
}

template <class Config, class OutputMapBlock, class MissingKeys,
          class InvalidKeys, class PixelAxes, class MapmakingConfig,
          class PostProcessingConfig, class Logger>
void read_output_map_block_config(
    Config &config, OutputMapBlock &omb, MissingKeys &missing_keys,
    InvalidKeys &invalid_keys, const PixelAxes &pixel_axes,
    const std::string &redu_type, double rad_to_arcsec,
    MapmakingConfig &typed_mapmaking_config,
    PostProcessingConfig &typed_post_processing_config,
    const Logger &logger) {
    logger->info("getting omb config options");
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    omb.get_config(config, missing_keys, invalid_keys, pixel_axes, redu_type);
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        citlali::pipeline::mirror_output_map_block_config(
            typed_mapmaking_config, omb, rad_to_arcsec,
            typed_post_processing_config);
    }
}

template <class Config, class CoaddMapBlock, class MissingKeys,
          class InvalidKeys, class PixelAxes, class Logger>
void read_coadd_map_block_config(
    Config &config, bool run_coadd, CoaddMapBlock &cmb,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    const PixelAxes &pixel_axes, const std::string &redu_type,
    const Logger &logger) {
    if (!run_coadd) {
        return;
    }
    logger->info("getting cmb config options");
    cmb.get_config(config, missing_keys, invalid_keys, pixel_axes, redu_type);
}

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

}  // namespace citlali::engine_detail
