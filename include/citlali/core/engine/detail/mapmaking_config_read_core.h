#pragma once

// Included by mapmaking_config_read.h inside namespace citlali::engine_detail {

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

