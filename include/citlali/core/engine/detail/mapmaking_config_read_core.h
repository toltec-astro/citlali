#pragma once

// Included by mapmaking_config_read.h inside namespace citlali::engine_detail {

template <class Config, class MissingKeys, class InvalidKeys,
          class MapmakingConfig>
void read_mapmaking_enabled_config(Config &config, bool &enabled,
                                   MapmakingConfig &typed_config,
                                   MissingKeys &missing_keys,
                                   InvalidKeys &invalid_keys) {
    read_mirrored_config_value(
        config, std::tuple{"mapmaking", "enabled"}, enabled,
        typed_config.enabled, missing_keys, invalid_keys);
}

template <class Config, class MissingKeys, class InvalidKeys,
          class MapmakingConfig>
void read_map_grouping_config(Config &config, std::string &grouping,
                              MapmakingConfig &typed_config,
                              MissingKeys &missing_keys,
                              InvalidKeys &invalid_keys) {
    read_parsed_mirrored_config_value(
        config, std::tuple{"mapmaking", "grouping"}, grouping,
        typed_config.grouping, citlali::config::parse_map_grouping,
        missing_keys, invalid_keys, {"auto", "array", "nw", "detector", "fg"});
}

template <class Config, class MissingKeys, class InvalidKeys,
          class MapmakingConfig>
void read_map_method_config(Config &config, std::string &method,
                            MapmakingConfig &typed_config,
                            MissingKeys &missing_keys,
                            InvalidKeys &invalid_keys) {
    read_parsed_mirrored_config_value(
        config, std::tuple{"mapmaking", "method"}, method,
        typed_config.method, citlali::config::parse_map_method,
        missing_keys, invalid_keys, {"naive", "jinc", "maximum_likelihood"});
}

template <class Config, class MissingKeys, class InvalidKeys,
          class PixelAxes, class MapmakingConfig>
void read_map_pixel_axes_config(Config &config, PixelAxes &pixel_axes,
                                MapmakingConfig &typed_config,
                                MissingKeys &missing_keys,
                                InvalidKeys &invalid_keys) {
    read_config_value_if_clean(
        config, std::tuple{"mapmaking", "pixel_axes"}, pixel_axes,
        [&typed_config](const auto &value) {
            typed_config.pixel_axes = value;
            if (auto parsed = citlali::config::parse_map_pixel_axes(value)) {
                typed_config.pixel_axes_frame = *parsed;
            }
        },
        missing_keys, invalid_keys, {"radec", "altaz", "galactic"});
}

template <class Config, class MissingKeys, class InvalidKeys,
          class MapmakingConfig>
void read_map_regime_config(Config &config, std::string &map_regime,
                            MapmakingConfig &typed_config,
                            MissingKeys &missing_keys,
                            InvalidKeys &invalid_keys) {
    map_regime = "unknown";
    typed_config.source_map_regime = map_regime;
    const auto key = std::tuple{"source", "map_regime"};
    if (!config.template has_typed<std::string>(key)) {
        return;
    }
    map_regime = config.template get_typed<std::string>(key);
    typed_config.source_map_regime = map_regime;
    ::check_allowed(map_regime, missing_keys, invalid_keys,
                    citlali::pipeline::allowed_map_regimes(), key);
}

template <class Config, class MissingKeys, class InvalidKeys,
          class CoaddConfig>
void read_coadd_enabled_config(Config &config, bool &enabled,
                               CoaddConfig &typed_config,
                               MissingKeys &missing_keys,
                               InvalidKeys &invalid_keys) {
    read_mirrored_config_value(
        config, std::tuple{"coadd", "enabled"}, enabled,
        typed_config.enabled, missing_keys, invalid_keys);
}
