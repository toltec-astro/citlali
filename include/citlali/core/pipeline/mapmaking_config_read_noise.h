#pragma once

// Included by mapmaking_config_read.h inside namespace citlali::pipeline {

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseConfig>
void read_noise_maps_enabled_config(Config &config, bool &enabled,
                                    NoiseConfig &typed_config,
                                    MissingKeys &missing_keys,
                                    InvalidKeys &invalid_keys) {
    citlali::pipeline::read_mirrored_config_value(
        config, std::tuple{"noise_maps", "enabled"}, enabled,
        typed_config.enabled, missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class NoiseConfig>
void read_noise_maps_enabled_config(Config &config, bool &enabled,
                                    NoiseConfig &typed_config,
                                    Diagnostics &diagnostics) {
    read_noise_maps_enabled_config(
        config, enabled, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseCount, class NoiseConfig>
void read_noise_map_count_config(Config &config, NoiseCount &n_noise,
                                 NoiseConfig &typed_config,
                                 MissingKeys &missing_keys,
                                 InvalidKeys &invalid_keys) {
    using value_type = std::decay_t<NoiseCount>;
    citlali::pipeline::read_config_value_if_clean(
        config,
        std::tuple{"noise_maps", "n_noise_maps"},
        n_noise,
        [&typed_config](value_type count) {
            typed_config.n_noise_maps = static_cast<int>(count);
        },
        missing_keys, invalid_keys,
        std::vector<value_type>{}, std::vector<value_type>{0},
        std::vector<value_type>{});
}

template <class Config, class Diagnostics, class NoiseCount,
          class NoiseConfig>
void read_noise_map_count_config(Config &config, NoiseCount &n_noise,
                                 NoiseConfig &typed_config,
                                 Diagnostics &diagnostics) {
    read_noise_map_count_config(
        config, n_noise, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseConfig>
void read_noise_randomize_dets_config(Config &config, bool &randomize_dets,
                                      NoiseConfig &typed_config,
                                      MissingKeys &missing_keys,
                                      InvalidKeys &invalid_keys) {
    citlali::pipeline::read_mirrored_config_value(
        config, std::tuple{"noise_maps", "randomize_dets"}, randomize_dets,
        typed_config.randomize_dets, missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class NoiseConfig>
void read_noise_randomize_dets_config(Config &config, bool &randomize_dets,
                                      NoiseConfig &typed_config,
                                      Diagnostics &diagnostics) {
    read_noise_randomize_dets_config(
        config, randomize_dets, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
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
    citlali::pipeline::read_optional_mirrored_config_value(
        config, key, write_realizations, typed_config.write_realizations,
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class NoiseConfig>
void read_noise_write_realizations_config(Config &config,
                                          bool &write_realizations,
                                          NoiseConfig &typed_config,
                                          Diagnostics &diagnostics) {
    read_noise_write_realizations_config(
        config, write_realizations, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
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
    citlali::pipeline::read_optional_mirrored_config_value(
        config, key, products_enabled, typed_config.products_enabled,
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class NoiseConfig>
void read_noise_products_enabled_config(Config &config,
                                        bool &products_enabled,
                                        bool default_enabled,
                                        NoiseConfig &typed_config,
                                        Diagnostics &diagnostics) {
    read_noise_products_enabled_config(
        config, products_enabled, default_enabled, typed_config,
        diagnostics.missing_key_paths(), diagnostics.invalid_key_paths());
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
    citlali::pipeline::read_optional_mirrored_config_value(
        config, key, apply_weights, typed_config.apply_empirical_weights,
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class NoiseConfig>
void read_noise_empirical_weights_config(Config &config,
                                         bool &apply_weights,
                                         bool default_enabled,
                                         NoiseConfig &typed_config,
                                         Diagnostics &diagnostics) {
    read_noise_empirical_weights_config(
        config, apply_weights, default_enabled, typed_config,
        diagnostics.missing_key_paths(), diagnostics.invalid_key_paths());
}
