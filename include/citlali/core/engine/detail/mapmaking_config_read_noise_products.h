#pragma once

// Included by mapmaking_config_read.h inside namespace citlali::engine_detail {

template <class Config, class OutputMapBlock, class CoaddMapBlock,
          class CoaddConfig,
          class MissingKeys, class InvalidKeys, class NoiseConfig>
void read_noise_map_config(Config &config, bool &run_noise,
                           const CoaddConfig &typed_coadd_config,
                           OutputMapBlock &omb, CoaddMapBlock &cmb,
                           NoiseConfig &typed_config,
                           MissingKeys &missing_keys,
                           InvalidKeys &invalid_keys) {
    read_noise_maps_enabled_config(
        config, run_noise, typed_config, missing_keys, invalid_keys);
    if (!citlali::config::noise_maps_active(typed_config)) {
        citlali::pipeline::disable_noise_map_settings(omb, cmb, typed_config);
        return;
    }
    read_noise_map_count_config(
        config, omb.n_noise, typed_config, missing_keys, invalid_keys);
    read_noise_randomize_dets_config(
        config, omb.randomize_dets, typed_config, missing_keys,
        invalid_keys);
    if (citlali::config::coadd_active(typed_coadd_config)) {
        citlali::pipeline::mirror_noise_map_settings_to_coadd(omb, cmb);
    }
}

template <class Config, class OutputMapBlock, class CoaddMapBlock,
          class CoaddConfig, class Diagnostics, class NoiseConfig>
void read_noise_map_config(Config &config, bool &run_noise,
                           const CoaddConfig &typed_coadd_config,
                           OutputMapBlock &omb, CoaddMapBlock &cmb,
                           NoiseConfig &typed_config,
                           Diagnostics &diagnostics) {
    read_noise_map_config(
        config, run_noise, typed_coadd_config, omb, cmb, typed_config,
        diagnostics.missing_key_paths(), diagnostics.invalid_key_paths());
}

template <class Config, class MissingKeys, class InvalidKeys,
          class NoiseConfig>
void read_noise_product_config(Config &config, bool &write_realizations,
                               bool &products_enabled,
                               bool &apply_empirical_weights,
                               NoiseConfig &typed_config,
                               MissingKeys &missing_keys,
                               InvalidKeys &invalid_keys) {
    read_noise_write_realizations_config(
        config, write_realizations, typed_config, missing_keys, invalid_keys);
    read_noise_products_enabled_config(
        config, products_enabled,
        citlali::config::noise_maps_active(typed_config), typed_config,
        missing_keys, invalid_keys);
    read_noise_empirical_weights_config(
        config, apply_empirical_weights,
        citlali::config::noise_maps_active(typed_config), typed_config,
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class NoiseConfig>
void read_noise_product_config(Config &config, bool &write_realizations,
                               bool &products_enabled,
                               bool &apply_empirical_weights,
                               NoiseConfig &typed_config,
                               Diagnostics &diagnostics) {
    read_noise_product_config(
        config, write_realizations, products_enabled, apply_empirical_weights,
        typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}
