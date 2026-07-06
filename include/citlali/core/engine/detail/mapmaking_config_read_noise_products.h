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
    if (!typed_config.enabled) {
        citlali::pipeline::disable_noise_map_settings(omb, cmb, typed_config);
        return;
    }
    read_noise_map_count_config(
        config, omb.n_noise, typed_config, missing_keys, invalid_keys);
    read_noise_randomize_dets_config(
        config, omb.randomize_dets, typed_config, missing_keys,
        invalid_keys);
    if (typed_coadd_config.enabled) {
        citlali::pipeline::mirror_noise_map_settings_to_coadd(omb, cmb);
    }
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
        config, products_enabled, typed_config.enabled, typed_config,
        missing_keys, invalid_keys);
    read_noise_empirical_weights_config(
        config, apply_empirical_weights, typed_config.enabled, typed_config,
        missing_keys, invalid_keys);
}
