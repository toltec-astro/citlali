#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_iteration_config(Config &config, MissingKeys &missing_keys,
                                   InvalidKeys &invalid_keys,
                                   int &iter_max,
                                   double &iter_tolerance,
                                   double &convergence_radius_arcsec) {
    ::get_config_value(config, iter_max, missing_keys, invalid_keys,
                       std::tuple{"beammap", "iter_max"});
    ::get_config_value(config, iter_tolerance, missing_keys, invalid_keys,
                       std::tuple{"beammap", "iter_tolerance"});
    convergence_radius_arcsec = 10.0;
    if (config.template has_typed<double>(
            std::tuple{"beammap", "convergence_radius_arcsec"})) {
        ::get_config_value(
            config, convergence_radius_arcsec, missing_keys, invalid_keys,
            std::tuple{"beammap", "convergence_radius_arcsec"}, {}, {0.0});
    }
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_phase_strategy_config(Config &config,
                                        MissingKeys &missing_keys,
                                        InvalidKeys &invalid_keys,
                                        bool &enabled,
                                        int &locator_iter,
                                        int &measurement_start_iter) {
    enabled = true;
    locator_iter = 0;
    measurement_start_iter = 1;
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "phase_strategy", "enabled"})) {
        ::get_config_value(
            config, enabled, missing_keys, invalid_keys,
            std::tuple{"beammap", "phase_strategy", "enabled"});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "phase_strategy", "locator_iter"})) {
        ::get_config_value(
            config, locator_iter, missing_keys, invalid_keys,
            std::tuple{"beammap", "phase_strategy", "locator_iter"}, {}, {0});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "phase_strategy",
                       "measurement_start_iter"})) {
        ::get_config_value(
            config, measurement_start_iter, missing_keys, invalid_keys,
            std::tuple{"beammap", "phase_strategy",
                       "measurement_start_iter"},
            {}, {1});
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class ReferenceDetector>
void read_beammap_reference_config(Config &config, MissingKeys &missing_keys,
                                   InvalidKeys &invalid_keys,
                                   ReferenceDetector &reference_det,
                                   bool &subtract_reference,
                                   bool &derotate) {
    ::get_config_value(config, reference_det, missing_keys, invalid_keys,
                       std::tuple{"beammap", "reference_det"});
    ::get_config_value(config, subtract_reference, missing_keys, invalid_keys,
                       std::tuple{"beammap", "subtract_reference_det"});
    ::get_config_value(config, derotate, missing_keys, invalid_keys,
                       std::tuple{"beammap", "derotate"});
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_rfi_mask_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    bool &enabled, int &block_size_samples, int &min_good_samples,
    int &dilate_blocks, double &sigma_threshold, double &sigma_floor,
    double &max_flagged_fraction) {
    enabled = false;
    block_size_samples = 64;
    min_good_samples = 32;
    dilate_blocks = 1;
    sigma_threshold = 6.0;
    sigma_floor = 0.0;
    max_flagged_fraction = 0.35;
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "rfi_mask", "enabled"})) {
        ::get_config_value(config, enabled, missing_keys, invalid_keys,
                           std::tuple{"beammap", "rfi_mask", "enabled"});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "rfi_mask", "block_size_samples"})) {
        ::get_config_value(
            config, block_size_samples, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "block_size_samples"}, {}, {8});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "rfi_mask", "min_good_samples"})) {
        ::get_config_value(
            config, min_good_samples, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "min_good_samples"}, {}, {4});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "rfi_mask", "dilate_blocks"})) {
        ::get_config_value(
            config, dilate_blocks, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "dilate_blocks"}, {}, {0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "rfi_mask", "sigma_threshold"})) {
        ::get_config_value(
            config, sigma_threshold, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "sigma_threshold"}, {}, {1.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "rfi_mask", "sigma_floor"})) {
        ::get_config_value(
            config, sigma_floor, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "sigma_floor"}, {}, {0.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "rfi_mask", "max_flagged_fraction"})) {
        ::get_config_value(
            config, max_flagged_fraction, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "max_flagged_fraction"}, {},
            {0.0}, {1.0});
    }
}
