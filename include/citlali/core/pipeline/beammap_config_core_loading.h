#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

struct BeammapCoreConfigValues {
    citlali::config::BeammapIterationConfig iteration;
    citlali::config::BeammapPhaseStrategyConfig phase_strategy;
    citlali::config::BeammapReferenceConfig reference;
    citlali::config::BeammapRfiMaskConfig rfi_mask;
};

template <class Logger>
void normalize_beammap_phase_strategy(int iter_max, int &locator_iter,
                                      int &measurement_start_iter,
                                      const Logger &logger) {
    if (locator_iter != 0) {
        logger->warn(
            "beammap.phase_strategy.locator_iter={} requested, but the locator pass must be iter 0; using 0",
            locator_iter);
        locator_iter = 0;
    }
    if (measurement_start_iter <= locator_iter) {
        logger->warn(
            "beammap.phase_strategy.measurement_start_iter={} must be after locator_iter={}; using {}",
            measurement_start_iter, locator_iter, locator_iter + 1);
        measurement_start_iter = locator_iter + 1;
    }
    if (iter_max <= measurement_start_iter) {
        logger->warn(
            "beammap.iter_max={} will not run a measurement pass with measurement_start_iter={}",
            iter_max, measurement_start_iter);
    }
}

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapIterationConfig read_beammap_iteration_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    citlali::config::BeammapIterationConfig values;
    ::get_config_value(config, values.max_iterations, missing_keys, invalid_keys,
                       std::tuple{"beammap", "iter_max"});
    ::get_config_value(config, values.tolerance, missing_keys, invalid_keys,
                       std::tuple{"beammap", "iter_tolerance"});
    read_optional_beammap_config_value(
        config, values.convergence_radius_arcsec, missing_keys, invalid_keys,
        std::tuple{"beammap", "convergence_radius_arcsec"}, {}, {0.0});
    return values;
}

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapPhaseStrategyConfig read_beammap_phase_strategy_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    citlali::config::BeammapPhaseStrategyConfig values;
    read_optional_beammap_config_value(
        config, values.enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "phase_strategy", "enabled"});
    read_optional_beammap_config_value(
        config, values.locator_iter, missing_keys, invalid_keys,
        std::tuple{"beammap", "phase_strategy", "locator_iter"}, {}, {0});
    read_optional_beammap_config_value(
        config, values.measurement_start_iter, missing_keys, invalid_keys,
        std::tuple{"beammap", "phase_strategy", "measurement_start_iter"},
        {}, {1});
    return values;
}

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapReferenceConfig read_beammap_reference_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    citlali::config::BeammapReferenceConfig values;
    ::get_config_value(config, values.reference_detector, missing_keys, invalid_keys,
                       std::tuple{"beammap", "reference_det"});
    ::get_config_value(config, values.subtract_reference_detector,
                       missing_keys, invalid_keys,
                       std::tuple{"beammap", "subtract_reference_det"});
    ::get_config_value(config, values.derotate, missing_keys, invalid_keys,
                       std::tuple{"beammap", "derotate"});
    return values;
}

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapRfiMaskConfig read_beammap_rfi_mask_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    citlali::config::BeammapRfiMaskConfig values;
    read_optional_beammap_config_value(
        config, values.enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "rfi_mask", "enabled"});
    read_optional_beammap_config_value(
        config, values.block_size_samples, missing_keys, invalid_keys,
        std::tuple{"beammap", "rfi_mask", "block_size_samples"}, {}, {8});
    read_optional_beammap_config_value(
        config, values.min_good_samples, missing_keys, invalid_keys,
        std::tuple{"beammap", "rfi_mask", "min_good_samples"}, {}, {4});
    read_optional_beammap_config_value(
        config, values.dilate_blocks, missing_keys, invalid_keys,
        std::tuple{"beammap", "rfi_mask", "dilate_blocks"}, {}, {0});
    read_optional_beammap_config_value(
        config, values.sigma_threshold, missing_keys, invalid_keys,
        std::tuple{"beammap", "rfi_mask", "sigma_threshold"}, {}, {1.0});
    read_optional_beammap_config_value(
        config, values.sigma_floor, missing_keys, invalid_keys,
        std::tuple{"beammap", "rfi_mask", "sigma_floor"}, {}, {0.0});
    read_optional_beammap_config_value(
        config, values.max_flagged_fraction, missing_keys, invalid_keys,
        std::tuple{"beammap", "rfi_mask", "max_flagged_fraction"}, {},
        {0.0}, {1.0});
    return values;
}

template <class Config, class MissingKeys, class InvalidKeys, class Logger>
BeammapCoreConfigValues read_beammap_core_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    const Logger &logger) {
    BeammapCoreConfigValues values;
    values.iteration =
        read_beammap_iteration_config(config, missing_keys, invalid_keys);
    values.phase_strategy =
        read_beammap_phase_strategy_config(config, missing_keys, invalid_keys);
    normalize_beammap_phase_strategy(
        values.iteration.max_iterations, values.phase_strategy.locator_iter,
        values.phase_strategy.measurement_start_iter, logger);
    values.reference =
        read_beammap_reference_config(config, missing_keys, invalid_keys);
    values.rfi_mask =
        read_beammap_rfi_mask_config(config, missing_keys, invalid_keys);
    return values;
}
