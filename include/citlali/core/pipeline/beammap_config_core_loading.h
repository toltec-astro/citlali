#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

struct BeammapCoreConfigValues {
    citlali::config::BeammapDirectionMode direction_mode =
        citlali::config::BeammapDirectionMode::standard;
    citlali::config::BeammapIterationConfig iteration;
    citlali::config::BeammapPhaseStrategyConfig phase_strategy;
    citlali::config::BeammapReferenceConfig reference;
    citlali::config::BeammapRfiMaskConfig rfi_mask;
};

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapDirectionMode read_beammap_direction_mode(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    auto value = std::string{citlali::config::to_string(
        citlali::config::BeammapDirectionMode::standard)};
    read_optional_beammap_config_value(
        config, value, missing_keys, invalid_keys,
        std::tuple{"beammap", "direction_mode"},
        {"standard", "left", "right", "all"});
    const auto parsed = citlali::config::parse_beammap_direction_mode(value);
    return parsed.value_or(citlali::config::BeammapDirectionMode::standard);
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

template <class Config, class MissingKeys, class InvalidKeys>
BeammapCoreConfigValues read_beammap_core_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    BeammapCoreConfigValues values;
    values.direction_mode =
        read_beammap_direction_mode(config, missing_keys, invalid_keys);
    values.iteration =
        read_beammap_iteration_config(config, missing_keys, invalid_keys);
    values.phase_strategy =
        read_beammap_phase_strategy_config(config, missing_keys, invalid_keys);
    values.reference =
        read_beammap_reference_config(config, missing_keys, invalid_keys);
    values.rfi_mask =
        read_beammap_rfi_mask_config(config, missing_keys, invalid_keys);
    return values;
}

template <class Config, class Diagnostics>
BeammapCoreConfigValues read_beammap_core_config(
    Config &config, Diagnostics &diagnostics) {
    return read_beammap_core_config(
        config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}
