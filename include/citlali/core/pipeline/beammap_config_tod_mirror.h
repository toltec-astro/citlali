#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapDetectorTodOutputConfig
read_beammap_detector_tod_output_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    citlali::config::BeammapDetectorTodOutputConfig values;
    read_optional_beammap_config_value(
        config, values.enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "enabled"});
    read_optional_beammap_config_value(
        config, values.subdir_name, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "subdir_name"});
    read_optional_beammap_config_value(
        config, values.n_uniform, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "n_uniform"}, {}, {0});
    read_optional_beammap_config_value(
        config, values.n_source_dense, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "n_source_dense"},
        {}, {0});
    return values;
}

template <class Config, class Diagnostics>
citlali::config::BeammapDetectorTodOutputConfig
read_beammap_detector_tod_output_config(
    Config &config, Diagnostics &diagnostics) {
    return read_beammap_detector_tod_output_config(
        config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

inline int default_beammap_tod_output_iter() {
    return -1;
}

template <class MapFitter>
void sync_beammap_map_fitter(
    const citlali::config::BeammapFittingConfig &fitting,
    MapFitter &map_fitter) {
    map_fitter.beammap_fit_radius_fwhm =
        fitting.fit_radius_fwhm;
}

inline void apply_beammap_typed_config(
    citlali::config::BeammapConfig &target,
    const BeammapCoreConfigValues &core_values,
    const BeammapFittingConfigValues &fitting_values,
    const citlali::config::BeammapScanBandMaskConfig &scan_band_mask,
    const citlali::config::BeammapSplitFitsByFlagConfig &split_fits_by_flag,
    const BeammapPriorsConfigValues &priors,
    const citlali::config::BeammapDetectorTodOutputConfig &detector_tod_output,
    const citlali::config::BeammapFlaggingConfig &flagging,
    const BeammapSensitivityConfigValues &sensitivity) {
    target = citlali::config::BeammapConfig{};
    target.direction_mode = core_values.direction_mode;
    target.iteration = core_values.iteration;
    target.phase_strategy = core_values.phase_strategy;
    target.reference = core_values.reference;
    target.rfi_mask = core_values.rfi_mask;
    target.detector_weighting_mode = fitting_values.detector_weighting_mode;
    target.fitting = fitting_values.fitting;
    target.scan_band_mask = scan_band_mask;
    target.split_fits_by_flag = split_fits_by_flag;
    target.priors = priors;
    target.detector_tod_output = detector_tod_output;
    target.flagging = flagging;
    target.flagging.sens_factors = sensitivity.sens_factors;
    target.flagging.sens_psd_limits_hz = sensitivity.sens_psd_limits_hz;
}
