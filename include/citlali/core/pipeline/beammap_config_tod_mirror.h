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

inline int default_beammap_tod_output_iter() {
    return -1;
}

template <class MapFitter>
void sync_beammap_map_fitter(
    const BeammapFittingConfigValues &fitting_values, MapFitter &map_fitter) {
    map_fitter.beammap_fit_radius_fwhm =
        fitting_values.fitting.fit_radius_fwhm;
}

template <class BeammapControls, class ArrayNameMap>
void sync_beammap_flagging_controls(
    BeammapControls &controls,
    const citlali::config::BeammapFlaggingConfig &flagging,
    const BeammapSensitivityConfigValues &sensitivity,
    const ArrayNameMap &array_name_map) {
    assign_beammap_array_flag_limits(
        array_name_map, flagging, controls.lower_fwhm_arcsec,
        controls.upper_fwhm_arcsec, controls.lower_sig2noise,
        controls.upper_sig2noise, controls.max_dist_arcsec,
        controls.network_robust_z);
    controls.lower_sens_factor = sensitivity.sens_factors[0];
    controls.upper_sens_factor = sensitivity.sens_factors[1];
    controls.sens_psd_limits_Hz.resize(
        static_cast<Eigen::Index>(sensitivity.sens_psd_limits_hz.size()));
    controls.sens_psd_limits_Hz =
        Eigen::Map<const Eigen::VectorXd>(
            sensitivity.sens_psd_limits_hz.data(),
            static_cast<Eigen::Index>(
                sensitivity.sens_psd_limits_hz.size()));
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
