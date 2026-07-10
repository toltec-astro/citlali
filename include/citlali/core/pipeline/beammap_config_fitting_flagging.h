#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

struct BeammapFittingConfigValues {
    citlali::config::BeammapDetectorWeightingMode detector_weighting_mode =
        citlali::config::BeammapDetectorWeightingMode::constant;
    citlali::config::BeammapFittingConfig fitting;
};

template <class Config, class MissingKeys, class InvalidKeys>
BeammapFittingConfigValues read_beammap_fitting_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    BeammapFittingConfigValues values;
    std::string detector_weighting_mode_name =
        std::string(citlali::config::to_string(values.detector_weighting_mode));
    read_optional_beammap_config_value(
        config, detector_weighting_mode_name, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_weighting", "mode"},
        {"const", "ptc", "ptc_after_iter0"});
    if (auto parsed = citlali::config::parse_beammap_detector_weighting_mode(
            detector_weighting_mode_name)) {
        values.detector_weighting_mode = *parsed;
    }
    read_optional_beammap_config_value(
        config, values.fitting.fit_radius_fwhm, missing_keys, invalid_keys,
        std::tuple{"beammap", "fitting", "fit_radius_fwhm"}, {}, {0.0});
    return values;
}

template <class Config, class Diagnostics>
BeammapFittingConfigValues read_beammap_fitting_config(
    Config &config, Diagnostics &diagnostics) {
    return read_beammap_fitting_config(
        config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapScanBandMaskConfig
read_beammap_scan_band_mask_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    citlali::config::BeammapScanBandMaskConfig values;
    read_optional_beammap_config_value(
        config, values.enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "enabled"});
    read_optional_beammap_config_value(
        config, values.edge_rows, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "edge_rows"}, {}, {2});
    read_optional_beammap_config_value(
        config, values.min_row_pixels, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "min_row_pixels"}, {}, {1});
    read_optional_beammap_config_value(
        config, values.min_contiguous_rows, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "min_contiguous_rows"}, {},
        {1});
    read_optional_beammap_config_value(
        config, values.row_median_sigma_threshold, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask",
                   "row_median_sigma_threshold"},
        {}, {0.0});
    read_optional_beammap_config_value(
        config, values.row_sigma_ratio_threshold, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask",
                   "row_sigma_ratio_threshold"},
        {}, {0.0});
    read_optional_beammap_config_value(
        config, values.max_flagged_fraction, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "max_flagged_fraction"}, {},
        {0.0}, {1.0});
    return values;
}

template <class Config, class Diagnostics>
citlali::config::BeammapScanBandMaskConfig
read_beammap_scan_band_mask_config(Config &config, Diagnostics &diagnostics) {
    return read_beammap_scan_band_mask_config(
        config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class InvalidKeys>
std::vector<double> beammap_fixed_double_vector(
    Config &config, const std::vector<std::string> &path,
    std::size_t expected_size, InvalidKeys &invalid_keys) {
    std::vector<double> values;
    if (path.size() == 2) {
        values = config.template get_typed<std::vector<double>>(
            std::make_tuple(path[0], path[1]));
    }
    else {
        values = config.template get_typed<std::vector<double>>(
            std::make_tuple(path[0], path[1], path[2]));
    }
    if (values.size() != expected_size) {
        invalid_keys.push_back(path);
        values.resize(expected_size, 0.0);
    }
    return values;
}

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapFlaggingConfig read_beammap_flagging_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    std::size_t n_arrays) {
    citlali::config::BeammapFlaggingConfig values;
    values.array_lower_fwhm_arcsec = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_lower_fwhm_arcsec"},
        n_arrays, invalid_keys);
    values.array_upper_fwhm_arcsec = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_upper_fwhm_arcsec"},
        n_arrays, invalid_keys);
    values.array_lower_sig2noise = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_lower_sig2noise"},
        n_arrays, invalid_keys);
    values.array_upper_sig2noise = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_upper_sig2noise"},
        n_arrays, invalid_keys);
    values.array_max_dist_arcsec = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_max_dist_arcsec"},
        n_arrays, invalid_keys);
    values.array_network_robust_z = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_network_robust_z"},
        n_arrays, invalid_keys);
    read_optional_beammap_config_value(
        config, values.max_prior_d2, missing_keys, invalid_keys,
        std::tuple{"beammap", "flagging", "max_prior_d2"}, {}, {0.0});
    return values;
}

template <class Config, class Diagnostics>
citlali::config::BeammapFlaggingConfig read_beammap_flagging_config(
    Config &config, Diagnostics &diagnostics, std::size_t n_arrays) {
    return read_beammap_flagging_config(
        config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths(), n_arrays);
}

template <class ArrayNameMap, class ValueMap>
void assign_beammap_array_flag_limits(
    const ArrayNameMap &array_name_map,
    const citlali::config::BeammapFlaggingConfig &flagging,
    ValueMap &lower_fwhm_arcsec,
    ValueMap &upper_fwhm_arcsec,
    ValueMap &lower_sig2noise,
    ValueMap &upper_sig2noise,
    ValueMap &max_dist_arcsec,
    ValueMap &network_robust_z) {
    std::size_t i = 0;
    for (auto const& [arr_index, arr_name] : array_name_map) {
        (void)arr_index;
        lower_fwhm_arcsec[arr_name] =
            flagging.array_lower_fwhm_arcsec[i];
        upper_fwhm_arcsec[arr_name] =
            flagging.array_upper_fwhm_arcsec[i];
        lower_sig2noise[arr_name] =
            flagging.array_lower_sig2noise[i];
        upper_sig2noise[arr_name] =
            flagging.array_upper_sig2noise[i];
        max_dist_arcsec[arr_name] =
            flagging.array_max_dist_arcsec[i];
        network_robust_z[arr_name] =
            flagging.array_network_robust_z[i];
        ++i;
    }
}

struct BeammapArrayFlaggingLimits {
    std::map<std::string, double> lower_fwhm_arcsec;
    std::map<std::string, double> upper_fwhm_arcsec;
    std::map<std::string, double> lower_sig2noise;
    std::map<std::string, double> upper_sig2noise;
    std::map<std::string, double> max_dist_arcsec;
    std::map<std::string, double> network_robust_z;
};

template <class ArrayNameMap>
BeammapArrayFlaggingLimits make_beammap_array_flagging_limits(
    const ArrayNameMap &array_name_map,
    const citlali::config::BeammapFlaggingConfig &flagging) {
    BeammapArrayFlaggingLimits limits;
    assign_beammap_array_flag_limits(
        array_name_map, flagging, limits.lower_fwhm_arcsec,
        limits.upper_fwhm_arcsec, limits.lower_sig2noise,
        limits.upper_sig2noise, limits.max_dist_arcsec,
        limits.network_robust_z);
    return limits;
}
