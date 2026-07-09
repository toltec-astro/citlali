#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

template <class Config, class MissingKeys, class InvalidKeys,
          class MapFitter>
void read_beammap_fitting_config(Config &config, MissingKeys &missing_keys,
                                 InvalidKeys &invalid_keys,
                                 std::string &detector_weighting_mode,
                                 double &fit_radius_fwhm,
                                 MapFitter &map_fitter) {
    detector_weighting_mode = "const";
    read_optional_beammap_config_value(
        config, detector_weighting_mode, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_weighting", "mode"},
        {"const", "ptc", "ptc_after_iter0"});
    fit_radius_fwhm = 0.0;
    read_optional_beammap_config_value(
        config, fit_radius_fwhm, missing_keys, invalid_keys,
        std::tuple{"beammap", "fitting", "fit_radius_fwhm"}, {}, {0.0});
    map_fitter.beammap_fit_radius_fwhm = fit_radius_fwhm;
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_scan_band_mask_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    bool &enabled, int &edge_rows, int &min_row_pixels,
    int &min_contiguous_rows, double &row_median_sigma_threshold,
    double &row_sigma_ratio_threshold, double &max_flagged_fraction) {
    enabled = false;
    edge_rows = 24;
    min_row_pixels = 8;
    min_contiguous_rows = 2;
    row_median_sigma_threshold = 4.0;
    row_sigma_ratio_threshold = 2.5;
    max_flagged_fraction = 0.30;
    read_optional_beammap_config_value(
        config, enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "enabled"});
    read_optional_beammap_config_value(
        config, edge_rows, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "edge_rows"}, {}, {2});
    read_optional_beammap_config_value(
        config, min_row_pixels, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "min_row_pixels"}, {}, {1});
    read_optional_beammap_config_value(
        config, min_contiguous_rows, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "min_contiguous_rows"}, {},
        {1});
    read_optional_beammap_config_value(
        config, row_median_sigma_threshold, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask",
                   "row_median_sigma_threshold"},
        {}, {0.0});
    read_optional_beammap_config_value(
        config, row_sigma_ratio_threshold, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask",
                   "row_sigma_ratio_threshold"},
        {}, {0.0});
    read_optional_beammap_config_value(
        config, max_flagged_fraction, missing_keys, invalid_keys,
        std::tuple{"beammap", "scan_band_mask", "max_flagged_fraction"}, {},
        {0.0}, {1.0});
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

struct BeammapFlaggingVectors {
    std::vector<double> lower_fwhm_arcsec;
    std::vector<double> upper_fwhm_arcsec;
    std::vector<double> lower_sig2noise;
    std::vector<double> upper_sig2noise;
    std::vector<double> max_dist_arcsec;
    std::vector<double> network_robust_z;
    double max_prior_d2 = 0.0;
};

template <class Config, class MissingKeys, class InvalidKeys>
BeammapFlaggingVectors read_beammap_flagging_vectors(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    std::size_t n_arrays) {
    BeammapFlaggingVectors vectors;
    vectors.lower_fwhm_arcsec = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_lower_fwhm_arcsec"},
        n_arrays, invalid_keys);
    vectors.upper_fwhm_arcsec = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_upper_fwhm_arcsec"},
        n_arrays, invalid_keys);
    vectors.lower_sig2noise = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_lower_sig2noise"},
        n_arrays, invalid_keys);
    vectors.upper_sig2noise = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_upper_sig2noise"},
        n_arrays, invalid_keys);
    vectors.max_dist_arcsec = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_max_dist_arcsec"},
        n_arrays, invalid_keys);
    vectors.network_robust_z = beammap_fixed_double_vector(
        config, {"beammap", "flagging", "array_network_robust_z"},
        n_arrays, invalid_keys);
    read_optional_beammap_config_value(
        config, vectors.max_prior_d2, missing_keys, invalid_keys,
        std::tuple{"beammap", "flagging", "max_prior_d2"}, {}, {0.0});
    return vectors;
}

template <class ArrayNameMap, class ValueMap>
void assign_beammap_array_flag_limits(
    const ArrayNameMap &array_name_map,
    const std::vector<double> &lower_fwhm_arcsec_vec,
    const std::vector<double> &upper_fwhm_arcsec_vec,
    const std::vector<double> &lower_sig2noise_vec,
    const std::vector<double> &upper_sig2noise_vec,
    const std::vector<double> &max_dist_arcsec_vec,
    const std::vector<double> &network_robust_z_vec,
    ValueMap &lower_fwhm_arcsec,
    ValueMap &upper_fwhm_arcsec,
    ValueMap &lower_sig2noise,
    ValueMap &upper_sig2noise,
    ValueMap &max_dist_arcsec,
    ValueMap &network_robust_z) {
    std::size_t i = 0;
    for (auto const& [arr_index, arr_name] : array_name_map) {
        (void)arr_index;
        lower_fwhm_arcsec[arr_name] = lower_fwhm_arcsec_vec[i];
        upper_fwhm_arcsec[arr_name] = upper_fwhm_arcsec_vec[i];
        lower_sig2noise[arr_name] = lower_sig2noise_vec[i];
        upper_sig2noise[arr_name] = upper_sig2noise_vec[i];
        max_dist_arcsec[arr_name] = max_dist_arcsec_vec[i];
        network_robust_z[arr_name] = network_robust_z_vec[i];
        ++i;
    }
}
