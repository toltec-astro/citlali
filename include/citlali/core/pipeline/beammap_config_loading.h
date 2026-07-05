#pragma once

#include <citlali/core/config/beammap_config.h>

#include <cstddef>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

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

inline std::vector<int> normalized_beammap_split_flag_values(
    std::vector<int> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

inline void mirror_beammap_core_config(
    citlali::config::BeammapConfig &target,
    int iter_max, double iter_tolerance, double convergence_radius_arcsec,
    bool phase_split_enabled, int locator_iter, int measurement_start_iter,
    bool subtract_reference, long reference_det, bool derotate,
    bool rfi_mask_enabled, int rfi_mask_block_size_samples,
    int rfi_mask_min_good_samples, int rfi_mask_dilate_blocks,
    double rfi_mask_sigma_threshold, double rfi_mask_sigma_floor,
    double rfi_mask_max_flagged_fraction,
    const std::string &detector_weighting_mode,
    double fit_radius_fwhm,
    bool scan_band_mask_enabled, int scan_band_mask_edge_rows,
    int scan_band_mask_min_row_pixels,
    int scan_band_mask_min_contiguous_rows,
    double scan_band_mask_row_median_sigma_threshold,
    double scan_band_mask_row_sigma_ratio_threshold,
    double scan_band_mask_max_flagged_fraction,
    bool split_fits_by_flag,
    const std::vector<int> &split_flag_values) {
    target.iteration.max_iterations = iter_max;
    target.iteration.tolerance = iter_tolerance;
    target.iteration.convergence_radius_arcsec = convergence_radius_arcsec;
    target.phase_strategy.enabled = phase_split_enabled;
    target.phase_strategy.locator_iter = locator_iter;
    target.phase_strategy.measurement_start_iter = measurement_start_iter;
    target.reference.subtract_reference_detector = subtract_reference;
    target.reference.reference_detector = reference_det;
    target.reference.derotate = derotate;
    target.rfi_mask.enabled = rfi_mask_enabled;
    target.rfi_mask.block_size_samples = rfi_mask_block_size_samples;
    target.rfi_mask.min_good_samples = rfi_mask_min_good_samples;
    target.rfi_mask.dilate_blocks = rfi_mask_dilate_blocks;
    target.rfi_mask.sigma_threshold = rfi_mask_sigma_threshold;
    target.rfi_mask.sigma_floor = rfi_mask_sigma_floor;
    target.rfi_mask.max_flagged_fraction = rfi_mask_max_flagged_fraction;
    if (auto parsed = citlali::config::parse_beammap_detector_weighting_mode(
            detector_weighting_mode)) {
        target.detector_weighting_mode = *parsed;
    }
    target.fitting.fit_radius_fwhm = fit_radius_fwhm;
    target.scan_band_mask.enabled = scan_band_mask_enabled;
    target.scan_band_mask.edge_rows = scan_band_mask_edge_rows;
    target.scan_band_mask.min_row_pixels = scan_band_mask_min_row_pixels;
    target.scan_band_mask.min_contiguous_rows =
        scan_band_mask_min_contiguous_rows;
    target.scan_band_mask.row_median_sigma_threshold =
        scan_band_mask_row_median_sigma_threshold;
    target.scan_band_mask.row_sigma_ratio_threshold =
        scan_band_mask_row_sigma_ratio_threshold;
    target.scan_band_mask.max_flagged_fraction =
        scan_band_mask_max_flagged_fraction;
    target.split_fits_by_flag.enabled = split_fits_by_flag;
    target.split_fits_by_flag.flag_values = split_flag_values;
}

}  // namespace citlali::pipeline
