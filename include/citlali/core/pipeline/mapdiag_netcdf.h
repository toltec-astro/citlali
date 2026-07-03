#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <netcdf>

namespace citlali::pipeline {

inline double mapdiag_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr int mapdiag_fill_int() {
    return -2147483647;
}

struct MapdiagSizeContext {
    std::size_t n_maps;
    std::size_t n_obsnums;
    bool is_coadd;
};

MapdiagSizeContext make_mapdiag_size_context(std::size_t n_maps,
                                             std::size_t obsnum_count,
                                             bool is_coadd) {
    return {n_maps, obsnum_count, is_coadd};
}

inline std::size_t mapdiag_obs_table_size(const MapdiagSizeContext &context) {
    return context.n_maps * context.n_obsnums;
}

inline void put_netcdf_string_1d(
    netCDF::NcFile &fo, const std::string &name, netCDF::NcDim dim,
    const std::vector<std::string> &values,
    const std::string &comment = "") {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncString, dim);
    if (!comment.empty()) {
        v.putAtt("comment", comment);
    }
    for (std::size_t i=0; i<values.size(); ++i) {
        const std::vector<std::size_t> idx = {i};
        std::string value = values[i];
        v.putVar(idx, value);
    }
}

inline void add_mapdiag_double_1d(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, netCDF::NcDim dim,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, dim);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void add_mapdiag_int_1d(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, netCDF::NcDim dim,
    const std::vector<int> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, dim);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void add_mapdiag_double_2d(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &dims,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, dims);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void add_mapdiag_int_2d(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &dims,
    const std::vector<int> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, dims);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

struct MapdiagMapDoubleValues {
    const std::vector<double> &median_err;
    const std::vector<double> &median_rms;
    const std::vector<double> &weight_thresholds;
    const std::vector<double> &weight_sum;
    const std::vector<double> &core_weight_sum;
    const std::vector<double> &coverage_sum;
    const std::vector<double> &coverage_max;
    const std::vector<double> &coverage_median_core;
    const std::vector<double> &empirical_to_formal_noise_ratio;
    const std::vector<double> &noise_weight_median_ratio;
    const std::vector<double> &noise_weight_scale;
    const std::vector<double> &noise_products_s2n_sigma;
    const std::vector<double> &noise_products_valid_pixels;
    const std::vector<double> &peak_signal;
    const std::vector<double> &peak_abs_sig2noise;
    const std::vector<double> &core_peak_abs_sig2noise;
    const std::vector<double> &noise_rms_p16;
    const std::vector<double> &noise_rms_p84;
    const std::vector<double> &core_tail_frac_abs3;
    const std::vector<double> &core_tail_frac_pos3;
    const std::vector<double> &core_tail_frac_neg3;
    const std::vector<double> &core_tail_excess_abs3;
    const std::vector<double> &core_tail_excess_pos3;
    const std::vector<double> &core_tail_excess_neg3;
    const std::vector<double> &core_sig2noise_skew;
    const std::vector<double> &noise_tail_frac_abs3;
    const std::vector<double> &noise_tail_frac_pos3;
    const std::vector<double> &noise_tail_frac_neg3;
    const std::vector<double> &noise_tail_excess_abs3;
    const std::vector<double> &noise_tail_excess_pos3;
    const std::vector<double> &noise_tail_excess_neg3;
    const std::vector<double> &noise_sig2noise_skew;
    const std::vector<double> &edge_guard_weight_thresholds;
    const std::vector<double> &edge_guard_hits_thresholds;
    const std::vector<double> &edge_guard_background_levels;
    const std::vector<double> &edge_guard_science_frac;
    const std::vector<double> &edge_guard_support_frac;
    const std::vector<double> &edge_guard_guardband_rms_pre;
    const std::vector<double> &edge_guard_guardband_rms_post;
    const std::vector<double> &edge_guard_exterior_rms_pre;
    const std::vector<double> &edge_guard_exterior_rms_post;
    const std::vector<double> &edge_guard_exterior_max_abs_pre;
    const std::vector<double> &edge_guard_exterior_max_abs_post;
};

template <class AddDouble>
void add_mapdiag_map_double_vars(
    const AddDouble &add_double, const MapdiagMapDoubleValues &values) {
    add_double("map_median_err",
               "median error derived from the map weight product",
               values.median_err);
    add_double("map_median_rms",
               "median RMS of the map noise realization or background estimator",
               values.median_rms);
    add_double("map_weight_threshold",
               "coverage-derived weight threshold used to define the core map support",
               values.weight_thresholds);
    add_double("map_weight_sum",
               "sum of positive map weights over all valid pixels",
               values.weight_sum);
    add_double("map_core_weight_sum",
               "sum of positive map weights over pixels above map_weight_threshold",
               values.core_weight_sum);
    add_double("map_coverage_sum",
               "sum of coverage values over the map; NaN if no coverage map exists",
               values.coverage_sum);
    add_double("map_coverage_max",
               "maximum coverage value in the map; NaN if no coverage map exists",
               values.coverage_max);
    add_double("map_core_coverage_median",
               "median coverage over the core support; NaN if no coverage map exists",
               values.coverage_median_core);
    add_double("map_empirical_to_formal_noise_ratio",
               "ratio of map_median_rms to map_median_err over the core support",
               values.empirical_to_formal_noise_ratio);
    add_double("map_noise_weight_median_ratio",
               "median of formal weight times jackknife variance over the valid support",
               values.noise_weight_median_ratio);
    add_double("map_noise_weight_scale",
               "empirical scalar applied to formal weights",
               values.noise_weight_scale);
    add_double("map_noise_products_s2n_sigma",
               "standard deviation of jackknife noise multiplied by sqrt(formal weight)",
               values.noise_products_s2n_sigma);
    add_double("map_noise_products_valid_pixels",
               "number of pixels used for empirical noise-product calibration",
               values.noise_products_valid_pixels);
    add_double("map_peak_signal", "maximum signal value in the map",
               values.peak_signal);
    add_double("map_peak_abs_sig2noise",
               "maximum absolute signal-to-noise value in the map",
               values.peak_abs_sig2noise);
    add_double("map_core_peak_abs_sig2noise",
               "maximum absolute signal-to-noise value over pixels with weight >= map_weight_threshold",
               values.core_peak_abs_sig2noise);
    add_double("map_noise_rms_p16",
               "16th percentile of core RMS values across noise realizations",
               values.noise_rms_p16);
    add_double("map_noise_rms_p84",
               "84th percentile of core RMS values across noise realizations",
               values.noise_rms_p84);
    add_double("map_core_tail_fraction_abs_gt3",
               "fraction of core sig2noise pixels with |robust-z| >= 3",
               values.core_tail_frac_abs3);
    add_double("map_core_tail_fraction_pos_gt3",
               "fraction of core sig2noise pixels with robust-z >= 3",
               values.core_tail_frac_pos3);
    add_double("map_core_tail_fraction_neg_lt3",
               "fraction of core sig2noise pixels with robust-z <= -3",
               values.core_tail_frac_neg3);
    add_double("map_core_tail_excess_abs_gt3",
               "ratio of map_core_tail_fraction_abs_gt3 to Gaussian expectation",
               values.core_tail_excess_abs3);
    add_double("map_core_tail_excess_pos_gt3",
               "ratio of map_core_tail_fraction_pos_gt3 to Gaussian expectation",
               values.core_tail_excess_pos3);
    add_double("map_core_tail_excess_neg_lt3",
               "ratio of map_core_tail_fraction_neg_lt3 to Gaussian expectation",
               values.core_tail_excess_neg3);
    add_double("map_core_sig2noise_skew",
               "mean robust-z^3 of core sig2noise pixels",
               values.core_sig2noise_skew);
    add_double("map_noise_tail_fraction_abs_gt3",
               "median fraction across noise realizations with |robust-z| >= 3 in the core support",
               values.noise_tail_frac_abs3);
    add_double("map_noise_tail_fraction_pos_gt3",
               "median fraction across noise realizations with robust-z >= 3 in the core support",
               values.noise_tail_frac_pos3);
    add_double("map_noise_tail_fraction_neg_lt3",
               "median fraction across noise realizations with robust-z <= -3 in the core support",
               values.noise_tail_frac_neg3);
    add_double("map_noise_tail_excess_abs_gt3",
               "median ratio across noise realizations of abs tail fraction to Gaussian expectation",
               values.noise_tail_excess_abs3);
    add_double("map_noise_tail_excess_pos_gt3",
               "median ratio across noise realizations of positive tail fraction to Gaussian expectation",
               values.noise_tail_excess_pos3);
    add_double("map_noise_tail_excess_neg_lt3",
               "median ratio across noise realizations of negative tail fraction to Gaussian expectation",
               values.noise_tail_excess_neg3);
    add_double("map_noise_sig2noise_skew",
               "median mean robust-z^3 across noise realizations in the core support",
               values.noise_sig2noise_skew);
    add_double("map_edge_guard_weight_threshold",
               "runtime weight threshold used by the filter edge guard; NaN when not applied",
               values.edge_guard_weight_thresholds);
    add_double("map_edge_guard_hits_threshold",
               "runtime coverage threshold used by the filter edge guard; NaN when not applied or no coverage map exists",
               values.edge_guard_hits_thresholds);
    add_double("map_edge_guard_background_level",
               "background fill level applied outside the edge-guard support mask before filtering",
               values.edge_guard_background_levels);
    add_double("map_edge_guard_science_fraction",
               "fraction of map pixels in the edge-guard science mask",
               values.edge_guard_science_frac);
    add_double("map_edge_guard_support_fraction",
               "fraction of map pixels in the edge-guard support mask",
               values.edge_guard_support_frac);
    add_double("map_edge_guard_guardband_rms_pre",
               "RMS of signal values in the effective edge-guard guard band before applying fill/taper",
               values.edge_guard_guardband_rms_pre);
    add_double("map_edge_guard_guardband_rms_post",
               "RMS of signal values in the effective edge-guard guard band after applying fill/taper and before filtering",
               values.edge_guard_guardband_rms_post);
    add_double("map_edge_guard_exterior_rms_pre",
               "RMS of signal values outside the effective edge-guard support before applying fill/taper",
               values.edge_guard_exterior_rms_pre);
    add_double("map_edge_guard_exterior_rms_post",
               "RMS of signal values outside the effective edge-guard support after applying fill/taper and before filtering",
               values.edge_guard_exterior_rms_post);
    add_double("map_edge_guard_exterior_max_abs_pre",
               "maximum absolute signal value outside the effective edge-guard support before applying fill/taper",
               values.edge_guard_exterior_max_abs_pre);
    add_double("map_edge_guard_exterior_max_abs_post",
               "maximum absolute signal value outside the effective edge-guard support after applying fill/taper and before filtering",
               values.edge_guard_exterior_max_abs_post);
}

struct MapdiagMapIntValues {
    const std::vector<int> &n_valid_pixels;
    const std::vector<int> &n_core_pixels;
    const std::vector<int> &peak_row;
    const std::vector<int> &peak_col;
    const std::vector<int> &edge_guard_applied;
    const std::vector<int> &edge_guard_support_radius_pix;
    const std::vector<int> &edge_guard_science_npix;
    const std::vector<int> &edge_guard_support_npix;
    const std::vector<int> &edge_guard_guardband_npix;
};

template <class AddInt>
void add_mapdiag_map_int_vars(
    const AddInt &add_int, const MapdiagMapIntValues &values) {
    add_int("map_n_valid_pixels",
            "count of pixels with strictly positive weight",
            values.n_valid_pixels);
    add_int("map_n_core_pixels",
            "count of pixels with weight >= map_weight_threshold",
            values.n_core_pixels);
    add_int("map_peak_row",
            "row index of the maximum absolute signal-to-noise pixel",
            values.peak_row);
    add_int("map_peak_col",
            "column index of the maximum absolute signal-to-noise pixel",
            values.peak_col);
    add_int("map_edge_guard_applied",
            "1 when the filter edge guard was applied to this map, 0 otherwise",
            values.edge_guard_applied);
    add_int("map_edge_guard_support_radius_pix",
            "support-mask dilation radius in pixels used by the filter edge guard",
            values.edge_guard_support_radius_pix);
    add_int("map_edge_guard_science_npix",
            "number of pixels in the filter edge-guard science mask",
            values.edge_guard_science_npix);
    add_int("map_edge_guard_support_npix",
            "number of pixels in the filter edge-guard support mask",
            values.edge_guard_support_npix);
    add_int("map_edge_guard_guardband_npix",
            "number of pixels in the filter edge-guard guard band (support minus science)",
            values.edge_guard_guardband_npix);
}

struct MapdiagObservationDoubleValues {
    const std::vector<double> &obs_weight_sum;
    const std::vector<double> &obs_weight_frac;
    const std::vector<double> &obs_core_weight_sum;
    const std::vector<double> &obs_core_weight_frac;
};

struct MapdiagObservationIntValues {
    const std::vector<int> &obs_valid_pixels;
    const std::vector<int> &obs_core_pixels;
};

template <class AddDouble, class AddInt>
void add_mapdiag_observation_contribution_vars(
    const AddDouble &add_double, const AddInt &add_int,
    const MapdiagObservationDoubleValues &double_values,
    const MapdiagObservationIntValues &int_values) {
    add_double("coadd_obs_weight_sum",
               "sum of positive observation-level raw weight values aligned onto this map grid",
               double_values.obs_weight_sum);
    add_double("coadd_obs_weight_frac",
               "fractional contribution of each obsnum to coadd_obs_weight_sum for a given map",
               double_values.obs_weight_frac);
    add_double("coadd_obs_core_weight_sum",
               "sum of positive observation-level raw weight values within the final map core support",
               double_values.obs_core_weight_sum);
    add_double("coadd_obs_core_weight_frac",
               "fractional contribution of each obsnum within the final map core support",
               double_values.obs_core_weight_frac);
    add_int("coadd_obs_n_valid_pixels",
            "count of aligned observation pixels with positive raw weight",
            int_values.obs_valid_pixels);
    add_int("coadd_obs_n_core_pixels",
            "count of aligned observation pixels with positive raw weight inside the final map core support",
            int_values.obs_core_pixels);
}

}  // namespace citlali::pipeline
