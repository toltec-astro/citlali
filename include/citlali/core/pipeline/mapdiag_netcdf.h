#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <netcdf>

#include <citlali/core/pipeline/mapdiag_labels.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/utils/netcdf_io.h>

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

inline std::string mapdiag_map_dim_name() {
    return "n_maps";
}

inline std::string mapdiag_obsnum_dim_name() {
    return "n_obsnums";
}

inline std::string mapdiag_netcdf_filename(
    const std::string &base_filename) {
    return base_filename + ".nc";
}

struct MapdiagNetcdfDims {
    netCDF::NcDim maps;
    netCDF::NcDim obsnums;
    std::vector<netCDF::NcDim> map_obs;
};

struct MapdiagIdentityVars {
    const std::string &stage_name;
    const std::string &buffer_name;
    const std::string &map_regime;
    const std::string &source_name;
    const std::string &project_id;
    const std::string &obs_goal;
};

struct MapdiagRuntimeVars {
    double pixel_size_rad;
    double coverage_cut;
    const std::string &signal_unit;
};

struct MapdiagEdgeGuardConfigVars {
    bool enabled;
    const std::string &weight_threshold_mode;
    const std::string &hits_threshold_mode;
    const std::string &fill_mode;
    const std::string &taper_mode;
    double hits_core_fraction;
    double radius_fwhm;
    double taper_min_fraction;
};

struct MapdiagMetadataVars {
    MapdiagIdentityVars identity;
    MapdiagRuntimeVars runtime;
    MapdiagEdgeGuardConfigVars edge_guard;
};

struct MapdiagLabelVars {
    const std::vector<std::string> &array_names;
    const std::vector<std::string> &stokes_names;
    const std::vector<std::string> &map_names;
    const std::vector<std::string> &obsnums;
    const std::string &fallback_obsnum;
    const std::vector<std::string> &date_obs;
    std::size_t n_obsnums;
};

inline MapdiagNetcdfDims add_mapdiag_netcdf_dims(
    netCDF::NcFile &fo, const MapdiagSizeContext &context) {
    netCDF::NcDim maps_dim =
        fo.addDim(mapdiag_map_dim_name(), context.n_maps);
    netCDF::NcDim obsnums_dim =
        fo.addDim(mapdiag_obsnum_dim_name(), context.n_obsnums);
    return {maps_dim, obsnums_dim, {maps_dim, obsnums_dim}};
}

MapdiagSizeContext make_mapdiag_size_context(std::size_t n_maps,
                                             std::size_t obsnum_count,
                                             bool is_coadd) {
    return {n_maps, obsnum_count, is_coadd};
}

inline std::size_t mapdiag_obs_table_size(const MapdiagSizeContext &context) {
    return context.n_maps * context.n_obsnums;
}

inline int mapdiag_obsnum_value(const MapdiagSizeContext &context,
                                const std::string &obsnum) {
    return context.is_coadd ? -1 : std::stoi(obsnum);
}

inline std::size_t mapdiag_obs_flat_index(const MapdiagSizeContext &context,
                                          std::size_t map_index,
                                          std::size_t obs_index) {
    return map_index * context.n_obsnums + obs_index;
}

inline void add_mapdiag_identity_vars(
    netCDF::NcFile &fo, const MapdiagIdentityVars &values) {
    add_netcdf_var<std::string>(fo, "MAP_STAGE", values.stage_name);
    add_netcdf_var<std::string>(fo, "MAP_BUFFER", values.buffer_name);
    add_netcdf_var<std::string>(fo, "MAP_REGIME", values.map_regime);
    add_netcdf_var<std::string>(fo, "SOURCE", values.source_name);
    add_netcdf_var<std::string>(fo, "PROJID", values.project_id);
    add_netcdf_var<std::string>(fo, "OBSGOAL", values.obs_goal);
}

inline void add_mapdiag_runtime_vars(
    netCDF::NcFile &fo, const MapdiagRuntimeVars &values) {
    add_netcdf_var(fo, "MAP_PIXEL_SIZE_RAD", values.pixel_size_rad);
    add_netcdf_var(fo, "MAP_COVERAGE_CUT", values.coverage_cut);
    add_netcdf_var<std::string>(fo, "MAP_SIG_UNIT", values.signal_unit);
}

inline void add_mapdiag_edge_guard_config_vars(
    netCDF::NcFile &fo, const MapdiagEdgeGuardConfigVars &values) {
    add_netcdf_var(fo, "MAP_EDGE_GUARD_ENABLED", values.enabled);
    add_netcdf_var<std::string>(
        fo, "MAP_EDGE_GUARD_WEIGHT_THRESHOLD_MODE",
        values.weight_threshold_mode);
    add_netcdf_var<std::string>(
        fo, "MAP_EDGE_GUARD_HITS_THRESHOLD_MODE",
        values.hits_threshold_mode);
    add_netcdf_var<std::string>(
        fo, "MAP_EDGE_GUARD_FILL_MODE", values.fill_mode);
    add_netcdf_var<std::string>(
        fo, "MAP_EDGE_GUARD_TAPER_MODE", values.taper_mode);
    add_netcdf_var(
        fo, "MAP_EDGE_GUARD_HITS_CORE_FRACTION",
        values.hits_core_fraction);
    add_netcdf_var(
        fo, "MAP_EDGE_GUARD_RADIUS_FWHM", values.radius_fwhm);
    add_netcdf_var(
        fo, "MAP_EDGE_GUARD_TAPER_MIN_FRACTION",
        values.taper_min_fraction);
}

inline void add_mapdiag_metadata_vars(
    netCDF::NcFile &fo, const MapdiagIdentityVars &identity,
    const MapdiagRuntimeVars &runtime,
    const MapdiagEdgeGuardConfigVars &edge_guard) {
    add_mapdiag_identity_vars(fo, identity);
    add_mapdiag_runtime_vars(fo, runtime);
    add_mapdiag_edge_guard_config_vars(fo, edge_guard);
}

inline void add_mapdiag_metadata_vars(
    netCDF::NcFile &fo, const MapdiagMetadataVars &values) {
    add_mapdiag_metadata_vars(
        fo, values.identity, values.runtime, values.edge_guard);
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

inline void add_mapdiag_map_label_vars(
    netCDF::NcFile &fo, netCDF::NcDim maps_dim,
    const std::vector<std::string> &array_names,
    const std::vector<std::string> &stokes_names,
    const std::vector<std::string> &map_names) {
    put_netcdf_string_1d(
        fo, "map_array_name", maps_dim, array_names,
        "array label for each map row");
    put_netcdf_string_1d(
        fo, "map_stokes", maps_dim, stokes_names,
        "stokes parameter label for each map row");
    put_netcdf_string_1d(
        fo, "map_name", maps_dim, map_names,
        "grouping-derived map label prefix for each map row");
}

inline void add_mapdiag_observation_label_vars(
    netCDF::NcFile &fo, netCDF::NcDim obsnums_dim,
    const std::vector<std::string> &obsnums,
    const std::string &fallback_obsnum,
    const std::vector<std::string> &date_obs,
    std::size_t n_obsnums) {
    const auto obsnum_strings =
        mapdiag_obsnum_labels(obsnums, fallback_obsnum);
    put_netcdf_string_1d(
        fo, "coadd_obsnum", obsnums_dim, obsnum_strings,
        "obsnum ordering for map x obsnum contribution tables");

    const auto dateobs_strings =
        mapdiag_dateobs_labels(date_obs, n_obsnums);
    put_netcdf_string_1d(
        fo, "coadd_dateobs", obsnums_dim, dateobs_strings,
        "DATEOBS ordering matching coadd_obsnum");
}

inline void add_mapdiag_label_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::vector<std::string> &array_names,
    const std::vector<std::string> &stokes_names,
    const std::vector<std::string> &map_names,
    const std::vector<std::string> &obsnums,
    const std::string &fallback_obsnum,
    const std::vector<std::string> &date_obs,
    std::size_t n_obsnums) {
    add_mapdiag_map_label_vars(
        fo, dims.maps, array_names, stokes_names, map_names);
    add_mapdiag_observation_label_vars(
        fo, dims.obsnums, obsnums, fallback_obsnum, date_obs, n_obsnums);
}

inline void add_mapdiag_label_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagLabelVars &values) {
    add_mapdiag_label_vars(
        fo, dims, values.array_names, values.stokes_names,
        values.map_names, values.obsnums, values.fallback_obsnum,
        values.date_obs, values.n_obsnums);
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

inline void add_mapdiag_map_double_var(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::string &name, const std::string &comment,
    const std::vector<double> &values) {
    add_mapdiag_double_1d(fo, name, comment, dims.maps, values);
}

inline void add_mapdiag_map_int_var(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::string &name, const std::string &comment,
    const std::vector<int> &values) {
    add_mapdiag_int_1d(fo, name, comment, dims.maps, values);
}

inline void add_mapdiag_obs_double_var(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::string &name, const std::string &comment,
    const std::vector<double> &values) {
    add_mapdiag_double_2d(fo, name, comment, dims.map_obs, values);
}

inline void add_mapdiag_obs_int_var(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::string &name, const std::string &comment,
    const std::vector<int> &values) {
    add_mapdiag_int_2d(fo, name, comment, dims.map_obs, values);
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

inline void add_mapdiag_map_double_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagMapDoubleValues &values) {
    auto add_double = [&](const std::string &name,
                          const std::string &comment,
                          const std::vector<double> &var_values) {
        add_mapdiag_map_double_var(fo, dims, name, comment, var_values);
    };
    add_mapdiag_map_double_vars(add_double, values);
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

inline void add_mapdiag_map_int_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagMapIntValues &values) {
    auto add_int = [&](const std::string &name,
                       const std::string &comment,
                       const std::vector<int> &var_values) {
        add_mapdiag_map_int_var(fo, dims, name, comment, var_values);
    };
    add_mapdiag_map_int_vars(add_int, values);
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

struct MapdiagValueVars {
    MapdiagMapDoubleValues map_double;
    MapdiagMapIntValues map_int;
    MapdiagObservationDoubleValues observation_double;
    MapdiagObservationIntValues observation_int;
};

struct MapdiagNetcdfVars {
    const MapdiagSizeContext &size;
    const std::string &obsnum;
    const MapdiagMetadataVars &metadata;
    const MapdiagLabelVars &labels;
    const MapdiagValueVars &values;
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

inline void add_mapdiag_observation_contribution_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagObservationDoubleValues &double_values,
    const MapdiagObservationIntValues &int_values) {
    auto add_double = [&](const std::string &name,
                          const std::string &comment,
                          const std::vector<double> &var_values) {
        add_mapdiag_obs_double_var(fo, dims, name, comment, var_values);
    };
    auto add_int = [&](const std::string &name,
                       const std::string &comment,
                       const std::vector<int> &var_values) {
        add_mapdiag_obs_int_var(fo, dims, name, comment, var_values);
    };
    add_mapdiag_observation_contribution_vars(
        add_double, add_int, double_values, int_values);
}

inline void add_mapdiag_value_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagValueVars &values) {
    add_mapdiag_map_double_vars(fo, dims, values.map_double);
    add_mapdiag_map_int_vars(fo, dims, values.map_int);
    add_mapdiag_observation_contribution_vars(
        fo, dims, values.observation_double, values.observation_int);
}

inline void add_mapdiag_netcdf_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfVars &values) {
    add_obsnum_var(fo, mapdiag_obsnum_value(values.size, values.obsnum));

    const auto dims = add_mapdiag_netcdf_dims(fo, values.size);
    add_mapdiag_metadata_vars(fo, values.metadata);
    add_mapdiag_label_vars(fo, dims, values.labels);
    add_mapdiag_value_vars(fo, dims, values.values);
}

}  // namespace citlali::pipeline
