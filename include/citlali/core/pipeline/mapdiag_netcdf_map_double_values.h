#pragma once

// Included by mapdiag_netcdf_map_values.h inside namespace citlali::pipeline.

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

inline std::string mapdiag_noise_product_comment(
    const std::string &description, const std::string &variable,
    const std::string &product_identity,
    const std::string &semantic_digest, const std::string &validity,
    const std::string &restriction) {
    return description +
        "; citlali_noise_product_join_v1"
        "|variable=" + variable +
        "|package_id=citlali-noise-products"
        "|provenance_id=noise_products_provenance.yaml"
        "|product_identity=" + product_identity +
        "|product_version=SCI-NOI-002-v1"
        "|semantic_digest=" + semantic_digest +
        "|digest_kind=semantic_contract_sha256"
        "|missingness=nonfinite_unavailable"
        "|scope=map_summary"
        "|validity=" + validity +
        "|restriction=" + restriction;
}

template <class AddDouble>
void add_mapdiag_map_double_vars(
    const AddDouble &add_double, const MapdiagMapDoubleValues &values) {
    add_double("map_median_err",
               "legacy reciprocal-sqrt normalization-coefficient scale; not an uncertainty unless SCI-PTC-001 precision conditions are established",
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
               "ratio of empirical map RMS to the legacy reciprocal-sqrt coefficient scale; not an uncertainty calibration claim",
               values.empirical_to_formal_noise_ratio);
    add_double("map_noise_weight_median_ratio",
               mapdiag_noise_product_comment(
                   "median of the nonprecision normalization coefficient times conditional finite-stack scatter over the realized calibration support",
                   "map_noise_weight_median_ratio",
                   "global_nonprecision_scale_diagnostic",
                   "sha256:bfb6d1ea365d1b8e82fd88aad0c2aac3ebb0a2f40f3b78c244f5b1ce9498a655",
                   "available_when_finite_positive_calibration_support_exists",
                   "engineering_scale_diagnostic_not_precision_or_significance"),
               values.noise_weight_median_ratio);
    add_double("map_noise_weight_scale",
               mapdiag_noise_product_comment(
                   "existing-use-only global scalar applied to the nonprecision normalization coefficient",
                   "map_noise_weight_scale",
                   "global_nonprecision_scale_diagnostic",
                   "sha256:bfb6d1ea365d1b8e82fd88aad0c2aac3ebb0a2f40f3b78c244f5b1ce9498a655",
                   "available_when_finite_positive_median_ratio_exists",
                   "nonprecision_scale_not_inverse_variance_or_precision"),
               values.noise_weight_scale);
    add_double("map_noise_products_s2n_sigma",
               mapdiag_noise_product_comment(
                   "pooled completed-stack scale of realization amplitudes multiplied by sqrt(nonprecision normalization coefficient)",
                   "map_noise_products_s2n_sigma",
                   "pooled_stack_scale_diagnostic",
                   "sha256:1b3a38d18a451b9e35ffe9f9fed21b1f3107a8f9cd229386d16998ddff359e79",
                   "available_when_finite_pooled_stack_scale_exists",
                   "engineering_scale_diagnostic_not_calibrated_significance"),
               values.noise_products_s2n_sigma);
    add_double("map_noise_products_valid_pixels",
               "number of pixels used for empirical noise-product calibration",
               values.noise_products_valid_pixels);
    add_double("map_peak_signal", "maximum signal value in the map",
               values.peak_signal);
    add_double("map_peak_abs_sig2noise",
               "maximum absolute legacy coefficient-standardized amplitude; not a signal-to-noise or significance claim",
               values.peak_abs_sig2noise);
    add_double("map_core_peak_abs_sig2noise",
               "maximum absolute legacy coefficient-standardized amplitude over core support; not a signal-to-noise or significance claim",
               values.core_peak_abs_sig2noise);
    add_double("map_noise_rms_p16",
               "16th percentile of core RMS values across noise realizations",
               values.noise_rms_p16);
    add_double("map_noise_rms_p84",
               "84th percentile of core RMS values across noise realizations",
               values.noise_rms_p84);
    add_double("map_core_tail_fraction_abs_gt3",
               "fraction of core legacy coefficient-standardized amplitudes with |robust-z| >= 3",
               values.core_tail_frac_abs3);
    add_double("map_core_tail_fraction_pos_gt3",
               "fraction of core legacy coefficient-standardized amplitudes with robust-z >= 3",
               values.core_tail_frac_pos3);
    add_double("map_core_tail_fraction_neg_lt3",
               "fraction of core legacy coefficient-standardized amplitudes with robust-z <= -3",
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
               "mean robust-z^3 of core legacy coefficient-standardized amplitudes",
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
               "median mean robust-z^3 of legacy coefficient-standardized amplitudes across noise realizations in core support",
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
