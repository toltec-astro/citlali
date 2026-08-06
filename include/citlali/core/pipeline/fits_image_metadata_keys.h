#pragma once

// Included by fits_image_metadata.h inside namespace citlali::pipeline.

template <class Hdu>
void add_image_unit_keys(Hdu &hdu, const std::string &unit) {
    hdu.addKey("UNIT", unit, "Unit of map");
    hdu.addKey("BUNIT", unit, "Physical unit of image values");
}

template <class Hdu>
void add_image_description_key(Hdu &hdu, const std::string &description) {
    hdu.addKey("DESCRIP", description, "Image product description");
}

template <class Hdu>
void add_image_type_key(Hdu &hdu, const std::string &type,
                        const std::string &comment) {
    hdu.addKey("ESTTYPE", type, comment + " (authoritative)");
    hdu.addKey("TYPE", type, comment);
}

template <class Hdu>
void add_image_data_type_key(Hdu &hdu, const std::string &data_type) {
    hdu.addKey("DATTYP", data_type, "Logical image scalar type");
}

template <class Hdu>
void add_image_validity_authority_key(Hdu &hdu, bool is_authority) {
    hdu.addKey("VALAUTH", is_authority ? std::string{"true"}
                                       : std::string{"false"},
               "Authoritative raw science-validity mask");
}

template <class Hdu>
void add_raw_parent_identity_keys(Hdu &hdu,
                                  const std::string &raw_parent_digest) {
    hdu.addKey("RAWSTATE", std::string{"immutable_input"},
               "Relationship to raw science-map authority");
    hdu.addKey("RAWPDGST", raw_parent_digest,
               "Exact raw-parent/product digest", true);
}

template <class Hdu>
void add_image_alias_keys(Hdu &hdu, const std::string &canonical_name,
                          bool deprecated) {
    hdu.addKey("ALIASOF", canonical_name, "Canonical image product");
    hdu.addKey("DEPRCATD", deprecated ? std::string{"true"}
                                      : std::string{"false"},
               "Compatibility alias is deprecated");
    add_image_validity_authority_key(hdu, false);
}

template <class Hdu>
void add_image_type_description_keys(Hdu &hdu, const std::string &type,
                                     const std::string &type_comment,
                                     const std::string &description) {
    add_image_type_key(hdu, type, type_comment);
    add_image_description_key(hdu, description);
}

template <class Hdu>
void add_image_unit_type_description_keys(Hdu &hdu, const std::string &unit,
                                          const std::string &type,
                                          const std::string &type_comment,
                                          const std::string &description) {
    add_image_unit_keys(hdu, unit);
    add_image_type_description_keys(hdu, type, type_comment, description);
}

template <class Hdu>
void add_image_median_error_key(Hdu &hdu, double median_error,
                                const std::string &unit) {
    hdu.addKey("MEDERR", median_error, "Median Error (" + unit + ")");
}

template <class Hdu>
void add_image_median_rms_key(Hdu &hdu, double median_rms,
                              const std::string &unit) {
    hdu.addKey("MEDRMS", median_rms,
               "Median jackknife-map RMS (" + unit + ")");
}

template <class Hdu>
void add_image_weight_threshold_key(Hdu &hdu, double weight_threshold) {
    hdu.addKey("WTTHRESH", weight_threshold, "Weight threshold");
}

template <class Hdu>
void add_empirical_weight_scale_key(Hdu &hdu, double scale) {
    hdu.addKey("EMP_SCALE", scale,
               "Existing-use-only global nonprecision coefficient scale");
}

template <class Hdu>
void add_weight_variance_median_key(Hdu &hdu, double median_ratio) {
    hdu.addKey("WVARMED", median_ratio,
               "Median q_p times conditional finite-stack scatter");
}

template <class Hdu>
void add_point_source_response_norm_key(Hdu &hdu, double response_norm) {
    hdu.addKey("RESPNORM", response_norm,
               "Legacy identity response; no aperture/template calibration");
}

template <class Hdu>
void add_noise_product_join_keys(
    Hdu &hdu, const std::string &package_id,
    const std::string &provenance_id, const std::string &product_identity,
    const std::string &product_version, const std::string &semantic_digest,
    const std::string &scope, const std::string &validity,
    const std::string &restriction) {
    hdu.addKey("NOIPKG", package_id, "Noise-product package identity", true);
    hdu.addKey("NOIPROV", provenance_id,
               "Required authoritative package sidecar", true);
    hdu.addKey("NOIPRID", product_identity,
               "Noise product identity", true);
    hdu.addKey("NOIPVER", product_version,
               "Noise product contract version", true);
    hdu.addKey("NOIDGST", semantic_digest,
               "Product semantic-contract digest", true);
    hdu.addKey("NOIDGKND", std::string{"semantic_contract_sha256"},
               "Product digest kind", true);
    hdu.addKey("NOISCOPE", scope, "Product scope", true);
    hdu.addKey("NOIVALID", validity, "Product-level validity", true);
    hdu.addKey("NOIRESTR", restriction,
               "Detached-product restriction", true);
    hdu.addKey("NOIMISS", std::string{"nonfinite_unavailable"},
               "Missing/invalid value policy");
}

template <class Hdu>
void add_global_nonprecision_scale_diagnostic_keys(
    Hdu &hdu, bool valid, Eigen::Index valid_pixel_count,
    double median_ratio, double scale) {
    hdu.addKey("SCLID", std::string{"global_nonprecision_scale_diagnostic"},
               "Scale diagnostic identity", true);
    hdu.addKey("SCLVALID", valid ? std::string{"true"}
                                  : std::string{"false"},
               "Scale diagnostic validity");
    hdu.addKey("SCLNPIX", valid_pixel_count,
               "Finite positive calibration-region pixels");
    if (valid) {
        hdu.addKey("SCLMED", median_ratio,
                   "Median q_p times conditional stack scatter");
        hdu.addKey("SCLALPH", scale,
                   "Reciprocal-median nonprecision scale");
    }
}

template <class Hdu>
void add_not_significance_keys(Hdu &hdu) {
    hdu.addKey("SIGSTAT", std::string{"not_significance"},
               "Statistical-significance status");
    hdu.addKey("NULLSTAT", std::string{"unavailable"},
               "Null/tail calibration status");
}

template <class Hdu>
void add_kernel_fwhm_key(Hdu &hdu, double fwhm_arcsec) {
    hdu.addKey("FWHM", fwhm_arcsec, "Kernel fwhm (arcsec)");
}

template <class Hdu>
void add_noise_image_summary_keys(Hdu &hdu, const std::string &unit,
                                  double median_rms) {
    hdu.addKey("UNIT", unit, "Unit of map");
    hdu.addKey("MEDRMS", median_rms, "Median RMS of noise maps");
}

template <class Hdu>
void add_image_unit_description_keys(Hdu &hdu, const std::string &unit,
                                     const std::string &description) {
    add_image_unit_keys(hdu, unit);
    add_image_description_key(hdu, description);
}

template <class Hdu>
void add_signal_map_metadata(Hdu &hdu, const std::string &signal_unit) {
    add_image_unit_description_keys(hdu, signal_unit,
                                    signal_map_description());
}

template <class Hdu>
void add_weight_map_metadata(Hdu &hdu, const std::string &weight_unit,
                             bool empirical_weight_calibration) {
    add_image_unit_type_description_keys(
        hdu, weight_unit, normalization_coefficient_estimator_type(),
        "Coefficient estimator type",
        weight_map_description(empirical_weight_calibration));
    hdu.addKey("CALTYPE",
               std::string{weight_calibration_type(
                   empirical_weight_calibration)},
               "Coefficient scaling type; not precision calibration");
    hdu.addKey("PRECSTAT", std::string{"not_established"},
               "Marginal-precision interpretation");
    hdu.addKey("COVSTAT", std::string{"unavailable"},
               "Cross-pixel/observation covariance status");
}

template <class Hdu>
void add_formal_weight_map_metadata(Hdu &hdu,
    const std::string &weight_unit) {
    add_image_unit_type_description_keys(
        hdu, weight_unit, formal_coefficient_snapshot_estimator_type(),
        "Coefficient estimator type", formal_weight_map_description());
    hdu.addKey("CALTYPE", std::string{formal_weight_calibration_type()},
               "Coefficient calibration type");
    hdu.addKey("PRECSTAT", std::string{"conditional_SCI-PTC-001"},
               "Marginal-precision interpretation");
    hdu.addKey("COVSTAT", std::string{"unavailable"},
               "Cross-pixel/observation covariance status");
}

template <class Hdu>
void add_noise_variance_map_metadata(Hdu &hdu,
                                     const std::string &variance_unit,
                                     const std::string &canonical_name) {
    add_image_unit_type_description_keys(
        hdu, variance_unit, conditional_stack_scatter_estimator_type(),
        "Finite-stack estimator type", noise_variance_map_description());
    add_image_alias_keys(hdu, canonical_name, true);
    hdu.addKey("COVSTAT", std::string{"conditional_stack_only"},
               "Physical-noise covariance status");
}

template <class Hdu>
void add_conditional_stack_scatter_map_metadata(
    Hdu &hdu, const std::string &variance_unit) {
    add_image_unit_type_description_keys(
        hdu, variance_unit, conditional_stack_scatter_estimator_type(),
        "Finite-stack estimator type",
        conditional_stack_scatter_map_description());
    add_image_validity_authority_key(hdu, false);
    hdu.addKey("COVSTAT", std::string{"conditional_stack_only"},
               "Physical-noise covariance status");
}

template <class Hdu>
void add_kernel_map_metadata(Hdu &hdu, const std::string &signal_unit) {
    add_image_unit_description_keys(hdu, signal_unit,
                                    kernel_map_description());
}

template <class Hdu>
void add_coverage_map_metadata(
    Hdu &hdu,
    const std::string &canonical_name = "retained_exposure_I") {
    add_image_unit_type_description_keys(
        hdu, coverage_time_unit(), retained_exposure_estimator_type(),
        "Exposure estimator type", coverage_map_description());
    add_image_data_type_key(hdu, "float64");
    add_image_alias_keys(hdu, canonical_name, false);
}

template <class Hdu>
void add_coverage_mask_map_metadata(
    Hdu &hdu,
    const std::string &canonical_name = "science_policy_support_I") {
    add_image_unit_type_description_keys(
        hdu, science_map_mask_unit(), science_policy_support_estimator_type(),
        "Support estimator type", coverage_mask_map_description());
    add_image_data_type_key(hdu, "uint8");
    add_image_alias_keys(hdu, canonical_name, true);
}

template <class Hdu>
void add_science_map_product_metadata(
    Hdu &hdu, const std::string &unit, const std::string &estimator_type,
    const std::string &description, const std::string &data_type,
    bool is_validity_authority = false) {
    add_image_unit_type_description_keys(
        hdu, unit, estimator_type, "Science-map product estimator type",
        description);
    add_image_data_type_key(hdu, data_type);
    add_image_validity_authority_key(hdu, is_validity_authority);
}

template <class Hdu>
void add_geometric_hits_map_metadata(Hdu &hdu) {
    add_science_map_product_metadata(
        hdu, science_map_count_unit(), geometric_hits_estimator_type(),
        geometric_hits_map_description(), "int64");
}

template <class Hdu>
void add_contributing_hits_map_metadata(Hdu &hdu) {
    add_science_map_product_metadata(
        hdu, science_map_count_unit(), contributing_hits_estimator_type(),
        contributing_hits_map_description(), "int64");
}

template <class Hdu>
void add_coadd_observation_count_map_metadata(Hdu &hdu) {
    add_science_map_product_metadata(
        hdu, science_map_count_unit(),
        coadd_observation_count_estimator_type(),
        coadd_observation_count_map_description(), "int64");
}

template <class Hdu>
void add_upstream_eligible_exposure_map_metadata(Hdu &hdu) {
    add_science_map_product_metadata(
        hdu, coverage_time_unit(),
        upstream_eligible_exposure_estimator_type(),
        upstream_eligible_exposure_map_description(), "float64");
}

template <class Hdu>
void add_retained_exposure_map_metadata(Hdu &hdu) {
    add_science_map_product_metadata(
        hdu, coverage_time_unit(), retained_exposure_estimator_type(),
        retained_exposure_map_description(), "float64");
}

template <class Hdu>
void add_normalization_support_map_metadata(Hdu &hdu) {
    add_science_map_product_metadata(
        hdu, science_map_mask_unit(), normalization_support_estimator_type(),
        normalization_support_map_description(), "uint8");
}

template <class Hdu>
void add_science_policy_support_map_metadata(Hdu &hdu) {
    add_science_map_product_metadata(
        hdu, science_map_mask_unit(), science_policy_support_estimator_type(),
        science_policy_support_map_description(), "uint8");
}

template <class Hdu>
void add_science_valid_map_metadata(Hdu &hdu) {
    add_science_map_product_metadata(
        hdu, science_map_mask_unit(), science_valid_estimator_type(),
        science_valid_map_description(), "uint8", true);
}

template <class Hdu>
void add_legacy_pixel_snr_map_metadata(
    Hdu &hdu, const std::string &canonical_name) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(),
        coefficient_standardized_signal_estimator_type(),
        snr_estimator_type_comment(), legacy_pixel_snr_map_description());
    add_image_alias_keys(hdu, canonical_name, true);
    add_not_significance_keys(hdu);
}

template <class Hdu>
void add_pixel_snr_map_metadata(
    Hdu &hdu, const std::string &canonical_name) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(),
        coefficient_standardized_signal_estimator_type(),
        snr_estimator_type_comment(), pixel_snr_map_description());
    add_image_alias_keys(hdu, canonical_name, true);
    add_not_significance_keys(hdu);
}

template <class Hdu>
void add_coefficient_standardized_signal_map_metadata(Hdu &hdu) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(),
        coefficient_standardized_signal_estimator_type(),
        standardized_signal_estimator_type_comment(),
        coefficient_standardized_signal_map_description());
    add_image_validity_authority_key(hdu, false);
    add_not_significance_keys(hdu);
}

template <class Hdu>
void add_formal_standardized_signal_map_metadata(Hdu &hdu) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(),
        formal_standardized_signal_estimator_type(),
        standardized_signal_estimator_type_comment(),
        formal_standardized_signal_map_description());
    add_not_significance_keys(hdu);
}

template <class Hdu>
void add_point_source_flux_map_metadata(Hdu &hdu,
                                        const std::string &signal_unit,
                                        const std::string &canonical_name) {
    add_image_unit_description_keys(hdu, signal_unit,
                                    point_source_flux_map_description());
    add_image_alias_keys(hdu, canonical_name, true);
    add_not_significance_keys(hdu);
}

template <class Hdu>
void add_point_source_uncertainty_map_metadata(
    Hdu &hdu, const std::string &signal_unit,
    const std::string &canonical_name) {
    add_image_unit_description_keys(
        hdu, signal_unit, point_source_uncertainty_map_description());
    add_image_type_key(
        hdu, filtered_pixel_stack_scatter_estimator_type(),
        "Finite-stack estimator type");
    add_image_alias_keys(hdu, canonical_name, true);
    add_not_significance_keys(hdu);
}

template <class Hdu>
void add_point_source_snr_map_metadata(
    Hdu &hdu, const std::string &canonical_name) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(),
        conditional_stack_scatter_ratio_estimator_type(),
        snr_estimator_type_comment(), point_source_snr_map_description());
    add_image_alias_keys(hdu, canonical_name, true);
    add_not_significance_keys(hdu);
}

template <class Hdu>
void add_filtered_pixel_stack_scatter_map_metadata(
    Hdu &hdu, const std::string &signal_unit) {
    add_image_unit_type_description_keys(
        hdu, signal_unit, filtered_pixel_stack_scatter_estimator_type(),
        "Finite-stack estimator type",
        filtered_pixel_stack_scatter_map_description());
    add_image_validity_authority_key(hdu, false);
    add_not_significance_keys(hdu);
}

template <class Hdu>
void add_conditional_stack_scatter_ratio_map_metadata(Hdu &hdu) {
    add_image_unit_type_description_keys(
        hdu, not_applicable_image_unit(),
        conditional_stack_scatter_ratio_estimator_type(),
        "Finite-stack ratio type",
        conditional_stack_scatter_ratio_map_description());
    add_image_validity_authority_key(hdu, false);
    add_not_significance_keys(hdu);
}
