#pragma once

// Included by tod_output_reduction_metadata.h inside namespace citlali::pipeline.

template <class Arrays, class ShapeParams, class ArrayNameMap>
void add_jinc_shape_config_vars(netCDF::NcFile &fo, const Arrays &arrays,
                                ShapeParams &shape_params,
                                ArrayNameMap &array_name_map,
                                double r_max) {
    add_netcdf_var(fo, "JINC_R", r_max);
    for (const auto &arr: arrays) {
        const auto &name = array_name_map[arr];
        add_netcdf_var(fo, "JINC_A_" + name, shape_params[arr][0]);
        add_netcdf_var(fo, "JINC_B_" + name, shape_params[arr][1]);
        add_netcdf_var(fo, "JINC_C_" + name, shape_params[arr][2]);
    }
}

template <class Arrays, class ShapeParams, class ArrayNameMap>
void add_jinc_shape_config_vars_if_needed(
    netCDF::NcFile &fo, citlali::config::MapMethod map_method,
    const Arrays &arrays,
    ShapeParams &shape_params, ArrayNameMap &array_name_map, double r_max) {
    if (citlali::config::is_jinc_map_method(map_method)) {
        add_jinc_shape_config_vars(
            fo, arrays, shape_params, array_name_map, r_max);
    }
}

template <class TauByFrequency, class Calib, class ArrayNameMap>
void add_mean_tau_vars(netCDF::NcFile &fo, const TauByFrequency &tau_freq,
                       const Calib &calib, ArrayNameMap &array_name_map) {
    decltype(calib.arrays.size()) i = 0;
    for (auto const& [key, val] : tau_freq) {
        add_netcdf_var(
            fo, "MEAN_TAU_" + array_name_map[calib.arrays(i)], val[0]);
        i++;
    }
}

template <class Calib, class ArrayNameMap>
void add_zero_mean_tau_vars(netCDF::NcFile &fo, const Calib &calib,
                            ArrayNameMap &array_name_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        add_netcdf_var(
            fo, "MEAN_TAU_" + array_name_map[calib.arrays(i)], 0.);
    }
}

template <class Rtcproc, class TelescopeData, class Calib, class ArrayNameMap>
void add_tod_mean_tau_vars(netCDF::NcFile &fo, bool extinction_enabled,
                           Rtcproc &rtcproc,
                           TelescopeData &tel_data, double tau_225_ghz,
                           const Calib &calib, ArrayNameMap &array_name_map) {
    if (extinction_enabled) {
        Eigen::VectorXd tau_el(1);
        tau_el << tel_data["TelElAct"].mean();
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, tau_225_ghz);
        add_mean_tau_vars(fo, tau_freq, calib, array_name_map);
    }
    else {
        add_zero_mean_tau_vars(fo, calib, array_name_map);
    }
    const auto requested =
        rtcproc.calibration.requested_reference_spectral_index_alpha();
    add_netcdf_var(
        fo, "CAL.ALPHA.REQUESTED_AVAILABLE", requested.has_value());
    if (requested) {
        add_netcdf_var(fo, "CAL.ALPHA.REQUESTED", *requested);
    }
    add_netcdf_var(
        fo, "CAL.ALPHA.EFFECTIVE",
        rtcproc.calibration.effective_reference_spectral_index_alpha());
    add_netcdf_var(
        fo, "CAL.ALPHA.REALIZED",
        rtcproc.calibration.effective_reference_spectral_index_alpha());
    add_netcdf_var(
        fo, "CAL.ALPHA.DEFAULT_APPLIED",
        rtcproc.calibration.reference_spectral_index_default_applied());
    add_netcdf_var<std::string>(
        fo, "CAL.OPERATOR_ID",
        std::string{rtcproc.calibration.operator_id()});
    add_netcdf_var<std::string>(
        fo, "CAL.OPERATOR_CONTRACT_SHA256",
        std::string{rtcproc.calibration.operator_contract_sha256()});
    add_netcdf_var<std::string>(
        fo, "CAL.NODE_TABLE_SHA256",
        std::string{rtcproc.calibration.operator_nodes_sha256()});
    add_netcdf_var<std::string>(
        fo, "CAL.PASSBAND_SET_ID",
        std::string{rtcproc.calibration.passband_set_id()});
    add_netcdf_var<std::string>(
        fo, "CAL.REFERENCE_PROFILE_ID",
        std::string{rtcproc.calibration.reference_profile_id()});
    add_netcdf_var<std::string>(
        fo, "CAL.QUALITY_REGIME",
        rtcproc.calibration.calibration_quality_regime);
    add_netcdf_var(
        fo, "CAL.VALID", rtcproc.calibration.calibration_valid);
    add_netcdf_var<std::string>(
        fo, "CAL.VALIDITY_REASON",
        rtcproc.calibration.calibration_validity_reason);
    const auto &product = rtcproc.calibration.product;
    if (product.valid()) {
        timestream::require_finalized_calibration_product_join(product);
    }
    add_netcdf_var<std::string>(
        fo, "CAL.PRODUCT_SCHEMA", std::string{product.schema_version});
    add_netcdf_var<std::string>(
        fo, "CAL.VALIDITY_DETAIL", product.validity_detail);
    add_netcdf_var<std::string>(fo, "CAL.TARGET_UNIT", product.target_unit);
    add_netcdf_var<std::string>(
        fo, "CAL.PHOTOMETRY_POLICY", std::string{product.photometry_policy});
    add_netcdf_var<std::string>(
        fo, "CAL.FACTOR_COMPOSITION", std::string{product.factor_composition});
    add_netcdf_var<std::string>(
        fo, "CAL.FACTOR_PROVENANCE", std::string{product.factor_provenance});
    add_netcdf_var<std::string>(
        fo, "CAL.COMPATIBILITY_FCF_SEMANTICS",
        std::string{product.compatibility_fcf_semantics});
    add_netcdf_var<std::string>(
        fo, "CAL.WEIGHT_RECIPIENT_SEMANTICS",
        std::string{product.weight_recipient_semantics});
    add_netcdf_var<std::string>(
        fo, "CAL.COMPACT_COVARIANCE_STATE",
        std::string{product.compact_covariance_state});
    add_netcdf_var(
        fo, "CAL.OBSERVATION_FLXSCALE_CORRECTION_APPLIED",
        product.observation_flxscale_correction_applied);
    add_netcdf_var(
        fo, "CAL.APPLIED_OBSERVATION_FLXSCALE_CORRECTION",
        product.applied_observation_flxscale_correction);
    add_netcdf_var<std::string>(
        fo, "CAL.OBSERVATION_FLXSCALE_CORRECTION_STATE",
        product.observation_flxscale_correction_state);
    add_netcdf_var<std::string>(
        fo, "CAL.OBSERVATION_FLXSCALE_CORRECTION_SOURCE_IDENTITY",
        product.observation_flxscale_correction_source_identity);
    add_netcdf_var<std::string>(
        fo, "CAL.OBSERVATION_FLXSCALE_CORRECTION_RECIPIENT_IDENTITY",
        product.observation_flxscale_correction_recipient_identity);
    add_netcdf_var<std::string>(
        fo, "CAL.APT_ARTIFACT_SHA256", product.apt_artifact_sha256);
    add_netcdf_var<std::string>(
        fo, "CAL.ACQUISITION_BINDING_SHA256",
        product.acquisition_binding_sha256);
    add_netcdf_var<std::string>(
        fo, "CAL.RAW_OBSERVATION_IDENTITY", product.raw_observation_identity);
    add_netcdf_var<std::string>(
        fo, "CAL.ACQUISITION_BINDING_MODE", product.acquisition_binding_mode);
    add_netcdf_var<std::string>(
        fo, "CAL.ACQUISITION_KEY_SCHEMA", product.acquisition_key_schema);
    add_netcdf_var<std::string>(
        fo, "CAL.RESPONSE_IDENTITY", product.response_identity);
    add_netcdf_var(fo, "CAL.JOIN_AVAILABLE", product.valid());
    if (product.valid()) {
        add_netcdf_var<std::string>(
            fo, "CAL.CALIBRATION_IDENTITY", product.calibration_identity);
        add_netcdf_var<std::string>(
            fo, "CAL.PACKAGE_IDENTITY", product.package_identity);
        add_netcdf_var<std::string>(fo, "CALID", product.calibration_identity);
        add_netcdf_var<std::string>(fo, "CALPKGID", product.package_identity);
    }
    add_netcdf_var<std::string>(
        fo, "CAL.CONDITIONAL_VARIANCE_TRANSFER",
        std::string{product.conditional_variance_transfer});
    add_netcdf_var<std::string>(
        fo, "CAL.CONDITIONAL_INVERSE_VARIANCE_TRANSFER",
        std::string{product.conditional_inverse_variance_transfer});
    add_netcdf_var<std::string>(
        fo, "CAL.PRECISION_LIMITATION", std::string{product.precision_limitation});
    add_netcdf_var<std::string>(
        fo, "CAL.NUISANCE_STATES",
        timestream::calibration_nuisance_state_summary(product));
    const auto minimum_total_multiplier =
        timestream::minimum_total_signal_multiplier(product);
    const auto maximum_total_multiplier =
        timestream::maximum_total_signal_multiplier(product);
    const bool total_multiplier_extrema_available =
        std::isfinite(minimum_total_multiplier) &&
        std::isfinite(maximum_total_multiplier);
    add_netcdf_var(
        fo, "CAL.TOTAL_MULTIPLIER_EXTREMA_AVAILABLE",
        total_multiplier_extrema_available);
    if (total_multiplier_extrema_available) {
        add_netcdf_var(
            fo, "CAL.MINIMUM_TOTAL_MULTIPLIER", minimum_total_multiplier);
        add_netcdf_var(
            fo, "CAL.MAXIMUM_TOTAL_MULTIPLIER", maximum_total_multiplier);
    }
    const bool tau225_available =
        std::isfinite(rtcproc.calibration.realized_tau225);
    add_netcdf_var(fo, "CAL.TAU225_AVAILABLE", tau225_available);
    if (tau225_available) {
        add_netcdf_var(
            fo, "CAL.TAU225", rtcproc.calibration.realized_tau225);
    }
    const bool reduction_max_tau225_available =
        std::isfinite(rtcproc.calibration.reduction_maximum_tau225);
    add_netcdf_var(
        fo, "CAL.REDUCTION_MAX_TAU225_AVAILABLE",
        reduction_max_tau225_available);
    if (reduction_max_tau225_available) {
        add_netcdf_var(
            fo, "CAL.REDUCTION_MAX_TAU225",
            rtcproc.calibration.reduction_maximum_tau225);
    }
    add_netcdf_var<std::string>(
        fo, "CAL.REDUCTION_QUALITY_REGIME",
        rtcproc.calibration.reduction_calibration_quality_regime);
    add_netcdf_var<std::string>(
        fo, "CAL.TAU_FRAME",
        std::string{"line_of_sight_at_sample_elevation"});
    add_netcdf_var(fo, "CAL.X_REF", 0.0);
}
