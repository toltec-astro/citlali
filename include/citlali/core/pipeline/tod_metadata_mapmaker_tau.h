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
