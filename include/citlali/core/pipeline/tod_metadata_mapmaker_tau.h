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
}

template <class Rtcproc, class TelescopeData, class Alignment, class Calib,
          class ArrayNameMap>
void add_tod_mean_tau_vars(
    netCDF::NcFile &fo, bool extinction_enabled, Rtcproc &rtcproc,
    TelescopeData &tel_data, const Alignment &alignment,
    double tau_225_ghz, const Calib &calib,
    ArrayNameMap &array_name_map) {
    if (extinction_enabled) {
        Eigen::VectorXd tau_el(1);
        const auto tel_el_it = tel_data.find("TelElAct");
        if (tel_el_it == tel_data.end()) {
            throw std::logic_error(
                "TelElAct is unavailable for governing-compatible TOD tau");
        }
        tau_el << governing_compatibility_mean(
            tel_el_it->second, alignment);
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, tau_225_ghz);
        add_mean_tau_vars(fo, tau_freq, calib, array_name_map);
    }
    else {
        add_zero_mean_tau_vars(fo, calib, array_name_map);
    }
}
