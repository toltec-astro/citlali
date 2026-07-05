#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

namespace citlali::pipeline {

inline std::string stats_unit_or_empty(
    const std::map<std::string, std::string> &units,
    const std::string &stat) {
    const auto it = units.find(stat);
    return it == units.end() ? "" : it->second;
}

inline std::map<std::string, std::string>
detector_stats_units(const std::string &signal_unit) {
    return {
        {"rms", signal_unit},
        {"stddev", signal_unit},
        {"median", signal_unit},
        {"flagged_frac", "N/A"},
        {"weights", "1/(" + signal_unit + ")^2"}};
}

inline std::map<std::string, std::string>
group_stats_units(const std::string &signal_unit) {
    return {{"median_weights", "1/(" + signal_unit + ")^2"}};
}

template <class Values>
void add_stats_double_var(netCDF::NcFile &fo, const std::string &name,
                          const std::vector<netCDF::NcDim> &dims,
                          const Values &values,
                          const std::string &units) {
    netCDF::NcVar stat_v = fo.addVar(name, netCDF::ncDouble, dims);
    stat_v.putVar(values.data());
    stat_v.putAtt("units", units);
}

template <class Calib>
void add_stats_apt_double_vars(netCDF::NcFile &fo, const Calib &calib,
                               netCDF::NcDim n_dets_dim) {
    for (const auto &x : calib.apt) {
        netCDF::NcVar apt_v =
            fo.addVar("apt_" + x.first, netCDF::ncDouble, n_dets_dim);
        apt_v.putVar(x.second.data());
        apt_v.putAtt("units",
                     stats_unit_or_empty(calib.apt_header_units, x.first));
    }
}

template <class Calib, class AdcSnapData>
void add_stats_adc_snap_vars(netCDF::NcFile &fo, const Calib &calib,
                             const AdcSnapData &adc_snap_data) {
    if (adc_snap_data.empty()) {
        return;
    }

    netCDF::NcDim adc_snap_dim =
        fo.addDim("adcSnapDim", adc_snap_data[0].cols());
    netCDF::NcDim adc_snap_data_dim =
        fo.addDim("adcSnapDataDim", adc_snap_data[0].rows());
    const std::vector<netCDF::NcDim> adc_snap_dims = {
        adc_snap_dim, adc_snap_data_dim};

    Eigen::Index network_index = 0;
    for (const auto &x : adc_snap_data) {
        netCDF::NcVar adc_snap_v =
            fo.addVar("toltec" +
                          std::to_string(calib.nws(network_index)) +
                          "_adc_snap_data",
                      netCDF::ncDouble, adc_snap_dims);
        adc_snap_v.putVar(x.data());
        ++network_index;
    }
}

template <class Diagnostics, class Cleaner>
bool should_write_stats_eigenvalues(const Diagnostics &diagnostics,
                                    const Cleaner &cleaner) {
    return !diagnostics.evals.empty() && cleaner.n_calc > 0;
}

}  // namespace citlali::pipeline
