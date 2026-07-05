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

}  // namespace citlali::pipeline
