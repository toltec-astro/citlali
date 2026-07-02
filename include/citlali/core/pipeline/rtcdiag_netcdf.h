#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/utils/netcdf_io.h>

namespace citlali::pipeline {

template <class Calib>
std::vector<int> diagnostic_array_ids(const Calib &calib, int fill_value) {
    std::vector<int> ids(static_cast<std::size_t>(calib.n_arrays),
                         fill_value);
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.arrays(i));
    }
    return ids;
}

template <class Calib>
void add_rtcdiag_network_ids(netCDF::NcFile &fo, const Calib &calib,
                             netCDF::NcDim n_nws_rtcdiag_dim,
                             int fill_value) {
    netCDF::NcVar nw_ids_v =
        fo.addVar("rtc_diag_network_ids", netCDF::ncInt,
                  n_nws_rtcdiag_dim);
    nw_ids_v.putAtt("units", "N/A");
    nw_ids_v.putAtt("comment",
                    "network IDs corresponding to n_nws_rtcdiag axis");
    std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws),
                            fill_value);
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
    }
    nw_ids_v.putVar(nw_ids.data());
}

template <class Calib>
void add_rtcdiag_apt_double_vars(netCDF::NcFile &fo, Calib &calib,
                                 netCDF::NcDim n_dets_dim) {
    for (auto const &x : calib.apt) {
        netCDF::NcVar apt_v =
            fo.addVar("apt_" + x.first, netCDF::ncDouble, n_dets_dim);
        apt_v.putAtt("units", calib.apt_header_units[x.first]);
        apt_v.putVar(x.second.data());
    }
}

inline double rtcdiag_percentile_sorted(
    const std::vector<double> &sorted_values, double pct) {
    if (sorted_values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (sorted_values.size() == 1) {
        return sorted_values.front();
    }
    pct = std::min(100.0, std::max(0.0, pct));
    const double pos =
        (pct / 100.0) * static_cast<double>(sorted_values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}

inline void add_rtcdiag_scan_double(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment, netCDF::NcDim n_scans_dim,
    const std::vector<std::size_t> &scan_chunks,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, n_scans_dim);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, scan_chunks, 1);
    v.putVar(values.data());
}

inline void add_rtcdiag_scan_array_double(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment,
    const std::vector<netCDF::NcDim> &scan_array_dims,
    const std::vector<std::size_t> &scan_array_chunks,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, scan_array_dims);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, scan_array_chunks, 1);
    v.putVar(values.data());
}

inline void add_rtcdiag_det_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, det_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_det_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, det_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

}  // namespace citlali::pipeline
