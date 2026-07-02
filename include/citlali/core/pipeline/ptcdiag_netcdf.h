#pragma once

#include <cstddef>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

namespace citlali::pipeline {

inline std::vector<int> diagnostic_output_scan_indices(Eigen::Index n_scans,
                                                       int fill_value) {
    std::vector<int> output_scan_index(static_cast<std::size_t>(n_scans),
                                       fill_value);
    for (Eigen::Index i=0; i<n_scans; ++i) {
        output_scan_index[static_cast<std::size_t>(i)] =
            static_cast<int>(i + 1);
    }
    return output_scan_index;
}

inline std::vector<int> ptcdiag_output_scan_indices(Eigen::Index n_scans,
                                                    int fill_value) {
    return diagnostic_output_scan_indices(n_scans, fill_value);
}

template <class Calib>
std::vector<int> ptcdiag_apt_int_values(const Calib &calib,
                                        const std::string &key,
                                        int fill_value) {
    std::vector<int> values(static_cast<std::size_t>(calib.n_dets),
                            fill_value);
    const auto it = calib.apt.find(key);
    if (it != calib.apt.end() && it->second.size() == calib.n_dets) {
        for (Eigen::Index i=0; i<calib.n_dets; ++i) {
            values[static_cast<std::size_t>(i)] =
                static_cast<int>(std::lround(it->second(i)));
        }
    }
    return values;
}

template <class Calib>
std::vector<int> diagnostic_network_ids(const Calib &calib,
                                        int fill_value) {
    std::vector<int> ids(static_cast<std::size_t>(calib.n_nws),
                         fill_value);
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
    }
    return ids;
}

inline void add_ptcdiag_det_meta_int(netCDF::NcFile &fo,
                                     const std::string &name,
                                     const std::string &comment,
                                     netCDF::NcDim n_dets_dim,
                                     const std::vector<int> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, n_dets_dim);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void add_ptcdiag_det_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    auto chunks = det_chunks;
    v.setChunking(netCDF::NcVar::nc_CHUNKED, chunks);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_ptcdiag_det_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    auto chunks = det_chunks;
    v.setChunking(netCDF::NcVar::nc_CHUNKED, chunks);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

template <class Calib>
void add_ptcdiag_network_block(
    netCDF::NcFile &fo, const Calib &calib, netCDF::NcDim n_scans_dim,
    Eigen::Index n_scans, const std::string &dim_name,
    const std::string &id_name, const std::string &id_comment,
    const std::vector<std::pair<std::string, std::string>> &int_vars,
    const std::vector<std::pair<std::string, std::string>> &double_vars,
    int fill_int, double fill_double) {
    netCDF::NcDim n_nws_dim = fo.addDim(dim_name, calib.n_nws);
    netCDF::NcVar nw_ids_v = fo.addVar(id_name, netCDF::ncInt, n_nws_dim);
    nw_ids_v.putAtt("units", "N/A");
    nw_ids_v.putAtt("comment", id_comment);
    const auto nw_ids = diagnostic_network_ids(calib, fill_int);
    nw_ids_v.putVar(nw_ids.data());

    std::vector<netCDF::NcDim> dims = {n_scans_dim, n_nws_dim};
    const auto n_values =
        static_cast<std::size_t>(n_scans) *
        static_cast<std::size_t>(calib.n_nws);
    for (const auto &[name, comment] : int_vars) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", comment);
        std::vector<int> init(n_values, fill_int);
        v.putVar(init.data());
    }
    for (const auto &[name, comment] : double_vars) {
        netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, dims);
        v.putAtt("units", "N/A");
        v.putAtt("comment", comment);
        std::vector<double> init(n_values, fill_double);
        v.putVar(init.data());
    }
}

}  // namespace citlali::pipeline
