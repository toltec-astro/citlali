#pragma once

#include <cstddef>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

namespace citlali::pipeline {

inline std::vector<int> ptcdiag_output_scan_indices(Eigen::Index n_scans,
                                                    int fill_value) {
    std::vector<int> output_scan_index(static_cast<std::size_t>(n_scans),
                                       fill_value);
    for (Eigen::Index i=0; i<n_scans; ++i) {
        output_scan_index[static_cast<std::size_t>(i)] =
            static_cast<int>(i + 1);
    }
    return output_scan_index;
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
    v.setChunking(netCDF::NcVar::nc_CHUNKED, det_chunks);
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
    v.setChunking(netCDF::NcVar::nc_CHUNKED, det_chunks);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

}  // namespace citlali::pipeline
