#pragma once

// Beammap detector TOD NetCDF output helpers.

#include <Eigen/Core>

#include <netcdf>

#include <cmath>
#include <string>
#include <vector>

namespace beammap_detector_tod_netcdf_helpers {

inline void put_detector_int(netCDF::NcFile &fo,
                             const std::vector<netCDF::NcDim> &det_dims,
                             const std::string &name,
                             const std::string &comment,
                             const std::vector<int> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void put_detector_double(netCDF::NcFile &fo,
                                const std::vector<netCDF::NcDim> &det_dims,
                                const std::string &name,
                                const std::string &units,
                                const std::string &comment,
                                const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_dims);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void put_slot_int(netCDF::NcFile &fo,
                         const std::vector<netCDF::NcDim> &det_slot_dims,
                         const std::vector<std::size_t> &det_slot_chunks,
                         const std::string &name,
                         const std::string &comment,
                         const std::vector<int> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, det_slot_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, det_slot_chunks, 1);
    v.putVar(values.data());
}

inline void put_slot_double(netCDF::NcFile &fo,
                            const std::vector<netCDF::NcDim> &det_slot_dims,
                            const std::vector<std::size_t> &det_slot_chunks,
                            const std::string &name,
                            const std::string &units,
                            const std::string &comment,
                            const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_slot_dims);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, det_slot_chunks, 1);
    v.putVar(values.data());
}

template <class AptTable>
std::vector<int> apt_int_values(const AptTable &apt,
                                const std::string &key,
                                Eigen::Index n_dets,
                                int fill_value) {
    std::vector<int> values(static_cast<std::size_t>(n_dets), fill_value);
    auto it = apt.find(key);
    if (it != apt.end() && it->second.size() == n_dets) {
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            values[static_cast<std::size_t>(det)] =
                static_cast<int>(std::llround(it->second(det)));
        }
    }
    return values;
}

} // namespace beammap_detector_tod_netcdf_helpers
