#pragma once

// Beammap detector TOD NetCDF output helpers.

#include <citlali/core/engine/detail/beammap_detector_tod_selection.h>

#include <Eigen/Core>

#include <netcdf>

#include <algorithm>
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

template <class Ptcs>
void put_detector_tod_signal_flags(
    netCDF::NcVar &signal_v,
    netCDF::NcVar &flags_v,
    const Ptcs &ptcs,
    const std::vector<int> &slot_scan_index,
    Eigen::Index n_dets,
    Eigen::Index n_slots,
    Eigen::Index n_samples_max,
    float fill_signal,
    signed char fill_flag) {
    std::vector<float> signal_block(
        static_cast<std::size_t>(n_slots) *
            static_cast<std::size_t>(n_samples_max),
        fill_signal);
    std::vector<signed char> flags_block(
        static_cast<std::size_t>(n_slots) *
            static_cast<std::size_t>(n_samples_max),
        fill_flag);

    for (Eigen::Index det = 0; det < n_dets; ++det) {
        std::fill(signal_block.begin(), signal_block.end(), fill_signal);
        std::fill(flags_block.begin(), flags_block.end(), fill_flag);
        for (Eigen::Index slot = 0; slot < n_slots; ++slot) {
            const auto meta_idx =
                beammap_detector_tod_selection::flat_detector_slot(
                    det, slot, n_slots);
            const int scan_1based = slot_scan_index[meta_idx];
            if (scan_1based <= 0) {
                continue;
            }
            const Eigen::Index scan_index =
                static_cast<Eigen::Index>(scan_1based - 1);
            if (scan_index < 0 ||
                scan_index >= static_cast<Eigen::Index>(ptcs.size())) {
                continue;
            }
            const auto &ptc = ptcs[scan_index];
            if (det >= ptc.scans.data.cols() ||
                det >= ptc.flags.data.cols()) {
                continue;
            }
            const Eigen::Index n_copy =
                std::min<Eigen::Index>(n_samples_max,
                                       ptc.scans.data.rows());
            for (Eigen::Index sample = 0; sample < n_copy; ++sample) {
                const auto data_idx =
                    static_cast<std::size_t>(slot) *
                        static_cast<std::size_t>(n_samples_max) +
                    static_cast<std::size_t>(sample);
                signal_block[data_idx] =
                    static_cast<float>(ptc.scans.data(sample, det));
                flags_block[data_idx] =
                    ptc.flags.data(sample, det)
                        ? static_cast<signed char>(1)
                        : static_cast<signed char>(0);
            }
        }

        std::vector<std::size_t> start = {
            static_cast<std::size_t>(det), 0, 0};
        std::vector<std::size_t> size = {
            1, static_cast<std::size_t>(n_slots),
            static_cast<std::size_t>(n_samples_max)};
        signal_v.putVar(start, size, signal_block.data());
        flags_v.putVar(start, size, flags_block.data());
    }
}

} // namespace beammap_detector_tod_netcdf_helpers
