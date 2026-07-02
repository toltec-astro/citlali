#pragma once

#include <cstddef>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

namespace citlali::pipeline {

using PtcDiagVarList = std::vector<std::pair<std::string, std::string>>;

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
    const PtcDiagVarList &int_vars,
    const PtcDiagVarList &double_vars,
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

inline PtcDiagVarList ptcdiag_corr_network_int_vars() {
    return {
        {"corr_nw_n_groups", "number of final corr_nw cleaning groups per network"},
        {"corr_nw_n_groups_raw", "number of raw connected components before min_group_size filtering"},
        {"corr_nw_n_det_input", "input detector count in each network block"},
        {"corr_nw_n_det_candidates", "detectors passing apt flag and min_good_frac"},
        {"corr_nw_n_det_used", "candidate detectors with finite non-zero std for correlation"},
        {"corr_nw_n_det_grouped", "detectors included in final cleaned corr_nw groups"},
        {"corr_nw_n_det_ungrouped", "detectors excluded from final cleaned corr_nw groups"},
        {"corr_nw_sample_step", "time decimation factor used for corr_nw grouping"},
    };
}

inline PtcDiagVarList ptcdiag_weight_corr_int_vars() {
    return {
        {"weight_corr_penalty_n_det_input", "detector count in each network block"},
        {"weight_corr_penalty_n_det_candidates", "detectors passing apt flag and min_good_frac"},
        {"weight_corr_penalty_n_det_used", "candidate detectors with finite non-zero std"},
        {"weight_corr_penalty_n_det_weighted", "detectors with positive map weight multiplied by penalty factor"},
        {"weight_corr_penalty_sample_step", "time decimation factor used for penalty metrics"},
    };
}

inline PtcDiagVarList ptcdiag_weight_corr_double_vars(
    const std::string &factor_comment) {
    return {
        {"weight_corr_penalty_factor", factor_comment},
        {"weight_corr_penalty_severity", "normalized [0,1] severity used to derive weight_corr_penalty_factor"},
        {"weight_corr_penalty_pair_med_abs_corr", "median absolute sampled detector-detector correlation per network"},
        {"weight_corr_penalty_cm_el_abs_corr", "absolute correlation between network common mode and TelElAct"},
        {"weight_corr_penalty_cm_low_mid_ratio", "common-mode low/mid bandpower ratio"},
    };
}

inline PtcDiagVarList ptcdiag_busy_row_int_vars() {
    return {
        {"weight_busy_row_suppression_applied", "1 if busy-row weight suppression was applied to this scan/network block, else 0"},
        {"weight_busy_row_suppression_busy_network_vetoed", "1 if this scan/network exceeded the second-pass busy-network veto threshold, else 0"},
        {"weight_busy_row_suppression_n_candidate_clusters", "candidate second-pass residual cluster count used by the busy-row suppression rule"},
        {"weight_busy_row_suppression_n_det_weighted", "detectors with positive map weight multiplied by the busy-row suppression factor"},
    };
}

inline PtcDiagVarList ptcdiag_busy_row_double_vars() {
    return {
        {"weight_busy_row_suppression_factor", "multiplicative factor applied by busy-row suppression to positive detector map weights"},
        {"weight_busy_row_suppression_max_unflagged_residual_z", "largest absolute unflagged post-PCA residual z used by the busy-row suppression rule"},
    };
}

}  // namespace citlali::pipeline
