#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/pipeline/ptcdiag_netcdf.h>

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

template <class EvalMap>
bool has_stats_eigenvalue_groups(const EvalMap &evals) {
    if (evals.empty()) {
        return false;
    }
    const auto first_it = evals.begin();
    return !first_it->second.empty() && !first_it->second[0].empty();
}

inline std::vector<netCDF::NcDim> add_stats_eigenvalue_dims(
    netCDF::NcFile &fo, Eigen::Index n_calc, std::size_t n_eig_groups) {
    netCDF::NcDim n_eigs_dim = fo.addDim("n_eigs", n_calc);
    netCDF::NcDim n_eig_grp_dim = fo.addDim("n_eig_grp", n_eig_groups);
    return {n_eig_grp_dim, n_eigs_dim};
}

inline std::string stats_eigenvalue_var_name(
    const std::string &grouping_name, Eigen::Index grouping_index,
    Eigen::Index chunk_index) {
    return "evals_" + grouping_name + "_" +
           std::to_string(grouping_index) + "_chunk_" +
           std::to_string(chunk_index);
}

inline std::vector<std::size_t> stats_eigenvalue_start_index() {
    return {0, 0};
}

inline std::vector<std::size_t> stats_eigenvalue_write_shape(
    Eigen::Index n_calc) {
    return {1, static_cast<std::size_t>(n_calc)};
}

inline netCDF::NcVar add_stats_eigenvalue_var(
    netCDF::NcFile &fo, const std::string &name,
    const std::vector<netCDF::NcDim> &dims) {
    return fo.addVar(name, netCDF::ncDouble, dims);
}

template <class EvalVectors>
void write_stats_eigenvalue_rows(netCDF::NcVar &eval_v,
                                 const EvalVectors &eval_vectors,
                                 Eigen::Index n_cleaner_eigenvalues,
                                 double fill_value) {
    auto start_eig_index = stats_eigenvalue_start_index();
    const auto eig_write_shape =
        stats_eigenvalue_write_shape(n_cleaner_eigenvalues);
    for (const auto &evals : eval_vectors) {
        Eigen::VectorXd padded_evals =
            ptcdiag_padded_eigenvalues(
                evals, n_cleaner_eigenvalues, fill_value);
        eval_v.putVar(start_eig_index, eig_write_shape, padded_evals.data());
        start_eig_index[0] += 1;
    }
}

template <class EvalVectors>
void add_stats_eigenvalue_group_var(
    netCDF::NcFile &fo, const std::string &name,
    const std::vector<netCDF::NcDim> &dims,
    const EvalVectors &eval_vectors,
    Eigen::Index n_cleaner_eigenvalues, double fill_value) {
    netCDF::NcVar eval_v = add_stats_eigenvalue_var(fo, name, dims);
    write_stats_eigenvalue_rows(
        eval_v, eval_vectors, n_cleaner_eigenvalues, fill_value);
}

}  // namespace citlali::pipeline
