#pragma once

// Included by stats_netcdf.h inside namespace citlali::pipeline.

struct StatsDims {
    netCDF::NcDim n_dets;
    netCDF::NcDim n_arrays;
    netCDF::NcDim n_chunks;
    std::vector<netCDF::NcDim> det_stat;
    std::vector<netCDF::NcDim> grp_stat;
};

inline StatsDims add_stats_dims(netCDF::NcFile &fo, Eigen::Index n_dets,
                                Eigen::Index n_arrays,
                                Eigen::Index n_chunks) {
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);
    netCDF::NcDim n_arrays_dim = fo.addDim("n_arrays", n_arrays);
    netCDF::NcDim n_chunks_dim = fo.addDim("n_chunks", n_chunks);

    return {
        n_dets_dim,
        n_arrays_dim,
        n_chunks_dim,
        {n_chunks_dim, n_dets_dim},
        {n_chunks_dim, n_arrays_dim}};
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

template <class Diagnostics>
void add_detector_stats_vars(
    netCDF::NcFile &fo, Diagnostics &diagnostics,
    const std::vector<netCDF::NcDim> &det_stat_dims,
    const std::map<std::string, std::string> &det_stats_header_units) {
    for (const auto &stat : diagnostics.det_stats_header) {
        add_stats_double_var(
            fo, stat, det_stat_dims, diagnostics.stats[stat],
            stats_unit_or_empty(det_stats_header_units, stat));
    }
}

template <class Diagnostics>
void add_group_stats_vars(
    netCDF::NcFile &fo, Diagnostics &diagnostics,
    const std::vector<netCDF::NcDim> &grp_stat_dims,
    const std::map<std::string, std::string> &grp_stats_header_units) {
    for (const auto &stat : diagnostics.grp_stats_header) {
        add_stats_double_var(
            fo, stat, grp_stat_dims, diagnostics.stats[stat],
            stats_unit_or_empty(grp_stats_header_units, stat));
    }
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

