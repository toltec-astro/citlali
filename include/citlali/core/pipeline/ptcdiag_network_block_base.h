#pragma once

// Included by ptcdiag_network_blocks.h inside namespace citlali::pipeline.

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

