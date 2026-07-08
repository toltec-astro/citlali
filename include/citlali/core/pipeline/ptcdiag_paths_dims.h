#pragma once

// Included by ptcdiag_netcdf.h inside namespace citlali::pipeline.

using PtcDiagVarList = std::vector<std::pair<std::string, std::string>>;

inline std::string diagnostic_raw_directory(
    const std::string &obsnum_dir_name,
    const std::string &tod_output_subdir_name) {
    std::string dir_name = obsnum_dir_name + "raw/";
    if (citlali::config::has_config_value(tod_output_subdir_name)) {
        dir_name += tod_output_subdir_name + "/";
    }
    return dir_name;
}

inline std::string diagnostic_netcdf_filename(
    const std::string &filename) {
    return filename + ".nc";
}

template <auto DataType, auto ProductType, auto FilterType, class ToltecIo>
std::string diagnostic_output_netcdf_filename(
    ToltecIo &toltec_io, const std::string &obsnum_dir_name,
    const std::string &tod_output_subdir_name,
    citlali::config::ReductionType reduction_type, const std::string &obsnum,
    bool simulated_observation) {
    const std::string reduction_type_name{
        citlali::config::to_string(reduction_type)};
    const std::string dir_name =
        diagnostic_raw_directory(obsnum_dir_name, tod_output_subdir_name);
    const auto filename =
        toltec_io.template create_filename<DataType, ProductType, FilterType>(
            dir_name, reduction_type_name, "", obsnum, simulated_observation);
    return diagnostic_netcdf_filename(filename);
}

inline double ptcdiag_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr int ptcdiag_fill_int() {
    return -2147483647;
}

inline void add_ptc_eigenvalue_dim(netCDF::NcFile &fo,
                                   Eigen::Index n_eigenvalues) {
    fo.addDim("n_eigs", n_eigenvalues);
}

struct PtcDiagDims {
    netCDF::NcDim n_scans;
    netCDF::NcDim n_dets;
    std::vector<netCDF::NcDim> det;
    std::vector<std::size_t> det_chunks;
    std::size_t n_det_values;
};

inline PtcDiagDims add_ptcdiag_dims(netCDF::NcFile &fo,
                                    Eigen::Index n_scans,
                                    Eigen::Index n_dets) {
    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_scans);
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);
    const std::vector<netCDF::NcDim> det_dims = {
        n_scans_dim, n_dets_dim};
    const std::vector<std::size_t> det_chunks = {
        1, static_cast<std::size_t>(n_dets)};
    const auto n_scan_values = static_cast<std::size_t>(n_scans);
    const auto n_det_values = static_cast<std::size_t>(n_dets);

    return {
        n_scans_dim,
        n_dets_dim,
        det_dims,
        det_chunks,
        n_scan_values * n_det_values};
}

inline void add_ptc_weights_var(netCDF::NcFile &fo,
                                const std::vector<netCDF::NcDim> &dims,
                                const std::string &signal_unit) {
    netCDF::NcVar weights_v = fo.addVar("weights", netCDF::ncDouble, dims);
    weights_v.putAtt("units", "(" + signal_unit + ")^-2");
}

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

inline void add_diagnostic_output_scan_index(netCDF::NcFile &fo,
                                             netCDF::NcDim n_scans_dim,
                                             Eigen::Index n_scans,
                                             int fill_value) {
    netCDF::NcVar v = fo.addVar("output_scan_index", netCDF::ncInt,
                                n_scans_dim);
    v.putAtt("units", "N/A");
    v.putAtt("comment", "1-based original scan index from the full observation");
    const auto values = diagnostic_output_scan_indices(n_scans, fill_value);
    v.putVar(values.data());
}
