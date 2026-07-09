#pragma once

// Included by ptcdiag_network_blocks.h inside namespace citlali::pipeline.

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

template <class Ptcproc>
bool ptcdiag_corr_nw_requested(const Ptcproc &ptcproc) {
    for (const auto &grouping : ptcproc.cleaner.grouping) {
        if (citlali::config::is_corr_network_processed_cleaner_grouping(
                grouping)) {
            return true;
        }
    }
    return false;
}

inline void add_ptcdiag_corr_group_id(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &corr_det_dims,
    std::size_t n_values, int fill_value) {
    netCDF::NcVar corr_group_id_v =
        fo.addVar("corr_nw_group_id", netCDF::ncInt, corr_det_dims);
    corr_group_id_v.putAtt("units", "N/A");
    corr_group_id_v.putAtt(
        "comment",
        "corr_nw group index for each detector in each output scan; -2147483647 means not assigned");
    std::vector<int> init(n_values, fill_value);
    corr_group_id_v.putVar(init.data());
}

inline void add_ptcdiag_second_pass_added_flag(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    netCDF::NcVar added_flag_v =
        fo.addVar("ptc_second_pass_added_flag", netCDF::ncByte, dims);
    added_flag_v.putAtt("units", "N/A");
    added_flag_v.putAtt(
        "comment",
        "0=not added by PTC second-pass residual deglitching, 1=newly flagged by that pass");
    auto chunks = chunk_sizes;
    added_flag_v.setChunking(chunk_mode, chunks);
}

inline Eigen::VectorXd ptcdiag_padded_eigenvalues(
    const Eigen::VectorXd &evals, Eigen::Index n_calc, double fill_value) {
    Eigen::VectorXd values = Eigen::VectorXd::Constant(n_calc, fill_value);
    const Eigen::Index n_copy = std::min<Eigen::Index>(evals.size(), n_calc);
    if (n_copy > 0) {
        values.head(n_copy) = evals.head(n_copy);
    }
    return values;
}

template <class Calib>
void add_ptcdiag_corr_network_block(netCDF::NcFile &fo, const Calib &calib,
                                    netCDF::NcDim n_scans_dim,
                                    Eigen::Index n_scans,
                                    int fill_int, double fill_double) {
    add_ptcdiag_network_block(
        fo, calib, n_scans_dim, n_scans,
        "n_nws_corr", "corr_nw_network_ids",
        "network IDs corresponding to n_nws_corr axis",
        ptcdiag_corr_network_int_vars(), {}, fill_int, fill_double);
}
