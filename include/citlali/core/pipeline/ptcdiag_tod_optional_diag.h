#pragma once

// Included by ptcdiag_netcdf.h inside namespace citlali::pipeline.

template <class Calib, class Ptcproc>
void add_ptcdiag_tod_optional_diag(
    netCDF::NcFile &fo, const Calib &calib, const Ptcproc &ptcproc,
    const std::vector<netCDF::NcDim> &signal_dims,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes,
    netCDF::NcDim n_scans_dim, netCDF::NcDim n_dets_dim,
    Eigen::Index n_scans, int fill_int, double fill_double) {
    if (ptcproc.second_pass_local.enabled) {
        add_ptcdiag_second_pass_added_flag(
            fo, signal_dims, chunk_mode, chunk_sizes);

        add_ptcdiag_second_pass_network_block(
            fo, calib, n_scans_dim, n_scans,
            "1 if this network had more candidate second-pass clusters than the auto-flag limit and was diagnostic-only",
            false, fill_int, fill_double);
    }

    if (ptcproc.cleaner.corr_grouping.enabled &&
        ptcdiag_corr_nw_requested(ptcproc)) {
        std::vector<netCDF::NcDim> corr_det_dims = {n_scans_dim, n_dets_dim};
        add_ptcdiag_corr_group_id(
            fo, corr_det_dims,
            static_cast<std::size_t>(n_scans) *
                static_cast<std::size_t>(calib.n_dets),
            fill_int);

        add_ptcdiag_corr_network_block(
            fo, calib, n_scans_dim, n_scans, fill_int, fill_double);
    }

    if (ptcproc.weight_corr_penalty.enabled) {
        add_ptcdiag_weight_corr_network_block(
            fo, calib, n_scans_dim, n_scans,
            "multiplicative weight penalty factor applied per network in each output scan",
            fill_int, fill_double);
    }

    if (ptcproc.busy_row_suppression.enabled) {
        add_ptcdiag_busy_row_network_block(
            fo, calib, n_scans_dim, n_scans, fill_int, fill_double);
    }

    if (ptcproc.cleaner.adaptive_selector.enabled) {
        add_ptcdiag_adaptive_pca_network_block(
            fo, calib, n_scans_dim, n_scans, fill_int, fill_double);
    }
}

