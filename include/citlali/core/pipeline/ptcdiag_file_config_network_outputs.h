#pragma once

// Included by ptcdiag_netcdf.h inside namespace citlali::pipeline.

template <class PtcProc, class ReductionLearning>
void add_ptcdiag_file_config_vars(netCDF::NcFile &fo,
                                  const PtcProc &ptcproc,
                                  const ReductionLearning &learning,
                                  const citlali::config::ProcessedTimeChunkConfig
                                      &processed_config,
                                  const citlali::config::TimestreamFruitLoopsConfig
                                      &fruit_config) {
    add_weight_selection_config_vars(fo, ptcproc);
    add_reduction_learning_config_vars(fo, learning);
    add_ptc_weight_cutoff_config_vars(fo, ptcproc, true);
    add_ptcdiag_compact_config_vars(
        fo, processed_config, fruit_config);
}

template <class Calib>
void add_ptcdiag_standard_network_blocks(
    netCDF::NcFile &fo, const Calib &calib, netCDF::NcDim n_scans_dim,
    Eigen::Index n_scans, int fill_int, double fill_double) {
    add_ptcdiag_corr_network_block(
        fo, calib, n_scans_dim, n_scans, fill_int, fill_double);
    add_ptcdiag_weight_corr_network_block(
        fo, calib, n_scans_dim, n_scans,
        ptcdiag_weight_corr_factor_comment(), fill_int, fill_double);
    add_ptcdiag_busy_row_network_block(
        fo, calib, n_scans_dim, n_scans, fill_int, fill_double);
    add_ptcdiag_adaptive_pca_network_block(
        fo, calib, n_scans_dim, n_scans, fill_int, fill_double);
    add_ptcdiag_second_pass_network_block(
        fo, calib, n_scans_dim, n_scans,
        ptcdiag_second_pass_busy_network_comment(), true, fill_int,
        fill_double);
}
