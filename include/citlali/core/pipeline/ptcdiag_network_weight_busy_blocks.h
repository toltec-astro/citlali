#pragma once

// Included by ptcdiag_network_blocks.h inside namespace citlali::pipeline.

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

template <class Calib>
void add_ptcdiag_weight_corr_network_block(
    netCDF::NcFile &fo, const Calib &calib, netCDF::NcDim n_scans_dim,
    Eigen::Index n_scans, const std::string &factor_comment,
    int fill_int, double fill_double) {
    add_ptcdiag_network_block(
        fo, calib, n_scans_dim, n_scans,
        "n_nws_wcorr", "weight_corr_penalty_network_ids",
        "network IDs corresponding to n_nws_wcorr axis",
        ptcdiag_weight_corr_int_vars(),
        ptcdiag_weight_corr_double_vars(factor_comment),
        fill_int, fill_double);
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

template <class Calib>
void add_ptcdiag_busy_row_network_block(
    netCDF::NcFile &fo, const Calib &calib, netCDF::NcDim n_scans_dim,
    Eigen::Index n_scans, int fill_int, double fill_double) {
    add_ptcdiag_network_block(
        fo, calib, n_scans_dim, n_scans,
        "n_nws_busy_row_suppression",
        "weight_busy_row_suppression_network_ids",
        "network IDs corresponding to n_nws_busy_row_suppression axis",
        ptcdiag_busy_row_int_vars(), ptcdiag_busy_row_double_vars(),
        fill_int, fill_double);
}

