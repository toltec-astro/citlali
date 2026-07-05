#pragma once

// Included by ptcdiag_network_blocks.h inside namespace citlali::pipeline.

inline PtcDiagVarList ptcdiag_adaptive_pca_int_vars() {
    return {
        {"adaptive_pca_selector_used", "1 if the bounded adaptive PCA selector evaluated this scan/network block, else 0"},
        {"adaptive_pca_selector_fallback", "1 if adaptive PCA selector fell back to the configured baseline cut, else 0"},
        {"adaptive_pca_baseline_k", "configured baseline PCA cut for this scan/network block"},
        {"adaptive_pca_chosen_k", "adaptive PCA cut selected for this scan/network block"},
        {"adaptive_pca_runnerup_k", "second-best adaptive PCA cut for this scan/network block"},
        {"adaptive_pca_n_candidates", "number of candidate PCA cuts evaluated for this scan/network block"},
        {"adaptive_pca_n_det_input", "input detector count in this scan/network block before selector filtering"},
        {"adaptive_pca_n_det_used", "detector count retained for adaptive selector scoring"},
        {"adaptive_pca_n_time_used", "sample count retained for adaptive selector scoring"},
        {"adaptive_pca_sample_step", "time decimation factor used by the adaptive selector"},
    };
}

inline PtcDiagVarList ptcdiag_adaptive_pca_double_vars() {
    return {
        {"adaptive_pca_chosen_score", "final normalized adaptive selector score for the chosen PCA cut"},
        {"adaptive_pca_runnerup_score", "final normalized adaptive selector score for the runner-up PCA cut"},
        {"adaptive_pca_score_margin", "chosen minus runner-up score margin; more negative is a clearer adaptive choice"},
        {"adaptive_pca_chosen_med_abs_corr", "median absolute detector-detector correlation for the chosen adaptive PCA cut"},
        {"adaptive_pca_chosen_cm_low_mid_ratio", "common-mode low/mid bandpower ratio for the chosen adaptive PCA cut"},
        {"adaptive_pca_chosen_tail4_binom_z", "tail-excess metric for the chosen adaptive PCA cut"},
        {"adaptive_pca_chosen_top_mode_frac", "top residual covariance mode fraction for the chosen adaptive PCA cut"},
        {"adaptive_pca_eig_solve_msec", "milliseconds spent solving eigenmodes before adaptive scoring"},
        {"adaptive_pca_candidate_eval_msec", "milliseconds spent scoring candidate PCA cuts after eigen solve"},
        {"adaptive_pca_total_msec", "total adaptive PCA milliseconds for this scan/network block"},
    };
}

template <class Calib>
void add_ptcdiag_adaptive_pca_network_block(
    netCDF::NcFile &fo, const Calib &calib, netCDF::NcDim n_scans_dim,
    Eigen::Index n_scans, int fill_int, double fill_double) {
    add_ptcdiag_network_block(
        fo, calib, n_scans_dim, n_scans,
        "n_nws_adaptive_pca", "adaptive_pca_network_ids",
        "network IDs corresponding to n_nws_adaptive_pca axis",
        ptcdiag_adaptive_pca_int_vars(), ptcdiag_adaptive_pca_double_vars(),
        fill_int, fill_double);
}

