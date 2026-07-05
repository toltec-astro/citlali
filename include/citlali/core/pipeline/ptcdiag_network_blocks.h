#pragma once

// Included by ptcdiag_netcdf.h inside namespace citlali::pipeline.

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

template <class Ptcproc>
bool ptcdiag_corr_nw_requested(const Ptcproc &ptcproc) {
    for (const auto &grouping : ptcproc.cleaner.grouping) {
        if (grouping == "corr_nw") {
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

inline PtcDiagVarList ptcdiag_second_pass_double_vars() {
    return {
        {"ptc_second_pass_existing_flagged_fraction", "fraction of detector-samples already flagged before the PTC second pass in this scan/network"},
        {"ptc_second_pass_proposed_flagged_fraction", "fraction of detector-samples that the accepted PTC second-pass flags would cover in this scan/network"},
        {"ptc_second_pass_newly_flagged_fraction", "fraction of previously good detector-samples newly flagged by the PTC second pass in this scan/network"},
        {"ptc_second_pass_max_unflagged_residual_z", "largest absolute standardized residual remaining on previously unflagged PTC samples in this scan/network"},
        {"ptc_second_pass_top_candidate_cluster_peak_score", "peak event score of the strongest candidate second-pass cluster in this scan/network"},
        {"ptc_second_pass_top_event_score", "score of the strongest accepted second-pass event; NaN means none"},
    };
}

inline PtcDiagVarList ptcdiag_second_pass_int_vars(
    const std::string &busy_network_veto_comment,
    bool include_rejection_policy_vars) {
    PtcDiagVarList vars = {
        {"ptc_second_pass_busy_network_vetoed", busy_network_veto_comment},
        {"ptc_second_pass_n_candidate_clusters", "number of candidate second-pass residual clusters in this scan/network"},
        {"ptc_second_pass_n_candidate_events", "number of candidate detector-local residual events contributing to candidate clusters"},
        {"ptc_second_pass_n_accepted_clusters", "number of candidate clusters accepted for auto-flagging after the busy-network veto"},
        {"ptc_second_pass_n_accepted_events", "number of accepted detector-local residual events contributing to auto-flagging"},
    };
    if (include_rejection_policy_vars) {
        vars.insert(vars.end(), {
            {"ptc_second_pass_n_rejected_clusters", "number of candidate clusters rejected by busy-network/source-protection second-pass policy"},
            {"ptc_second_pass_n_rejected_events", "number of detector-local residual events in rejected second-pass clusters"},
            {"ptc_second_pass_n_source_protected_clusters", "number of candidate clusters with at least one source-protected detector event"},
            {"ptc_second_pass_n_source_protected_events", "number of detector-local residual events protected by the second-pass source mask"},
        });
    }
    vars.insert(vars.end(), {
        {"ptc_second_pass_n_det_with_added_flags", "number of detectors in this scan/network with at least one sample newly flagged by the PTC second pass"},
        {"ptc_second_pass_max_unflagged_residual_uid", "UID of the detector with the largest absolute unflagged post-PCA residual in this scan/network"},
        {"ptc_second_pass_top_candidate_cluster_sample", "median sample of the strongest candidate second-pass cluster; -2147483647 means none"},
        {"ptc_second_pass_top_candidate_cluster_n_detectors", "number of distinct detectors contributing to the strongest candidate second-pass cluster"},
        {"ptc_second_pass_top_candidate_cluster_n_events", "number of merged detector events contributing to the strongest candidate second-pass cluster"},
        {"ptc_second_pass_top_event_kind", "kind code of the strongest accepted second-pass event (0=raw_like,1=delta_like,-2147483647 means none)"},
        {"ptc_second_pass_top_event_uid", "UID of the strongest accepted second-pass event; -2147483647 means none"},
        {"ptc_second_pass_top_event_sample", "sample of the strongest accepted second-pass event; -2147483647 means none"},
    });
    return vars;
}

template <class Calib>
void add_ptcdiag_second_pass_network_block(
    netCDF::NcFile &fo, const Calib &calib, netCDF::NcDim n_scans_dim,
    Eigen::Index n_scans, const std::string &busy_network_veto_comment,
    bool include_rejection_policy_vars, int fill_int, double fill_double) {
    add_ptcdiag_network_block(
        fo, calib, n_scans_dim, n_scans,
        "n_nws_ptc_second_pass", "ptc_second_pass_network_ids",
        "network IDs corresponding to n_nws_ptc_second_pass axis",
        ptcdiag_second_pass_int_vars(
            busy_network_veto_comment, include_rejection_policy_vars),
        ptcdiag_second_pass_double_vars(), fill_int, fill_double);
}

inline std::string ptcdiag_weight_corr_factor_comment() {
    return "multiplicative weight penalty factor applied per network in each scan";
}

inline std::string ptcdiag_second_pass_busy_network_comment() {
    return "1 if this network had more candidate second-pass clusters than the normal auto-flag limit";
}

