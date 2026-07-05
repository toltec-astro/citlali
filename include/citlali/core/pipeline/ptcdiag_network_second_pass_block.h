#pragma once

// Included by ptcdiag_network_blocks.h inside namespace citlali::pipeline.

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

