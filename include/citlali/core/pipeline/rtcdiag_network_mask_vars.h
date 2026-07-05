#pragma once

// Included by rtcdiag_network_outputs.h inside namespace citlali::pipeline.

template <class AddInt, class AddDouble>
void add_rtcdiag_network_step_mask_diag(const AddInt &add_int,
                                        const AddDouble &add_double) {
    add_int("rtc_network_step_mask_applied",
            "1 if network_step_mask flagged a time window for this RTC network block, else 0");
    add_int("rtc_network_step_mask_start_sample",
            "inclusive starting sample of the applied network_step_mask window; -2147483647 means none");
    add_int("rtc_network_step_mask_end_sample",
            "inclusive ending sample of the applied network_step_mask window; -2147483647 means none");
    add_int("rtc_network_step_mask_window_samples",
            "number of RTC time samples in the applied network_step_mask window");
    add_int("rtc_network_step_mask_n_det_masked",
            "number of detectors included in the applied network_step_mask window");
    add_int("rtc_network_step_mask_n_det_samples_flagged",
            "number of previously good detector-samples newly flagged by network_step_mask");
    add_double("rtc_network_step_mask_flagged_fraction",
               "fraction of previously good detector-samples in the network block newly flagged by network_step_mask");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_impulsive_mask_window_diag(
    const AddInt &add_int, const AddDouble &add_double) {
    add_int("rtc_network_impulsive_mask_applied",
            "1 if impulsive_coincidence_mask flagged a time window for this RTC network block, else 0");
    add_int("rtc_network_impulsive_mask_start_sample",
            "inclusive starting sample of the applied impulsive_coincidence_mask window; -2147483647 means none");
    add_int("rtc_network_impulsive_mask_end_sample",
            "inclusive ending sample of the applied impulsive_coincidence_mask window; -2147483647 means none");
    add_int("rtc_network_impulsive_mask_window_samples",
            "number of RTC time samples in the applied impulsive_coincidence_mask window");
    add_int("rtc_network_impulsive_mask_n_det_masked",
            "number of detectors included in the applied impulsive_coincidence_mask window");
    add_int("rtc_network_impulsive_mask_n_det_samples_flagged",
            "number of previously good detector-samples newly flagged by impulsive_coincidence_mask");
    add_double("rtc_network_impulsive_mask_flagged_fraction",
               "fraction of previously good detector-samples in the network block newly flagged by impulsive_coincidence_mask");
}

template <class AddInt>
void add_rtcdiag_network_impulsive_mask_trigger_diag(
    const AddInt &add_int) {
    add_int("rtc_network_impulsive_mask_candidate_available",
            "1 if impulsive_coincidence_mask found a candidate for this RTC network block, else 0");
    add_int("rtc_network_impulsive_mask_local_trigger",
            "1 if the selected impulsive candidate satisfied the within-network trigger thresholds, else 0");
    add_int("rtc_network_impulsive_mask_cross_network_trigger",
            "1 if the selected impulsive candidate satisfied a cross-network alignment trigger, else 0");
    add_int("rtc_network_impulsive_mask_high_score_override_trigger",
            "1 if the selected impulsive candidate satisfied the looser high-score cross-network override, else 0");
    add_int("rtc_network_impulsive_mask_rejected_max_fraction",
            "1 if the selected impulsive candidate was rejected only because its proposed flagged fraction exceeded the configured limit");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_impulsive_mask_candidate_diag(
    const AddInt &add_int, const AddDouble &add_double) {
    add_int("rtc_network_impulsive_mask_candidate_center_sample",
            "center sample of the selected impulsive candidate before any cross-network recentering; -2147483647 means unavailable");
    add_int("rtc_network_impulsive_mask_cluster_center_sample",
            "median aligned sample of the selected cross-network impulsive cluster; -2147483647 means unavailable");
    add_int("rtc_network_impulsive_mask_cluster_network_count",
            "number of distinct networks participating in the selected impulsive candidate cluster");
    add_int("rtc_network_impulsive_mask_cluster_active_count",
            "number of detector-level impulsive events in the selected within-network cluster");
    add_int("rtc_network_impulsive_mask_total_active_count",
            "total number of detector-level impulsive events above threshold in the selected network block");
    add_double("rtc_network_impulsive_mask_cluster_peak_score",
               "maximum impulsive-event score found within the selected cross-network impulsive cluster");
    add_double("rtc_network_impulsive_mask_override_score",
               "score used by the high-score override path after combining the selected cluster peak with the strongest candidate score seen in participating networks");
    add_int("rtc_network_impulsive_mask_override_uses_network_peak",
            "1 if rtc_network_impulsive_mask_override_score came from a participating network's strongest candidate rather than the selected cluster peak");
    add_double("rtc_network_impulsive_mask_proposed_flagged_fraction",
               "fraction of previously good detector-samples that the selected impulsive mask window would newly flag before any rejection");
}

