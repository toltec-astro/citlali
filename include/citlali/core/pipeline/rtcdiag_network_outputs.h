#pragma once

// Included by rtcdiag_netcdf.h inside namespace citlali::pipeline.

inline void add_rtcdiag_network_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, nw_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, nw_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_network_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, nw_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, nw_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

template <class AddInt>
void add_rtcdiag_network_detector_count_diag(const AddInt &add_int) {
    add_int("rtc_network_n_det_input",
            "input detector count in each RTC network block");
    add_int("rtc_network_n_det_used",
            "detectors passing the step-mask valid-sample threshold and finite robust scale");
    add_int("rtc_network_impulsive_n_det_used",
            "detectors passing the impulsive-coincidence valid-sample threshold and finite robust scale");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_line_audit_base_diag(const AddInt &add_int,
                                              const AddDouble &add_double) {
    add_int("rtc_network_line_audit_n_det_used",
            "detectors analyzed by the pre-filter RTC line audit in each network block");
    add_double("rtc_network_line_audit_shared_freq_hz",
               "frequency of the strongest shared narrowband RTC line family in each network block");
    add_int("rtc_network_line_audit_shared_detector_count",
            "number of detectors participating in the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_detector_frac",
               "fraction of audited detectors participating in the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_median_prominence",
               "median detector-level PSD prominence of the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_max_prominence",
               "maximum detector-level PSD prominence of the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_width_hz",
               "median linewidth of the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_line_power_frac",
               "median detector-level line-power fraction of the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_common_mode_freq_hz",
               "matched common-mode line frequency for the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_common_mode_prominence",
               "matched common-mode PSD prominence for the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_notch_score",
               "shared-line notch score, detector fraction times median prominence");
    add_int("rtc_network_line_audit_shared_recommend_notch",
            "1 if the strongest shared narrowband RTC line family met the current notch-candidate criteria");
    add_int("rtc_network_line_audit_n_applied_notches",
            "number of chunk-level shared-line RTC notches actually applied to this scan");
    add_int("rtc_network_line_audit_shared_applied_notch",
            "1 if the strongest shared narrowband RTC line family in this network matched an applied chunk-level RTC notch");
    add_double("rtc_network_line_audit_shared_applied_freq_hz",
               "center frequency of the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_applied_width_hz",
               "full-width bandwidth of the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_int("rtc_network_line_audit_shared_applied_support_network_count",
            "number of networks supporting the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_int("rtc_network_line_audit_detector_candidate_uid",
            "UID of the strongest detector-local RTC line candidate in each network block; -2147483647 means none");
    add_double("rtc_network_line_audit_detector_candidate_freq_hz",
               "frequency of the strongest detector-local RTC line candidate");
    add_double("rtc_network_line_audit_detector_candidate_prominence",
               "PSD prominence of the strongest detector-local RTC line candidate");
    add_double("rtc_network_line_audit_detector_candidate_line_power_frac",
               "line-power fraction of the strongest detector-local RTC line candidate");
    add_double("rtc_network_line_audit_detector_candidate_cluster_detector_frac",
               "shared-cluster detector fraction associated with the strongest detector-local RTC line candidate");
    add_int("rtc_network_line_audit_detector_candidate_recommend_flag",
            "1 if the strongest detector-local RTC line candidate met the current bad-detector criteria");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_line_audit_diag(
    const AddInt &add_int, const AddDouble &add_double,
    const std::string &prefix, const std::string &stage) {
    add_int(prefix + "_n_det_used",
            "detectors analyzed by the " + stage +
                " RTC line audit in each network block");
    add_double(prefix + "_shared_freq_hz",
               "frequency of the strongest shared narrowband " + stage +
                   " RTC line family in each network block");
    add_int(prefix + "_shared_detector_count",
            "number of detectors participating in the strongest shared narrowband " +
                stage + " RTC line family");
    add_double(prefix + "_shared_detector_frac",
               "fraction of audited detectors participating in the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_median_prominence",
               "median detector-level PSD prominence of the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_max_prominence",
               "maximum detector-level PSD prominence of the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_width_hz",
               "median linewidth of the strongest shared narrowband " + stage +
                   " RTC line family");
    add_double(prefix + "_shared_line_power_frac",
               "median detector-level line-power fraction of the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_common_mode_freq_hz",
               "matched common-mode line frequency for the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_common_mode_prominence",
               "matched common-mode PSD prominence for the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_notch_score",
               "shared-line notch score, detector fraction times median prominence");
    add_int(prefix + "_shared_recommend_notch",
            "1 if the strongest shared narrowband " + stage +
                " RTC line family met the current notch-candidate criteria");
    add_int(prefix + "_n_applied_notches",
            "number of chunk-level shared-line RTC notches actually applied in the " +
                stage + " stage");
    add_int(prefix + "_shared_applied_notch",
            "1 if the strongest shared narrowband " + stage +
                " RTC line family in this network matched an applied chunk-level RTC notch");
    add_double(prefix + "_shared_applied_freq_hz",
               "center frequency of the applied chunk-level RTC notch matched to the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_applied_width_hz",
               "full-width bandwidth of the applied chunk-level RTC notch matched to the strongest shared narrowband " +
                   stage + " RTC line family");
    add_int(prefix + "_shared_applied_support_network_count",
            "number of networks supporting the applied chunk-level RTC notch matched to the strongest shared narrowband " +
                stage + " RTC line family");
    add_int(prefix + "_detector_candidate_uid",
            "UID of the strongest detector-local " + stage +
                " RTC line candidate in each network block; -2147483647 means none");
    add_double(prefix + "_detector_candidate_freq_hz",
               "frequency of the strongest detector-local " + stage +
                   " RTC line candidate");
    add_double(prefix + "_detector_candidate_prominence",
               "PSD prominence of the strongest detector-local " + stage +
                   " RTC line candidate");
    add_double(prefix + "_detector_candidate_line_power_frac",
               "line-power fraction of the strongest detector-local " + stage +
                   " RTC line candidate");
    add_double(prefix + "_detector_candidate_cluster_detector_frac",
               "shared-cluster detector fraction associated with the strongest detector-local " +
                   stage + " RTC line candidate");
    add_int(prefix + "_detector_candidate_recommend_flag",
            "1 if the strongest detector-local " + stage +
                " RTC line candidate met the current bad-detector criteria");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_step_summary_diag(const AddInt &add_int,
                                           const AddDouble &add_double) {
    add_double("rtc_network_step_score_median",
               "median detector step score within each RTC network block");
    add_double("rtc_network_step_score_max",
               "maximum detector step score within each RTC network block");
    add_double("rtc_network_step_det_frac",
               "fraction of diagnostic-used detectors with strong step-like score in each RTC network block");
    add_double("rtc_network_step_alignment_frac",
               "fraction of strong-step detectors aligned in the dominant step-time cluster");
    add_int("rtc_network_step_dominant_sample",
            "dominant aligned step sample within each RTC network block; -2147483647 means unavailable");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_impulsive_summary_diag(
    const AddInt &add_int, const AddDouble &add_double) {
    add_double("rtc_network_impulsive_score_median",
               "median detector impulsive-event score within each RTC network block");
    add_double("rtc_network_impulsive_score_max",
               "maximum detector impulsive-event score within each RTC network block");
    add_double("rtc_network_impulsive_det_frac",
               "fraction of diagnostic-used detectors with impulsive-event score above the impulsive coincidence threshold");
    add_double("rtc_network_impulsive_alignment_frac",
               "fraction of impulsive-active detectors aligned in the dominant impulsive time cluster");
    add_int("rtc_network_impulsive_dominant_sample",
            "dominant aligned impulsive sample within each RTC network block; -2147483647 means unavailable");
}

template <class AddDouble>
void add_rtcdiag_network_common_mode_diag(const AddDouble &add_double) {
    add_double("rtc_network_cm_low_mid_ratio",
               "low-band to mid-band common-mode power ratio for each RTC network block");
    add_double("rtc_network_cm_peak_freq_hz",
               "frequency of the strongest common-mode spectral peak for each RTC network block");
    add_double("rtc_network_cm_peak_prominence",
               "prominence of the strongest common-mode spectral peak for each RTC network block");
}

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

template <class AddInt, class AddDouble>
void add_rtcdiag_standard_network_diag(const AddInt &add_int,
                                       const AddDouble &add_double) {
    add_rtcdiag_network_detector_count_diag(add_int);
    add_rtcdiag_network_line_audit_base_diag(add_int, add_double);
    add_rtcdiag_network_line_audit_diag(
        add_int, add_double, "rtc_network_post_line_audit", "post-filter");
    add_rtcdiag_network_step_summary_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_summary_diag(add_int, add_double);
    add_rtcdiag_network_common_mode_diag(add_double);
    add_rtcdiag_network_step_mask_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_mask_window_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_mask_trigger_diag(add_int);
    add_rtcdiag_network_impulsive_mask_candidate_diag(add_int, add_double);
}

inline void add_rtcdiag_standard_network_outputs(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_nw_values,
    int fill_int, double fill_double) {
    auto add_rtc_nw_double = [&](const std::string &name,
                                 const std::string &comment) {
        add_rtcdiag_network_double(
            fo, name, comment, nw_dims, nw_chunks, n_nw_values,
            fill_double);
    };
    auto add_rtc_nw_int = [&](const std::string &name,
                              const std::string &comment) {
        add_rtcdiag_network_int(
            fo, name, comment, nw_dims, nw_chunks, n_nw_values,
            fill_int);
    };

    add_rtcdiag_standard_network_diag(add_rtc_nw_int, add_rtc_nw_double);
}

