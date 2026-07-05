#pragma once

// Included by rtcdiag_network_outputs.h inside namespace citlali::pipeline.

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

