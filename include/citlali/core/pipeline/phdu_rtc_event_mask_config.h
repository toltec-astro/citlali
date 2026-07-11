#pragma once

// Included by phdu_rtc_config.h inside namespace citlali::pipeline.

template <class FitsEntry, class FlaggingConfig, class Logger>
void add_phdu_rtc_event_mask_config(FitsEntry &fits_entry,
                                    const std::string &array_name,
                                    const Logger &logger,
                                    const FlaggingConfig &flagging) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.RTC.STEP_MASK.ENABLED",
               flagging.network_step_mask.enabled,
               "Enable RTC network-window step masking");
    add_double_key("CONFIG.RTC.STEP_MASK.STEP_WINDOW_SEC",
                   flagging.network_step_mask.step_window_sec,
                   "Window used for RTC step-score estimation");
    add_double_key("CONFIG.RTC.STEP_MASK.STEP_SCORE_THRESH",
                   flagging.network_step_mask.step_score_thresh,
                   "Detector step-score threshold for RTC step masking");
    add_double_key("CONFIG.RTC.STEP_MASK.MIN_GOOD_FRAC",
                   flagging.network_step_mask.min_good_frac,
                   "Minimum good-sample fraction for RTC step-mask metrics");
    hdu.addKey("CONFIG.RTC.STEP_MASK.MIN_DET_USED",
               static_cast<int>(flagging.network_step_mask.min_det_used),
               "Minimum detectors required in a network for RTC step masking");
    add_double_key("CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC",
                   flagging.network_step_mask.min_step_det_frac,
                   "Minimum step-like detector fraction for RTC step masking");
    add_double_key(
        "CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC",
        flagging.network_step_mask.min_alignment_frac,
        "Minimum aligned-step detector fraction for RTC step masking");
    add_double_key(
        "CONFIG.RTC.STEP_MASK.CLUSTER_TOL_SEC",
        flagging.network_step_mask.cluster_tol_sec,
        "Allowed timing tolerance for aligned RTC step clusters");
    add_double_key("CONFIG.RTC.STEP_MASK.HALF_WIDTH_SEC",
                   flagging.network_step_mask.mask_half_width_sec,
                   "Half-width of the applied RTC step-mask window");
    add_double_key(
        "CONFIG.RTC.STEP_MASK.MAX_FLAGGED_FRAC",
        flagging.network_step_mask.max_flagged_fraction,
        "Maximum allowed newly flagged detector-sample fraction per RTC network mask");

    hdu.addKey("CONFIG.RTC.IMPULSIVE.ENABLED",
               flagging.impulsive_capture.enabled,
               "Enable RTC impulsive-event snippet capture");
    add_double_key("CONFIG.RTC.IMPULSIVE.MIN_GOOD_FRAC",
                   flagging.impulsive_capture.min_good_frac,
                   "Minimum good-sample fraction for RTC impulsive capture");
    add_double_key("CONFIG.RTC.IMPULSIVE.MIN_EVENT_Z",
                   flagging.impulsive_capture.min_event_z,
                   "Minimum event score for RTC impulsive capture");
    add_double_key("CONFIG.RTC.IMPULSIVE.NEAR_EVENT_Z",
                   flagging.impulsive_capture.near_event_z,
                   "Near-threshold z for RTC impulsive counts");
    hdu.addKey("CONFIG.RTC.IMPULSIVE.MAX_EVENTS",
               static_cast<int>(
                   flagging.impulsive_capture.max_events_per_network),
               "Maximum captured impulsive detectors per network");
    add_double_key("CONFIG.RTC.IMPULSIVE.PRE_WINDOW_SEC",
                   flagging.impulsive_capture.snippet_pre_window_sec,
                   "Pre-event window of captured RTC impulsive snippets");
    add_double_key("CONFIG.RTC.IMPULSIVE.POST_WINDOW_SEC",
                   flagging.impulsive_capture.snippet_post_window_sec,
                   "Post-event window of captured RTC impulsive snippets");

    hdu.addKey("CONFIG.RTC.IMPULSIVE_COINCIDENCE.ENABLED",
               flagging.impulsive_coincidence.enabled,
               "Enable RTC impulsive coincidence masking");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_GOOD_FRAC",
        flagging.impulsive_coincidence.min_good_frac,
        "Minimum good-sample fraction for RTC impulsive coincidence metrics");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.EVENT_SCORE_THRESH",
        flagging.impulsive_coincidence.event_score_thresh,
        "Detector impulsive-event score threshold for RTC impulsive coincidence masking");
    hdu.addKey(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED",
        static_cast<int>(flagging.impulsive_coincidence.min_det_used),
        "Minimum detectors required in a network for RTC impulsive coincidence masking");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC",
        flagging.impulsive_coincidence.min_impulsive_det_frac,
        "Minimum impulsive-active detector fraction for RTC impulsive coincidence masking");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC",
        flagging.impulsive_coincidence.min_alignment_frac,
        "Minimum aligned-impulsive detector fraction for RTC impulsive coincidence masking");
    hdu.addKey(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_NETWORKS_ALIGNED",
        static_cast<int>(flagging.impulsive_coincidence.min_networks_aligned),
        "Minimum aligned networks required for cross-network RTC impulsive coincidence masking");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_OVERRIDE_THRESH",
        flagging.impulsive_coincidence.high_score_override_thresh,
        "High-score threshold enabling a looser cross-network RTC impulsive coincidence trigger");
    hdu.addKey(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_MIN_NETWORKS",
        static_cast<int>(
            flagging.impulsive_coincidence.high_score_min_networks_aligned),
        "Minimum aligned networks for the high-score override RTC impulsive coincidence trigger");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.CLUSTER_TOL_SEC",
        flagging.impulsive_coincidence.cluster_tol_sec,
        "Allowed timing tolerance for aligned RTC impulsive coincidence clusters");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.PRE_WINDOW_SEC",
        flagging.impulsive_coincidence.mask_pre_window_sec,
        "Pre-event window of the applied RTC impulsive coincidence mask");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.POST_WINDOW_SEC",
        flagging.impulsive_coincidence.mask_post_window_sec,
        "Post-event window of the applied RTC impulsive coincidence mask");
    add_double_key(
        "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MAX_FLAGGED_FRAC",
        flagging.impulsive_coincidence.max_flagged_fraction,
        "Maximum allowed newly flagged detector-sample fraction per RTC impulsive coincidence mask");
}
