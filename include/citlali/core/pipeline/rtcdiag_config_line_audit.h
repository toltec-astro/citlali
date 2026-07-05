#pragma once

// Included by rtcdiag_layout_config.h inside namespace citlali::pipeline.

template <class LineAudit>
void add_rtc_line_audit_config_vars(netCDF::NcFile &fo,
                                    const LineAudit &line_audit) {
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.ENABLED",
                   line_audit.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MIN_HZ",
                   line_audit.line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MAX_HZ",
                   line_audit.line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.SEGMENT_SEC",
                   line_audit.segment_sec);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_SEGMENT_SEC",
                   line_audit.min_segment_sec);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.OVERLAP_FRAC",
                   line_audit.overlap_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CONTINUUM_RADIUS_BINS",
                   line_audit.continuum_radius_bins);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PROMINENCE_THRESH",
                   line_audit.prominence_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CM_PROMINENCE_THRESH",
                   line_audit.cm_prominence_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_GOOD_FRAC",
                   line_audit.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_WINDOWS",
                   line_audit.min_windows);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_PEAKS_PER_DETECTOR",
                   line_audit.max_peaks_per_detector);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_DET",
                   line_audit.max_det);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_DET_FOR_NETWORK",
                   line_audit.min_det_for_network);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CLUSTER_TOL_HZ",
                   line_audit.cluster_tol_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTOR_FRAC",
                   line_audit.notch_min_detector_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTORS",
                   line_audit.notch_min_detectors);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_CM_PROMINENCE",
                   line_audit.notch_min_cm_prominence);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_PROMINENCE",
                   line_audit.detector_min_prominence);
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_LINE_POWER_FRAC",
        line_audit.detector_min_line_power_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.BAD_DETECTOR_MAX_CLUSTER_FRAC",
                   line_audit.bad_detector_max_cluster_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PRE_FILTER_ENABLED",
                   line_audit.pre_filter_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_ENABLED",
                   line_audit.post_filter_enabled);
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_SHARED_NOTCHES",
        line_audit.post_filter_apply_shared_notches);
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_DETECTOR_NOTCHES",
        line_audit.post_filter_apply_detector_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_ITERATIONS",
                   line_audit.post_filter_apply_iterations);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MIN_HZ",
                   line_audit.post_filter_line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MAX_HZ",
                   line_audit.post_filter_line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_MODEL_PROTECTED_ENABLED",
                   line_audit.ptc_model_protected_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_REQUIRE_MODEL_SUBTRACTED",
                   line_audit.ptc_require_model_subtracted);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_FIXED_NOTCHES",
                   line_audit.ptc_apply_fixed_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_SHARED_NOTCHES",
                   line_audit.ptc_apply_shared_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_DETECTOR_NOTCHES",
                   line_audit.ptc_apply_detector_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_ITERATIONS",
                   line_audit.ptc_apply_iterations);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MIN_HZ",
                   line_audit.ptc_line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MAX_HZ",
                   line_audit.ptc_line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_ENABLED",
                   line_audit.fixed_notch_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_COUNT",
                   static_cast<int>(line_audit.fixed_notch_freqs_hz.size()));
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_WIDTH_COUNT",
                   static_cast<int>(line_audit.fixed_notch_widths_hz.size()));
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_EXCLUSION_HALF_WIDTH_HZ",
        line_audit.fixed_notch_exclusion_half_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_SHARED_NOTCHES",
                   line_audit.apply_shared_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_SUPPORT_NETWORKS",
                   line_audit.apply_min_support_networks);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_DETECTOR_FRAC",
                   line_audit.apply_min_detector_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_CM_PROMINENCE",
                   line_audit.apply_min_common_mode_prominence);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_WIDTH_SCALE",
                   line_audit.apply_width_scale);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_WIDTH_HZ",
                   line_audit.apply_min_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_WIDTH_HZ",
                   line_audit.apply_max_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_NOTCHES",
                   line_audit.apply_max_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_CLUSTER_TOL_HZ",
                   line_audit.apply_cluster_tol_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_PROMINENCE",
                   line_audit.detector_notch_min_prominence);
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_LINE_POWER_FRAC",
        line_audit.detector_notch_min_line_power_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_NOTCHES",
                   line_audit.detector_notch_max_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_WIDTH_SCALE",
                   line_audit.detector_notch_width_scale);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_WIDTH_HZ",
                   line_audit.detector_notch_min_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_WIDTH_HZ",
                   line_audit.detector_notch_max_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_CONTEXT_SAMPLES",
                   line_audit.detector_notch_context_samples);
}

