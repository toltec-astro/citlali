#pragma once

// Included by rtcdiag_netcdf.h inside namespace citlali::pipeline.

inline double rtcdiag_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr int rtcdiag_fill_int() {
    return -2147483647;
}

template <class RtcProc>
double rtc_tod_stream_sample_rate(const RtcProc &rtcproc, double fsmp,
                                  double downsampled_fsmp) {
    return rtcproc.run_downsample ? downsampled_fsmp : fsmp;
}

struct RtcDiagDims {
    netCDF::NcDim n_scans;
    netCDF::NcDim n_dets;
    netCDF::NcDim n_arrays;
    netCDF::NcDim n_nws;
    std::vector<netCDF::NcDim> scan_array;
    std::vector<netCDF::NcDim> det;
    std::vector<netCDF::NcDim> nw;
    std::vector<std::size_t> scan_chunks;
    std::vector<std::size_t> scan_array_chunks;
    std::vector<std::size_t> det_chunks;
    std::vector<std::size_t> nw_chunks;
    std::size_t n_scan_values;
    std::size_t n_array_values;
    std::size_t n_scan_array_values;
    std::size_t n_det_values;
    std::size_t n_nw_values;
};

inline RtcDiagDims add_rtcdiag_dims(netCDF::NcFile &fo,
                                    Eigen::Index n_scans,
                                    Eigen::Index n_dets,
                                    Eigen::Index n_arrays,
                                    Eigen::Index n_nws) {
    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_scans);
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);
    netCDF::NcDim n_arrays_dim = fo.addDim("n_arrays", n_arrays);
    netCDF::NcDim n_nws_dim = fo.addDim("n_nws_rtcdiag", n_nws);

    const auto n_scan_values = static_cast<std::size_t>(n_scans);
    const auto n_array_values = static_cast<std::size_t>(n_arrays);
    const auto n_det_values = static_cast<std::size_t>(n_dets);
    const auto n_nw_values = static_cast<std::size_t>(n_nws);
    const std::vector<netCDF::NcDim> scan_array_dims = {
        n_scans_dim, n_arrays_dim};
    const std::vector<netCDF::NcDim> det_dims = {
        n_scans_dim, n_dets_dim};
    const std::vector<netCDF::NcDim> nw_dims = {
        n_scans_dim, n_nws_dim};
    const std::vector<std::size_t> scan_chunks = {
        static_cast<std::size_t>(std::max<Eigen::Index>(n_scans, 1))};
    const std::vector<std::size_t> scan_array_chunks = {
        1, static_cast<std::size_t>(std::max<Eigen::Index>(n_arrays, 1))};
    const std::vector<std::size_t> det_chunks = {
        1, n_det_values};
    const std::vector<std::size_t> nw_chunks = {
        1, n_nw_values};

    return {
        n_scans_dim,
        n_dets_dim,
        n_arrays_dim,
        n_nws_dim,
        scan_array_dims,
        det_dims,
        nw_dims,
        scan_chunks,
        scan_array_chunks,
        det_chunks,
        nw_chunks,
        n_scan_values,
        n_array_values,
        n_scan_values * n_array_values,
        n_scan_values * n_det_values,
        n_scan_values * n_nw_values};
}

template <class Calib>
std::vector<int> diagnostic_array_ids(const Calib &calib, int fill_value) {
    std::vector<int> ids(static_cast<std::size_t>(calib.n_arrays),
                         fill_value);
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.arrays(i));
    }
    return ids;
}

template <class Calib>
void add_rtcdiag_array_ids(netCDF::NcFile &fo, const Calib &calib,
                           netCDF::NcDim n_arrays_dim, int fill_value) {
    netCDF::NcVar array_ids_v =
        fo.addVar("rtc_diag_array_ids", netCDF::ncInt, n_arrays_dim);
    array_ids_v.putAtt("units", "N/A");
    array_ids_v.putAtt("comment",
                       "array IDs corresponding to n_arrays axis");
    const auto array_ids = diagnostic_array_ids(calib, fill_value);
    array_ids_v.putVar(array_ids.data());
}

template <class Calib>
void add_rtcdiag_network_ids(netCDF::NcFile &fo, const Calib &calib,
                             netCDF::NcDim n_nws_rtcdiag_dim,
                             int fill_value) {
    netCDF::NcVar nw_ids_v =
        fo.addVar("rtc_diag_network_ids", netCDF::ncInt,
                  n_nws_rtcdiag_dim);
    nw_ids_v.putAtt("units", "N/A");
    nw_ids_v.putAtt("comment",
                    "network IDs corresponding to n_nws_rtcdiag axis");
    std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws),
                            fill_value);
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
    }
    nw_ids_v.putVar(nw_ids.data());
}

template <class Calib>
void add_rtcdiag_apt_double_vars(netCDF::NcFile &fo, Calib &calib,
                                 netCDF::NcDim n_dets_dim) {
    for (auto const &x : calib.apt) {
        netCDF::NcVar apt_v =
            fo.addVar("apt_" + x.first, netCDF::ncDouble, n_dets_dim);
        apt_v.putAtt("units", calib.apt_header_units[x.first]);
        apt_v.putVar(x.second.data());
    }
}

template <class LocalResidual>
void add_rtc_local_despike_config_vars(
    netCDF::NcFile &fo, const LocalResidual &local_residual) {
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.ENABLED",
                   local_residual.enabled);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.WINDOW_SEC",
                   local_residual.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.SIGMA_SCALE",
                   local_residual.sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE",
                   local_residual.delta_sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.EXPAND_WITH_FILTER",
                   local_residual.expand_with_filter);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.EVENT_PADDING_SEC",
                   local_residual.event_padding_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.HIGH_SCORE_EVENT_OVERRIDE",
                   local_residual.high_score_event_override);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.MAX_ADDED_FLAGGED_FRAC",
                   local_residual.max_added_flagged_fraction);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED",
                   local_residual.compact_raw_gate.enabled);
    add_netcdf_var(
        fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE",
        local_residual.compact_raw_gate.candidate_rel_sigma_scale);
    add_netcdf_var(
        fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE",
        local_residual.compact_raw_gate.candidate_rel_sigma_scale *
            local_residual.sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC",
                   local_residual.compact_raw_gate.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC",
                   local_residual.compact_raw_gate.half_peak_frac);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC",
                   local_residual.compact_raw_gate.max_width_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z",
                   local_residual.compact_raw_gate.max_step_shift_z);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED",
                   local_residual.compact_delta_gate.enabled);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC",
                   local_residual.compact_delta_gate.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC",
                   local_residual.compact_delta_gate.half_peak_frac);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC",
                   local_residual.compact_delta_gate.max_width_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z",
                   local_residual.compact_delta_gate.max_step_shift_z);
}

template <class StepMask>
void add_rtc_step_mask_config_vars(netCDF::NcFile &fo,
                                   const StepMask &step_mask) {
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.ENABLED",
                   step_mask.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.STEP_WINDOW_SEC",
                   step_mask.step_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.STEP_SCORE_THRESH",
                   step_mask.step_score_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_GOOD_FRAC",
                   step_mask.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_DET_USED",
                   step_mask.min_det_used);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC",
                   step_mask.min_step_det_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC",
                   step_mask.min_alignment_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.CLUSTER_TOL_SEC",
                   step_mask.cluster_tol_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.HALF_WIDTH_SEC",
                   step_mask.mask_half_width_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MAX_FLAGGED_FRAC",
                   step_mask.max_flagged_fraction);
}

template <class ImpulsiveCapture>
void add_rtc_impulsive_capture_config_vars(
    netCDF::NcFile &fo, const ImpulsiveCapture &impulsive_capture) {
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.ENABLED",
                   impulsive_capture.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MIN_GOOD_FRAC",
                   impulsive_capture.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MIN_EVENT_Z",
                   impulsive_capture.min_event_z);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.NEAR_EVENT_Z",
                   impulsive_capture.near_event_z);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MAX_EVENTS",
                   impulsive_capture.max_events_per_network);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.PRE_WINDOW_SEC",
                   impulsive_capture.snippet_pre_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.POST_WINDOW_SEC",
                   impulsive_capture.snippet_post_window_sec);
}

template <class ImpulsiveCoincidence>
void add_rtc_impulsive_coincidence_config_vars(
    netCDF::NcFile &fo, const ImpulsiveCoincidence &impulsive_coincidence) {
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.ENABLED",
                   impulsive_coincidence.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_GOOD_FRAC",
                   impulsive_coincidence.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.EVENT_SCORE_THRESH",
                   impulsive_coincidence.event_score_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED",
                   impulsive_coincidence.min_det_used);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC",
                   impulsive_coincidence.min_impulsive_det_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC",
                   impulsive_coincidence.min_alignment_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_NETWORKS_ALIGNED",
                   impulsive_coincidence.min_networks_aligned);
    add_netcdf_var(
        fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_OVERRIDE_THRESH",
        impulsive_coincidence.high_score_override_thresh);
    add_netcdf_var(
        fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_MIN_NETWORKS",
        impulsive_coincidence.high_score_min_networks_aligned);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.CLUSTER_TOL_SEC",
                   impulsive_coincidence.cluster_tol_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.PRE_WINDOW_SEC",
                   impulsive_coincidence.mask_pre_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.POST_WINDOW_SEC",
                   impulsive_coincidence.mask_post_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MAX_FLAGGED_FRAC",
                   impulsive_coincidence.max_flagged_fraction);
}

template <class RtcProc>
void add_rtc_event_mask_config_vars(netCDF::NcFile &fo,
                                    const RtcProc &rtcproc) {
    add_rtc_step_mask_config_vars(fo, rtcproc.network_step_mask);
    add_rtc_impulsive_capture_config_vars(fo, rtcproc.impulsive_capture);
    add_rtc_impulsive_coincidence_config_vars(
        fo, rtcproc.impulsive_coincidence);
}

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

template <class RtcProc, class ReductionLearning, class OuterContext>
void add_rtcdiag_file_config_vars(netCDF::NcFile &fo,
                                  const RtcProc &rtcproc,
                                  const ReductionLearning &learning,
                                  bool verbose_mode,
                                  OuterContext outer_context_samples,
                                  double rtc_sample_rate_hz) {
    add_netcdf_var(fo, "RTC_SAMPRATE", rtc_sample_rate_hz);
    add_netcdf_var(fo, "CONFIG.TODFILTERED", rtcproc.run_tod_filter);
    add_netcdf_var(fo, "CONFIG.TODFILTER.FREQ_HIGH_HZ",
                   rtcproc.filter.freq_high_Hz);
    add_netcdf_var(fo, "CONFIG.TODFILTER.FREQ_LOW_HZ",
                   rtcproc.filter.freq_low_Hz);
    add_netcdf_var(fo, "CONFIG.TODFILTER.N_TERMS",
                   rtcproc.filter.n_terms);
    add_tod_filter_edge_guard_config_vars(
        fo, rtcproc.filter_edge_guard, outer_context_samples,
        rtcproc.tod_output_outer_context_samples);

    // Keep a compact provenance subset so rtcdiag is interpretable without the RTC TOD.
    add_netcdf_var(fo, "CONFIG.VERBOSE", verbose_mode);
    add_reduction_learning_config_vars(fo, learning, false);
    add_netcdf_var(fo, "CONFIG.DESPIKED", rtcproc.run_despike);
    add_rtc_local_despike_config_vars(fo, rtcproc.despiker.local_residual);
    add_rtc_event_mask_config_vars(fo, rtcproc);
    add_rtc_line_audit_config_vars(fo, rtcproc.line_audit);
    add_netcdf_var(fo, "CONFIG.INV_VAR.WINDOW_SEC",
                   rtcproc.remove_bad_dets_window_sec);
}

template <class LineAudit>
void add_rtc_line_audit_config_vars_if_absent(
    netCDF::NcFile &fo, const LineAudit &line_audit) {
    if (fo.getVar("CONFIG.RTC.LINE_AUDIT.ENABLED").isNull()) {
        add_rtc_line_audit_config_vars(fo, line_audit);
    }
}

