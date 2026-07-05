#pragma once

// Included by timestream_config_mirror.h inside namespace citlali::pipeline.

template <class DespikeConfig, class RtcProc>
void mirror_raw_despike_config(DespikeConfig &target,
                               const RtcProc &rtcproc) {
    target.enabled = rtcproc.run_despike;
    target.source_protection.enabled =
        rtcproc.despike_source_protection_config_enabled;
    target.source_protection.radius_arcsec =
        rtcproc.despiker.source_protection_radius_arcsec;
    if (!rtcproc.run_despike) {
        return;
    }

    target.min_spike_sigma = rtcproc.despiker.min_spike_sigma;
    target.time_constant_sec = rtcproc.despiker.time_constant_sec;
    target.window_size = rtcproc.despiker.window_size;
    target.legacy_enabled = rtcproc.despiker.run_legacy;

    const auto &local = rtcproc.despiker.local_residual;
    auto &typed_local = target.local_residual;
    typed_local.enabled = local.enabled;
    typed_local.window_sec = local.window_sec;
    typed_local.sigma_scale = local.sigma_scale;
    typed_local.delta_sigma_scale = local.delta_sigma_scale;
    typed_local.expand_with_filter = local.expand_with_filter;
    typed_local.event_padding_sec = local.event_padding_sec;
    typed_local.high_score_event_override = local.high_score_event_override;
    typed_local.max_added_flagged_fraction = local.max_added_flagged_fraction;
    typed_local.compact_raw_gate.enabled = local.compact_raw_gate.enabled;
    typed_local.compact_raw_gate.candidate_rel_sigma_scale =
        local.compact_raw_gate.candidate_rel_sigma_scale;
    typed_local.compact_raw_gate.window_sec =
        local.compact_raw_gate.window_sec;
    typed_local.compact_raw_gate.half_peak_frac =
        local.compact_raw_gate.half_peak_frac;
    typed_local.compact_raw_gate.max_width_sec =
        local.compact_raw_gate.max_width_sec;
    typed_local.compact_raw_gate.max_step_shift_z =
        local.compact_raw_gate.max_step_shift_z;
    typed_local.compact_delta_gate.enabled =
        local.compact_delta_gate.enabled;
    typed_local.compact_delta_gate.window_sec =
        local.compact_delta_gate.window_sec;
    typed_local.compact_delta_gate.half_peak_frac =
        local.compact_delta_gate.half_peak_frac;
    typed_local.compact_delta_gate.max_width_sec =
        local.compact_delta_gate.max_width_sec;
    typed_local.compact_delta_gate.max_step_shift_z =
        local.compact_delta_gate.max_step_shift_z;
}

template <class RawFlaggingConfig, class RtcProc>
void mirror_raw_flagging_config(RawFlaggingConfig &target,
                                const RtcProc &rtcproc) {
    target.delta_f_min_Hz = rtcproc.delta_f_min_Hz;
    target.lower_tod_inv_var_factor = rtcproc.lower_inv_var_factor;
    target.upper_tod_inv_var_factor = rtcproc.upper_inv_var_factor;

    const auto &network_step_mask = rtcproc.network_step_mask;
    auto &typed_network_step = target.network_step_mask;
    typed_network_step.enabled = network_step_mask.enabled;
    typed_network_step.step_window_sec = network_step_mask.step_window_sec;
    typed_network_step.step_score_thresh = network_step_mask.step_score_thresh;
    typed_network_step.min_good_frac = network_step_mask.min_good_frac;
    typed_network_step.min_det_used =
        static_cast<int>(network_step_mask.min_det_used);
    typed_network_step.min_step_det_frac =
        network_step_mask.min_step_det_frac;
    typed_network_step.min_alignment_frac =
        network_step_mask.min_alignment_frac;
    typed_network_step.cluster_tol_sec = network_step_mask.cluster_tol_sec;
    typed_network_step.mask_half_width_sec =
        network_step_mask.mask_half_width_sec;
    typed_network_step.max_flagged_fraction =
        network_step_mask.max_flagged_fraction;

    const auto &impulsive_capture = rtcproc.impulsive_capture;
    auto &typed_capture = target.impulsive_capture;
    typed_capture.enabled = impulsive_capture.enabled;
    typed_capture.min_good_frac = impulsive_capture.min_good_frac;
    typed_capture.min_event_z = impulsive_capture.min_event_z;
    typed_capture.near_event_z = impulsive_capture.near_event_z;
    typed_capture.max_events_per_network =
        static_cast<int>(impulsive_capture.max_events_per_network);
    typed_capture.snippet_pre_window_sec =
        impulsive_capture.snippet_pre_window_sec;
    typed_capture.snippet_post_window_sec =
        impulsive_capture.snippet_post_window_sec;

    const auto &impulsive_coincidence = rtcproc.impulsive_coincidence;
    auto &typed_coincidence = target.impulsive_coincidence;
    typed_coincidence.enabled = impulsive_coincidence.enabled;
    typed_coincidence.min_good_frac = impulsive_coincidence.min_good_frac;
    typed_coincidence.event_score_thresh =
        impulsive_coincidence.event_score_thresh;
    typed_coincidence.min_det_used =
        static_cast<int>(impulsive_coincidence.min_det_used);
    typed_coincidence.min_impulsive_det_frac =
        impulsive_coincidence.min_impulsive_det_frac;
    typed_coincidence.min_alignment_frac =
        impulsive_coincidence.min_alignment_frac;
    typed_coincidence.min_networks_aligned =
        static_cast<int>(impulsive_coincidence.min_networks_aligned);
    typed_coincidence.high_score_override_thresh =
        impulsive_coincidence.high_score_override_thresh;
    typed_coincidence.high_score_min_networks_aligned =
        static_cast<int>(impulsive_coincidence.high_score_min_networks_aligned);
    typed_coincidence.cluster_tol_sec = impulsive_coincidence.cluster_tol_sec;
    typed_coincidence.mask_pre_window_sec =
        impulsive_coincidence.mask_pre_window_sec;
    typed_coincidence.mask_post_window_sec =
        impulsive_coincidence.mask_post_window_sec;
    typed_coincidence.max_flagged_fraction =
        impulsive_coincidence.max_flagged_fraction;
}

template <class LineAuditConfig, class LineAudit>
void mirror_raw_line_audit_config(LineAuditConfig &target,
                                  const LineAudit &source) {
    target.enabled = source.enabled;
    target.line_min_hz = source.line_min_hz;
    target.line_max_hz = source.line_max_hz;
    target.segment_sec = source.segment_sec;
    target.min_segment_sec = source.min_segment_sec;
    target.overlap_frac = source.overlap_frac;
    target.continuum_radius_bins =
        static_cast<int>(source.continuum_radius_bins);
    target.prominence_thresh = source.prominence_thresh;
    target.cm_prominence_thresh = source.cm_prominence_thresh;
    target.min_good_frac = source.min_good_frac;
    target.min_windows = static_cast<int>(source.min_windows);
    target.max_peaks_per_detector =
        static_cast<int>(source.max_peaks_per_detector);
    target.max_det = static_cast<int>(source.max_det);
    target.min_det_for_network = static_cast<int>(source.min_det_for_network);
    target.cluster_tol_hz = source.cluster_tol_hz;
    target.notch_min_detector_frac = source.notch_min_detector_frac;
    target.notch_min_detectors =
        static_cast<int>(source.notch_min_detectors);
    target.notch_min_cm_prominence = source.notch_min_cm_prominence;
    target.detector_min_prominence = source.detector_min_prominence;
    target.detector_min_line_power_frac =
        source.detector_min_line_power_frac;
    target.bad_detector_max_cluster_frac =
        source.bad_detector_max_cluster_frac;
    target.pre_filter_enabled = source.pre_filter_enabled;
    target.post_filter_enabled = source.post_filter_enabled;
    target.post_filter_apply_shared_notches =
        source.post_filter_apply_shared_notches;
    target.post_filter_apply_detector_notches =
        source.post_filter_apply_detector_notches;
    target.post_filter_apply_iterations =
        static_cast<int>(source.post_filter_apply_iterations);
    target.post_filter_line_min_hz = source.post_filter_line_min_hz;
    target.post_filter_line_max_hz = source.post_filter_line_max_hz;
    target.ptc_model_protected_enabled = source.ptc_model_protected_enabled;
    target.ptc_require_model_subtracted = source.ptc_require_model_subtracted;
    target.ptc_apply_fixed_notches = source.ptc_apply_fixed_notches;
    target.ptc_apply_shared_notches = source.ptc_apply_shared_notches;
    target.ptc_apply_detector_notches = source.ptc_apply_detector_notches;
    target.ptc_apply_iterations =
        static_cast<int>(source.ptc_apply_iterations);
    target.ptc_line_min_hz = source.ptc_line_min_hz;
    target.ptc_line_max_hz = source.ptc_line_max_hz;
    target.fixed_notch_enabled = source.fixed_notch_enabled;
    target.fixed_notch_freqs_hz = source.fixed_notch_freqs_hz;
    target.fixed_notch_widths_hz = source.fixed_notch_widths_hz;
    target.fixed_notch_exclusion_half_width_hz =
        source.fixed_notch_exclusion_half_width_hz;
    target.apply_shared_notches = source.apply_shared_notches;
    target.apply_min_support_networks =
        static_cast<int>(source.apply_min_support_networks);
    target.apply_min_detector_frac = source.apply_min_detector_frac;
    target.apply_min_common_mode_prominence =
        source.apply_min_common_mode_prominence;
    target.apply_width_scale = source.apply_width_scale;
    target.apply_min_width_hz = source.apply_min_width_hz;
    target.apply_max_width_hz = source.apply_max_width_hz;
    target.apply_max_notches = static_cast<int>(source.apply_max_notches);
    target.apply_cluster_tol_hz = source.apply_cluster_tol_hz;
    target.detector_notch_min_prominence =
        source.detector_notch_min_prominence;
    target.detector_notch_min_line_power_frac =
        source.detector_notch_min_line_power_frac;
    target.detector_notch_max_notches =
        static_cast<int>(source.detector_notch_max_notches);
    target.detector_notch_width_scale = source.detector_notch_width_scale;
    target.detector_notch_min_width_hz =
        source.detector_notch_min_width_hz;
    target.detector_notch_max_width_hz =
        source.detector_notch_max_width_hz;
    target.detector_notch_context_samples =
        static_cast<int>(source.detector_notch_context_samples);
}

