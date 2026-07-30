#pragma once

#include <citlali/core/config/timestream_config.h>

namespace citlali::pipeline {

template <class RtcProc>
void adapt_raw_flagging_config_one_way(
    const citlali::config::RawTimeChunkConfig &raw, RtcProc &rtcproc) {
    const auto &despike = raw.despike;
    rtcproc.run_despike = despike.enabled;
    rtcproc.despiker.min_spike_sigma = despike.min_spike_sigma;
    rtcproc.despiker.time_constant_sec = despike.time_constant_sec;
    rtcproc.despiker.window_size = raw.filter.enabled
                                       ? raw.filter.n_terms
                                       : despike.window_size;
    rtcproc.despiker.run_filter = raw.filter.enabled;
    rtcproc.despiker.run_legacy = despike.legacy_enabled;
    rtcproc.despiker.grouping = "nw";
    rtcproc.despike_source_protection_config_enabled =
        despike.enabled ? despike.source_protection.enabled : true;
    rtcproc.despiker.source_protection_enabled = false;
    rtcproc.despiker.source_protection_radius_arcsec =
        despike.enabled ? despike.source_protection.radius_arcsec : 20.0;

    const auto &local = despike.local_residual;
    auto &target_local = rtcproc.despiker.local_residual;
    target_local.enabled = local.enabled;
    target_local.window_sec = local.window_sec;
    target_local.sigma_scale = local.sigma_scale;
    target_local.delta_sigma_scale = local.delta_sigma_scale;
    target_local.expand_with_filter = local.expand_with_filter;
    target_local.event_padding_sec = local.event_padding_sec;
    target_local.high_score_event_override = local.high_score_event_override;
    target_local.max_added_flagged_fraction =
        local.max_added_flagged_fraction;
    target_local.compact_raw_gate.enabled = local.compact_raw_gate.enabled;
    target_local.compact_raw_gate.candidate_rel_sigma_scale =
        local.compact_raw_gate.candidate_rel_sigma_scale;
    target_local.compact_raw_gate.window_sec =
        local.compact_raw_gate.window_sec;
    target_local.compact_raw_gate.half_peak_frac =
        local.compact_raw_gate.half_peak_frac;
    target_local.compact_raw_gate.max_width_sec =
        local.compact_raw_gate.max_width_sec;
    target_local.compact_raw_gate.max_step_shift_z =
        local.compact_raw_gate.max_step_shift_z;
    target_local.compact_delta_gate.enabled =
        local.compact_delta_gate.enabled;
    target_local.compact_delta_gate.window_sec =
        local.compact_delta_gate.window_sec;
    target_local.compact_delta_gate.half_peak_frac =
        local.compact_delta_gate.half_peak_frac;
    target_local.compact_delta_gate.max_width_sec =
        local.compact_delta_gate.max_width_sec;
    target_local.compact_delta_gate.max_step_shift_z =
        local.compact_delta_gate.max_step_shift_z;

    const auto &flagging = raw.flagging;
    rtcproc.coherent_iq_mode_observer_enabled =
        raw.coherent_iq_mode_observer.enabled;
    rtcproc.coherent_iq_mode_candidate_step_score_min =
        raw.coherent_iq_mode_observer.candidate_step_score_min;
    rtcproc.coherent_iq_mode_candidate_impulsive_score_min =
        raw.coherent_iq_mode_observer.candidate_impulsive_score_min;
    rtcproc.delta_f_min_Hz = flagging.delta_f_min_Hz;
    rtcproc.lower_inv_var_factor = flagging.lower_tod_inv_var_factor;
    rtcproc.upper_inv_var_factor = flagging.upper_tod_inv_var_factor;

    const auto &step = flagging.network_step_mask;
    auto &target_step = rtcproc.network_step_mask;
    target_step.enabled = step.enabled;
    target_step.step_window_sec = step.step_window_sec;
    target_step.step_score_thresh = step.step_score_thresh;
    target_step.min_good_frac = step.min_good_frac;
    target_step.min_det_used = step.min_det_used;
    target_step.min_step_det_frac = step.min_step_det_frac;
    target_step.min_alignment_frac = step.min_alignment_frac;
    target_step.cluster_tol_sec = step.cluster_tol_sec;
    target_step.mask_half_width_sec = step.mask_half_width_sec;
    target_step.max_flagged_fraction = step.max_flagged_fraction;

    const auto &capture = flagging.impulsive_capture;
    auto &target_capture = rtcproc.impulsive_capture;
    target_capture.enabled = capture.enabled;
    target_capture.min_good_frac = capture.min_good_frac;
    target_capture.min_event_z = capture.min_event_z;
    target_capture.near_event_z = capture.near_event_z;
    target_capture.max_events_per_network = capture.max_events_per_network;
    target_capture.snippet_pre_window_sec = capture.snippet_pre_window_sec;
    target_capture.snippet_post_window_sec = capture.snippet_post_window_sec;

    const auto &coincidence = flagging.impulsive_coincidence;
    auto &target_coincidence = rtcproc.impulsive_coincidence;
    target_coincidence.enabled = coincidence.enabled;
    target_coincidence.min_good_frac = coincidence.min_good_frac;
    target_coincidence.event_score_thresh = coincidence.event_score_thresh;
    target_coincidence.min_det_used = coincidence.min_det_used;
    target_coincidence.min_impulsive_det_frac =
        coincidence.min_impulsive_det_frac;
    target_coincidence.min_alignment_frac =
        coincidence.min_alignment_frac;
    target_coincidence.min_networks_aligned =
        coincidence.min_networks_aligned;
    target_coincidence.high_score_override_thresh =
        coincidence.high_score_override_thresh;
    target_coincidence.high_score_min_networks_aligned =
        coincidence.high_score_min_networks_aligned;
    target_coincidence.cluster_tol_sec = coincidence.cluster_tol_sec;
    target_coincidence.mask_pre_window_sec =
        coincidence.mask_pre_window_sec;
    target_coincidence.mask_post_window_sec =
        coincidence.mask_post_window_sec;
    target_coincidence.max_flagged_fraction =
        coincidence.max_flagged_fraction;
}

}  // namespace citlali::pipeline
