#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <string>

namespace citlali::pipeline {

inline YAML::Node raw_rtc_contract_node(
    const RawRtcContractState &contract) {
    YAML::Node node;
    node["assigned_grid_authority"] = contract.assigned_grid_authority;
    node["assigned_time_semantics"] = contract.assigned_time_semantics;
    node["physical_event_semantics"] =
        contract.physical_event_semantics;
    node["lattice_label"] = contract.lattice_label;
    node["phase_label"] = contract.phase_label;
    node["representative_assigned_time_rule"] =
        contract.representative_assigned_time_rule;
    node["edge_rule"] = contract.edge_rule;
    node["influence_support_policy"] =
        contract.influence_support_policy;
    node["operator_ordering"] = contract.operator_ordering;
    node["fir_normalization"] = contract.fir_normalization;
    node["downsample_normalization"] =
        contract.downsample_normalization;
    node["timing_sensitive_mask_accuracy"] =
        contract.timing_sensitive_mask_accuracy;
    node["detector_ordering"] = contract.detector_ordering;
    node["scientific_eligibility_required"] =
        contract.scientific_eligibility_required;
    node["complete_response_or_unavailable_required"] =
        contract.complete_response_or_unavailable_required;
    node["source_mask_fail_closed_required"] =
        contract.source_mask_fail_closed_required;
    return node;
}

inline YAML::Node raw_source_protection_request_node(
    const citlali::config::TimestreamSourceProtectionConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["radius_arcsec"] = config.radius_arcsec;
    return node;
}

inline YAML::Node raw_despike_request_node(
    const citlali::config::RawTimeChunkDespikeConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["min_spike_sigma"] = config.min_spike_sigma;
    node["time_constant_sec"] = config.time_constant_sec;
    node["window_size"] = config.window_size;
    node["legacy"]["enabled"] = config.legacy_enabled;
    node["source_protection"] =
        raw_source_protection_request_node(config.source_protection);

    const auto &local = config.local_residual;
    auto local_node = node["local_residual"];
    local_node["enabled"] = local.enabled;
    local_node["window_sec"] = local.window_sec;
    local_node["sigma_scale"] = local.sigma_scale;
    local_node["delta_sigma_scale"] = local.delta_sigma_scale;
    local_node["expand_with_filter"] = local.expand_with_filter;
    local_node["event_padding_sec"] = local.event_padding_sec;
    local_node["high_score_event_override"] =
        local.high_score_event_override;
    local_node["max_added_flagged_fraction"] =
        local.max_added_flagged_fraction;
    const auto &raw_gate = local.compact_raw_gate;
    auto raw_gate_node = local_node["compact_raw_gate"];
    raw_gate_node["enabled"] = raw_gate.enabled;
    raw_gate_node["candidate_rel_sigma_scale"] =
        raw_gate.candidate_rel_sigma_scale;
    raw_gate_node["window_sec"] = raw_gate.window_sec;
    raw_gate_node["half_peak_frac"] = raw_gate.half_peak_frac;
    raw_gate_node["max_width_sec"] = raw_gate.max_width_sec;
    raw_gate_node["max_step_shift_z"] = raw_gate.max_step_shift_z;
    const auto &delta_gate = local.compact_delta_gate;
    auto delta_gate_node = local_node["compact_delta_gate"];
    delta_gate_node["enabled"] = delta_gate.enabled;
    delta_gate_node["window_sec"] = delta_gate.window_sec;
    delta_gate_node["half_peak_frac"] = delta_gate.half_peak_frac;
    delta_gate_node["max_width_sec"] = delta_gate.max_width_sec;
    delta_gate_node["max_step_shift_z"] = delta_gate.max_step_shift_z;
    return node;
}

inline YAML::Node raw_downsample_request_node(
    const citlali::config::RawTimeChunkDownsampleConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["factor"] = config.factor;
    node["downsampled_freq_Hz"] = config.downsampled_freq_Hz;
    return node;
}

inline YAML::Node raw_filter_request_node(
    const citlali::config::RawTimeChunkFilterConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["a_gibbs"] = config.a_gibbs;
    node["freq_low_Hz"] = config.freq_low_Hz;
    node["freq_high_Hz"] = config.freq_high_Hz;
    node["n_terms"] = config.n_terms;
    node["notch"]["enabled"] = config.notch.enabled;
    node["notch"]["zero_phase"] = config.notch.zero_phase;
    node["notch"]["freqs_Hz"] = config.notch.freqs_Hz;
    node["notch"]["delta_f_Hz"] = config.notch.delta_f_Hz;
    const auto &guard = config.edge_guard;
    auto guard_node = node["edge_guard"];
    guard_node["enabled"] = guard.enabled;
    guard_node["mode"] =
        std::string{citlali::config::to_string(guard.mode)};
    guard_node["combine"] =
        std::string{citlali::config::to_string(guard.combine)};
    guard_node["min_samples"] = guard.min_samples;
    guard_node["extra_samples"] = guard.extra_samples;
    guard_node["max_samples"] = guard.max_samples;
    guard_node["iir_settle_attenuation"] = guard.iir_settle_attenuation;
    guard_node["apply_fir"] = guard.apply_fir;
    guard_node["apply_notch"] = guard.apply_notch;
    guard_node["apply_dynamic_notch"] = guard.apply_dynamic_notch;
    guard_node["apply_iir_highpass"] = guard.apply_iir_highpass;
    guard_node["apply_downsample"] = guard.apply_downsample;
    return node;
}

inline YAML::Node raw_iir_filter_request_node(
    const citlali::config::RawTimeChunkIirFilterConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["freq_Hz"] = config.freq_Hz;
    node["order"] = config.order;
    node["zero_phase"] = config.zero_phase;
    return node;
}

inline YAML::Node raw_flagging_request_node(
    const citlali::config::RawTimeChunkFlaggingConfig &config) {
    YAML::Node node;
    node["delta_f_min_Hz"] = config.delta_f_min_Hz;
    node["lower_tod_inv_var_factor"] = config.lower_tod_inv_var_factor;
    node["upper_tod_inv_var_factor"] = config.upper_tod_inv_var_factor;
    const auto &step = config.network_step_mask;
    auto step_node = node["network_step_mask"];
    step_node["enabled"] = step.enabled;
    step_node["step_window_sec"] = step.step_window_sec;
    step_node["step_score_thresh"] = step.step_score_thresh;
    step_node["min_good_frac"] = step.min_good_frac;
    step_node["min_det_used"] = step.min_det_used;
    step_node["min_step_det_frac"] = step.min_step_det_frac;
    step_node["min_alignment_frac"] = step.min_alignment_frac;
    step_node["cluster_tol_sec"] = step.cluster_tol_sec;
    step_node["mask_half_width_sec"] = step.mask_half_width_sec;
    step_node["max_flagged_fraction"] = step.max_flagged_fraction;
    const auto &capture = config.impulsive_capture;
    auto capture_node = node["impulsive_capture"];
    capture_node["enabled"] = capture.enabled;
    capture_node["min_good_frac"] = capture.min_good_frac;
    capture_node["min_event_z"] = capture.min_event_z;
    capture_node["near_event_z"] = capture.near_event_z;
    capture_node["max_events_per_network"] = capture.max_events_per_network;
    capture_node["snippet_pre_window_sec"] = capture.snippet_pre_window_sec;
    capture_node["snippet_post_window_sec"] =
        capture.snippet_post_window_sec;
    const auto &coincidence = config.impulsive_coincidence;
    auto coincidence_node = node["impulsive_coincidence"];
    coincidence_node["enabled"] = coincidence.enabled;
    coincidence_node["min_good_frac"] = coincidence.min_good_frac;
    coincidence_node["event_score_thresh"] = coincidence.event_score_thresh;
    coincidence_node["min_det_used"] = coincidence.min_det_used;
    coincidence_node["min_impulsive_det_frac"] =
        coincidence.min_impulsive_det_frac;
    coincidence_node["min_alignment_frac"] =
        coincidence.min_alignment_frac;
    coincidence_node["min_networks_aligned"] =
        coincidence.min_networks_aligned;
    coincidence_node["high_score_override_thresh"] =
        coincidence.high_score_override_thresh;
    coincidence_node["high_score_min_networks_aligned"] =
        coincidence.high_score_min_networks_aligned;
    coincidence_node["cluster_tol_sec"] = coincidence.cluster_tol_sec;
    coincidence_node["mask_pre_window_sec"] =
        coincidence.mask_pre_window_sec;
    coincidence_node["mask_post_window_sec"] =
        coincidence.mask_post_window_sec;
    coincidence_node["max_flagged_fraction"] =
        coincidence.max_flagged_fraction;
    return node;
}

inline YAML::Node coherent_iq_mode_observer_request_node(
    const citlali::config::RawTimeChunkCoherentIqModeObserverConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["template_paths"] = config.template_paths;
    node["candidate_step_score_min"] = config.candidate_step_score_min;
    node["candidate_impulsive_score_min"] =
        config.candidate_impulsive_score_min;
    node["candidate_cluster_tolerance_sec"] =
        config.candidate_cluster_tolerance_sec;
    node["pre_window_sec"] = config.pre_window_sec;
    node["guard_window_sec"] = config.guard_window_sec;
    node["post_window_sec"] = config.post_window_sec;
    node["cross_network_tolerance_sec"] =
        config.cross_network_tolerance_sec;
    node["max_candidates_per_scan_per_network"] =
        config.max_candidates_per_scan_per_network;
    return node;
}

inline YAML::Node raw_kernel_request_node(
    const citlali::config::RawTimeChunkKernelConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["filepath"] = config.filepath;
    node["type"] = config.type;
    node["fwhm_arcsec"] = config.fwhm_arcsec;
    node["image_ext_names"] = config.image_ext_names;
    return node;
}

inline YAML::Node raw_altaz_destripe_request_node(
    const citlali::config::RawTimeChunkAltAzDestripeConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["grouping"] = config.grouping;
    node["fit_time_trend"] = config.fit_time_trend;
    node["fit_derivs"] = config.fit_derivs;
    node["min_samples"] = config.min_samples;
    return node;
}

inline YAML::Node raw_line_audit_request_node(
    const citlali::config::RawTimeChunkLineAuditConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["line_min_hz"] = config.line_min_hz;
    node["line_max_hz"] = config.line_max_hz;
    node["segment_sec"] = config.segment_sec;
    node["min_segment_sec"] = config.min_segment_sec;
    node["overlap_frac"] = config.overlap_frac;
    node["continuum_radius_bins"] = config.continuum_radius_bins;
    node["prominence_thresh"] = config.prominence_thresh;
    node["cm_prominence_thresh"] = config.cm_prominence_thresh;
    node["min_good_frac"] = config.min_good_frac;
    node["min_windows"] = config.min_windows;
    node["max_peaks_per_detector"] = config.max_peaks_per_detector;
    node["max_det"] = config.max_det;
    node["min_det_for_network"] = config.min_det_for_network;
    node["cluster_tol_hz"] = config.cluster_tol_hz;
    node["notch_min_detector_frac"] = config.notch_min_detector_frac;
    node["notch_min_detectors"] = config.notch_min_detectors;
    node["notch_min_cm_prominence"] = config.notch_min_cm_prominence;
    node["detector_min_prominence"] = config.detector_min_prominence;
    node["detector_min_line_power_frac"] =
        config.detector_min_line_power_frac;
    node["bad_detector_max_cluster_frac"] =
        config.bad_detector_max_cluster_frac;
    node["pre_filter_enabled"] = config.pre_filter_enabled;
    node["post_filter_enabled"] = config.post_filter_enabled;
    node["post_filter_apply_shared_notches"] =
        config.post_filter_apply_shared_notches;
    node["post_filter_apply_detector_notches"] =
        config.post_filter_apply_detector_notches;
    node["post_filter_apply_iterations"] =
        config.post_filter_apply_iterations;
    node["post_filter_line_min_hz"] = config.post_filter_line_min_hz;
    node["post_filter_line_max_hz"] = config.post_filter_line_max_hz;
    node["ptc_model_protected_enabled"] =
        config.ptc_model_protected_enabled;
    node["ptc_require_model_subtracted"] =
        config.ptc_require_model_subtracted;
    node["ptc_apply_fixed_notches"] = config.ptc_apply_fixed_notches;
    node["ptc_apply_shared_notches"] = config.ptc_apply_shared_notches;
    node["ptc_apply_detector_notches"] =
        config.ptc_apply_detector_notches;
    node["ptc_apply_iterations"] = config.ptc_apply_iterations;
    node["ptc_line_min_hz"] = config.ptc_line_min_hz;
    node["ptc_line_max_hz"] = config.ptc_line_max_hz;
    node["fixed_notch_enabled"] = config.fixed_notch_enabled;
    node["fixed_notch_freqs_hz"] = config.fixed_notch_freqs_hz;
    node["fixed_notch_widths_hz"] = config.fixed_notch_widths_hz;
    node["fixed_notch_exclusion_half_width_hz"] =
        config.fixed_notch_exclusion_half_width_hz;
    node["apply_shared_notches"] = config.apply_shared_notches;
    node["apply_min_support_networks"] = config.apply_min_support_networks;
    node["apply_min_detector_frac"] = config.apply_min_detector_frac;
    node["apply_min_common_mode_prominence"] =
        config.apply_min_common_mode_prominence;
    node["apply_width_scale"] = config.apply_width_scale;
    node["apply_min_width_hz"] = config.apply_min_width_hz;
    node["apply_max_width_hz"] = config.apply_max_width_hz;
    node["apply_max_notches"] = config.apply_max_notches;
    node["apply_cluster_tol_hz"] = config.apply_cluster_tol_hz;
    node["detector_notch_min_prominence"] =
        config.detector_notch_min_prominence;
    node["detector_notch_min_line_power_frac"] =
        config.detector_notch_min_line_power_frac;
    node["detector_notch_max_notches"] =
        config.detector_notch_max_notches;
    node["detector_notch_width_scale"] =
        config.detector_notch_width_scale;
    node["detector_notch_min_width_hz"] =
        config.detector_notch_min_width_hz;
    node["detector_notch_max_width_hz"] =
        config.detector_notch_max_width_hz;
    node["detector_notch_context_samples"] =
        config.detector_notch_context_samples;
    return node;
}

inline YAML::Node raw_timestream_request_node(
    const citlali::config::RawTimeChunkConfig &config) {
    YAML::Node node;
    node["despike"] = raw_despike_request_node(config.despike);
    node["downsample"] = raw_downsample_request_node(config.downsample);
    node["filter"] = raw_filter_request_node(config.filter);
    node["IIR_filter"] = raw_iir_filter_request_node(config.iir_filter);
    node["flagging"] = raw_flagging_request_node(config.flagging);
    node["coherent_iq_mode_observer"] =
        coherent_iq_mode_observer_request_node(
            config.coherent_iq_mode_observer);
    node["kernel"] = raw_kernel_request_node(config.kernel);
    node["altaz_destripe"] =
        raw_altaz_destripe_request_node(config.altaz_destripe);
    node["line_audit"] = raw_line_audit_request_node(config.line_audit);
    node["flux_calibration"]["enabled"] =
        config.flux_calibration_enabled;
    node["extinction_correction"]["enabled"] =
        config.extinction_correction_enabled;
    return node;
}

}  // namespace citlali::pipeline
