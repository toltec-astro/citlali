#pragma once

#include <citlali/core/config/config_value.h>
#include <citlali/core/config/config_error.h>
#include <citlali/core/config/timestream_config.h>

#include <string>

namespace citlali::config {

inline void validate(const TodStreamOutputConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.outer_context_samples, 0,
                  append_config_path(path, {"outer_context_samples"}), report);
    check_minimum(config.selection_n_uniform, 0,
                  append_config_path(path, {"selection", "n_uniform"}), report);
    check_minimum(config.selection_n_source_dense, 0,
                  append_config_path(path, {"selection", "n_source_dense"}), report);
    if (config.selection_mode == TodOutputSelectionMode::uniform_plus_source_crossing &&
        config.selection_n_uniform + config.selection_n_source_dense <= 0) {
        report.add_error(append_config_path(path, {"selection"}),
                         "uniform_plus_source_crossing requires at least one selected chunk");
    }
    for (const auto chunk : config.chunks_1based) {
        check_minimum(chunk, 1,
                      append_config_path(path, {"indices"}), report);
    }
}

inline void validate(const TimestreamChunkingConfig &config,
                     ValidationReport &report) {
    check_minimum(config.value, 0.0, {"timestream", "chunking", "value"}, report);
}

inline void validate(const TimestreamSourceProtectionConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.radius_arcsec, 0.0,
                  append_config_path(path, {"radius_arcsec"}), report);
}

inline void validate(const RawTimeChunkDespikeCompactRawGateConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.candidate_rel_sigma_scale, 0.0,
                  append_config_path(path, {"candidate_rel_sigma_scale"}),
                  report);
    check_minimum(config.window_sec, 0.0,
                  append_config_path(path, {"window_sec"}), report);
    check_minimum(config.half_peak_frac, 0.0,
                  append_config_path(path, {"half_peak_frac"}), report);
    check_maximum(config.half_peak_frac, 1.0,
                  append_config_path(path, {"half_peak_frac"}), report);
    check_minimum(config.max_width_sec, 0.0,
                  append_config_path(path, {"max_width_sec"}), report);
    check_minimum(config.max_step_shift_z, 0.0,
                  append_config_path(path, {"max_step_shift_z"}), report);
}

inline void validate(const RawTimeChunkDespikeCompactDeltaGateConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.window_sec, 0.0,
                  append_config_path(path, {"window_sec"}), report);
    check_minimum(config.half_peak_frac, 0.0,
                  append_config_path(path, {"half_peak_frac"}), report);
    check_maximum(config.half_peak_frac, 1.0,
                  append_config_path(path, {"half_peak_frac"}), report);
    check_minimum(config.max_width_sec, 0.0,
                  append_config_path(path, {"max_width_sec"}), report);
    check_minimum(config.max_step_shift_z, 0.0,
                  append_config_path(path, {"max_step_shift_z"}), report);
}

inline void validate(const RawTimeChunkDespikeLocalResidualConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{
        "timestream", "raw_time_chunk", "despike", "local_residual"};
    check_minimum(config.window_sec, 0.0,
                  append_config_path(path, {"window_sec"}), report);
    check_minimum(config.sigma_scale, 0.0,
                  append_config_path(path, {"sigma_scale"}), report);
    check_minimum(config.delta_sigma_scale, 0.0,
                  append_config_path(path, {"delta_sigma_scale"}), report);
    check_minimum(config.event_padding_sec, 0.0,
                  append_config_path(path, {"event_padding_sec"}), report);
    check_minimum(config.high_score_event_override, 0.0,
                  append_config_path(path, {"high_score_event_override"}),
                  report);
    check_minimum(config.max_added_flagged_fraction, 0.0,
                  append_config_path(path, {"max_added_flagged_fraction"}),
                  report);
    check_maximum(config.max_added_flagged_fraction, 1.0,
                  append_config_path(path, {"max_added_flagged_fraction"}),
                  report);
    validate(config.compact_raw_gate,
             append_config_path(path, {"compact_raw_gate"}), report);
    validate(config.compact_delta_gate,
             append_config_path(path, {"compact_delta_gate"}), report);
}

inline void validate(const RawTimeChunkDespikeConfig &config,
                     ValidationReport &report) {
    validate(config.source_protection,
             {"timestream", "raw_time_chunk", "despike", "source_protection"},
             report);
    validate(config.local_residual, report);
}

inline void validate(const RawTimeChunkDownsampleConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{"timestream", "raw_time_chunk", "downsample"};
    check_minimum(config.factor, 0, append_config_path(path, {"factor"}),
                  report);
    check_minimum(config.downsampled_freq_Hz, 0.0,
                  append_config_path(path, {"downsampled_freq_Hz"}), report);
}

inline void validate(const RawTimeChunkFilterNotchConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{"timestream", "raw_time_chunk", "filter", "notch"};
    if (!config.zero_phase) {
        report.add_error(append_config_path(path, {"zero_phase"}),
                         "must be true to avoid phase shifts");
    }
    if (config.freqs_Hz.empty()) {
        report.add_error(append_config_path(path, {"freqs_Hz"}),
                         "must contain at least one notch frequency");
    }
    if (config.delta_f_Hz.size() != 1 &&
        config.delta_f_Hz.size() != config.freqs_Hz.size()) {
        report.add_error(append_config_path(path, {"delta_f_Hz"}),
                         "must have length 1 or match freqs_Hz length");
    }
    for (const auto freq_Hz : config.freqs_Hz) {
        check_minimum(freq_Hz, 1e-12, append_config_path(path, {"freqs_Hz"}),
                      report);
    }
    for (const auto delta_f_Hz : config.delta_f_Hz) {
        check_minimum(delta_f_Hz, 1e-12,
                      append_config_path(path, {"delta_f_Hz"}), report);
    }
}

inline void validate(const RawTimeChunkFilterEdgeGuardConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "raw_time_chunk", "filter", "edge_guard"};
    check_minimum(config.min_samples, 0,
                  append_config_path(path, {"min_samples"}), report);
    check_minimum(config.extra_samples, 0,
                  append_config_path(path, {"extra_samples"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.iir_settle_attenuation, 0.0,
                  append_config_path(path, {"iir_settle_attenuation"}), report);
    check_maximum(config.iir_settle_attenuation, 1.0,
                  append_config_path(path, {"iir_settle_attenuation"}), report);
}

inline void validate(const RawTimeChunkFilterConfig &config,
                     ValidationReport &report) {
    if (config.enabled) {
        const ConfigPath path{"timestream", "raw_time_chunk", "filter"};
        check_minimum(config.a_gibbs, 0.0,
                      append_config_path(path, {"a_gibbs"}), report);
        check_minimum(config.freq_low_Hz, 0.0,
                      append_config_path(path, {"freq_low_Hz"}), report);
        check_minimum(config.freq_high_Hz, 0.0,
                      append_config_path(path, {"freq_high_Hz"}), report);
        if (config.freq_high_Hz < config.freq_low_Hz) {
            report.add_error(append_config_path(path, {"freq_high_Hz"}),
                             "must be greater than or equal to freq_low_Hz");
        }
        check_minimum(config.n_terms, 0, append_config_path(path, {"n_terms"}),
                      report);
        validate(config.notch, report);
    }
    validate(config.edge_guard, report);
}

inline void validate(const RawTimeChunkIirFilterConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{"timestream", "raw_time_chunk", "IIR_filter"};
    check_minimum(config.freq_Hz, 1e-12,
                  append_config_path(path, {"freq_Hz"}), report);
    check_minimum(config.order, 1, append_config_path(path, {"order"}), report);
    if (!config.zero_phase) {
        report.add_error(append_config_path(path, {"zero_phase"}),
                         "must be true to avoid phase shifts");
    }
}

inline void validate(const RawTimeChunkNetworkStepMaskConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "raw_time_chunk", "flagging", "network_step_mask"};
    check_minimum(config.step_window_sec, 0.01,
                  append_config_path(path, {"step_window_sec"}), report);
    check_minimum(config.step_score_thresh, 0.0,
                  append_config_path(path, {"step_score_thresh"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_det_used, 1,
                  append_config_path(path, {"min_det_used"}), report);
    check_minimum(config.min_step_det_frac, 0.0,
                  append_config_path(path, {"min_step_det_frac"}), report);
    check_maximum(config.min_step_det_frac, 1.0,
                  append_config_path(path, {"min_step_det_frac"}), report);
    check_minimum(config.min_alignment_frac, 0.0,
                  append_config_path(path, {"min_alignment_frac"}), report);
    check_maximum(config.min_alignment_frac, 1.0,
                  append_config_path(path, {"min_alignment_frac"}), report);
    check_minimum(config.cluster_tol_sec, 0.0,
                  append_config_path(path, {"cluster_tol_sec"}), report);
    check_minimum(config.mask_half_width_sec, 0.0,
                  append_config_path(path, {"mask_half_width_sec"}), report);
    check_minimum(config.max_flagged_fraction, 0.0,
                  append_config_path(path, {"max_flagged_fraction"}), report);
    check_maximum(config.max_flagged_fraction, 1.0,
                  append_config_path(path, {"max_flagged_fraction"}), report);
}

inline void validate(const RawTimeChunkImpulsiveCaptureConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "raw_time_chunk", "flagging", "impulsive_capture"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_event_z, 0.0,
                  append_config_path(path, {"min_event_z"}), report);
    check_minimum(config.near_event_z, 0.0,
                  append_config_path(path, {"near_event_z"}), report);
    check_minimum(config.max_events_per_network, 1,
                  append_config_path(path, {"max_events_per_network"}), report);
    check_minimum(config.snippet_pre_window_sec, 0.0,
                  append_config_path(path, {"snippet_pre_window_sec"}), report);
    check_minimum(config.snippet_post_window_sec, 0.0,
                  append_config_path(path, {"snippet_post_window_sec"}), report);
}

inline void validate(const RawTimeChunkImpulsiveCoincidenceConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "raw_time_chunk", "flagging",
        "impulsive_coincidence"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.event_score_thresh, 0.0,
                  append_config_path(path, {"event_score_thresh"}), report);
    check_minimum(config.min_det_used, 1,
                  append_config_path(path, {"min_det_used"}), report);
    check_minimum(config.min_impulsive_det_frac, 0.0,
                  append_config_path(path, {"min_impulsive_det_frac"}), report);
    check_maximum(config.min_impulsive_det_frac, 1.0,
                  append_config_path(path, {"min_impulsive_det_frac"}), report);
    check_minimum(config.min_alignment_frac, 0.0,
                  append_config_path(path, {"min_alignment_frac"}), report);
    check_maximum(config.min_alignment_frac, 1.0,
                  append_config_path(path, {"min_alignment_frac"}), report);
    check_minimum(config.min_networks_aligned, 1,
                  append_config_path(path, {"min_networks_aligned"}), report);
    check_minimum(config.high_score_override_thresh, 0.0,
                  append_config_path(path, {"high_score_override_thresh"}),
                  report);
    check_minimum(config.high_score_min_networks_aligned, 0,
                  append_config_path(path,
                                     {"high_score_min_networks_aligned"}),
                  report);
    check_minimum(config.cluster_tol_sec, 0.0,
                  append_config_path(path, {"cluster_tol_sec"}), report);
    check_minimum(config.mask_pre_window_sec, 0.0,
                  append_config_path(path, {"mask_pre_window_sec"}), report);
    check_minimum(config.mask_post_window_sec, 0.0,
                  append_config_path(path, {"mask_post_window_sec"}), report);
    check_minimum(config.max_flagged_fraction, 0.0,
                  append_config_path(path, {"max_flagged_fraction"}), report);
    check_maximum(config.max_flagged_fraction, 1.0,
                  append_config_path(path, {"max_flagged_fraction"}), report);
}

inline void validate(const RawTimeChunkFlaggingConfig &config,
                     ValidationReport &report) {
    validate(config.network_step_mask, report);
    validate(config.impulsive_capture, report);
    validate(config.impulsive_coincidence, report);
}

inline void validate(const RawTimeChunkKernelConfig &, ValidationReport &) {}

inline void validate(const RawTimeChunkAltAzDestripeConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    check_minimum(config.min_samples, 4,
                  {"timestream", "raw_time_chunk", "altaz_destripe",
                   "min_samples"},
                  report);
}

inline void validate(const RawTimeChunkLineAuditConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{"timestream", "raw_time_chunk", "line_audit"};
    check_minimum(config.line_min_hz, 0.0,
                  append_config_path(path, {"line_min_hz"}), report);
    check_minimum(config.line_max_hz, 0.0,
                  append_config_path(path, {"line_max_hz"}), report);
    check_minimum(config.segment_sec, 0.1,
                  append_config_path(path, {"segment_sec"}), report);
    check_minimum(config.min_segment_sec, 0.1,
                  append_config_path(path, {"min_segment_sec"}), report);
    check_minimum(config.overlap_frac, 0.0,
                  append_config_path(path, {"overlap_frac"}), report);
    check_maximum(config.overlap_frac, 0.95,
                  append_config_path(path, {"overlap_frac"}), report);
    check_minimum(config.continuum_radius_bins, 1,
                  append_config_path(path, {"continuum_radius_bins"}), report);
    check_minimum(config.prominence_thresh, 1.0,
                  append_config_path(path, {"prominence_thresh"}), report);
    check_minimum(config.cm_prominence_thresh, 1.0,
                  append_config_path(path, {"cm_prominence_thresh"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_windows, 1,
                  append_config_path(path, {"min_windows"}), report);
    check_minimum(config.max_peaks_per_detector, 1,
                  append_config_path(path, {"max_peaks_per_detector"}), report);
    check_minimum(config.max_det, 0, append_config_path(path, {"max_det"}),
                  report);
    check_minimum(config.min_det_for_network, 1,
                  append_config_path(path, {"min_det_for_network"}), report);
    check_minimum(config.cluster_tol_hz, 0.0,
                  append_config_path(path, {"cluster_tol_hz"}), report);
    check_minimum(config.notch_min_detector_frac, 0.0,
                  append_config_path(path, {"notch_min_detector_frac"}),
                  report);
    check_maximum(config.notch_min_detector_frac, 1.0,
                  append_config_path(path, {"notch_min_detector_frac"}),
                  report);
    check_minimum(config.notch_min_detectors, 1,
                  append_config_path(path, {"notch_min_detectors"}), report);
    check_minimum(config.notch_min_cm_prominence, 1.0,
                  append_config_path(path, {"notch_min_cm_prominence"}),
                  report);
    check_minimum(config.detector_min_prominence, 1.0,
                  append_config_path(path, {"detector_min_prominence"}),
                  report);
    check_minimum(config.detector_min_line_power_frac, 0.0,
                  append_config_path(path, {"detector_min_line_power_frac"}),
                  report);
    check_maximum(config.detector_min_line_power_frac, 1.0,
                  append_config_path(path, {"detector_min_line_power_frac"}),
                  report);
    check_minimum(config.bad_detector_max_cluster_frac, 0.0,
                  append_config_path(path, {"bad_detector_max_cluster_frac"}),
                  report);
    check_maximum(config.bad_detector_max_cluster_frac, 1.0,
                  append_config_path(path, {"bad_detector_max_cluster_frac"}),
                  report);
    check_minimum(config.post_filter_apply_iterations, 1,
                  append_config_path(path, {"post_filter_apply_iterations"}),
                  report);
    check_optional_minimum(config.post_filter_line_min_hz, 0.0,
                           append_config_path(path,
                                              {"post_filter_line_min_hz"}),
                           report);
    check_optional_minimum(config.post_filter_line_max_hz, 0.0,
                           append_config_path(path,
                                              {"post_filter_line_max_hz"}),
                           report);
    check_minimum(config.ptc_apply_iterations, 1,
                  append_config_path(path, {"ptc_apply_iterations"}), report);
    check_optional_minimum(config.ptc_line_min_hz, 0.0,
                           append_config_path(path, {"ptc_line_min_hz"}),
                           report);
    check_optional_minimum(config.ptc_line_max_hz, 0.0,
                           append_config_path(path, {"ptc_line_max_hz"}),
                           report);
    if (std::isfinite(config.ptc_line_min_hz) &&
        std::isfinite(config.ptc_line_max_hz) &&
        config.ptc_line_max_hz < config.ptc_line_min_hz) {
        report.add_error(append_config_path(path, {"ptc_line_max_hz"}),
                         "must be greater than or equal to ptc_line_min_hz");
    }
    if (config.fixed_notch_enabled) {
        if (config.fixed_notch_freqs_hz.empty()) {
            report.add_error(append_config_path(path, {"fixed_notch_freqs_hz"}),
                             "must contain at least one fixed notch when enabled");
        }
        if (config.fixed_notch_widths_hz.empty()) {
            report.add_error(append_config_path(path, {"fixed_notch_widths_hz"}),
                             "must contain at least one fixed notch width");
        }
        if (!config.fixed_notch_widths_hz.empty() &&
            config.fixed_notch_widths_hz.size() != 1 &&
            config.fixed_notch_widths_hz.size() !=
                config.fixed_notch_freqs_hz.size()) {
            report.add_error(append_config_path(path, {"fixed_notch_widths_hz"}),
                             "must have length 1 or match fixed_notch_freqs_hz");
        }
        for (const auto freq_hz : config.fixed_notch_freqs_hz) {
            if (!std::isfinite(freq_hz) || freq_hz <= 0.0) {
                report.add_error(
                    append_config_path(path, {"fixed_notch_freqs_hz"}),
                    "values must be finite and greater than 0");
            }
        }
        for (const auto width_hz : config.fixed_notch_widths_hz) {
            if (!std::isfinite(width_hz) || width_hz <= 0.0) {
                report.add_error(
                    append_config_path(path, {"fixed_notch_widths_hz"}),
                    "values must be finite and greater than 0");
            }
        }
    }
    check_minimum(config.fixed_notch_exclusion_half_width_hz, 0.0,
                  append_config_path(path,
                                     {"fixed_notch_exclusion_half_width_hz"}),
                  report);
    check_minimum(config.apply_min_support_networks, 1,
                  append_config_path(path, {"apply_min_support_networks"}),
                  report);
    check_minimum(config.apply_min_detector_frac, 0.0,
                  append_config_path(path, {"apply_min_detector_frac"}), report);
    check_maximum(config.apply_min_detector_frac, 1.0,
                  append_config_path(path, {"apply_min_detector_frac"}), report);
    check_minimum(config.apply_min_common_mode_prominence, 1.0,
                  append_config_path(path,
                                     {"apply_min_common_mode_prominence"}),
                  report);
    check_minimum(config.apply_width_scale, 0.01,
                  append_config_path(path, {"apply_width_scale"}), report);
    check_minimum(config.apply_min_width_hz, 0.0,
                  append_config_path(path, {"apply_min_width_hz"}), report);
    check_minimum(config.apply_max_width_hz, 0.0,
                  append_config_path(path, {"apply_max_width_hz"}), report);
    if (config.apply_max_width_hz < config.apply_min_width_hz) {
        report.add_error(append_config_path(path, {"apply_max_width_hz"}),
                         "must be greater than or equal to apply_min_width_hz");
    }
    check_minimum(config.apply_max_notches, 0,
                  append_config_path(path, {"apply_max_notches"}), report);
    check_minimum(config.apply_cluster_tol_hz, 0.0,
                  append_config_path(path, {"apply_cluster_tol_hz"}), report);
    check_minimum(config.detector_notch_min_prominence, 1.0,
                  append_config_path(path, {"detector_notch_min_prominence"}),
                  report);
    check_minimum(config.detector_notch_min_line_power_frac, 0.0,
                  append_config_path(path,
                                     {"detector_notch_min_line_power_frac"}),
                  report);
    check_maximum(config.detector_notch_min_line_power_frac, 1.0,
                  append_config_path(path,
                                     {"detector_notch_min_line_power_frac"}),
                  report);
    check_minimum(config.detector_notch_max_notches, 0,
                  append_config_path(path, {"detector_notch_max_notches"}),
                  report);
    check_minimum(config.detector_notch_width_scale, 0.01,
                  append_config_path(path, {"detector_notch_width_scale"}),
                  report);
    check_minimum(config.detector_notch_min_width_hz, 0.0,
                  append_config_path(path, {"detector_notch_min_width_hz"}),
                  report);
    check_minimum(config.detector_notch_max_width_hz, 0.0,
                  append_config_path(path, {"detector_notch_max_width_hz"}),
                  report);
    if (config.detector_notch_max_width_hz <
        config.detector_notch_min_width_hz) {
        report.add_error(
            append_config_path(path, {"detector_notch_max_width_hz"}),
            "must be greater than or equal to detector_notch_min_width_hz");
    }
    check_minimum(config.detector_notch_context_samples, 0,
                  append_config_path(path, {"detector_notch_context_samples"}),
                  report);
}

inline void validate(const RawTimeChunkConfig &config, ValidationReport &report) {
    validate(config.despike, report);
    validate(config.downsample, report);
    validate(config.filter, report);
    validate(config.iir_filter, report);
    validate(config.flagging, report);
    validate(config.kernel, report);
    validate(config.altaz_destripe, report);
    validate(config.line_audit, report);
    if (config.downsample.enabled && !config.filter.enabled) {
        report.add_error({"timestream", "raw_time_chunk", "downsample"},
                         "requires raw_time_chunk.filter.enabled=true");
    }
}

inline void validate(const ProcessedTimeChunkSecondPassLocalConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{
        "timestream", "processed_time_chunk", "flagging", "second_pass_local"};
    check_minimum(config.min_spike_sigma, 0.0,
                  append_config_path(path, {"min_spike_sigma"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.baseline_window_sec, 0.0,
                  append_config_path(path, {"baseline_window_sec"}), report);
    check_minimum(config.sigma_scale, 0.0,
                  append_config_path(path, {"sigma_scale"}), report);
    check_minimum(config.delta_sigma_scale, 0.0,
                  append_config_path(path, {"delta_sigma_scale"}), report);
    check_minimum(config.raw_candidate_rel_sigma_scale, 0.0,
                  append_config_path(path, {"raw_candidate_rel_sigma_scale"}),
                  report);
    check_minimum(config.raw_window_sec, 0.0,
                  append_config_path(path, {"raw_window_sec"}), report);
    check_minimum(config.raw_half_peak_frac, 0.0,
                  append_config_path(path, {"raw_half_peak_frac"}), report);
    check_minimum(config.raw_max_width_sec, 0.0,
                  append_config_path(path, {"raw_max_width_sec"}), report);
    check_minimum(config.delta_window_sec, 0.0,
                  append_config_path(path, {"delta_window_sec"}), report);
    check_minimum(config.delta_half_peak_frac, 0.0,
                  append_config_path(path, {"delta_half_peak_frac"}), report);
    check_minimum(config.delta_max_width_sec, 0.0,
                  append_config_path(path, {"delta_max_width_sec"}), report);
    check_minimum(config.max_step_shift_z, 0.0,
                  append_config_path(path, {"max_step_shift_z"}), report);
    check_minimum(config.high_score_event_override, 0.0,
                  append_config_path(path, {"high_score_event_override"}), report);
    check_minimum(config.merge_within_detector_sec, 0.0,
                  append_config_path(path, {"merge_within_detector_sec"}), report);
    check_minimum(config.cluster_events_sec, 0.0,
                  append_config_path(path, {"cluster_events_sec"}), report);
    check_minimum(config.min_cluster_detectors, 1,
                  append_config_path(path, {"min_cluster_detectors"}), report);
    check_minimum(config.high_score_cluster_override, 0.0,
                  append_config_path(path, {"high_score_cluster_override"}),
                  report);
    check_minimum(config.max_auto_flag_clusters_per_network, 1,
                  append_config_path(path,
                                     {"max_auto_flag_clusters_per_network"}),
                  report);
    validate(config.source_protection,
             {"timestream", "processed_time_chunk", "flagging",
              "second_pass_local", "source_protection"},
             report);
}

inline void validate(const ProcessedTimeChunkStandardPcaConfig &config,
                     ValidationReport &report) {
    check_minimum(config.n_calc, 0,
                  {"timestream", "processed_time_chunk", "clean",
                   "standard_pca", "n_calc"},
                  report);
}

inline void validate(const ProcessedTimeChunkCorrGroupingConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "clean", "corr_grouping"};
    check_minimum(config.corr_min, 0.0,
                  append_config_path(path, {"corr_min"}), report);
    check_maximum(config.corr_min, 1.0,
                  append_config_path(path, {"corr_min"}), report);
    check_minimum(config.min_overlap, 1,
                  append_config_path(path, {"min_overlap"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_group_size, 2,
                  append_config_path(path, {"min_group_size"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
}

inline void validate(const ProcessedTimeChunkNullModelConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "clean", "null_model"};
    check_minimum(config.n_surrogates, 4,
                  append_config_path(path, {"n_surrogates"}), report);
    check_minimum(config.quantile, 0.5,
                  append_config_path(path, {"quantile"}), report);
    check_maximum(config.quantile, 0.999999,
                  append_config_path(path, {"quantile"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.max_modes, 0,
                  append_config_path(path, {"max_modes"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.seed, 0, append_config_path(path, {"seed"}), report);
}

inline void validate(const ProcessedTimeChunkMarchenkoPasturConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "clean", "marchenko_pastur"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.max_modes, 0,
                  append_config_path(path, {"max_modes"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.band_low_Hz, 0.0,
                  append_config_path(path, {"band_low_Hz"}), report);
    check_minimum(config.band_high_Hz, 0.0,
                  append_config_path(path, {"band_high_Hz"}), report);
    check_minimum(config.bulk_keep_frac, 0.1,
                  append_config_path(path, {"bulk_keep_frac"}), report);
    check_maximum(config.bulk_keep_frac, 1.0,
                  append_config_path(path, {"bulk_keep_frac"}), report);
    check_minimum(config.q_grid_size, 8,
                  append_config_path(path, {"q_grid_size"}), report);
}

inline void validate(const ProcessedTimeChunkAdaptiveSelectorConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "clean", "adaptive_selector"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.max_det, 0, append_config_path(path, {"max_det"}),
                  report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.max_pairs, 0,
                  append_config_path(path, {"max_pairs"}), report);
    check_minimum(config.seed, 0, append_config_path(path, {"seed"}), report);
    check_minimum(config.low_weight, 0.0,
                  append_config_path(path, {"low_weight"}), report);
    check_minimum(config.tail_weight, 0.0,
                  append_config_path(path, {"tail_weight"}), report);
    check_minimum(config.topmode_weight, 0.0,
                  append_config_path(path, {"topmode_weight"}), report);
    check_minimum(config.reg_weight, 0.0,
                  append_config_path(path, {"reg_weight"}), report);
    check_minimum(config.low_band_Hz[0], 0.0,
                  append_config_path(path, {"low_band_Hz"}), report);
    if (config.low_band_Hz[1] <= config.low_band_Hz[0]) {
        report.add_error(append_config_path(path, {"low_band_Hz"}),
                         "must be [fmin, fmax] with fmax greater than fmin");
    }
    check_minimum(config.mid_band_Hz[0], 0.0,
                  append_config_path(path, {"mid_band_Hz"}), report);
    if (config.mid_band_Hz[1] <= config.mid_band_Hz[0]) {
        report.add_error(append_config_path(path, {"mid_band_Hz"}),
                         "must be [fmin, fmax] with fmax greater than fmin");
    }
}

inline void validate(const ProcessedTimeChunkCleanConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const auto enabled_cleaners =
        static_cast<int>(config.standard_pca.enabled) +
        static_cast<int>(config.null_model.enabled) +
        static_cast<int>(config.marchenko_pastur.enabled) +
        static_cast<int>(config.adaptive_selector.enabled);
    if (enabled_cleaners != 1) {
        report.add_error(
            {"timestream", "processed_time_chunk", "clean"},
            "exactly one cleaner must be enabled when cleaning is enabled");
    }
    validate(config.standard_pca, report);
    validate(config.corr_grouping, report);
    validate(config.null_model, report);
    validate(config.marchenko_pastur, report);
    validate(config.adaptive_selector, report);
}

inline void validate(const ProcessedTimeChunkBusyRowSuppressionConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "weighting",
        "busy_row_suppression"};
    check_minimum(config.min_candidate_clusters, 0,
                  append_config_path(path, {"min_candidate_clusters"}), report);
    check_minimum(config.min_max_unflagged_residual_z, 0.0,
                  append_config_path(path, {"min_max_unflagged_residual_z"}),
                  report);
    check_minimum(config.factor, 0.0, append_config_path(path, {"factor"}),
                  report);
    check_maximum(config.factor, 1.0, append_config_path(path, {"factor"}),
                  report);
}

inline void validate(const ProcessedTimeChunkWeightValidationConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "weighting", "validation"};
    check_minimum(config.accumulation_iters, 1,
                  append_config_path(path, {"accumulation_iters"}), report);
    check_minimum(config.apply_start_iter, 0,
                  append_config_path(path, {"apply_start_iter"}), report);
    check_minimum(config.min_valid_scans, 1,
                  append_config_path(path, {"min_valid_scans"}), report);
    check_minimum(config.min_factor, 0.0,
                  append_config_path(path, {"min_factor"}), report);
    check_maximum(config.min_factor, 1.0,
                  append_config_path(path, {"min_factor"}), report);
    check_minimum(config.unvalidated_factor, 0.0,
                  append_config_path(path, {"unvalidated_factor"}), report);
    check_maximum(config.unvalidated_factor, 1.0,
                  append_config_path(path, {"unvalidated_factor"}), report);
    check_minimum(config.ratio_power, 0.0,
                  append_config_path(path, {"ratio_power"}), report);
    check_minimum(config.transient_ratio_power, 0.0,
                  append_config_path(path, {"transient_ratio_power"}), report);
    check_minimum(config.upward_max_factor, 1.0,
                  append_config_path(path, {"upward_max_factor"}), report);
    check_minimum(config.upward_power, 0.0,
                  append_config_path(path, {"upward_power"}), report);
    check_minimum(config.upward_min_base_factor, 0.0,
                  append_config_path(path, {"upward_min_base_factor"}), report);
    check_maximum(config.upward_min_base_factor, 1.0,
                  append_config_path(path, {"upward_min_base_factor"}), report);
    check_minimum(config.upward_min_atmospheric_factor, 0.0,
                  append_config_path(path, {"upward_min_atmospheric_factor"}),
                  report);
    check_maximum(config.upward_min_atmospheric_factor, 1.0,
                  append_config_path(path, {"upward_min_atmospheric_factor"}),
                  report);
    check_minimum(config.atmospheric_min_detectors, 2,
                  append_config_path(path, {"atmospheric_min_detectors"}),
                  report);
    check_minimum(config.atmospheric_ref, 0.0,
                  append_config_path(path, {"atmospheric_ref"}), report);
    check_maximum(config.atmospheric_ref, 1.0,
                  append_config_path(path, {"atmospheric_ref"}), report);
    check_minimum(config.atmospheric_span, 1e-12,
                  append_config_path(path, {"atmospheric_span"}), report);
    check_minimum(config.atmospheric_power, 0.0,
                  append_config_path(path, {"atmospheric_power"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_overlap, 2,
                  append_config_path(path, {"min_overlap"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.high_weight_min_group_detectors, 2,
                  append_config_path(path,
                                     {"high_weight_min_group_detectors"}),
                  report);
    check_minimum(config.high_weight_log_robust_z, 0.0,
                  append_config_path(path, {"high_weight_log_robust_z"}),
                  report);
    check_minimum(config.high_weight_max_median_factor, 1.0,
                  append_config_path(path,
                                     {"high_weight_max_median_factor"}),
                  report);
    check_minimum(config.high_weight_cap_median_factor, 1.0,
                  append_config_path(path,
                                     {"high_weight_cap_median_factor"}),
                  report);
    check_minimum(config.high_weight_min_validated_factor, 0.0,
                  append_config_path(path,
                                     {"high_weight_min_validated_factor"}),
                  report);
}

inline void validate(const ProcessedTimeChunkWeightCorrPenaltyTermConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.span, 1e-12, append_config_path(path, {"span"}),
                  report);
    check_minimum(config.weight, 0.0, append_config_path(path, {"weight"}),
                  report);
}

inline void validate(const ProcessedTimeChunkWeightCorrPenaltyBandConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.span, 1e-12, append_config_path(path, {"span"}),
                  report);
    check_minimum(config.weight, 0.0, append_config_path(path, {"weight"}),
                  report);
    check_minimum(config.low_band_Hz[0], 0.0,
                  append_config_path(path, {"low_band_Hz"}), report);
    if (config.low_band_Hz[1] <= config.low_band_Hz[0]) {
        report.add_error(append_config_path(path, {"low_band_Hz"}),
                         "must be [fmin, fmax] with fmax greater than fmin");
    }
    check_minimum(config.mid_band_Hz[0], 0.0,
                  append_config_path(path, {"mid_band_Hz"}), report);
    if (config.mid_band_Hz[1] <= config.mid_band_Hz[0]) {
        report.add_error(append_config_path(path, {"mid_band_Hz"}),
                         "must be [fmin, fmax] with fmax greater than fmin");
    }
}

inline void validate(const ProcessedTimeChunkWeightCorrPenaltyConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "weighting", "corr_penalty"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_overlap, 2,
                  append_config_path(path, {"min_overlap"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.max_pairs, 0,
                  append_config_path(path, {"max_pairs"}), report);
    check_minimum(config.seed, 0, append_config_path(path, {"seed"}), report);
    check_minimum(config.floor, 0.0, append_config_path(path, {"floor"}),
                  report);
    check_maximum(config.floor, 1.0, append_config_path(path, {"floor"}),
                  report);
    check_minimum(config.exponent, 0.0,
                  append_config_path(path, {"exponent"}), report);
    validate(config.pair_corr, append_config_path(path, {"pair_corr"}), report);
    validate(config.cm_el_corr, append_config_path(path, {"cm_el_corr"}),
             report);
    validate(config.cm_low_mid_ratio,
             append_config_path(path, {"cm_low_mid_ratio"}), report);
}

inline void validate(const ProcessedTimeChunkWeightingConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{"timestream", "processed_time_chunk", "weighting"};
    check_minimum(config.source_mask_radius_arcsec, 0.0,
                  append_config_path(path, {"source_mask_radius_arcsec"}),
                  report);
    check_minimum(config.hybrid_correction_min_factor, 0.0,
                  append_config_path(path, {"hybrid_correction_min_factor"}),
                  report);
    check_minimum(config.hybrid_correction_max_factor, 0.0,
                  append_config_path(path, {"hybrid_correction_max_factor"}),
                  report);
    if (config.hybrid_correction_max_factor <
        config.hybrid_correction_min_factor) {
        report.add_error(
            append_config_path(path, {"hybrid_correction_max_factor"}),
            "must be greater than or equal to hybrid_correction_min_factor");
    }
    validate(config.validation, report);
    validate(config.corr_penalty, report);
    validate(config.busy_row_suppression, report);
}

inline void validate(const ProcessedTimeChunkFlaggingConfig &config,
                     ValidationReport &report) {
    validate(config.second_pass_local, report);
}

inline void validate(const ProcessedTimeChunkConfig &config,
                     ValidationReport &report) {
    validate(config.clean, report);
    validate(config.weighting, report);
    validate(config.flagging, report);
}

inline void validate(const FruitLoopsWeightFeedbackConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{"timestream", "fruit_loops", "weight_feedback"};
    check_minimum(config.low_relative_weight, 0.0,
                  append_config_path(path, {"low_relative_weight"}), report);
    check_minimum(config.high_relative_weight, 0.0,
                  append_config_path(path, {"high_relative_weight"}), report);
    if (config.enabled &&
        config.high_relative_weight <= config.low_relative_weight) {
        report.add_error(append_config_path(path, {"high_relative_weight"}),
                         "must be greater than low_relative_weight when enabled");
    }
}

inline void validate(const FruitLoopsInjectedSourceTestConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "fruit_loops", "injected_source_test"};
    check_minimum(
        config.start_iteration, 1,
        append_config_path(path, {"start_iteration"}), report);
    if (config.array_amplitude_mjy_beam.size() != 3) {
        report.add_error(
            append_config_path(path, {"array_amplitude_mjy_beam"}),
            "must contain exactly three values ordered "
            "[a1100, a1400, a2000]");
    }
    bool any_positive = false;
    for (const double amplitude : config.array_amplitude_mjy_beam) {
        check_minimum(
            amplitude, 0.0,
            append_config_path(path, {"array_amplitude_mjy_beam"}), report);
        any_positive = any_positive || amplitude > 0.0;
    }
    if (!any_positive) {
        report.add_error(
            append_config_path(path, {"array_amplitude_mjy_beam"}),
            "must contain at least one positive amplitude");
    }
}

inline void validate(const TimestreamFruitLoopsConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        if (config.injected_source_test.enabled) {
            report.add_error(
                {"timestream", "fruit_loops", "injected_source_test",
                 "enabled"},
                "requires timestream.fruit_loops.enabled=true");
        }
        return;
    }
    const ConfigPath path{"timestream", "fruit_loops"};
    if (has_nonempty_config_value(config.restart_path) &&
        has_nonempty_config_value(config.path)) {
        report.add_error(
            append_config_path(path, {"restart_path"}),
            "cannot be combined with path; restart_path supplies the initial map and learning state");
    }
    check_minimum(config.peak_fraction_limit, 0.0,
                  append_config_path(path, {"peak_fraction_limit"}), report);
    check_minimum(config.local_snr_floor, 0.0,
                  append_config_path(path, {"local_snr_floor"}), report);
    check_minimum(config.local_sigma_inner_radius_arcsec, 0.0,
                  append_config_path(path, {"local_sigma_inner_radius_arcsec"}),
                  report);
    check_minimum(config.local_sigma_outer_radius_arcsec, 0.0,
                  append_config_path(path, {"local_sigma_outer_radius_arcsec"}),
                  report);
    check_minimum(config.local_sigma_inner_fwhm, 0.0,
                  append_config_path(path, {"local_sigma_inner_fwhm"}), report);
    check_minimum(config.local_sigma_outer_fwhm, 0.0,
                  append_config_path(path, {"local_sigma_outer_fwhm"}), report);
    check_minimum(config.local_sigma_edge_guard_arcsec, 0.0,
                  append_config_path(path, {"local_sigma_edge_guard_arcsec"}),
                  report);
    check_minimum(config.local_sigma_min_pixels, 1,
                  append_config_path(path, {"local_sigma_min_pixels"}), report);
    check_minimum(config.adaptive_support_radius_arcsec, 0.0,
                  append_config_path(path, {"adaptive_support_radius_arcsec"}),
                  report);
    check_minimum(config.adaptive_support_radius_fwhm, 0.0,
                  append_config_path(path, {"adaptive_support_radius_fwhm"}),
                  report);
    validate(config.weight_feedback, report);
    validate(config.injected_source_test, report);
    if (config.injected_source_test.enabled) {
        if (!config.diagnostics_enabled) {
            report.add_error(
                append_config_path(
                    path, {"injected_source_test", "enabled"}),
                "requires fruit-loop diagnostics_enabled=true");
        }
        if (!config.save_all_iters) {
            report.add_error(
                append_config_path(
                    path, {"injected_source_test", "enabled"}),
                "requires save_all_iters=true");
        }
        if (config.max_iters <=
            config.injected_source_test.start_iteration) {
            report.add_error(
                append_config_path(
                    path, {"injected_source_test", "start_iteration"}),
                "must be less than max_iters");
        }
    }
    check_minimum(config.center_keep_radius_arcsec, 0.0,
                  append_config_path(path, {"center_keep_radius_arcsec"}),
                  report);
    check_minimum(config.max_iters, 0,
                  append_config_path(path, {"max_iters"}), report);
}

inline void validate(const TimestreamLearningMapPixelOutlierConfig &config,
                     ValidationReport &report) {
    check_minimum(config.top_n, 0,
                  {"timestream", "learning", "map_pixel_outlier_top_n"},
                  report);
    check_minimum(config.targeted_contributor_max_pixels, 0,
                  {"timestream", "learning",
                   "map_pixel_outlier_targeted_contributor_max_pixels"},
                  report);
    check_minimum(config.detector_exclusion_min_pixels, 1,
                  {"timestream", "learning",
                   "map_pixel_outlier_detector_exclusion_min_pixels"},
                  report);
    check_minimum(config.min_abs_z, 0.0,
                  {"timestream", "learning", "map_pixel_outlier_min_abs_z"},
                  report);
    check_minimum(config.min_n_eff, 0.0,
                  {"timestream", "learning", "map_pixel_outlier_min_n_eff"},
                  report);
    check_minimum(config.source_radius_arcsec, 0.0,
                  {"timestream", "learning",
                   "map_pixel_outlier_source_radius_arcsec"},
                  report);
}

inline void validate(const TimestreamLearningBusyDetectorConfig &,
                     ValidationReport &) {}

inline void validate(const TimestreamLearningScanNetworkPathologyConfig &config,
                     ValidationReport &report) {
    check_minimum(config.min_candidate_clusters, 0,
                  {"timestream", "learning",
                   "scan_network_pathology_min_candidate_clusters"},
                  report);
    check_minimum(config.min_candidate_events, 0,
                  {"timestream", "learning",
                   "scan_network_pathology_min_candidate_events"},
                  report);
    check_minimum(config.min_max_residual_z, 0.0,
                  {"timestream", "learning",
                   "scan_network_pathology_min_max_residual_z"},
                  report);
    check_minimum(config.severe_candidate_events, 0,
                  {"timestream", "learning",
                   "scan_network_pathology_severe_candidate_events"},
                  report);
    check_minimum(config.severe_max_residual_z, 0.0,
                  {"timestream", "learning",
                   "scan_network_pathology_severe_max_residual_z"},
                  report);
    check_minimum(config.max_new_flagged_fraction, 0.0,
                  {"timestream", "learning",
                   "scan_network_pathology_max_new_flagged_fraction"},
                  report);
}

inline void validate(const TimestreamLearningConfig &config,
                     ValidationReport &report) {
    check_minimum(config.learn_iters, 0,
                  {"timestream", "learning", "learn_iters"}, report);
    check_minimum(config.apply_start_iter, 0,
                  {"timestream", "learning", "apply_start_iter"}, report);
    check_minimum(config.max_records_per_type, 0,
                  {"timestream", "learning", "max_records_per_type"}, report);
    check_minimum(config.apply_max_new_flagged_fraction, 0.0,
                  {"timestream", "learning", "apply_max_new_flagged_fraction"},
                  report);
    validate(config.map_pixel_outlier, report);
    validate(config.busy_detector, report);
    validate(config.scan_network_pathology, report);
}

inline void validate(const AuxiliaryMeasuredChannelConfig &config,
                     const ConfigPath &path, ValidationReport &report) {
    if (config.enabled) {
        report.add_warning(path,
                           "auxiliary measured channel structure is parsed but "
                           "not executed by the current pipeline");
    }
    if (config.use_for_science_map) {
        report.add_warning(append_config_path(path, {"use_for_science_map"}),
                           "auxiliary measured channels are not supported as "
                           "science-map inputs yet");
    }
}

inline void validate(const TimestreamAuxiliaryChannelsConfig &config,
                     ValidationReport &report) {
    validate(config.quadrature_r,
             {"timestream", "auxiliary_channels", "quadrature_r"}, report);
}

inline void validate(const TimestreamConfig &config, ValidationReport &report) {
    if (!config.enabled) {
        report.add_error({"timestream", "enabled"},
                         "false is not supported by the current pipeline");
    }
    validate(config.output.raw_time_chunk,
             {"timestream", "raw_time_chunk", "output"}, report);
    validate(config.output.processed_time_chunk,
             {"timestream", "processed_time_chunk", "output"}, report);
    validate(config.chunking, report);
    validate(config.raw_time_chunk, report);
    validate(config.processed_time_chunk, report);
    validate(config.fruit_loops, report);
    if (config.fruit_loops.injected_source_test.enabled &&
        !config.raw_time_chunk.kernel.enabled) {
        report.add_error(
            {"timestream", "fruit_loops", "injected_source_test", "enabled"},
            "requires timestream.raw_time_chunk.kernel.enabled=true");
    }
    validate(config.learning, report);
    validate(config.auxiliary_channels, report);
}

}  // namespace citlali::config
