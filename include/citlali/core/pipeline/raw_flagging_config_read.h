#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_config_read_helpers.h>

#include <array>
#include <string_view>
#include <tuple>

namespace citlali::pipeline {

inline constexpr std::array<std::string_view, 60>
    raw_flagging_request_paths{
        "timestream.raw_time_chunk.despike.enabled",
        "timestream.raw_time_chunk.despike.min_spike_sigma",
        "timestream.raw_time_chunk.despike.time_constant_sec",
        "timestream.raw_time_chunk.despike.window_size",
        "timestream.raw_time_chunk.despike.legacy.enabled",
        "timestream.raw_time_chunk.despike.source_protection.enabled",
        "timestream.raw_time_chunk.despike.source_protection.radius_arcsec",
        "timestream.raw_time_chunk.despike.local_residual.enabled",
        "timestream.raw_time_chunk.despike.local_residual.window_sec",
        "timestream.raw_time_chunk.despike.local_residual.sigma_scale",
        "timestream.raw_time_chunk.despike.local_residual.delta_sigma_scale",
        "timestream.raw_time_chunk.despike.local_residual.expand_with_filter",
        "timestream.raw_time_chunk.despike.local_residual.event_padding_sec",
        "timestream.raw_time_chunk.despike.local_residual.high_score_event_override",
        "timestream.raw_time_chunk.despike.local_residual.max_added_flagged_fraction",
        "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate.enabled",
        "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate.candidate_rel_sigma_scale",
        "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate.candidate_sigma_scale",
        "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate.window_sec",
        "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate.half_peak_frac",
        "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate.max_width_sec",
        "timestream.raw_time_chunk.despike.local_residual.compact_raw_gate.max_step_shift_z",
        "timestream.raw_time_chunk.despike.local_residual.compact_delta_gate.enabled",
        "timestream.raw_time_chunk.despike.local_residual.compact_delta_gate.window_sec",
        "timestream.raw_time_chunk.despike.local_residual.compact_delta_gate.half_peak_frac",
        "timestream.raw_time_chunk.despike.local_residual.compact_delta_gate.max_width_sec",
        "timestream.raw_time_chunk.despike.local_residual.compact_delta_gate.max_step_shift_z",
        "timestream.raw_time_chunk.flagging.delta_f_min_Hz",
        "timestream.raw_time_chunk.flagging.lower_tod_inv_var_factor",
        "timestream.raw_time_chunk.flagging.upper_tod_inv_var_factor",
        "timestream.raw_time_chunk.flagging.network_step_mask.enabled",
        "timestream.raw_time_chunk.flagging.network_step_mask.step_window_sec",
        "timestream.raw_time_chunk.flagging.network_step_mask.step_score_thresh",
        "timestream.raw_time_chunk.flagging.network_step_mask.min_good_frac",
        "timestream.raw_time_chunk.flagging.network_step_mask.min_det_used",
        "timestream.raw_time_chunk.flagging.network_step_mask.min_step_det_frac",
        "timestream.raw_time_chunk.flagging.network_step_mask.min_alignment_frac",
        "timestream.raw_time_chunk.flagging.network_step_mask.cluster_tol_sec",
        "timestream.raw_time_chunk.flagging.network_step_mask.mask_half_width_sec",
        "timestream.raw_time_chunk.flagging.network_step_mask.max_flagged_fraction",
        "timestream.raw_time_chunk.flagging.impulsive_capture.enabled",
        "timestream.raw_time_chunk.flagging.impulsive_capture.min_good_frac",
        "timestream.raw_time_chunk.flagging.impulsive_capture.min_event_z",
        "timestream.raw_time_chunk.flagging.impulsive_capture.near_event_z",
        "timestream.raw_time_chunk.flagging.impulsive_capture.max_events_per_network",
        "timestream.raw_time_chunk.flagging.impulsive_capture.snippet_pre_window_sec",
        "timestream.raw_time_chunk.flagging.impulsive_capture.snippet_post_window_sec",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.enabled",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.min_good_frac",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.event_score_thresh",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.min_det_used",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.min_impulsive_det_frac",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.min_alignment_frac",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.min_networks_aligned",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.high_score_override_thresh",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.high_score_min_networks_aligned",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.cluster_tol_sec",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.mask_pre_window_sec",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.mask_post_window_sec",
        "timestream.raw_time_chunk.flagging.impulsive_coincidence.max_flagged_fraction",
    };

template <class Config, class Diagnostics>
void read_raw_despike_request_config(
    Config &config, citlali::config::RawTimeChunkDespikeConfig &despike,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "despike", name};
    };
    read_optional_raw_request_value(
        config, key("enabled"), despike.enabled, diagnostics);
    read_optional_raw_request_value(
        config, key("min_spike_sigma"), despike.min_spike_sigma,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("time_constant_sec"), despike.time_constant_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("window_size"), despike.window_size, diagnostics,
        {}, {0.0});
    read_optional_raw_request_value(
        config,
        std::tuple{"timestream", "raw_time_chunk", "despike", "legacy",
                   "enabled"},
        despike.legacy_enabled, diagnostics);
    read_optional_raw_request_value(
        config,
        std::tuple{"timestream", "raw_time_chunk", "despike",
                   "source_protection", "enabled"},
        despike.source_protection.enabled, diagnostics);
    read_optional_raw_request_value(
        config,
        std::tuple{"timestream", "raw_time_chunk", "despike",
                   "source_protection", "radius_arcsec"},
        despike.source_protection.radius_arcsec, diagnostics, {}, {0.0});

    auto &local = despike.local_residual;
    auto local_key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "despike", "local_residual",
            name};
    };
    read_optional_raw_request_value(
        config, local_key("enabled"), local.enabled, diagnostics);
    read_optional_raw_request_value(
        config, local_key("window_sec"), local.window_sec, diagnostics,
        {}, {0.0});
    read_optional_raw_request_value(
        config, local_key("sigma_scale"), local.sigma_scale, diagnostics,
        {}, {0.0});
    read_optional_raw_request_value(
        config, local_key("delta_sigma_scale"), local.delta_sigma_scale,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, local_key("expand_with_filter"), local.expand_with_filter,
        diagnostics);
    read_optional_raw_request_value(
        config, local_key("event_padding_sec"), local.event_padding_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, local_key("high_score_event_override"),
        local.high_score_event_override, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, local_key("max_added_flagged_fraction"),
        local.max_added_flagged_fraction, diagnostics, {}, {0.0}, {1.0});

    auto &raw_gate = local.compact_raw_gate;
    auto raw_gate_key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "despike", "local_residual",
            "compact_raw_gate", name};
    };
    read_optional_raw_request_value(
        config, raw_gate_key("enabled"), raw_gate.enabled, diagnostics);
    const auto relative_key = raw_gate_key("candidate_rel_sigma_scale");
    const auto legacy_key = raw_gate_key("candidate_sigma_scale");
    if (config.template has_typed<double>(relative_key)) {
        read_optional_raw_request_value(
            config, relative_key, raw_gate.candidate_rel_sigma_scale,
            diagnostics, {}, {0.0});
    } else if (config.template has_typed<double>(legacy_key)) {
        double legacy_sigma = local.sigma_scale;
        read_config_value_if_clean(
            config, legacy_key, legacy_sigma,
            [&raw_gate, &local](double value) {
                if (local.sigma_scale > 0.0) {
                    raw_gate.candidate_rel_sigma_scale =
                        value / local.sigma_scale;
                }
            },
            diagnostics, {}, {0.0});
    }
    read_optional_raw_request_value(
        config, raw_gate_key("window_sec"), raw_gate.window_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, raw_gate_key("half_peak_frac"), raw_gate.half_peak_frac,
        diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, raw_gate_key("max_width_sec"), raw_gate.max_width_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, raw_gate_key("max_step_shift_z"),
        raw_gate.max_step_shift_z, diagnostics, {}, {0.0});

    auto &delta_gate = local.compact_delta_gate;
    auto delta_gate_key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "despike", "local_residual",
            "compact_delta_gate", name};
    };
    read_optional_raw_request_value(
        config, delta_gate_key("enabled"), delta_gate.enabled,
        diagnostics);
    read_optional_raw_request_value(
        config, delta_gate_key("window_sec"), delta_gate.window_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, delta_gate_key("half_peak_frac"),
        delta_gate.half_peak_frac, diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, delta_gate_key("max_width_sec"),
        delta_gate.max_width_sec, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, delta_gate_key("max_step_shift_z"),
        delta_gate.max_step_shift_z, diagnostics, {}, {0.0});
}

template <class Config, class Diagnostics>
void read_raw_flagging_request_config(
    Config &config, citlali::config::RawTimeChunkFlaggingConfig &flagging,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "flagging", name};
    };
    read_optional_raw_request_value(
        config, key("delta_f_min_Hz"), flagging.delta_f_min_Hz,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("lower_tod_inv_var_factor"),
        flagging.lower_tod_inv_var_factor, diagnostics);
    read_optional_raw_request_value(
        config, key("upper_tod_inv_var_factor"),
        flagging.upper_tod_inv_var_factor, diagnostics);

    auto &step = flagging.network_step_mask;
    auto step_key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "flagging",
            "network_step_mask", name};
    };
    read_optional_raw_request_value(
        config, step_key("enabled"), step.enabled, diagnostics);
    read_optional_raw_request_value(
        config, step_key("step_window_sec"), step.step_window_sec,
        diagnostics, {}, {0.01});
    read_optional_raw_request_value(
        config, step_key("step_score_thresh"), step.step_score_thresh,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, step_key("min_good_frac"), step.min_good_frac,
        diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, step_key("min_det_used"), step.min_det_used, diagnostics,
        {}, {1});
    read_optional_raw_request_value(
        config, step_key("min_step_det_frac"), step.min_step_det_frac,
        diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, step_key("min_alignment_frac"), step.min_alignment_frac,
        diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, step_key("cluster_tol_sec"), step.cluster_tol_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, step_key("mask_half_width_sec"), step.mask_half_width_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, step_key("max_flagged_fraction"),
        step.max_flagged_fraction, diagnostics, {}, {0.0}, {1.0});

    auto &capture = flagging.impulsive_capture;
    auto capture_key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "flagging",
            "impulsive_capture", name};
    };
    read_optional_raw_request_value(
        config, capture_key("enabled"), capture.enabled, diagnostics);
    read_optional_raw_request_value(
        config, capture_key("min_good_frac"), capture.min_good_frac,
        diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, capture_key("min_event_z"), capture.min_event_z,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, capture_key("near_event_z"), capture.near_event_z,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, capture_key("max_events_per_network"),
        capture.max_events_per_network, diagnostics, {}, {0});
    read_optional_raw_request_value(
        config, capture_key("snippet_pre_window_sec"),
        capture.snippet_pre_window_sec, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, capture_key("snippet_post_window_sec"),
        capture.snippet_post_window_sec, diagnostics, {}, {0.0});

    auto &coincidence = flagging.impulsive_coincidence;
    auto coincidence_key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "flagging",
            "impulsive_coincidence", name};
    };
    read_optional_raw_request_value(
        config, coincidence_key("enabled"), coincidence.enabled,
        diagnostics);
    read_optional_raw_request_value(
        config, coincidence_key("min_good_frac"),
        coincidence.min_good_frac, diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, coincidence_key("event_score_thresh"),
        coincidence.event_score_thresh, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, coincidence_key("min_det_used"),
        coincidence.min_det_used, diagnostics, {}, {1});
    read_optional_raw_request_value(
        config, coincidence_key("min_impulsive_det_frac"),
        coincidence.min_impulsive_det_frac, diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, coincidence_key("min_alignment_frac"),
        coincidence.min_alignment_frac, diagnostics, {}, {0.0}, {1.0});
    read_optional_raw_request_value(
        config, coincidence_key("min_networks_aligned"),
        coincidence.min_networks_aligned, diagnostics, {}, {1});
    read_optional_raw_request_value(
        config, coincidence_key("high_score_override_thresh"),
        coincidence.high_score_override_thresh, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, coincidence_key("high_score_min_networks_aligned"),
        coincidence.high_score_min_networks_aligned, diagnostics, {}, {0});
    read_optional_raw_request_value(
        config, coincidence_key("cluster_tol_sec"),
        coincidence.cluster_tol_sec, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, coincidence_key("mask_pre_window_sec"),
        coincidence.mask_pre_window_sec, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, coincidence_key("mask_post_window_sec"),
        coincidence.mask_post_window_sec, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, coincidence_key("max_flagged_fraction"),
        coincidence.max_flagged_fraction, diagnostics, {}, {0.0}, {1.0});
}

template <class Config, class Diagnostics>
void read_raw_flagging_and_despike_request_config(
    Config &config, citlali::config::RawTimeChunkConfig &raw,
    Diagnostics &diagnostics) {
    read_raw_despike_request_config(config, raw.despike, diagnostics);
    read_raw_flagging_request_config(config, raw.flagging, diagnostics);
}

}  // namespace citlali::pipeline
