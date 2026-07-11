#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <tuple>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_second_pass_local_config(
    Config &config,
    citlali::config::ProcessedTimeChunkSecondPassLocalConfig &typed,
    Diagnostics &diagnostics) {
    const auto key = [](const char *name) {
        return std::tuple{"timestream", "processed_time_chunk", "flagging",
                          "second_pass_local", name};
    };
    bool enabled = typed.enabled;
    read_optional_mirrored_config_value(
        config, key("enabled"), enabled, typed.enabled, diagnostics);
    if (!typed.enabled) {
        return;
    }

    auto read_double = [&](const char *name, double &target) {
        double value = target;
        read_optional_mirrored_config_value(
            config, key(name), value, target, diagnostics);
    };
    read_double("min_spike_sigma", typed.min_spike_sigma);
    read_double("min_good_frac", typed.min_good_frac);
    read_double("baseline_window_sec", typed.baseline_window_sec);
    read_double("sigma_scale", typed.sigma_scale);
    read_double("delta_sigma_scale", typed.delta_sigma_scale);
    read_double(
        "raw_candidate_rel_sigma_scale",
        typed.raw_candidate_rel_sigma_scale);
    read_double("raw_window_sec", typed.raw_window_sec);
    read_double("raw_half_peak_frac", typed.raw_half_peak_frac);
    read_double("raw_max_width_sec", typed.raw_max_width_sec);
    read_double("delta_window_sec", typed.delta_window_sec);
    read_double("delta_half_peak_frac", typed.delta_half_peak_frac);
    read_double("delta_max_width_sec", typed.delta_max_width_sec);
    read_double("max_step_shift_z", typed.max_step_shift_z);
    read_double("high_score_event_override", typed.high_score_event_override);
    read_double("merge_within_detector_sec", typed.merge_within_detector_sec);
    read_double("cluster_events_sec", typed.cluster_events_sec);
    read_double(
        "high_score_cluster_override", typed.high_score_cluster_override);

    auto read_int = [&](const char *name, int &target) {
        int value = target;
        read_optional_mirrored_config_value(
            config, key(name), value, target, diagnostics);
    };
    read_int("min_cluster_detectors", typed.min_cluster_detectors);
    read_int(
        "max_auto_flag_clusters_per_network",
        typed.max_auto_flag_clusters_per_network);

    bool selective = typed.selective_busy_network_acceptance_enabled;
    read_optional_mirrored_config_value(
        config, key("selective_busy_network_acceptance_enabled"), selective,
        typed.selective_busy_network_acceptance_enabled, diagnostics);

    const auto source_key = [](const char *name) {
        return std::tuple{"timestream", "processed_time_chunk", "flagging",
                          "second_pass_local", "source_protection", name};
    };
    bool source_enabled = typed.source_protection.enabled;
    read_optional_mirrored_config_value(
        config, source_key("enabled"), source_enabled,
        typed.source_protection.enabled, diagnostics);
    double source_radius = typed.source_protection.radius_arcsec;
    read_optional_mirrored_config_value(
        config, source_key("radius_arcsec"), source_radius,
        typed.source_protection.radius_arcsec, diagnostics);
}

}  // namespace citlali::pipeline
