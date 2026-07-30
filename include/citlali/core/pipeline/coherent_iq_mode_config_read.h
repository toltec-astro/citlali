#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_config_read_helpers.h>

#include <array>
#include <string_view>
#include <tuple>

namespace citlali::pipeline {

inline constexpr std::array<std::string_view, 10>
    coherent_iq_mode_observer_request_paths{
        "timestream.raw_time_chunk.coherent_iq_mode_observer.enabled",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.template_paths",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.candidate_step_score_min",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.candidate_impulsive_score_min",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.candidate_cluster_tolerance_sec",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.pre_window_sec",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.guard_window_sec",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.post_window_sec",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.cross_network_tolerance_sec",
        "timestream.raw_time_chunk.coherent_iq_mode_observer.max_candidates_per_scan_per_network",
    };

template <class Config, class Diagnostics>
void read_coherent_iq_mode_observer_request_config(
    Config &config,
    citlali::config::RawTimeChunkCoherentIqModeObserverConfig &observer,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "coherent_iq_mode_observer",
            name};
    };
    read_optional_raw_request_value(
        config, key("enabled"), observer.enabled, diagnostics);
    read_optional_raw_request_value(
        config, key("template_paths"), observer.template_paths,
        diagnostics);
    read_optional_raw_request_value(
        config, key("candidate_step_score_min"),
        observer.candidate_step_score_min, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("candidate_impulsive_score_min"),
        observer.candidate_impulsive_score_min, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("candidate_cluster_tolerance_sec"),
        observer.candidate_cluster_tolerance_sec, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("pre_window_sec"), observer.pre_window_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("guard_window_sec"), observer.guard_window_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("post_window_sec"), observer.post_window_sec,
        diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("cross_network_tolerance_sec"),
        observer.cross_network_tolerance_sec, diagnostics, {}, {0.0});
    read_optional_raw_request_value(
        config, key("max_candidates_per_scan_per_network"),
        observer.max_candidates_per_scan_per_network, diagnostics, {}, {1});
}

}  // namespace citlali::pipeline
