#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_config_read_helpers.h>

#include <array>
#include <string_view>
#include <tuple>

namespace citlali::pipeline {

inline constexpr std::array<std::string_view, 56>
    raw_line_audit_request_paths{
        "timestream.raw_time_chunk.line_audit.enabled",
        "timestream.raw_time_chunk.line_audit.line_min_hz",
        "timestream.raw_time_chunk.line_audit.line_max_hz",
        "timestream.raw_time_chunk.line_audit.segment_sec",
        "timestream.raw_time_chunk.line_audit.min_segment_sec",
        "timestream.raw_time_chunk.line_audit.overlap_frac",
        "timestream.raw_time_chunk.line_audit.continuum_radius_bins",
        "timestream.raw_time_chunk.line_audit.prominence_thresh",
        "timestream.raw_time_chunk.line_audit.cm_prominence_thresh",
        "timestream.raw_time_chunk.line_audit.min_good_frac",
        "timestream.raw_time_chunk.line_audit.min_windows",
        "timestream.raw_time_chunk.line_audit.max_peaks_per_detector",
        "timestream.raw_time_chunk.line_audit.max_det",
        "timestream.raw_time_chunk.line_audit.min_det_for_network",
        "timestream.raw_time_chunk.line_audit.cluster_tol_hz",
        "timestream.raw_time_chunk.line_audit.notch_min_detector_frac",
        "timestream.raw_time_chunk.line_audit.notch_min_detectors",
        "timestream.raw_time_chunk.line_audit.notch_min_cm_prominence",
        "timestream.raw_time_chunk.line_audit.detector_min_prominence",
        "timestream.raw_time_chunk.line_audit.detector_min_line_power_frac",
        "timestream.raw_time_chunk.line_audit.bad_detector_max_cluster_frac",
        "timestream.raw_time_chunk.line_audit.pre_filter_enabled",
        "timestream.raw_time_chunk.line_audit.post_filter_enabled",
        "timestream.raw_time_chunk.line_audit.post_filter_apply_shared_notches",
        "timestream.raw_time_chunk.line_audit.post_filter_apply_detector_notches",
        "timestream.raw_time_chunk.line_audit.post_filter_apply_iterations",
        "timestream.raw_time_chunk.line_audit.post_filter_line_min_hz",
        "timestream.raw_time_chunk.line_audit.post_filter_line_max_hz",
        "timestream.raw_time_chunk.line_audit.ptc_model_protected_enabled",
        "timestream.raw_time_chunk.line_audit.ptc_require_model_subtracted",
        "timestream.raw_time_chunk.line_audit.ptc_apply_fixed_notches",
        "timestream.raw_time_chunk.line_audit.ptc_apply_shared_notches",
        "timestream.raw_time_chunk.line_audit.ptc_apply_detector_notches",
        "timestream.raw_time_chunk.line_audit.ptc_apply_iterations",
        "timestream.raw_time_chunk.line_audit.ptc_line_min_hz",
        "timestream.raw_time_chunk.line_audit.ptc_line_max_hz",
        "timestream.raw_time_chunk.line_audit.fixed_notch_enabled",
        "timestream.raw_time_chunk.line_audit.fixed_notch_freqs_hz",
        "timestream.raw_time_chunk.line_audit.fixed_notch_widths_hz",
        "timestream.raw_time_chunk.line_audit.fixed_notch_exclusion_half_width_hz",
        "timestream.raw_time_chunk.line_audit.apply_shared_notches",
        "timestream.raw_time_chunk.line_audit.apply_min_support_networks",
        "timestream.raw_time_chunk.line_audit.apply_min_detector_frac",
        "timestream.raw_time_chunk.line_audit.apply_min_common_mode_prominence",
        "timestream.raw_time_chunk.line_audit.apply_width_scale",
        "timestream.raw_time_chunk.line_audit.apply_min_width_hz",
        "timestream.raw_time_chunk.line_audit.apply_max_width_hz",
        "timestream.raw_time_chunk.line_audit.apply_max_notches",
        "timestream.raw_time_chunk.line_audit.apply_cluster_tol_hz",
        "timestream.raw_time_chunk.line_audit.detector_notch_min_prominence",
        "timestream.raw_time_chunk.line_audit.detector_notch_min_line_power_frac",
        "timestream.raw_time_chunk.line_audit.detector_notch_max_notches",
        "timestream.raw_time_chunk.line_audit.detector_notch_width_scale",
        "timestream.raw_time_chunk.line_audit.detector_notch_min_width_hz",
        "timestream.raw_time_chunk.line_audit.detector_notch_max_width_hz",
        "timestream.raw_time_chunk.line_audit.detector_notch_context_samples",
    };

template <class Config, class Diagnostics>
void read_raw_line_audit_request_config(
    Config &config, citlali::config::RawTimeChunkLineAuditConfig &audit,
    Diagnostics &diagnostics) {
    auto key = [](const char *name) {
        return std::tuple{
            "timestream", "raw_time_chunk", "line_audit", name};
    };
    auto read_bool = [&](const char *name, bool &target) {
        read_optional_raw_request_value(
            config, key(name), target, diagnostics);
    };
    auto read_int = [&](const char *name, int &target) {
        read_optional_raw_request_value(
            config, key(name), target, diagnostics);
    };
    auto read_double = [&](const char *name, double &target) {
        read_optional_raw_request_value(
            config, key(name), target, diagnostics);
    };

    read_bool("enabled", audit.enabled);
    read_double("line_min_hz", audit.line_min_hz);
    read_double("line_max_hz", audit.line_max_hz);
    read_double("segment_sec", audit.segment_sec);
    read_double("min_segment_sec", audit.min_segment_sec);
    read_double("overlap_frac", audit.overlap_frac);
    read_int("continuum_radius_bins", audit.continuum_radius_bins);
    read_double("prominence_thresh", audit.prominence_thresh);
    read_double("cm_prominence_thresh", audit.cm_prominence_thresh);
    read_double("min_good_frac", audit.min_good_frac);
    read_int("min_windows", audit.min_windows);
    read_int("max_peaks_per_detector", audit.max_peaks_per_detector);
    read_int("max_det", audit.max_det);
    read_int("min_det_for_network", audit.min_det_for_network);
    read_double("cluster_tol_hz", audit.cluster_tol_hz);
    read_double("notch_min_detector_frac", audit.notch_min_detector_frac);
    read_int("notch_min_detectors", audit.notch_min_detectors);
    read_double("notch_min_cm_prominence", audit.notch_min_cm_prominence);
    read_double("detector_min_prominence", audit.detector_min_prominence);
    read_double(
        "detector_min_line_power_frac",
        audit.detector_min_line_power_frac);
    read_double(
        "bad_detector_max_cluster_frac",
        audit.bad_detector_max_cluster_frac);
    read_bool("pre_filter_enabled", audit.pre_filter_enabled);
    read_bool("post_filter_enabled", audit.post_filter_enabled);
    read_bool(
        "post_filter_apply_shared_notches",
        audit.post_filter_apply_shared_notches);
    read_bool(
        "post_filter_apply_detector_notches",
        audit.post_filter_apply_detector_notches);
    read_int(
        "post_filter_apply_iterations",
        audit.post_filter_apply_iterations);
    read_double("post_filter_line_min_hz", audit.post_filter_line_min_hz);
    read_double("post_filter_line_max_hz", audit.post_filter_line_max_hz);
    read_bool(
        "ptc_model_protected_enabled", audit.ptc_model_protected_enabled);
    read_bool(
        "ptc_require_model_subtracted",
        audit.ptc_require_model_subtracted);
    read_bool("ptc_apply_fixed_notches", audit.ptc_apply_fixed_notches);
    read_bool("ptc_apply_shared_notches", audit.ptc_apply_shared_notches);
    read_bool(
        "ptc_apply_detector_notches", audit.ptc_apply_detector_notches);
    read_int("ptc_apply_iterations", audit.ptc_apply_iterations);
    read_double("ptc_line_min_hz", audit.ptc_line_min_hz);
    read_double("ptc_line_max_hz", audit.ptc_line_max_hz);
    read_bool("fixed_notch_enabled", audit.fixed_notch_enabled);
    read_optional_raw_request_value(
        config, key("fixed_notch_freqs_hz"), audit.fixed_notch_freqs_hz,
        diagnostics);
    read_optional_raw_request_value(
        config, key("fixed_notch_widths_hz"), audit.fixed_notch_widths_hz,
        diagnostics);
    read_double(
        "fixed_notch_exclusion_half_width_hz",
        audit.fixed_notch_exclusion_half_width_hz);
    read_bool("apply_shared_notches", audit.apply_shared_notches);
    read_int("apply_min_support_networks", audit.apply_min_support_networks);
    read_double("apply_min_detector_frac", audit.apply_min_detector_frac);
    read_double(
        "apply_min_common_mode_prominence",
        audit.apply_min_common_mode_prominence);
    read_double("apply_width_scale", audit.apply_width_scale);
    read_double("apply_min_width_hz", audit.apply_min_width_hz);
    read_double("apply_max_width_hz", audit.apply_max_width_hz);
    read_int("apply_max_notches", audit.apply_max_notches);
    read_double("apply_cluster_tol_hz", audit.apply_cluster_tol_hz);
    read_double(
        "detector_notch_min_prominence",
        audit.detector_notch_min_prominence);
    read_double(
        "detector_notch_min_line_power_frac",
        audit.detector_notch_min_line_power_frac);
    read_int(
        "detector_notch_max_notches", audit.detector_notch_max_notches);
    read_double(
        "detector_notch_width_scale", audit.detector_notch_width_scale);
    read_double(
        "detector_notch_min_width_hz", audit.detector_notch_min_width_hz);
    read_double(
        "detector_notch_max_width_hz", audit.detector_notch_max_width_hz);
    read_int(
        "detector_notch_context_samples",
        audit.detector_notch_context_samples);
}

}  // namespace citlali::pipeline
