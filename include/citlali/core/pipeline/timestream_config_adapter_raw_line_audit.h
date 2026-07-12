#pragma once

#include <citlali/core/config/timestream_config.h>

namespace citlali::pipeline {

template <class RtcLineAudit>
void adapt_raw_line_audit_config_one_way(
    const citlali::config::RawTimeChunkLineAuditConfig &source,
    RtcLineAudit &target) {
    target.enabled = source.enabled;
    target.line_min_hz = source.line_min_hz;
    target.line_max_hz = source.line_max_hz;
    target.segment_sec = source.segment_sec;
    target.min_segment_sec = source.min_segment_sec;
    target.overlap_frac = source.overlap_frac;
    target.continuum_radius_bins = source.continuum_radius_bins;
    target.prominence_thresh = source.prominence_thresh;
    target.cm_prominence_thresh = source.cm_prominence_thresh;
    target.min_good_frac = source.min_good_frac;
    target.min_windows = source.min_windows;
    target.max_peaks_per_detector = source.max_peaks_per_detector;
    target.max_det = source.max_det;
    target.min_det_for_network = source.min_det_for_network;
    target.cluster_tol_hz = source.cluster_tol_hz;
    target.notch_min_detector_frac = source.notch_min_detector_frac;
    target.notch_min_detectors = source.notch_min_detectors;
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
        source.post_filter_apply_iterations;
    target.post_filter_line_min_hz = source.post_filter_line_min_hz;
    target.post_filter_line_max_hz = source.post_filter_line_max_hz;
    target.ptc_model_protected_enabled =
        source.ptc_model_protected_enabled;
    target.ptc_require_model_subtracted =
        source.ptc_require_model_subtracted;
    target.ptc_apply_fixed_notches = source.ptc_apply_fixed_notches;
    target.ptc_apply_shared_notches = source.ptc_apply_shared_notches;
    target.ptc_apply_detector_notches = source.ptc_apply_detector_notches;
    target.ptc_apply_iterations = source.ptc_apply_iterations;
    target.ptc_line_min_hz = source.ptc_line_min_hz;
    target.ptc_line_max_hz = source.ptc_line_max_hz;
    target.fixed_notch_enabled = source.fixed_notch_enabled;
    target.fixed_notch_freqs_hz = source.fixed_notch_freqs_hz;
    target.fixed_notch_widths_hz = source.fixed_notch_widths_hz;
    if (target.fixed_notch_widths_hz.empty()) {
        target.fixed_notch_widths_hz.push_back(0.25);
    }
    if (target.fixed_notch_enabled &&
        target.fixed_notch_widths_hz.size() == 1 &&
        target.fixed_notch_freqs_hz.size() > 1) {
        target.fixed_notch_widths_hz.resize(
            target.fixed_notch_freqs_hz.size(),
            target.fixed_notch_widths_hz.front());
    }
    target.fixed_notch_exclusion_half_width_hz =
        source.fixed_notch_exclusion_half_width_hz;
    target.apply_shared_notches = source.apply_shared_notches;
    target.apply_min_support_networks = source.apply_min_support_networks;
    target.apply_min_detector_frac = source.apply_min_detector_frac;
    target.apply_min_common_mode_prominence =
        source.apply_min_common_mode_prominence;
    target.apply_width_scale = source.apply_width_scale;
    target.apply_min_width_hz = source.apply_min_width_hz;
    target.apply_max_width_hz = source.apply_max_width_hz;
    target.apply_max_notches = source.apply_max_notches;
    target.apply_cluster_tol_hz = source.apply_cluster_tol_hz;
    target.detector_notch_min_prominence =
        source.detector_notch_min_prominence;
    target.detector_notch_min_line_power_frac =
        source.detector_notch_min_line_power_frac;
    target.detector_notch_max_notches =
        source.detector_notch_max_notches;
    target.detector_notch_width_scale = source.detector_notch_width_scale;
    target.detector_notch_min_width_hz =
        source.detector_notch_min_width_hz;
    target.detector_notch_max_width_hz =
        source.detector_notch_max_width_hz;
    target.detector_notch_context_samples =
        source.detector_notch_context_samples;
}

}  // namespace citlali::pipeline
