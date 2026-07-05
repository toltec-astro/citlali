#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/timestream_config_mirror.h>

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    // get rtcproc config
    rtcproc.get_config(config, missing_keys, invalid_keys);
    citlali::pipeline::mirror_raw_despike_config(
        typed_timestream_config.raw_time_chunk.despike, rtcproc);

    auto &typed_raw = typed_timestream_config.raw_time_chunk;
    citlali::pipeline::mirror_raw_flagging_config(
        typed_raw.flagging, rtcproc);

    citlali::pipeline::mirror_raw_kernel_config(
        typed_raw.kernel, rtcproc, RAD_TO_ASEC);

    citlali::pipeline::mirror_raw_altaz_destripe_config(
        typed_raw.altaz_destripe, rtcproc);

    const auto &line_audit = rtcproc.line_audit;
    auto &typed_line_audit = typed_raw.line_audit;
    typed_line_audit.enabled = line_audit.enabled;
    typed_line_audit.line_min_hz = line_audit.line_min_hz;
    typed_line_audit.line_max_hz = line_audit.line_max_hz;
    typed_line_audit.segment_sec = line_audit.segment_sec;
    typed_line_audit.min_segment_sec = line_audit.min_segment_sec;
    typed_line_audit.overlap_frac = line_audit.overlap_frac;
    typed_line_audit.continuum_radius_bins =
        static_cast<int>(line_audit.continuum_radius_bins);
    typed_line_audit.prominence_thresh = line_audit.prominence_thresh;
    typed_line_audit.cm_prominence_thresh = line_audit.cm_prominence_thresh;
    typed_line_audit.min_good_frac = line_audit.min_good_frac;
    typed_line_audit.min_windows = static_cast<int>(line_audit.min_windows);
    typed_line_audit.max_peaks_per_detector =
        static_cast<int>(line_audit.max_peaks_per_detector);
    typed_line_audit.max_det = static_cast<int>(line_audit.max_det);
    typed_line_audit.min_det_for_network =
        static_cast<int>(line_audit.min_det_for_network);
    typed_line_audit.cluster_tol_hz = line_audit.cluster_tol_hz;
    typed_line_audit.notch_min_detector_frac =
        line_audit.notch_min_detector_frac;
    typed_line_audit.notch_min_detectors =
        static_cast<int>(line_audit.notch_min_detectors);
    typed_line_audit.notch_min_cm_prominence =
        line_audit.notch_min_cm_prominence;
    typed_line_audit.detector_min_prominence =
        line_audit.detector_min_prominence;
    typed_line_audit.detector_min_line_power_frac =
        line_audit.detector_min_line_power_frac;
    typed_line_audit.bad_detector_max_cluster_frac =
        line_audit.bad_detector_max_cluster_frac;
    typed_line_audit.pre_filter_enabled = line_audit.pre_filter_enabled;
    typed_line_audit.post_filter_enabled = line_audit.post_filter_enabled;
    typed_line_audit.post_filter_apply_shared_notches =
        line_audit.post_filter_apply_shared_notches;
    typed_line_audit.post_filter_apply_detector_notches =
        line_audit.post_filter_apply_detector_notches;
    typed_line_audit.post_filter_apply_iterations =
        static_cast<int>(line_audit.post_filter_apply_iterations);
    typed_line_audit.post_filter_line_min_hz =
        line_audit.post_filter_line_min_hz;
    typed_line_audit.post_filter_line_max_hz =
        line_audit.post_filter_line_max_hz;
    typed_line_audit.ptc_model_protected_enabled =
        line_audit.ptc_model_protected_enabled;
    typed_line_audit.ptc_require_model_subtracted =
        line_audit.ptc_require_model_subtracted;
    typed_line_audit.ptc_apply_fixed_notches =
        line_audit.ptc_apply_fixed_notches;
    typed_line_audit.ptc_apply_shared_notches =
        line_audit.ptc_apply_shared_notches;
    typed_line_audit.ptc_apply_detector_notches =
        line_audit.ptc_apply_detector_notches;
    typed_line_audit.ptc_apply_iterations =
        static_cast<int>(line_audit.ptc_apply_iterations);
    typed_line_audit.ptc_line_min_hz = line_audit.ptc_line_min_hz;
    typed_line_audit.ptc_line_max_hz = line_audit.ptc_line_max_hz;
    typed_line_audit.fixed_notch_enabled = line_audit.fixed_notch_enabled;
    typed_line_audit.fixed_notch_freqs_hz =
        line_audit.fixed_notch_freqs_hz;
    typed_line_audit.fixed_notch_widths_hz =
        line_audit.fixed_notch_widths_hz;
    typed_line_audit.fixed_notch_exclusion_half_width_hz =
        line_audit.fixed_notch_exclusion_half_width_hz;
    typed_line_audit.apply_shared_notches =
        line_audit.apply_shared_notches;
    typed_line_audit.apply_min_support_networks =
        static_cast<int>(line_audit.apply_min_support_networks);
    typed_line_audit.apply_min_detector_frac =
        line_audit.apply_min_detector_frac;
    typed_line_audit.apply_min_common_mode_prominence =
        line_audit.apply_min_common_mode_prominence;
    typed_line_audit.apply_width_scale = line_audit.apply_width_scale;
    typed_line_audit.apply_min_width_hz = line_audit.apply_min_width_hz;
    typed_line_audit.apply_max_width_hz = line_audit.apply_max_width_hz;
    typed_line_audit.apply_max_notches =
        static_cast<int>(line_audit.apply_max_notches);
    typed_line_audit.apply_cluster_tol_hz =
        line_audit.apply_cluster_tol_hz;
    typed_line_audit.detector_notch_min_prominence =
        line_audit.detector_notch_min_prominence;
    typed_line_audit.detector_notch_min_line_power_frac =
        line_audit.detector_notch_min_line_power_frac;
    typed_line_audit.detector_notch_max_notches =
        static_cast<int>(line_audit.detector_notch_max_notches);
    typed_line_audit.detector_notch_width_scale =
        line_audit.detector_notch_width_scale;
    typed_line_audit.detector_notch_min_width_hz =
        line_audit.detector_notch_min_width_hz;
    typed_line_audit.detector_notch_max_width_hz =
        line_audit.detector_notch_max_width_hz;
    typed_line_audit.detector_notch_context_samples =
        static_cast<int>(line_audit.detector_notch_context_samples);

    citlali::pipeline::mirror_raw_downsample_config(
        typed_raw.downsample, rtcproc);

    auto &typed_filter = typed_raw.filter;
    citlali::pipeline::mirror_raw_filter_config(typed_filter, rtcproc);

    citlali::pipeline::mirror_raw_iir_filter_config(
        typed_raw.iir_filter, rtcproc);

    citlali::pipeline::mirror_raw_correction_flags(typed_raw, rtcproc);

    rtcproc.configure_filter_edge_guard(telescope.fsmp);
    auto &typed_edge_guard = typed_filter.edge_guard;
    typed_edge_guard.enabled = rtcproc.filter_edge_guard.enabled;
    if (auto parsed = citlali::config::parse_raw_filter_edge_guard_mode(
            rtcproc.filter_edge_guard.mode)) {
        typed_edge_guard.mode = *parsed;
    }
    if (auto parsed = citlali::config::parse_raw_filter_edge_guard_combine(
            rtcproc.filter_edge_guard.combine)) {
        typed_edge_guard.combine = *parsed;
    }
    typed_edge_guard.min_samples =
        static_cast<int>(rtcproc.filter_edge_guard.min_samples);
    typed_edge_guard.extra_samples =
        static_cast<int>(rtcproc.filter_edge_guard.extra_samples);
    typed_edge_guard.max_samples =
        static_cast<int>(rtcproc.filter_edge_guard.max_samples);
    typed_edge_guard.iir_settle_attenuation =
        rtcproc.filter_edge_guard.iir_settle_attenuation;
    typed_edge_guard.apply_fir = rtcproc.filter_edge_guard.apply_fir;
    typed_edge_guard.apply_notch = rtcproc.filter_edge_guard.apply_notch;
    typed_edge_guard.apply_dynamic_notch =
        rtcproc.filter_edge_guard.apply_dynamic_notch;
    typed_edge_guard.apply_iir_highpass =
        rtcproc.filter_edge_guard.apply_iir_highpass;
    typed_edge_guard.apply_downsample =
        rtcproc.filter_edge_guard.apply_downsample;
    telescope.inner_scans_chunk = rtcproc.filter_edge_guard.context_samples;
    telescope.outer_scans_chunk = telescope.inner_scans_chunk;
    if (rtcproc.tod_output_outer) {
        telescope.outer_scans_chunk = std::max<Eigen::Index>(
            telescope.outer_scans_chunk,
            std::max<Eigen::Index>(0, rtcproc.tod_output_outer_context_samples));
    }
    if (rtcproc.line_audit.enabled &&
        rtcproc.line_audit.post_filter_enabled &&
        rtcproc.line_audit.post_filter_apply_detector_notches) {
        telescope.outer_scans_chunk = std::max<Eigen::Index>(
            telescope.outer_scans_chunk,
            std::max<Eigen::Index>(0, rtcproc.line_audit.detector_notch_context_samples));
    }

    // ignore hwpr?
    get_config_value(config, calib.ignore_hwpr, missing_keys, invalid_keys,
                     std::tuple{"timestream","polarimetry", "ignore_hwpr"});
}
