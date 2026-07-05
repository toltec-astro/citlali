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
    auto &typed_flagging = typed_raw.flagging;
    typed_flagging.delta_f_min_Hz = rtcproc.delta_f_min_Hz;
    typed_flagging.lower_tod_inv_var_factor = rtcproc.lower_inv_var_factor;
    typed_flagging.upper_tod_inv_var_factor = rtcproc.upper_inv_var_factor;

    const auto &network_step_mask = rtcproc.network_step_mask;
    auto &typed_network_step = typed_flagging.network_step_mask;
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
    auto &typed_capture = typed_flagging.impulsive_capture;
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
    auto &typed_coincidence = typed_flagging.impulsive_coincidence;
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
        static_cast<int>(
            impulsive_coincidence.high_score_min_networks_aligned);
    typed_coincidence.cluster_tol_sec = impulsive_coincidence.cluster_tol_sec;
    typed_coincidence.mask_pre_window_sec =
        impulsive_coincidence.mask_pre_window_sec;
    typed_coincidence.mask_post_window_sec =
        impulsive_coincidence.mask_post_window_sec;
    typed_coincidence.max_flagged_fraction =
        impulsive_coincidence.max_flagged_fraction;

    auto &typed_kernel = typed_raw.kernel;
    typed_kernel.enabled = rtcproc.run_kernel;
    if (rtcproc.run_kernel) {
        typed_kernel.filepath = rtcproc.kernel.filepath;
        typed_kernel.type = rtcproc.kernel.type;
        typed_kernel.fwhm_arcsec = rtcproc.kernel.fwhm_rad * RAD_TO_ASEC;
        typed_kernel.image_ext_names = rtcproc.kernel.img_ext_names;
    }

    auto &typed_altaz = typed_raw.altaz_destripe;
    typed_altaz.enabled = rtcproc.altaz_destripe.enabled;
    typed_altaz.grouping = rtcproc.altaz_destripe.grouping;
    typed_altaz.fit_time_trend = rtcproc.altaz_destripe.fit_time_trend;
    typed_altaz.fit_derivs = rtcproc.altaz_destripe.fit_derivs;
    typed_altaz.min_samples =
        static_cast<int>(rtcproc.altaz_destripe.min_samples);

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

    typed_raw.downsample.enabled = rtcproc.run_downsample;
    if (rtcproc.run_downsample) {
        typed_raw.downsample.factor = rtcproc.downsampler.factor;
        typed_raw.downsample.downsampled_freq_Hz =
            rtcproc.downsampler.downsampled_freq_Hz;
    }

    auto &typed_filter = typed_raw.filter;
    typed_filter.enabled = rtcproc.run_tod_filter;
    if (rtcproc.run_tod_filter) {
        typed_filter.a_gibbs = rtcproc.filter.a_gibbs;
        typed_filter.freq_low_Hz = rtcproc.filter.freq_low_Hz;
        typed_filter.freq_high_Hz = rtcproc.filter.freq_high_Hz;
        typed_filter.n_terms = static_cast<int>(rtcproc.filter.n_terms);
        typed_filter.notch.enabled = rtcproc.run_tod_notch;
        if (rtcproc.run_tod_notch) {
            typed_filter.notch.zero_phase = rtcproc.filter.notch_zero_phase;
            typed_filter.notch.freqs_Hz = rtcproc.filter.w0s;
            typed_filter.notch.delta_f_Hz.clear();
            typed_filter.notch.delta_f_Hz.reserve(rtcproc.filter.qs.size());
            for (std::size_t i = 0; i < rtcproc.filter.qs.size(); ++i) {
                const auto center_Hz = i < rtcproc.filter.w0s.size()
                                           ? rtcproc.filter.w0s[i]
                                           : 0.0;
                typed_filter.notch.delta_f_Hz.push_back(
                    rtcproc.filter.qs[i] > 0.0
                        ? center_Hz / rtcproc.filter.qs[i]
                        : 0.0);
            }
        }
    }

    auto &typed_iir_filter = typed_raw.iir_filter;
    typed_iir_filter.enabled = rtcproc.run_tod_iir_highpass;
    if (rtcproc.run_tod_iir_highpass) {
        typed_iir_filter.freq_Hz = rtcproc.filter.iir_highpass_freq_Hz;
        typed_iir_filter.order = rtcproc.filter.iir_highpass_order;
        typed_iir_filter.zero_phase = rtcproc.filter.iir_highpass_zero_phase;
    }

    typed_raw.flux_calibration_enabled = rtcproc.run_calibrate;
    typed_raw.extinction_correction_enabled = rtcproc.run_extinction;

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
