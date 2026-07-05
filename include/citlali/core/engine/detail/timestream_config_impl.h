#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    // get rtcproc config
    rtcproc.get_config(config, missing_keys, invalid_keys);
    auto &typed_despike = typed_timestream_config.raw_time_chunk.despike;
    typed_despike.enabled = rtcproc.run_despike;
    typed_despike.source_protection.enabled =
        rtcproc.despike_source_protection_config_enabled;
    typed_despike.source_protection.radius_arcsec =
        rtcproc.despiker.source_protection_radius_arcsec;
    if (rtcproc.run_despike) {
        typed_despike.min_spike_sigma = rtcproc.despiker.min_spike_sigma;
        typed_despike.time_constant_sec = rtcproc.despiker.time_constant_sec;
        typed_despike.window_size = rtcproc.despiker.window_size;
        typed_despike.legacy_enabled = rtcproc.despiker.run_legacy;

        const auto &local = rtcproc.despiker.local_residual;
        auto &typed_local = typed_despike.local_residual;
        typed_local.enabled = local.enabled;
        typed_local.window_sec = local.window_sec;
        typed_local.sigma_scale = local.sigma_scale;
        typed_local.delta_sigma_scale = local.delta_sigma_scale;
        typed_local.expand_with_filter = local.expand_with_filter;
        typed_local.event_padding_sec = local.event_padding_sec;
        typed_local.high_score_event_override = local.high_score_event_override;
        typed_local.max_added_flagged_fraction = local.max_added_flagged_fraction;
        typed_local.compact_raw_gate.enabled = local.compact_raw_gate.enabled;
        typed_local.compact_raw_gate.candidate_rel_sigma_scale =
            local.compact_raw_gate.candidate_rel_sigma_scale;
        typed_local.compact_raw_gate.window_sec = local.compact_raw_gate.window_sec;
        typed_local.compact_raw_gate.half_peak_frac =
            local.compact_raw_gate.half_peak_frac;
        typed_local.compact_raw_gate.max_width_sec =
            local.compact_raw_gate.max_width_sec;
        typed_local.compact_raw_gate.max_step_shift_z =
            local.compact_raw_gate.max_step_shift_z;
        typed_local.compact_delta_gate.enabled = local.compact_delta_gate.enabled;
        typed_local.compact_delta_gate.window_sec =
            local.compact_delta_gate.window_sec;
        typed_local.compact_delta_gate.half_peak_frac =
            local.compact_delta_gate.half_peak_frac;
        typed_local.compact_delta_gate.max_width_sec =
            local.compact_delta_gate.max_width_sec;
        typed_local.compact_delta_gate.max_step_shift_z =
            local.compact_delta_gate.max_step_shift_z;
    }

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

template<typename CT>
void Engine::get_ptc_config(CT &config) {
    logger->info("getting ptc config options");
    // get ptcproc config
    ptcproc.get_config(config, missing_keys, invalid_keys);
    auto &typed_fruit_loops = typed_timestream_config.fruit_loops;
    typed_fruit_loops.enabled = ptcproc.run_fruit_loops;
    if (ptcproc.run_fruit_loops) {
        typed_fruit_loops.save_all_iters = ptcproc.save_all_iters;
        typed_fruit_loops.path = ptcproc.fruit_loops_path;
        typed_fruit_loops.type = ptcproc.fruit_loops_type;
        if (auto parsed = citlali::config::parse_fruit_loops_mode(
                ptcproc.fruit_mode)) {
            typed_fruit_loops.mode = *parsed;
        }
        typed_fruit_loops.sig2noise_limit = ptcproc.fruit_loops_sig2noise;
        typed_fruit_loops.array_flux_limit.clear();
        typed_fruit_loops.array_flux_limit.reserve(
            static_cast<std::size_t>(ptcproc.fruit_loops_flux.size()));
        for (Eigen::Index i = 0; i < ptcproc.fruit_loops_flux.size(); ++i) {
            typed_fruit_loops.array_flux_limit.push_back(
                ptcproc.fruit_loops_flux(i));
        }
        typed_fruit_loops.peak_fraction_limit =
            ptcproc.fruit_loops_peak_fraction_limit;
        typed_fruit_loops.local_snr_floor =
            ptcproc.fruit_loops_local_snr_floor;
        typed_fruit_loops.local_sigma_inner_radius_arcsec =
            ptcproc.fruit_loops_local_sigma_inner_radius_arcsec;
        typed_fruit_loops.local_sigma_outer_radius_arcsec =
            ptcproc.fruit_loops_local_sigma_outer_radius_arcsec;
        typed_fruit_loops.local_sigma_inner_fwhm =
            ptcproc.fruit_loops_local_sigma_inner_fwhm;
        typed_fruit_loops.local_sigma_outer_fwhm =
            ptcproc.fruit_loops_local_sigma_outer_fwhm;
        typed_fruit_loops.local_sigma_edge_guard_arcsec =
            ptcproc.fruit_loops_local_sigma_edge_guard_arcsec;
        typed_fruit_loops.local_sigma_min_pixels =
            ptcproc.fruit_loops_local_sigma_min_pixels;
        typed_fruit_loops.adaptive_support_radius_arcsec =
            ptcproc.fruit_loops_adaptive_support_radius_arcsec;
        typed_fruit_loops.adaptive_support_radius_fwhm =
            ptcproc.fruit_loops_adaptive_support_radius_fwhm;
        typed_fruit_loops.weight_feedback.enabled =
            ptcproc.fruit_loops_weight_feedback_enabled;
        if (auto parsed =
                citlali::config::parse_fruit_loops_weight_feedback_reference(
                    ptcproc.fruit_loops_weight_feedback_reference)) {
            typed_fruit_loops.weight_feedback.reference = *parsed;
        }
        typed_fruit_loops.weight_feedback.low_relative_weight =
            ptcproc.fruit_loops_weight_feedback_low_relative_weight;
        typed_fruit_loops.weight_feedback.high_relative_weight =
            ptcproc.fruit_loops_weight_feedback_high_relative_weight;
        typed_fruit_loops.center_keep_radius_arcsec =
            ptcproc.fruit_loops_center_keep_radius_arcsec;
        if (auto parsed =
                citlali::config::parse_fruit_loops_interp_mode_override(
                    ptcproc.fruit_loops_interp_mode_override)) {
            typed_fruit_loops.interp_mode_override = *parsed;
        }
        typed_fruit_loops.legacy_center = ptcproc.fruit_loops_legacy_center;
        typed_fruit_loops.recompute_weights_after_addback =
            ptcproc.fruit_loops_recompute_weights_after_addback;
        typed_fruit_loops.max_iters = ptcproc.fruit_loops_iters;
    }
    auto &typed_clean = typed_timestream_config.processed_time_chunk.clean;
    typed_clean.enabled = ptcproc.run_clean;
    if (ptcproc.run_clean) {
        if (auto parsed = citlali::config::parse_processed_cleaner_mode(
                ptcproc.cleaner.active_cleaner_label())) {
            typed_clean.active = *parsed;
        }
        typed_clean.grouping = ptcproc.cleaner.grouping;
        typed_clean.mask_radius_arcsec = ptcproc.mask_radius_arcsec;
        typed_clean.tau = ptcproc.cleaner.tau;
        typed_clean.standard_pca.enabled =
            ptcproc.cleaner.standard_pca.enabled;
        typed_clean.standard_pca.stddev_limit = ptcproc.cleaner.stddev_limit;
        typed_clean.standard_pca.n_calc = ptcproc.cleaner.n_calc;
        typed_clean.standard_pca.n_eig_to_cut.clear();
        for (const auto &[arr_index, arr_name] : toltec_io.array_name_map) {
            const auto it = ptcproc.cleaner.n_eig_to_cut.find(arr_index);
            if (it == ptcproc.cleaner.n_eig_to_cut.end()) {
                continue;
            }
            std::vector<int> n_eig_to_cut;
            n_eig_to_cut.reserve(static_cast<std::size_t>(it->second.size()));
            for (Eigen::Index i = 0; i < it->second.size(); ++i) {
                n_eig_to_cut.push_back(static_cast<int>(it->second(i)));
            }
            typed_clean.standard_pca.n_eig_to_cut[arr_name] =
                std::move(n_eig_to_cut);
        }
        auto &typed_corr_grouping = typed_clean.corr_grouping;
        typed_corr_grouping.enabled = ptcproc.cleaner.corr_grouping.enabled;
        if (auto parsed = citlali::config::parse_processed_corr_grouping_metric(
                ptcproc.cleaner.corr_grouping.metric)) {
            typed_corr_grouping.metric = *parsed;
        }
        typed_corr_grouping.corr_min = ptcproc.cleaner.corr_grouping.corr_min;
        typed_corr_grouping.min_overlap =
            ptcproc.cleaner.corr_grouping.min_overlap;
        typed_corr_grouping.min_good_frac =
            ptcproc.cleaner.corr_grouping.min_good_frac;
        typed_corr_grouping.min_group_size =
            ptcproc.cleaner.corr_grouping.min_group_size;
        typed_corr_grouping.max_samples =
            ptcproc.cleaner.corr_grouping.max_samples;
        typed_corr_grouping.clean_residual =
            ptcproc.cleaner.corr_grouping.clean_residual;

        auto &typed_null_model = typed_clean.null_model;
        typed_null_model.enabled = ptcproc.cleaner.null_model.enabled;
        typed_null_model.n_surrogates =
            ptcproc.cleaner.null_model.n_surrogates;
        typed_null_model.quantile = ptcproc.cleaner.null_model.quantile;
        typed_null_model.min_good_frac =
            ptcproc.cleaner.null_model.min_good_frac;
        typed_null_model.max_modes = ptcproc.cleaner.null_model.max_modes;
        typed_null_model.max_samples = ptcproc.cleaner.null_model.max_samples;
        typed_null_model.seed = static_cast<int>(ptcproc.cleaner.null_model.seed);
        typed_null_model.grouping = ptcproc.cleaner.null_model.grouping;

        auto &typed_mp = typed_clean.marchenko_pastur;
        typed_mp.enabled = ptcproc.cleaner.marchenko_pastur.enabled;
        typed_mp.min_good_frac =
            ptcproc.cleaner.marchenko_pastur.min_good_frac;
        typed_mp.max_modes = ptcproc.cleaner.marchenko_pastur.max_modes;
        typed_mp.max_samples = ptcproc.cleaner.marchenko_pastur.max_samples;
        typed_mp.band_low_Hz = ptcproc.cleaner.marchenko_pastur.band_low_Hz;
        typed_mp.band_high_Hz = ptcproc.cleaner.marchenko_pastur.band_high_Hz;
        typed_mp.clip_z = ptcproc.cleaner.marchenko_pastur.clip_z;
        typed_mp.bulk_keep_frac =
            ptcproc.cleaner.marchenko_pastur.bulk_keep_frac;
        typed_mp.q_grid_size = ptcproc.cleaner.marchenko_pastur.q_grid_size;
        typed_mp.grouping = ptcproc.cleaner.marchenko_pastur.grouping;

        auto &typed_adaptive = typed_clean.adaptive_selector;
        typed_adaptive.enabled = ptcproc.cleaner.adaptive_selector.enabled;
        typed_adaptive.min_good_frac =
            ptcproc.cleaner.adaptive_selector.min_good_frac;
        typed_adaptive.max_det = ptcproc.cleaner.adaptive_selector.max_det;
        typed_adaptive.max_samples =
            ptcproc.cleaner.adaptive_selector.max_samples;
        typed_adaptive.max_pairs = ptcproc.cleaner.adaptive_selector.max_pairs;
        typed_adaptive.seed =
            static_cast<int>(ptcproc.cleaner.adaptive_selector.seed);
        typed_adaptive.clip_z = ptcproc.cleaner.adaptive_selector.clip_z;
        typed_adaptive.low_weight =
            ptcproc.cleaner.adaptive_selector.low_weight;
        typed_adaptive.tail_weight =
            ptcproc.cleaner.adaptive_selector.tail_weight;
        typed_adaptive.topmode_weight =
            ptcproc.cleaner.adaptive_selector.topmode_weight;
        typed_adaptive.reg_weight =
            ptcproc.cleaner.adaptive_selector.reg_weight;
        typed_adaptive.low_band_Hz =
            ptcproc.cleaner.adaptive_selector.low_band_Hz;
        typed_adaptive.mid_band_Hz =
            ptcproc.cleaner.adaptive_selector.mid_band_Hz;
        typed_adaptive.candidate_offsets =
            ptcproc.cleaner.adaptive_selector.candidate_offsets;
        typed_adaptive.grouping = ptcproc.cleaner.adaptive_selector.grouping;
        typed_adaptive.log_candidates =
            ptcproc.cleaner.adaptive_selector.log_candidates;
    }
    auto &typed_weighting =
        typed_timestream_config.processed_time_chunk.weighting;
    if (auto parsed =
            citlali::config::parse_processed_weighting_type(ptcproc.weighting_type)) {
        typed_weighting.type = *parsed;
    }
    typed_weighting.source_mask_radius_arcsec =
        ptcproc.source_mask_radius_arcsec;
    typed_weighting.hybrid_correction_min_factor =
        ptcproc.hybrid_correction_min_factor;
    typed_weighting.hybrid_correction_max_factor =
        ptcproc.hybrid_correction_max_factor;
    typed_weighting.median_map_weight_factor = ptcproc.med_weight_factor;
    typed_weighting.lower_map_weight_factor = ptcproc.lower_weight_factor;
    typed_weighting.upper_map_weight_factor = ptcproc.upper_weight_factor;
    auto &typed_flagging =
        typed_timestream_config.processed_time_chunk.flagging;
    typed_flagging.lower_tod_inv_var_factor = ptcproc.lower_inv_var_factor;
    typed_flagging.upper_tod_inv_var_factor = ptcproc.upper_inv_var_factor;
    auto &typed_busy_row = typed_weighting.busy_row_suppression;
    typed_busy_row.enabled = ptcproc.busy_row_suppression.enabled;
    typed_busy_row.require_busy_veto =
        ptcproc.busy_row_suppression.require_busy_veto;
    typed_busy_row.min_candidate_clusters =
        ptcproc.busy_row_suppression.min_candidate_clusters;
    typed_busy_row.min_max_unflagged_residual_z =
        ptcproc.busy_row_suppression.min_max_unflagged_residual_z;
    typed_busy_row.factor = ptcproc.busy_row_suppression.factor;
    const auto &weight_validation = ptcproc.weight_validation;
    auto &typed_weight_validation = typed_weighting.validation;
    typed_weight_validation.enabled = weight_validation.enabled;
    typed_weight_validation.accumulation_iters =
        weight_validation.accumulation_iters;
    typed_weight_validation.apply_start_iter =
        weight_validation.apply_start_iter;
    typed_weight_validation.min_valid_scans =
        weight_validation.min_valid_scans;
    typed_weight_validation.min_factor = weight_validation.min_factor;
    typed_weight_validation.unvalidated_factor =
        weight_validation.unvalidated_factor;
    typed_weight_validation.require_fruitloops_model =
        weight_validation.require_fruitloops_model;
    typed_weight_validation.transient_ratio_enabled =
        weight_validation.transient_ratio_enabled;
    typed_weight_validation.ratio_power = weight_validation.ratio_power;
    typed_weight_validation.transient_ratio_power =
        weight_validation.transient_ratio_power;
    typed_weight_validation.upward_enabled = weight_validation.upward_enabled;
    typed_weight_validation.upward_max_factor =
        weight_validation.upward_max_factor;
    typed_weight_validation.upward_power = weight_validation.upward_power;
    typed_weight_validation.upward_min_base_factor =
        weight_validation.upward_min_base_factor;
    typed_weight_validation.upward_require_atmospheric =
        weight_validation.upward_require_atmospheric;
    typed_weight_validation.upward_min_atmospheric_factor =
        weight_validation.upward_min_atmospheric_factor;
    typed_weight_validation.atmospheric_correlation_enabled =
        weight_validation.atmospheric_correlation_enabled;
    if (auto parsed = citlali::config::parse_processed_weight_grouping(
            weight_validation.atmospheric_grouping)) {
        typed_weight_validation.atmospheric_grouping = *parsed;
    }
    typed_weight_validation.atmospheric_min_detectors =
        weight_validation.atmospheric_min_detectors;
    typed_weight_validation.atmospheric_ref = weight_validation.atmospheric_ref;
    typed_weight_validation.atmospheric_span =
        weight_validation.atmospheric_span;
    typed_weight_validation.atmospheric_power =
        weight_validation.atmospheric_power;
    typed_weight_validation.min_good_frac = weight_validation.min_good_frac;
    typed_weight_validation.min_overlap = weight_validation.min_overlap;
    typed_weight_validation.max_samples = weight_validation.max_samples;
    typed_weight_validation.high_weight_validation_enabled =
        weight_validation.high_weight_validation_enabled;
    typed_weight_validation.high_weight_apply_caps =
        weight_validation.high_weight_apply_caps;
    if (auto parsed = citlali::config::parse_processed_weight_grouping(
            weight_validation.high_weight_grouping)) {
        typed_weight_validation.high_weight_grouping = *parsed;
    }
    typed_weight_validation.high_weight_min_group_detectors =
        weight_validation.high_weight_min_group_detectors;
    typed_weight_validation.high_weight_log_robust_z =
        weight_validation.high_weight_log_robust_z;
    typed_weight_validation.high_weight_max_median_factor =
        weight_validation.high_weight_max_median_factor;
    typed_weight_validation.high_weight_cap_median_factor =
        weight_validation.high_weight_cap_median_factor;
    typed_weight_validation.high_weight_min_validated_factor =
        weight_validation.high_weight_min_validated_factor;

    const auto &weight_corr_penalty = ptcproc.weight_corr_penalty;
    auto &typed_corr_penalty = typed_weighting.corr_penalty;
    typed_corr_penalty.enabled = weight_corr_penalty.enabled;
    typed_corr_penalty.min_good_frac = weight_corr_penalty.min_good_frac;
    typed_corr_penalty.min_overlap = weight_corr_penalty.min_overlap;
    typed_corr_penalty.max_samples = weight_corr_penalty.max_samples;
    typed_corr_penalty.max_pairs = weight_corr_penalty.max_pairs;
    typed_corr_penalty.seed = static_cast<int>(weight_corr_penalty.seed);
    typed_corr_penalty.floor = weight_corr_penalty.floor;
    typed_corr_penalty.exponent = weight_corr_penalty.exponent;
    typed_corr_penalty.pair_corr.enabled =
        weight_corr_penalty.pair_corr.enabled;
    typed_corr_penalty.pair_corr.ref = weight_corr_penalty.pair_corr.ref;
    typed_corr_penalty.pair_corr.span = weight_corr_penalty.pair_corr.span;
    typed_corr_penalty.pair_corr.weight = weight_corr_penalty.pair_corr.weight;
    typed_corr_penalty.cm_el_corr.enabled =
        weight_corr_penalty.cm_el_corr.enabled;
    typed_corr_penalty.cm_el_corr.ref = weight_corr_penalty.cm_el_corr.ref;
    typed_corr_penalty.cm_el_corr.span = weight_corr_penalty.cm_el_corr.span;
    typed_corr_penalty.cm_el_corr.weight =
        weight_corr_penalty.cm_el_corr.weight;
    typed_corr_penalty.cm_low_mid_ratio.enabled =
        weight_corr_penalty.cm_low_mid_ratio.enabled;
    typed_corr_penalty.cm_low_mid_ratio.ref =
        weight_corr_penalty.cm_low_mid_ratio.ref;
    typed_corr_penalty.cm_low_mid_ratio.span =
        weight_corr_penalty.cm_low_mid_ratio.span;
    typed_corr_penalty.cm_low_mid_ratio.weight =
        weight_corr_penalty.cm_low_mid_ratio.weight;
    typed_corr_penalty.cm_low_mid_ratio.low_band_Hz = {
        weight_corr_penalty.cm_low_mid_ratio.low_min_Hz,
        weight_corr_penalty.cm_low_mid_ratio.low_max_Hz};
    typed_corr_penalty.cm_low_mid_ratio.mid_band_Hz = {
        weight_corr_penalty.cm_low_mid_ratio.mid_min_Hz,
        weight_corr_penalty.cm_low_mid_ratio.mid_max_Hz};

    auto &typed_second_pass =
        typed_timestream_config.processed_time_chunk.flagging.second_pass_local;
    typed_second_pass.enabled = ptcproc.second_pass_local.enabled;
    typed_second_pass.min_spike_sigma =
        ptcproc.second_pass_local.min_spike_sigma;
    typed_second_pass.min_good_frac = ptcproc.second_pass_local.min_good_frac;
    typed_second_pass.baseline_window_sec =
        ptcproc.second_pass_local.baseline_window_sec;
    typed_second_pass.sigma_scale = ptcproc.second_pass_local.sigma_scale;
    typed_second_pass.delta_sigma_scale =
        ptcproc.second_pass_local.delta_sigma_scale;
    typed_second_pass.raw_candidate_rel_sigma_scale =
        ptcproc.second_pass_local.raw_candidate_rel_sigma_scale;
    typed_second_pass.raw_window_sec =
        ptcproc.second_pass_local.raw_window_sec;
    typed_second_pass.raw_half_peak_frac =
        ptcproc.second_pass_local.raw_half_peak_frac;
    typed_second_pass.raw_max_width_sec =
        ptcproc.second_pass_local.raw_max_width_sec;
    typed_second_pass.delta_window_sec =
        ptcproc.second_pass_local.delta_window_sec;
    typed_second_pass.delta_half_peak_frac =
        ptcproc.second_pass_local.delta_half_peak_frac;
    typed_second_pass.delta_max_width_sec =
        ptcproc.second_pass_local.delta_max_width_sec;
    typed_second_pass.max_step_shift_z =
        ptcproc.second_pass_local.max_step_shift_z;
    typed_second_pass.high_score_event_override =
        ptcproc.second_pass_local.high_score_event_override;
    typed_second_pass.merge_within_detector_sec =
        ptcproc.second_pass_local.merge_within_detector_sec;
    typed_second_pass.cluster_events_sec =
        ptcproc.second_pass_local.cluster_events_sec;
    typed_second_pass.min_cluster_detectors =
        ptcproc.second_pass_local.min_cluster_detectors;
    typed_second_pass.high_score_cluster_override =
        ptcproc.second_pass_local.high_score_cluster_override;
    typed_second_pass.max_auto_flag_clusters_per_network =
        ptcproc.second_pass_local.max_auto_flag_clusters_per_network;
    typed_second_pass.selective_busy_network_acceptance_enabled =
        ptcproc.second_pass_local.selective_busy_network_acceptance_enabled;
    typed_second_pass.source_protection.enabled =
        ptcproc.second_pass_local.source_protection_config_enabled;
    typed_second_pass.source_protection.radius_arcsec =
        ptcproc.second_pass_local.source_protection_radius_arcsec;

    // copy tod output bool for eigenvalues
    ptcproc.run_tod_output = run_tod_output;
    ptcproc.write_evals = diagnostics.write_evals;
}

inline double Engine::processed_time_chunk_fs_hz() const {
    double fs_hz = telescope.fsmp;
    if (rtcproc.run_downsample && rtcproc.downsampler.factor > 1) {
        fs_hz /= static_cast<double>(rtcproc.downsampler.factor);
    }
    return fs_hz;
}

template <class calib_t>
Eigen::Index Engine::apply_model_protected_ptc_line_audit(
    TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
    calib_t &calib_for_scan,
    bool model_subtracted) {

    const auto &base_audit = rtcproc.line_audit;
    if (!base_audit.enabled || !base_audit.ptc_model_protected_enabled) {
        return 0;
    }
    if (base_audit.ptc_require_model_subtracted && !model_subtracted) {
        logger->debug(
            "skipping model-protected PTC line-audit notch pass for scan {} because no model was subtracted",
            ptcdata.index.data + 1);
        return 0;
    }
    if (!base_audit.ptc_apply_fixed_notches &&
        !base_audit.ptc_apply_shared_notches &&
        !base_audit.ptc_apply_detector_notches) {
        return 0;
    }

    auto audit = base_audit;
    audit.pre_filter_enabled = false;
    audit.post_filter_enabled = false;
    audit.apply_shared_notches = audit.ptc_apply_shared_notches;
    audit.post_filter_apply_detector_notches = audit.ptc_apply_detector_notches;
    audit.fixed_notch_enabled = audit.fixed_notch_enabled && audit.ptc_apply_fixed_notches;
    if (std::isfinite(base_audit.ptc_line_min_hz)) {
        audit.line_min_hz = base_audit.ptc_line_min_hz;
    }
    if (std::isfinite(base_audit.ptc_line_max_hz)) {
        audit.line_max_hz = base_audit.ptc_line_max_hz;
    }

    const double fs_hz = processed_time_chunk_fs_hz();
    if (!std::isfinite(fs_hz) || fs_hz <= 0.0) {
        logger->warn("skipping model-protected PTC line-audit notch pass; invalid fs_hz={}", fs_hz);
        return 0;
    }

    Eigen::Index total_notches = 0;
    Eigen::Index max_notches_per_timestream = 0;

    if (audit.fixed_notch_enabled) {
        const Eigen::Index n_fixed_sections =
            rtcproc.count_rtc_line_audit_fixed_notches(fs_hz, audit);
        const auto n_fixed =
            rtcproc.apply_rtc_line_audit_fixed_notches(ptcdata, fs_hz, audit);
        total_notches += n_fixed;
        if (n_fixed > 0) {
            max_notches_per_timestream += n_fixed_sections;
        }
    }

    if (audit.apply_shared_notches) {
        const Eigen::Index n_iters = std::max<Eigen::Index>(1, audit.ptc_apply_iterations);
        for (Eigen::Index iter = 0; iter < n_iters; ++iter) {
            rtcproc.capture_rtc_line_audit(
                ptcdata, calib_for_scan, 0, ptcdata.scans.data.rows(), audit, true);
            const auto n_shared =
                rtcproc.apply_rtc_line_audit_shared_notches(ptcdata, fs_hz, audit, true);
            total_notches += n_shared;
            if (n_shared > 0) {
                max_notches_per_timestream += n_shared;
            }
            if (n_shared <= 0) {
                break;
            }
        }
    }

    if (audit.post_filter_apply_detector_notches) {
        const auto n_detector =
            rtcproc.apply_rtc_line_audit_detector_notches(
                ptcdata, fs_hz, audit, 0, ptcdata.scans.data.rows());
        total_notches += n_detector;
        if (n_detector > 0) {
            if (audit.detector_notch_max_notches > 0) {
                max_notches_per_timestream +=
                    std::min<Eigen::Index>(audit.detector_notch_max_notches, n_detector);
            }
            else {
                max_notches_per_timestream += n_detector;
            }
        }
    }

    if (total_notches > 0) {
        ptcdata.status.tod_filtered = true;
        if (rtcproc.filter_edge_guard.enabled &&
            rtcproc.filter_edge_guard.apply_dynamic_notch &&
            max_notches_per_timestream > 0) {
            const double min_width_hz =
                std::min(audit.apply_min_width_hz, audit.detector_notch_min_width_hz);
            Eigen::Index guard_samples =
                max_notches_per_timestream *
                timestream::Filter::notch_settle_samples_for_width(
                    fs_hz, min_width_hz, rtcproc.filter_edge_guard.iir_settle_attenuation);
            guard_samples = std::max(guard_samples, rtcproc.filter_edge_guard.min_samples);
            guard_samples += rtcproc.filter_edge_guard.extra_samples;
            if (rtcproc.filter_edge_guard.max_samples > 0) {
                guard_samples = std::min(guard_samples, rtcproc.filter_edge_guard.max_samples);
            }
            guard_samples = std::max<Eigen::Index>(0, guard_samples);
            if (guard_samples > 0) {
                rtcproc.apply_filter_edge_guard(ptcdata, 0, ptcdata.scans.data.rows(), guard_samples);
            }
        }
        logger->info(
            "model-protected PTC line-audit notch pass scan {}: total_notches={} fs_hz={} model_subtracted={} fixed={} shared={} detector={}",
            ptcdata.index.data + 1,
            total_notches,
            fs_hz,
            model_subtracted,
            audit.fixed_notch_enabled,
            audit.apply_shared_notches,
            audit.post_filter_apply_detector_notches);
    }

    return total_notches;
}
