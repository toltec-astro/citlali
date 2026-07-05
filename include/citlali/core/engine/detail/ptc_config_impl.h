#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

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

