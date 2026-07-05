#pragma once

#include <citlali/core/config/timestream_config.h>

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

template <class FruitLoopsConfig, class PtcProc>
void mirror_fruit_loops_config(FruitLoopsConfig &target,
                               const PtcProc &ptcproc) {
    target.enabled = ptcproc.run_fruit_loops;
    if (!ptcproc.run_fruit_loops) {
        return;
    }

    target.save_all_iters = ptcproc.save_all_iters;
    target.path = ptcproc.fruit_loops_path;
    target.type = ptcproc.fruit_loops_type;
    if (auto parsed =
            citlali::config::parse_fruit_loops_mode(ptcproc.fruit_mode)) {
        target.mode = *parsed;
    }
    target.sig2noise_limit = ptcproc.fruit_loops_sig2noise;
    target.array_flux_limit.clear();
    target.array_flux_limit.reserve(
        static_cast<std::size_t>(ptcproc.fruit_loops_flux.size()));
    for (Eigen::Index i = 0; i < ptcproc.fruit_loops_flux.size(); ++i) {
        target.array_flux_limit.push_back(ptcproc.fruit_loops_flux(i));
    }
    target.peak_fraction_limit = ptcproc.fruit_loops_peak_fraction_limit;
    target.local_snr_floor = ptcproc.fruit_loops_local_snr_floor;
    target.local_sigma_inner_radius_arcsec =
        ptcproc.fruit_loops_local_sigma_inner_radius_arcsec;
    target.local_sigma_outer_radius_arcsec =
        ptcproc.fruit_loops_local_sigma_outer_radius_arcsec;
    target.local_sigma_inner_fwhm = ptcproc.fruit_loops_local_sigma_inner_fwhm;
    target.local_sigma_outer_fwhm = ptcproc.fruit_loops_local_sigma_outer_fwhm;
    target.local_sigma_edge_guard_arcsec =
        ptcproc.fruit_loops_local_sigma_edge_guard_arcsec;
    target.local_sigma_min_pixels = ptcproc.fruit_loops_local_sigma_min_pixels;
    target.adaptive_support_radius_arcsec =
        ptcproc.fruit_loops_adaptive_support_radius_arcsec;
    target.adaptive_support_radius_fwhm =
        ptcproc.fruit_loops_adaptive_support_radius_fwhm;
    target.weight_feedback.enabled =
        ptcproc.fruit_loops_weight_feedback_enabled;
    if (auto parsed =
            citlali::config::parse_fruit_loops_weight_feedback_reference(
                ptcproc.fruit_loops_weight_feedback_reference)) {
        target.weight_feedback.reference = *parsed;
    }
    target.weight_feedback.low_relative_weight =
        ptcproc.fruit_loops_weight_feedback_low_relative_weight;
    target.weight_feedback.high_relative_weight =
        ptcproc.fruit_loops_weight_feedback_high_relative_weight;
    target.center_keep_radius_arcsec = ptcproc.fruit_loops_center_keep_radius_arcsec;
    if (auto parsed =
            citlali::config::parse_fruit_loops_interp_mode_override(
                ptcproc.fruit_loops_interp_mode_override)) {
        target.interp_mode_override = *parsed;
    }
    target.legacy_center = ptcproc.fruit_loops_legacy_center;
    target.recompute_weights_after_addback =
        ptcproc.fruit_loops_recompute_weights_after_addback;
    target.max_iters = ptcproc.fruit_loops_iters;
}

template <class CleanConfig, class PtcProc, class ArrayNameMap>
void mirror_processed_clean_config(CleanConfig &target, const PtcProc &ptcproc,
                                   const ArrayNameMap &array_name_map) {
    target.enabled = ptcproc.run_clean;
    if (!ptcproc.run_clean) {
        return;
    }

    if (auto parsed = citlali::config::parse_processed_cleaner_mode(
            ptcproc.cleaner.active_cleaner_label())) {
        target.active = *parsed;
    }
    target.grouping = ptcproc.cleaner.grouping;
    target.mask_radius_arcsec = ptcproc.mask_radius_arcsec;
    target.tau = ptcproc.cleaner.tau;
    target.standard_pca.enabled = ptcproc.cleaner.standard_pca.enabled;
    target.standard_pca.stddev_limit = ptcproc.cleaner.stddev_limit;
    target.standard_pca.n_calc = ptcproc.cleaner.n_calc;
    target.standard_pca.n_eig_to_cut.clear();
    for (const auto &[arr_index, arr_name] : array_name_map) {
        const auto it = ptcproc.cleaner.n_eig_to_cut.find(arr_index);
        if (it == ptcproc.cleaner.n_eig_to_cut.end()) {
            continue;
        }
        std::vector<int> n_eig_to_cut;
        n_eig_to_cut.reserve(static_cast<std::size_t>(it->second.size()));
        for (Eigen::Index i = 0; i < it->second.size(); ++i) {
            n_eig_to_cut.push_back(static_cast<int>(it->second(i)));
        }
        target.standard_pca.n_eig_to_cut[arr_name] =
            std::move(n_eig_to_cut);
    }

    auto &typed_corr_grouping = target.corr_grouping;
    typed_corr_grouping.enabled = ptcproc.cleaner.corr_grouping.enabled;
    if (auto parsed = citlali::config::parse_processed_corr_grouping_metric(
            ptcproc.cleaner.corr_grouping.metric)) {
        typed_corr_grouping.metric = *parsed;
    }
    typed_corr_grouping.corr_min = ptcproc.cleaner.corr_grouping.corr_min;
    typed_corr_grouping.min_overlap = ptcproc.cleaner.corr_grouping.min_overlap;
    typed_corr_grouping.min_good_frac =
        ptcproc.cleaner.corr_grouping.min_good_frac;
    typed_corr_grouping.min_group_size =
        ptcproc.cleaner.corr_grouping.min_group_size;
    typed_corr_grouping.max_samples = ptcproc.cleaner.corr_grouping.max_samples;
    typed_corr_grouping.clean_residual =
        ptcproc.cleaner.corr_grouping.clean_residual;

    auto &typed_null_model = target.null_model;
    typed_null_model.enabled = ptcproc.cleaner.null_model.enabled;
    typed_null_model.n_surrogates = ptcproc.cleaner.null_model.n_surrogates;
    typed_null_model.quantile = ptcproc.cleaner.null_model.quantile;
    typed_null_model.min_good_frac = ptcproc.cleaner.null_model.min_good_frac;
    typed_null_model.max_modes = ptcproc.cleaner.null_model.max_modes;
    typed_null_model.max_samples = ptcproc.cleaner.null_model.max_samples;
    typed_null_model.seed = static_cast<int>(ptcproc.cleaner.null_model.seed);
    typed_null_model.grouping = ptcproc.cleaner.null_model.grouping;

    auto &typed_mp = target.marchenko_pastur;
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

    auto &typed_adaptive = target.adaptive_selector;
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
    typed_adaptive.low_weight = ptcproc.cleaner.adaptive_selector.low_weight;
    typed_adaptive.tail_weight = ptcproc.cleaner.adaptive_selector.tail_weight;
    typed_adaptive.topmode_weight =
        ptcproc.cleaner.adaptive_selector.topmode_weight;
    typed_adaptive.reg_weight = ptcproc.cleaner.adaptive_selector.reg_weight;
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

template <class WeightingConfig, class FlaggingConfig, class PtcProc>
void mirror_processed_weighting_config(WeightingConfig &weighting,
                                       FlaggingConfig &flagging,
                                       const PtcProc &ptcproc) {
    if (auto parsed =
            citlali::config::parse_processed_weighting_type(
                ptcproc.weighting_type)) {
        weighting.type = *parsed;
    }
    weighting.source_mask_radius_arcsec = ptcproc.source_mask_radius_arcsec;
    weighting.hybrid_correction_min_factor =
        ptcproc.hybrid_correction_min_factor;
    weighting.hybrid_correction_max_factor =
        ptcproc.hybrid_correction_max_factor;
    weighting.median_map_weight_factor = ptcproc.med_weight_factor;
    weighting.lower_map_weight_factor = ptcproc.lower_weight_factor;
    weighting.upper_map_weight_factor = ptcproc.upper_weight_factor;
    flagging.lower_tod_inv_var_factor = ptcproc.lower_inv_var_factor;
    flagging.upper_tod_inv_var_factor = ptcproc.upper_inv_var_factor;

    auto &busy_row = weighting.busy_row_suppression;
    busy_row.enabled = ptcproc.busy_row_suppression.enabled;
    busy_row.require_busy_veto =
        ptcproc.busy_row_suppression.require_busy_veto;
    busy_row.min_candidate_clusters =
        ptcproc.busy_row_suppression.min_candidate_clusters;
    busy_row.min_max_unflagged_residual_z =
        ptcproc.busy_row_suppression.min_max_unflagged_residual_z;
    busy_row.factor = ptcproc.busy_row_suppression.factor;
}

template <class WeightValidationConfig, class WeightValidation>
void mirror_processed_weight_validation_config(
    WeightValidationConfig &target, const WeightValidation &source) {
    target.enabled = source.enabled;
    target.accumulation_iters = source.accumulation_iters;
    target.apply_start_iter = source.apply_start_iter;
    target.min_valid_scans = source.min_valid_scans;
    target.min_factor = source.min_factor;
    target.unvalidated_factor = source.unvalidated_factor;
    target.require_fruitloops_model = source.require_fruitloops_model;
    target.transient_ratio_enabled = source.transient_ratio_enabled;
    target.ratio_power = source.ratio_power;
    target.transient_ratio_power = source.transient_ratio_power;
    target.upward_enabled = source.upward_enabled;
    target.upward_max_factor = source.upward_max_factor;
    target.upward_power = source.upward_power;
    target.upward_min_base_factor = source.upward_min_base_factor;
    target.upward_require_atmospheric = source.upward_require_atmospheric;
    target.upward_min_atmospheric_factor =
        source.upward_min_atmospheric_factor;
    target.atmospheric_correlation_enabled =
        source.atmospheric_correlation_enabled;
    if (auto parsed =
            citlali::config::parse_processed_weight_grouping(
                source.atmospheric_grouping)) {
        target.atmospheric_grouping = *parsed;
    }
    target.atmospheric_min_detectors = source.atmospheric_min_detectors;
    target.atmospheric_ref = source.atmospheric_ref;
    target.atmospheric_span = source.atmospheric_span;
    target.atmospheric_power = source.atmospheric_power;
    target.min_good_frac = source.min_good_frac;
    target.min_overlap = source.min_overlap;
    target.max_samples = source.max_samples;
    target.high_weight_validation_enabled =
        source.high_weight_validation_enabled;
    target.high_weight_apply_caps = source.high_weight_apply_caps;
    if (auto parsed =
            citlali::config::parse_processed_weight_grouping(
                source.high_weight_grouping)) {
        target.high_weight_grouping = *parsed;
    }
    target.high_weight_min_group_detectors =
        source.high_weight_min_group_detectors;
    target.high_weight_log_robust_z = source.high_weight_log_robust_z;
    target.high_weight_max_median_factor =
        source.high_weight_max_median_factor;
    target.high_weight_cap_median_factor =
        source.high_weight_cap_median_factor;
    target.high_weight_min_validated_factor =
        source.high_weight_min_validated_factor;
}

template <class CorrPenaltyConfig, class CorrPenalty>
void mirror_processed_weight_corr_penalty_config(CorrPenaltyConfig &target,
                                                 const CorrPenalty &source) {
    target.enabled = source.enabled;
    target.min_good_frac = source.min_good_frac;
    target.min_overlap = source.min_overlap;
    target.max_samples = source.max_samples;
    target.max_pairs = source.max_pairs;
    target.seed = static_cast<int>(source.seed);
    target.floor = source.floor;
    target.exponent = source.exponent;
    target.pair_corr.enabled = source.pair_corr.enabled;
    target.pair_corr.ref = source.pair_corr.ref;
    target.pair_corr.span = source.pair_corr.span;
    target.pair_corr.weight = source.pair_corr.weight;
    target.cm_el_corr.enabled = source.cm_el_corr.enabled;
    target.cm_el_corr.ref = source.cm_el_corr.ref;
    target.cm_el_corr.span = source.cm_el_corr.span;
    target.cm_el_corr.weight = source.cm_el_corr.weight;
    target.cm_low_mid_ratio.enabled = source.cm_low_mid_ratio.enabled;
    target.cm_low_mid_ratio.ref = source.cm_low_mid_ratio.ref;
    target.cm_low_mid_ratio.span = source.cm_low_mid_ratio.span;
    target.cm_low_mid_ratio.weight = source.cm_low_mid_ratio.weight;
    target.cm_low_mid_ratio.low_band_Hz = {
        source.cm_low_mid_ratio.low_min_Hz,
        source.cm_low_mid_ratio.low_max_Hz};
    target.cm_low_mid_ratio.mid_band_Hz = {
        source.cm_low_mid_ratio.mid_min_Hz,
        source.cm_low_mid_ratio.mid_max_Hz};
}

template <class SecondPassConfig, class SecondPassLocal>
void mirror_second_pass_local_config(SecondPassConfig &target,
                                     const SecondPassLocal &source) {
    target.enabled = source.enabled;
    target.min_spike_sigma = source.min_spike_sigma;
    target.min_good_frac = source.min_good_frac;
    target.baseline_window_sec = source.baseline_window_sec;
    target.sigma_scale = source.sigma_scale;
    target.delta_sigma_scale = source.delta_sigma_scale;
    target.raw_candidate_rel_sigma_scale =
        source.raw_candidate_rel_sigma_scale;
    target.raw_window_sec = source.raw_window_sec;
    target.raw_half_peak_frac = source.raw_half_peak_frac;
    target.raw_max_width_sec = source.raw_max_width_sec;
    target.delta_window_sec = source.delta_window_sec;
    target.delta_half_peak_frac = source.delta_half_peak_frac;
    target.delta_max_width_sec = source.delta_max_width_sec;
    target.max_step_shift_z = source.max_step_shift_z;
    target.high_score_event_override = source.high_score_event_override;
    target.merge_within_detector_sec = source.merge_within_detector_sec;
    target.cluster_events_sec = source.cluster_events_sec;
    target.min_cluster_detectors = source.min_cluster_detectors;
    target.high_score_cluster_override = source.high_score_cluster_override;
    target.max_auto_flag_clusters_per_network =
        source.max_auto_flag_clusters_per_network;
    target.selective_busy_network_acceptance_enabled =
        source.selective_busy_network_acceptance_enabled;
    target.source_protection.enabled =
        source.source_protection_config_enabled;
    target.source_protection.radius_arcsec =
        source.source_protection_radius_arcsec;
}

template <class DespikeConfig, class RtcProc>
void mirror_raw_despike_config(DespikeConfig &target,
                               const RtcProc &rtcproc) {
    target.enabled = rtcproc.run_despike;
    target.source_protection.enabled =
        rtcproc.despike_source_protection_config_enabled;
    target.source_protection.radius_arcsec =
        rtcproc.despiker.source_protection_radius_arcsec;
    if (!rtcproc.run_despike) {
        return;
    }

    target.min_spike_sigma = rtcproc.despiker.min_spike_sigma;
    target.time_constant_sec = rtcproc.despiker.time_constant_sec;
    target.window_size = rtcproc.despiker.window_size;
    target.legacy_enabled = rtcproc.despiker.run_legacy;

    const auto &local = rtcproc.despiker.local_residual;
    auto &typed_local = target.local_residual;
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
    typed_local.compact_raw_gate.window_sec =
        local.compact_raw_gate.window_sec;
    typed_local.compact_raw_gate.half_peak_frac =
        local.compact_raw_gate.half_peak_frac;
    typed_local.compact_raw_gate.max_width_sec =
        local.compact_raw_gate.max_width_sec;
    typed_local.compact_raw_gate.max_step_shift_z =
        local.compact_raw_gate.max_step_shift_z;
    typed_local.compact_delta_gate.enabled =
        local.compact_delta_gate.enabled;
    typed_local.compact_delta_gate.window_sec =
        local.compact_delta_gate.window_sec;
    typed_local.compact_delta_gate.half_peak_frac =
        local.compact_delta_gate.half_peak_frac;
    typed_local.compact_delta_gate.max_width_sec =
        local.compact_delta_gate.max_width_sec;
    typed_local.compact_delta_gate.max_step_shift_z =
        local.compact_delta_gate.max_step_shift_z;
}

template <class RawFlaggingConfig, class RtcProc>
void mirror_raw_flagging_config(RawFlaggingConfig &target,
                                const RtcProc &rtcproc) {
    target.delta_f_min_Hz = rtcproc.delta_f_min_Hz;
    target.lower_tod_inv_var_factor = rtcproc.lower_inv_var_factor;
    target.upper_tod_inv_var_factor = rtcproc.upper_inv_var_factor;

    const auto &network_step_mask = rtcproc.network_step_mask;
    auto &typed_network_step = target.network_step_mask;
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
    auto &typed_capture = target.impulsive_capture;
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
    auto &typed_coincidence = target.impulsive_coincidence;
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
        static_cast<int>(impulsive_coincidence.high_score_min_networks_aligned);
    typed_coincidence.cluster_tol_sec = impulsive_coincidence.cluster_tol_sec;
    typed_coincidence.mask_pre_window_sec =
        impulsive_coincidence.mask_pre_window_sec;
    typed_coincidence.mask_post_window_sec =
        impulsive_coincidence.mask_post_window_sec;
    typed_coincidence.max_flagged_fraction =
        impulsive_coincidence.max_flagged_fraction;
}

}  // namespace citlali::pipeline
