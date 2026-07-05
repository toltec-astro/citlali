#pragma once

// Included by timestream_config_mirror.h inside namespace citlali::pipeline.

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

