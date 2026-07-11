#pragma once

#include <citlali/core/config/timestream_config.h>

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <utility>

namespace citlali::pipeline {

template <class PtcProc>
void apply_fruit_loops_config_to_processor(
    const citlali::config::TimestreamFruitLoopsConfig &config,
    PtcProc &ptcproc) {
    ptcproc.run_fruit_loops = config.enabled;
    ptcproc.fruit_loops_recompute_weights_after_addback =
        config.recompute_weights_after_addback;
    if (!config.enabled) {
        return;
    }

    ptcproc.save_all_iters = config.save_all_iters;
    ptcproc.fruit_loops_path = config.path;
    ptcproc.fruit_loops_type = config.type;
    ptcproc.fruit_mode =
        std::string{citlali::config::to_string(config.mode)};
    ptcproc.fruit_loops_sig2noise = config.sig2noise_limit;
    ptcproc.fruit_loops_flux.resize(
        static_cast<Eigen::Index>(config.array_flux_limit.size()));
    for (std::size_t i = 0; i < config.array_flux_limit.size(); ++i) {
        ptcproc.fruit_loops_flux(static_cast<Eigen::Index>(i)) =
            config.array_flux_limit[i];
    }
    ptcproc.fruit_loops_peak_fraction_limit = config.peak_fraction_limit;
    ptcproc.fruit_loops_local_snr_floor = config.local_snr_floor;
    ptcproc.fruit_loops_local_sigma_inner_radius_arcsec =
        config.local_sigma_inner_radius_arcsec;
    ptcproc.fruit_loops_local_sigma_outer_radius_arcsec =
        config.local_sigma_outer_radius_arcsec;
    ptcproc.fruit_loops_local_sigma_inner_fwhm =
        config.local_sigma_inner_fwhm;
    ptcproc.fruit_loops_local_sigma_outer_fwhm =
        config.local_sigma_outer_fwhm;
    ptcproc.fruit_loops_local_sigma_edge_guard_arcsec =
        config.local_sigma_edge_guard_arcsec;
    ptcproc.fruit_loops_local_sigma_min_pixels =
        config.local_sigma_min_pixels;
    ptcproc.fruit_loops_adaptive_support_radius_arcsec =
        config.adaptive_support_radius_arcsec;
    ptcproc.fruit_loops_adaptive_support_radius_fwhm =
        config.adaptive_support_radius_fwhm;
    ptcproc.fruit_loops_weight_feedback_enabled =
        config.weight_feedback.enabled;
    ptcproc.fruit_loops_weight_feedback_reference = std::string{
        citlali::config::to_string(config.weight_feedback.reference)};
    ptcproc.fruit_loops_weight_feedback_low_relative_weight =
        config.weight_feedback.low_relative_weight;
    ptcproc.fruit_loops_weight_feedback_high_relative_weight =
        config.weight_feedback.high_relative_weight;
    ptcproc.fruit_loops_center_keep_radius_arcsec =
        config.center_keep_radius_arcsec;
    ptcproc.fruit_loops_interp_mode_override = std::string{
        citlali::config::to_string(config.interp_mode_override)};
    ptcproc.fruit_loops_legacy_center = config.legacy_center;
    ptcproc.fruit_loops_iters = config.max_iters;
}

template <class PtcProc, class ArrayNameMap>
void apply_processed_clean_config_to_processor(
    const citlali::config::ProcessedTimeChunkCleanConfig &config,
    const ArrayNameMap &array_name_map, PtcProc &ptcproc) {
    ptcproc.run_clean = config.enabled;
    if (!config.enabled) {
        return;
    }

    auto &cleaner = ptcproc.cleaner;
    cleaner.grouping = config.grouping;
    ptcproc.mask_radius_arcsec = config.mask_radius_arcsec;
    cleaner.tau = config.tau;
    cleaner.standard_pca.enabled = config.standard_pca.enabled;
    cleaner.stddev_limit = config.standard_pca.stddev_limit;
    cleaner.n_calc = config.standard_pca.n_calc;
    cleaner.n_eig_to_cut.clear();
    for (const auto &[array_id, array_name] : array_name_map) {
        const auto it = config.standard_pca.n_eig_to_cut.find(array_name);
        if (it == config.standard_pca.n_eig_to_cut.end()) {
            continue;
        }
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> values(
            static_cast<Eigen::Index>(it->second.size()));
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            values(static_cast<Eigen::Index>(i)) = it->second[i];
        }
        cleaner.n_eig_to_cut[array_id] = std::move(values);
    }

    cleaner.corr_grouping.enabled = config.corr_grouping.enabled;
    cleaner.corr_grouping.metric = std::string{
        citlali::config::to_string(config.corr_grouping.metric)};
    cleaner.corr_grouping.corr_min = config.corr_grouping.corr_min;
    cleaner.corr_grouping.min_overlap = config.corr_grouping.min_overlap;
    cleaner.corr_grouping.min_good_frac = config.corr_grouping.min_good_frac;
    cleaner.corr_grouping.min_group_size = config.corr_grouping.min_group_size;
    cleaner.corr_grouping.max_samples = config.corr_grouping.max_samples;
    cleaner.corr_grouping.clean_residual = config.corr_grouping.clean_residual;

    cleaner.null_model.enabled = config.null_model.enabled;
    cleaner.null_model.n_surrogates = config.null_model.n_surrogates;
    cleaner.null_model.quantile = config.null_model.quantile;
    cleaner.null_model.min_good_frac = config.null_model.min_good_frac;
    cleaner.null_model.max_modes = config.null_model.max_modes;
    cleaner.null_model.max_samples = config.null_model.max_samples;
    cleaner.null_model.seed =
        static_cast<std::uint32_t>(config.null_model.seed);
    cleaner.null_model.grouping = config.null_model.grouping;

    cleaner.marchenko_pastur.enabled = config.marchenko_pastur.enabled;
    cleaner.marchenko_pastur.min_good_frac =
        config.marchenko_pastur.min_good_frac;
    cleaner.marchenko_pastur.max_modes = config.marchenko_pastur.max_modes;
    cleaner.marchenko_pastur.max_samples = config.marchenko_pastur.max_samples;
    cleaner.marchenko_pastur.band_low_Hz = config.marchenko_pastur.band_low_Hz;
    cleaner.marchenko_pastur.band_high_Hz =
        config.marchenko_pastur.band_high_Hz;
    cleaner.marchenko_pastur.clip_z = config.marchenko_pastur.clip_z;
    cleaner.marchenko_pastur.bulk_keep_frac =
        config.marchenko_pastur.bulk_keep_frac;
    cleaner.marchenko_pastur.q_grid_size = config.marchenko_pastur.q_grid_size;
    cleaner.marchenko_pastur.grouping = config.marchenko_pastur.grouping;

    cleaner.adaptive_selector.enabled = config.adaptive_selector.enabled;
    cleaner.adaptive_selector.min_good_frac =
        config.adaptive_selector.min_good_frac;
    cleaner.adaptive_selector.max_det = config.adaptive_selector.max_det;
    cleaner.adaptive_selector.max_samples = config.adaptive_selector.max_samples;
    cleaner.adaptive_selector.max_pairs = config.adaptive_selector.max_pairs;
    cleaner.adaptive_selector.seed =
        static_cast<std::uint32_t>(config.adaptive_selector.seed);
    cleaner.adaptive_selector.clip_z = config.adaptive_selector.clip_z;
    cleaner.adaptive_selector.low_weight = config.adaptive_selector.low_weight;
    cleaner.adaptive_selector.tail_weight = config.adaptive_selector.tail_weight;
    cleaner.adaptive_selector.topmode_weight =
        config.adaptive_selector.topmode_weight;
    cleaner.adaptive_selector.reg_weight = config.adaptive_selector.reg_weight;
    cleaner.adaptive_selector.low_band_Hz = config.adaptive_selector.low_band_Hz;
    cleaner.adaptive_selector.mid_band_Hz = config.adaptive_selector.mid_band_Hz;
    cleaner.adaptive_selector.candidate_offsets =
        config.adaptive_selector.candidate_offsets;
    cleaner.adaptive_selector.grouping = config.adaptive_selector.grouping;
    cleaner.adaptive_selector.log_candidates =
        config.adaptive_selector.log_candidates;
}

template <class PtcProc>
void apply_processed_weighting_config_to_processor(
    const citlali::config::ProcessedTimeChunkWeightingConfig &weighting,
    const citlali::config::ProcessedTimeChunkFlaggingConfig &flagging,
    PtcProc &ptcproc) {
    ptcproc.weighting_type =
        std::string{citlali::config::to_string(weighting.type)};
    ptcproc.source_mask_radius_arcsec = weighting.source_mask_radius_arcsec;
    ptcproc.hybrid_correction_min_factor =
        weighting.hybrid_correction_min_factor;
    ptcproc.hybrid_correction_max_factor =
        weighting.hybrid_correction_max_factor;
    ptcproc.med_weight_factor = weighting.median_map_weight_factor;
    ptcproc.lower_weight_factor = weighting.lower_map_weight_factor;
    ptcproc.upper_weight_factor = weighting.upper_map_weight_factor;
    ptcproc.lower_inv_var_factor = flagging.lower_tod_inv_var_factor;
    ptcproc.upper_inv_var_factor = flagging.upper_tod_inv_var_factor;

    const auto &validation = weighting.validation;
    auto &processor_validation = ptcproc.weight_validation;
    processor_validation.enabled = validation.enabled;
    processor_validation.accumulation_iters = validation.accumulation_iters;
    processor_validation.apply_start_iter = validation.apply_start_iter;
    processor_validation.min_valid_scans = validation.min_valid_scans;
    processor_validation.min_factor = validation.min_factor;
    processor_validation.unvalidated_factor = validation.unvalidated_factor;
    processor_validation.require_fruitloops_model =
        validation.require_fruitloops_model;
    processor_validation.transient_ratio_enabled =
        validation.transient_ratio_enabled;
    processor_validation.ratio_power = validation.ratio_power;
    processor_validation.transient_ratio_power =
        validation.transient_ratio_power;
    processor_validation.upward_enabled = validation.upward_enabled;
    processor_validation.upward_max_factor = validation.upward_max_factor;
    processor_validation.upward_power = validation.upward_power;
    processor_validation.upward_min_base_factor =
        validation.upward_min_base_factor;
    processor_validation.upward_require_atmospheric =
        validation.upward_require_atmospheric;
    processor_validation.upward_min_atmospheric_factor =
        validation.upward_min_atmospheric_factor;
    processor_validation.atmospheric_correlation_enabled =
        validation.atmospheric_correlation_enabled;
    processor_validation.atmospheric_grouping = std::string{
        citlali::config::to_string(validation.atmospheric_grouping)};
    processor_validation.atmospheric_min_detectors =
        validation.atmospheric_min_detectors;
    processor_validation.atmospheric_ref = validation.atmospheric_ref;
    processor_validation.atmospheric_span = validation.atmospheric_span;
    processor_validation.atmospheric_power = validation.atmospheric_power;
    processor_validation.min_good_frac = validation.min_good_frac;
    processor_validation.min_overlap = validation.min_overlap;
    processor_validation.max_samples = validation.max_samples;
    processor_validation.high_weight_validation_enabled =
        validation.high_weight_validation_enabled;
    processor_validation.high_weight_apply_caps =
        validation.high_weight_apply_caps;
    processor_validation.high_weight_grouping = std::string{
        citlali::config::to_string(validation.high_weight_grouping)};
    processor_validation.high_weight_min_group_detectors =
        validation.high_weight_min_group_detectors;
    processor_validation.high_weight_log_robust_z =
        validation.high_weight_log_robust_z;
    processor_validation.high_weight_max_median_factor =
        validation.high_weight_max_median_factor;
    processor_validation.high_weight_cap_median_factor =
        validation.high_weight_cap_median_factor;
    processor_validation.high_weight_min_validated_factor =
        validation.high_weight_min_validated_factor;

    const auto &penalty = weighting.corr_penalty;
    auto &processor_penalty = ptcproc.weight_corr_penalty;
    processor_penalty.enabled = penalty.enabled;
    processor_penalty.min_good_frac = penalty.min_good_frac;
    processor_penalty.min_overlap = penalty.min_overlap;
    processor_penalty.max_samples = penalty.max_samples;
    processor_penalty.max_pairs = penalty.max_pairs;
    processor_penalty.seed = static_cast<std::uint32_t>(penalty.seed);
    processor_penalty.floor = penalty.floor;
    processor_penalty.exponent = penalty.exponent;
    processor_penalty.pair_corr.enabled = penalty.pair_corr.enabled;
    processor_penalty.pair_corr.ref = penalty.pair_corr.ref;
    processor_penalty.pair_corr.span = penalty.pair_corr.span;
    processor_penalty.pair_corr.weight = penalty.pair_corr.weight;
    processor_penalty.cm_el_corr.enabled = penalty.cm_el_corr.enabled;
    processor_penalty.cm_el_corr.ref = penalty.cm_el_corr.ref;
    processor_penalty.cm_el_corr.span = penalty.cm_el_corr.span;
    processor_penalty.cm_el_corr.weight = penalty.cm_el_corr.weight;
    processor_penalty.cm_low_mid_ratio.enabled =
        penalty.cm_low_mid_ratio.enabled;
    processor_penalty.cm_low_mid_ratio.ref = penalty.cm_low_mid_ratio.ref;
    processor_penalty.cm_low_mid_ratio.span = penalty.cm_low_mid_ratio.span;
    processor_penalty.cm_low_mid_ratio.weight = penalty.cm_low_mid_ratio.weight;
    processor_penalty.cm_low_mid_ratio.low_min_Hz =
        penalty.cm_low_mid_ratio.low_band_Hz[0];
    processor_penalty.cm_low_mid_ratio.low_max_Hz =
        penalty.cm_low_mid_ratio.low_band_Hz[1];
    processor_penalty.cm_low_mid_ratio.mid_min_Hz =
        penalty.cm_low_mid_ratio.mid_band_Hz[0];
    processor_penalty.cm_low_mid_ratio.mid_max_Hz =
        penalty.cm_low_mid_ratio.mid_band_Hz[1];

    const auto &busy_row = weighting.busy_row_suppression;
    ptcproc.busy_row_suppression.enabled = busy_row.enabled;
    ptcproc.busy_row_suppression.require_busy_veto =
        busy_row.require_busy_veto;
    ptcproc.busy_row_suppression.min_candidate_clusters =
        busy_row.min_candidate_clusters;
    ptcproc.busy_row_suppression.min_max_unflagged_residual_z =
        busy_row.min_max_unflagged_residual_z;
    ptcproc.busy_row_suppression.factor = busy_row.factor;
}

template <class PtcProc>
void apply_second_pass_local_config_to_processor(
    const citlali::config::ProcessedTimeChunkSecondPassLocalConfig &config,
    PtcProc &ptcproc) {
    auto &target = ptcproc.second_pass_local;
    target.enabled = config.enabled;
    target.min_spike_sigma = config.min_spike_sigma;
    target.min_good_frac = config.min_good_frac;
    target.baseline_window_sec = config.baseline_window_sec;
    target.sigma_scale = config.sigma_scale;
    target.delta_sigma_scale = config.delta_sigma_scale;
    target.raw_candidate_rel_sigma_scale =
        config.raw_candidate_rel_sigma_scale;
    target.raw_window_sec = config.raw_window_sec;
    target.raw_half_peak_frac = config.raw_half_peak_frac;
    target.raw_max_width_sec = config.raw_max_width_sec;
    target.delta_window_sec = config.delta_window_sec;
    target.delta_half_peak_frac = config.delta_half_peak_frac;
    target.delta_max_width_sec = config.delta_max_width_sec;
    target.max_step_shift_z = config.max_step_shift_z;
    target.high_score_event_override = config.high_score_event_override;
    target.merge_within_detector_sec = config.merge_within_detector_sec;
    target.cluster_events_sec = config.cluster_events_sec;
    target.min_cluster_detectors = config.min_cluster_detectors;
    target.high_score_cluster_override = config.high_score_cluster_override;
    target.max_auto_flag_clusters_per_network =
        config.max_auto_flag_clusters_per_network;
    target.selective_busy_network_acceptance_enabled =
        config.selective_busy_network_acceptance_enabled;
    target.source_protection_config_enabled = config.source_protection.enabled;
    target.source_protection_radius_arcsec =
        config.source_protection.radius_arcsec;
}

}  // namespace citlali::pipeline
