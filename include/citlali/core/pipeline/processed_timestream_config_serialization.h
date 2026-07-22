#pragma once

#include <citlali/core/pipeline/processed_timestream_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <array>
#include <string>
#include <vector>

namespace citlali::pipeline {

template <class Value>
YAML::Node processed_config_sequence_node(const std::vector<Value> &values) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto &value : values) {
        node.push_back(value);
    }
    return node;
}

template <class Value, std::size_t Size>
YAML::Node processed_config_sequence_node(
    const std::array<Value, Size> &values) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto &value : values) {
        node.push_back(value);
    }
    return node;
}

inline YAML::Node timestream_source_protection_config_node(
    const citlali::config::TimestreamSourceProtectionConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["radius_arcsec"] = config.radius_arcsec;
    node["active"] = config.active;
    return node;
}

inline YAML::Node fruit_loops_weight_feedback_config_node(
    const citlali::config::FruitLoopsWeightFeedbackConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["reference"] =
        std::string{citlali::config::to_string(config.reference)};
    node["low_relative_weight"] = config.low_relative_weight;
    node["high_relative_weight"] = config.high_relative_weight;
    return node;
}

inline YAML::Node fruit_loops_config_node(
    const citlali::config::TimestreamFruitLoopsConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["save_all_iters"] = config.save_all_iters;
    node["path"] = config.path;
    node["type"] = config.type;
    node["mode"] = std::string{citlali::config::to_string(config.mode)};
    node["sig2noise_limit"] = config.sig2noise_limit;
    node["array_flux_limit"] =
        processed_config_sequence_node(config.array_flux_limit);
    node["peak_fraction_limit"] = config.peak_fraction_limit;
    node["local_snr_floor"] = config.local_snr_floor;
    node["local_sigma_inner_radius_arcsec"] =
        config.local_sigma_inner_radius_arcsec;
    node["local_sigma_outer_radius_arcsec"] =
        config.local_sigma_outer_radius_arcsec;
    node["local_sigma_inner_fwhm"] = config.local_sigma_inner_fwhm;
    node["local_sigma_outer_fwhm"] = config.local_sigma_outer_fwhm;
    node["local_sigma_edge_guard_arcsec"] =
        config.local_sigma_edge_guard_arcsec;
    node["local_sigma_min_pixels"] = config.local_sigma_min_pixels;
    node["adaptive_support_radius_arcsec"] =
        config.adaptive_support_radius_arcsec;
    node["adaptive_support_radius_fwhm"] =
        config.adaptive_support_radius_fwhm;
    node["source_center_mode"] =
        std::string{citlali::config::to_string(config.source_center_mode)};
    node["weight_feedback"] =
        fruit_loops_weight_feedback_config_node(config.weight_feedback);
    node["center_keep_radius_arcsec"] = config.center_keep_radius_arcsec;
    node["interp_mode_override"] = std::string{
        citlali::config::to_string(config.interp_mode_override)};
    node["legacy_center"] = config.legacy_center;
    node["recompute_weights_after_addback"] =
        config.recompute_weights_after_addback;
    node["max_iters"] = config.max_iters;
    return node;
}

inline YAML::Node learning_config_node(
    const citlali::config::TimestreamLearningConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["diagnostics_enabled"] = config.diagnostics_enabled;
    node["learn_iters"] = config.learn_iters;
    node["apply_start_iter"] = config.apply_start_iter;
    node["max_records_per_type"] = config.max_records_per_type;
    node["apply_sample_masks_enabled"] = config.apply_sample_masks_enabled;
    node["apply_max_new_flagged_fraction"] =
        config.apply_max_new_flagged_fraction;
    const auto &outlier = config.map_pixel_outlier;
    node["map_pixel_outlier_diagnostics_enabled"] =
        outlier.diagnostics_enabled;
    node["map_pixel_outlier_contributor_diagnostics_enabled"] =
        outlier.contributor_diagnostics_enabled;
    node["map_pixel_outlier_targeted_contributor_diagnostics_enabled"] =
        outlier.targeted_contributor_diagnostics_enabled;
    node["map_pixel_outlier_detector_exclusion_enabled"] =
        outlier.detector_exclusion_enabled;
    node["map_pixel_outlier_top_n"] = outlier.top_n;
    node["map_pixel_outlier_targeted_contributor_max_pixels"] =
        outlier.targeted_contributor_max_pixels;
    node["map_pixel_outlier_detector_exclusion_min_pixels"] =
        outlier.detector_exclusion_min_pixels;
    node["map_pixel_outlier_min_abs_z"] = outlier.min_abs_z;
    node["map_pixel_outlier_min_n_eff"] = outlier.min_n_eff;
    node["map_pixel_outlier_source_radius_arcsec"] =
        outlier.source_radius_arcsec;
    node["busy_detector_exclusion_enabled"] =
        config.busy_detector.exclusion_enabled;
    const auto &pathology = config.scan_network_pathology;
    node["scan_network_pathology_enabled"] = pathology.enabled;
    node["scan_network_pathology_apply_pre_rtc"] = pathology.apply_pre_rtc;
    node["scan_network_pathology_apply_pre_ptc"] = pathology.apply_pre_ptc;
    node["scan_network_pathology_apply_pre_mapmaking"] =
        pathology.apply_pre_mapmaking;
    node["scan_network_pathology_min_candidate_clusters"] =
        pathology.min_candidate_clusters;
    node["scan_network_pathology_min_candidate_events"] =
        pathology.min_candidate_events;
    node["scan_network_pathology_min_max_residual_z"] =
        pathology.min_max_residual_z;
    node["scan_network_pathology_severe_candidate_events"] =
        pathology.severe_candidate_events;
    node["scan_network_pathology_severe_max_residual_z"] =
        pathology.severe_max_residual_z;
    node["scan_network_pathology_max_new_flagged_fraction"] =
        pathology.max_new_flagged_fraction;
    return node;
}

inline YAML::Node processed_second_pass_local_config_node(
    const citlali::config::ProcessedTimeChunkSecondPassLocalConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["min_spike_sigma"] = config.min_spike_sigma;
    node["min_good_frac"] = config.min_good_frac;
    node["baseline_window_sec"] = config.baseline_window_sec;
    node["sigma_scale"] = config.sigma_scale;
    node["delta_sigma_scale"] = config.delta_sigma_scale;
    node["raw_candidate_rel_sigma_scale"] =
        config.raw_candidate_rel_sigma_scale;
    node["raw_window_sec"] = config.raw_window_sec;
    node["raw_half_peak_frac"] = config.raw_half_peak_frac;
    node["raw_max_width_sec"] = config.raw_max_width_sec;
    node["delta_window_sec"] = config.delta_window_sec;
    node["delta_half_peak_frac"] = config.delta_half_peak_frac;
    node["delta_max_width_sec"] = config.delta_max_width_sec;
    node["max_step_shift_z"] = config.max_step_shift_z;
    node["high_score_event_override"] = config.high_score_event_override;
    node["merge_within_detector_sec"] =
        config.merge_within_detector_sec;
    node["cluster_events_sec"] = config.cluster_events_sec;
    node["min_cluster_detectors"] = config.min_cluster_detectors;
    node["high_score_cluster_override"] =
        config.high_score_cluster_override;
    node["max_auto_flag_clusters_per_network"] =
        config.max_auto_flag_clusters_per_network;
    node["selective_busy_network_acceptance_enabled"] =
        config.selective_busy_network_acceptance_enabled;
    node["source_protection"] =
        timestream_source_protection_config_node(config.source_protection);
    return node;
}

inline YAML::Node processed_standard_pca_config_node(
    const citlali::config::ProcessedTimeChunkStandardPcaConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["stddev_limit"] = config.stddev_limit;
    node["n_calc"] = config.n_calc;
    node["n_eig_to_cut"] = YAML::Node(YAML::NodeType::Map);
    for (const auto &[array_name, values] : config.n_eig_to_cut) {
        node["n_eig_to_cut"][array_name] =
            processed_config_sequence_node(values);
    }
    return node;
}

inline YAML::Node processed_corr_grouping_config_node(
    const citlali::config::ProcessedTimeChunkCorrGroupingConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["metric"] =
        std::string{citlali::config::to_string(config.metric)};
    node["corr_min"] = config.corr_min;
    node["min_overlap"] = config.min_overlap;
    node["min_good_frac"] = config.min_good_frac;
    node["min_group_size"] = config.min_group_size;
    node["max_samples"] = config.max_samples;
    node["clean_residual"] = config.clean_residual;
    return node;
}

inline YAML::Node processed_null_model_config_node(
    const citlali::config::ProcessedTimeChunkNullModelConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["n_surrogates"] = config.n_surrogates;
    node["quantile"] = config.quantile;
    node["min_good_frac"] = config.min_good_frac;
    node["max_modes"] = config.max_modes;
    node["max_samples"] = config.max_samples;
    node["seed"] = config.seed;
    node["grouping"] = processed_config_sequence_node(config.grouping);
    return node;
}

inline YAML::Node processed_marchenko_pastur_config_node(
    const citlali::config::ProcessedTimeChunkMarchenkoPasturConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["min_good_frac"] = config.min_good_frac;
    node["max_modes"] = config.max_modes;
    node["max_samples"] = config.max_samples;
    node["band_low_Hz"] = config.band_low_Hz;
    node["band_high_Hz"] = config.band_high_Hz;
    node["clip_z"] = config.clip_z;
    node["bulk_keep_frac"] = config.bulk_keep_frac;
    node["q_grid_size"] = config.q_grid_size;
    node["grouping"] = processed_config_sequence_node(config.grouping);
    return node;
}

inline YAML::Node processed_adaptive_selector_config_node(
    const citlali::config::ProcessedTimeChunkAdaptiveSelectorConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["min_good_frac"] = config.min_good_frac;
    node["max_det"] = config.max_det;
    node["max_samples"] = config.max_samples;
    node["max_pairs"] = config.max_pairs;
    node["seed"] = config.seed;
    node["clip_z"] = config.clip_z;
    node["low_weight"] = config.low_weight;
    node["tail_weight"] = config.tail_weight;
    node["topmode_weight"] = config.topmode_weight;
    node["reg_weight"] = config.reg_weight;
    node["low_band_Hz"] =
        processed_config_sequence_node(config.low_band_Hz);
    node["mid_band_Hz"] =
        processed_config_sequence_node(config.mid_band_Hz);
    node["candidate_offsets"] =
        processed_config_sequence_node(config.candidate_offsets);
    node["grouping"] = processed_config_sequence_node(config.grouping);
    node["log_candidates"] = config.log_candidates;
    return node;
}

inline YAML::Node processed_clean_config_node(
    const citlali::config::ProcessedTimeChunkCleanConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["active"] = std::string{citlali::config::to_string(config.active)};
    node["grouping"] = processed_config_sequence_node(config.grouping);
    node["mask_radius_arcsec"] = config.mask_radius_arcsec;
    node["tau"] = config.tau;
    node["standard_pca"] =
        processed_standard_pca_config_node(config.standard_pca);
    node["corr_grouping"] =
        processed_corr_grouping_config_node(config.corr_grouping);
    node["null_model"] =
        processed_null_model_config_node(config.null_model);
    node["marchenko_pastur"] =
        processed_marchenko_pastur_config_node(config.marchenko_pastur);
    node["adaptive_selector"] =
        processed_adaptive_selector_config_node(config.adaptive_selector);
    return node;
}

inline YAML::Node processed_weight_validation_config_node(
    const citlali::config::ProcessedTimeChunkWeightValidationConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["accumulation_iters"] = config.accumulation_iters;
    node["apply_start_iter"] = config.apply_start_iter;
    node["min_valid_scans"] = config.min_valid_scans;
    node["min_factor"] = config.min_factor;
    node["unvalidated_factor"] = config.unvalidated_factor;
    node["require_fruitloops_model"] = config.require_fruitloops_model;
    node["transient_ratio_enabled"] = config.transient_ratio_enabled;
    node["ratio_power"] = config.ratio_power;
    node["transient_ratio_power"] = config.transient_ratio_power;
    node["upward_enabled"] = config.upward_enabled;
    node["upward_max_factor"] = config.upward_max_factor;
    node["upward_power"] = config.upward_power;
    node["upward_min_base_factor"] = config.upward_min_base_factor;
    node["upward_require_atmospheric"] =
        config.upward_require_atmospheric;
    node["upward_min_atmospheric_factor"] =
        config.upward_min_atmospheric_factor;
    node["atmospheric_correlation_enabled"] =
        config.atmospheric_correlation_enabled;
    node["atmospheric_grouping"] = std::string{
        citlali::config::to_string(config.atmospheric_grouping)};
    node["atmospheric_min_detectors"] = config.atmospheric_min_detectors;
    node["atmospheric_ref"] = config.atmospheric_ref;
    node["atmospheric_span"] = config.atmospheric_span;
    node["atmospheric_power"] = config.atmospheric_power;
    node["min_good_frac"] = config.min_good_frac;
    node["min_overlap"] = config.min_overlap;
    node["max_samples"] = config.max_samples;
    node["high_weight_validation_enabled"] =
        config.high_weight_validation_enabled;
    node["high_weight_apply_caps"] = config.high_weight_apply_caps;
    node["high_weight_grouping"] = std::string{
        citlali::config::to_string(config.high_weight_grouping)};
    node["high_weight_min_group_detectors"] =
        config.high_weight_min_group_detectors;
    node["high_weight_log_robust_z"] = config.high_weight_log_robust_z;
    node["high_weight_max_median_factor"] =
        config.high_weight_max_median_factor;
    node["high_weight_cap_median_factor"] =
        config.high_weight_cap_median_factor;
    node["high_weight_min_validated_factor"] =
        config.high_weight_min_validated_factor;
    return node;
}

inline YAML::Node processed_weight_corr_penalty_term_config_node(
    const citlali::config::ProcessedTimeChunkWeightCorrPenaltyTermConfig
        &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["ref"] = config.ref;
    node["span"] = config.span;
    node["weight"] = config.weight;
    return node;
}

inline YAML::Node processed_weight_corr_penalty_band_config_node(
    const citlali::config::ProcessedTimeChunkWeightCorrPenaltyBandConfig
        &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["ref"] = config.ref;
    node["span"] = config.span;
    node["weight"] = config.weight;
    node["low_band_Hz"] =
        processed_config_sequence_node(config.low_band_Hz);
    node["mid_band_Hz"] =
        processed_config_sequence_node(config.mid_band_Hz);
    return node;
}

inline YAML::Node processed_weight_corr_penalty_config_node(
    const citlali::config::ProcessedTimeChunkWeightCorrPenaltyConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["min_good_frac"] = config.min_good_frac;
    node["min_overlap"] = config.min_overlap;
    node["max_samples"] = config.max_samples;
    node["max_pairs"] = config.max_pairs;
    node["seed"] = config.seed;
    node["floor"] = config.floor;
    node["exponent"] = config.exponent;
    node["pair_corr"] =
        processed_weight_corr_penalty_term_config_node(config.pair_corr);
    node["cm_el_corr"] =
        processed_weight_corr_penalty_term_config_node(config.cm_el_corr);
    node["cm_low_mid_ratio"] =
        processed_weight_corr_penalty_band_config_node(
            config.cm_low_mid_ratio);
    return node;
}

inline YAML::Node processed_busy_row_suppression_config_node(
    const citlali::config::ProcessedTimeChunkBusyRowSuppressionConfig
        &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["require_busy_veto"] = config.require_busy_veto;
    node["min_candidate_clusters"] = config.min_candidate_clusters;
    node["min_max_unflagged_residual_z"] =
        config.min_max_unflagged_residual_z;
    node["factor"] = config.factor;
    return node;
}

inline YAML::Node processed_weighting_config_node(
    const citlali::config::ProcessedTimeChunkWeightingConfig &config) {
    YAML::Node node;
    node["type"] = std::string{citlali::config::to_string(config.type)};
    node["source_mask_radius_arcsec"] = config.source_mask_radius_arcsec;
    node["hybrid_correction_min_factor"] =
        config.hybrid_correction_min_factor;
    node["hybrid_correction_max_factor"] =
        config.hybrid_correction_max_factor;
    node["median_map_weight_factor"] = config.median_map_weight_factor;
    node["lower_map_weight_factor"] = config.lower_map_weight_factor;
    node["upper_map_weight_factor"] = config.upper_map_weight_factor;
    node["validation"] =
        processed_weight_validation_config_node(config.validation);
    node["corr_penalty"] =
        processed_weight_corr_penalty_config_node(config.corr_penalty);
    node["busy_row_suppression"] =
        processed_busy_row_suppression_config_node(
            config.busy_row_suppression);
    return node;
}

inline YAML::Node processed_flagging_config_node(
    const citlali::config::ProcessedTimeChunkFlaggingConfig &config) {
    YAML::Node node;
    node["lower_tod_inv_var_factor"] = config.lower_tod_inv_var_factor;
    node["upper_tod_inv_var_factor"] = config.upper_tod_inv_var_factor;
    node["second_pass_local"] =
        processed_second_pass_local_config_node(config.second_pass_local);
    return node;
}

inline YAML::Node processed_time_chunk_config_node(
    const citlali::config::ProcessedTimeChunkConfig &config) {
    YAML::Node node;
    node["clean"] = processed_clean_config_node(config.clean);
    node["weighting"] = processed_weighting_config_node(config.weighting);
    node["flagging"] = processed_flagging_config_node(config.flagging);
    return node;
}

inline YAML::Node processed_timestream_config_snapshot_node(
    const ProcessedTimestreamConfigSnapshot &snapshot) {
    YAML::Node node;
    node["fruit_loops"] = fruit_loops_config_node(snapshot.fruit_loops);
    node["learning"] = learning_config_node(snapshot.learning);
    node["processed_time_chunk"] =
        processed_time_chunk_config_node(snapshot.processed_time_chunk);
    return node;
}

template <class Value, class Serialize>
YAML::Node processed_optional_record_node(
    const std::optional<Value> &value, Serialize serialize) {
    YAML::Node node;
    node["available"] = value.has_value();
    if (value) {
        node["value"] = serialize(*value);
    }
    return node;
}

template <class Value>
YAML::Node processed_optional_scalar_node(
    const std::optional<Value> &value) {
    return processed_optional_record_node(
        value, [](const auto &item) {
            YAML::Node node;
            node = item;
            return node;
        });
}

inline YAML::Node processed_cleaner_mode_resolution_node(
    const ProcessedCleanerModeResolution &resolution) {
    YAML::Node node;
    node["effective"] =
        std::string{citlali::config::to_string(resolution.effective)};
    node["enabled_mode_count"] = resolution.enabled_mode_count;
    return node;
}

inline YAML::Node processed_weighting_source_mask_resolution_node(
    const ProcessedWeightingSourceMaskResolution &resolution) {
    YAML::Node node;
    node["requested_present"] = resolution.requested.has_value();
    if (resolution.requested) {
        node["requested"] = *resolution.requested;
    }
    node["effective"] = resolution.effective;
    node["inherited_from_cleaning"] = resolution.inherited_from_cleaning;
    return node;
}

inline YAML::Node processed_weighting_resolution_node(
    const ProcessedWeightingResolution &resolution) {
    YAML::Node node;
    node["validation_forced_by_weighting_type"] =
        resolution.validation_forced_by_weighting_type;
    node["busy_row_disabled_without_second_pass"] =
        resolution.busy_row_disabled_without_second_pass;
    return node;
}

inline YAML::Node fruit_loop_iteration_resolution_node(
    const FruitLoopIterationResolution &resolution) {
    YAML::Node node;
    node["effective_max_iters"] = resolution.effective_max_iters;
    node["effective_save_all_iters"] =
        resolution.effective_save_all_iters;
    node["forced_single_iteration_while_disabled"] =
        resolution.forced_single_iteration_while_disabled;
    node["forced_single_iteration_for_beammap"] =
        resolution.forced_single_iteration_for_beammap;
    return node;
}

inline YAML::Node fruit_loop_interpolation_resolution_node(
    const FruitLoopInterpolationResolution &resolution) {
    YAML::Node node;
    node["requested"] =
        std::string{citlali::config::to_string(resolution.requested)};
    node["mapmaking_default"] = std::string{
        citlali::config::to_string(resolution.mapmaking_default)};
    node["effective"] =
        std::string{citlali::config::to_string(resolution.effective)};
    node["override_applied"] = resolution.override_applied;
    node["jinc_fell_back_to_bilinear"] =
        resolution.jinc_fell_back_to_bilinear;
    return node;
}

inline YAML::Node source_protection_resolution_node(
    const SourceProtectionActivationResolution &resolution) {
    YAML::Node node;
    node["source_aware_reduction"] = resolution.source_aware_reduction;
    node["raw_activation_requested"] =
        resolution.raw_activation_requested;
    node["processed_activation_requested"] =
        resolution.processed_activation_requested;
    node["raw_active"] = resolution.raw_active;
    node["processed_active"] = resolution.processed_active;
    return node;
}

inline YAML::Node processed_timestream_effective_resolutions_node(
    const ProcessedTimestreamEffectiveResolutionRecord &resolutions) {
    YAML::Node node;
    node["cleaner_mode"] = processed_optional_record_node(
        resolutions.cleaner_mode, processed_cleaner_mode_resolution_node);
    node["weighting_source_mask"] = processed_optional_record_node(
        resolutions.weighting_source_mask,
        processed_weighting_source_mask_resolution_node);
    node["weighting_dependencies"] = processed_optional_record_node(
        resolutions.weighting_dependencies,
        processed_weighting_resolution_node);
    node["fruit_loop_iterations"] = processed_optional_record_node(
        resolutions.fruit_loop_iterations,
        fruit_loop_iteration_resolution_node);
    node["fruit_loop_interpolation"] = processed_optional_record_node(
        resolutions.fruit_loop_interpolation,
        fruit_loop_interpolation_resolution_node);
    return node;
}

inline YAML::Node processed_timestream_realized_state_node(
    const ProcessedTimestreamRealizedState &realized) {
    YAML::Node node;
    node["source_protection"] = processed_optional_record_node(
        realized.source_protection, source_protection_resolution_node);
    node["fruit_loop_iterations_completed"] =
        processed_optional_scalar_node(
            realized.fruit_loop_iterations_completed);
    node["fruit_loops_converged"] =
        processed_optional_scalar_node(realized.fruit_loops_converged);
    return node;
}

}  // namespace citlali::pipeline
