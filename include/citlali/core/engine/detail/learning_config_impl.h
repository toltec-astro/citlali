#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/learning_config_logging.h>
#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/engine/detail/learning_config_read.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_learning_config(CT &config) {
    ReductionLearningState::Options options;
    auto &learning_config =
        citlali::pipeline::timestream_config(*this).learning;

    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","enabled"},
        options.enabled, learning_config.enabled,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","diagnostics_enabled"},
        options.diagnostics_enabled,
        learning_config.diagnostics_enabled,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","learn_iters"},
        options.learn_iters, learning_config.learn_iters,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","apply_start_iter"},
        options.apply_start_iter,
        learning_config.apply_start_iter,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","max_records_per_type"},
        options.max_records_per_type,
        learning_config.max_records_per_type,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","apply_sample_masks_enabled"},
        options.apply_sample_masks_enabled,
        learning_config.apply_sample_masks_enabled,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","apply_max_new_flagged_fraction"},
        options.apply_max_new_flagged_fraction,
        learning_config.apply_max_new_flagged_fraction,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys, {0.0});
    auto &typed_map_outlier =
        learning_config.map_pixel_outlier;
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_diagnostics_enabled"},
        options.map_pixel_outlier_diagnostics_enabled,
        typed_map_outlier.diagnostics_enabled, config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_contributor_diagnostics_enabled"},
        options.map_pixel_outlier_contributor_diagnostics_enabled,
        typed_map_outlier.contributor_diagnostics_enabled, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_diagnostics_enabled"},
        options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
        typed_map_outlier.targeted_contributor_diagnostics_enabled,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_enabled"},
        options.map_pixel_outlier_detector_exclusion_enabled,
        typed_map_outlier.detector_exclusion_enabled, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_top_n"},
        options.map_pixel_outlier_top_n, typed_map_outlier.top_n,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_max_pixels"},
        options.map_pixel_outlier_targeted_contributor_max_pixels,
        typed_map_outlier.targeted_contributor_max_pixels, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_min_pixels"},
        options.map_pixel_outlier_detector_exclusion_min_pixels,
        typed_map_outlier.detector_exclusion_min_pixels, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys, {1});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_min_abs_z"},
        options.map_pixel_outlier_min_abs_z, typed_map_outlier.min_abs_z,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys, {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_min_n_eff"},
        options.map_pixel_outlier_min_n_eff, typed_map_outlier.min_n_eff,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys, {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_source_radius_arcsec"},
        options.map_pixel_outlier_source_radius_arcsec,
        typed_map_outlier.source_radius_arcsec, config_diagnostics.missing_keys, config_diagnostics.invalid_keys,
        {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","busy_detector_exclusion_enabled"},
        options.busy_detector_exclusion_enabled,
        learning_config.busy_detector.exclusion_enabled,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    auto &typed_scan_pathology =
        learning_config.scan_network_pathology;
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_enabled"},
        options.scan_network_pathology_enabled, typed_scan_pathology.enabled,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_apply_pre_rtc"},
        options.scan_network_pathology_apply_pre_rtc,
        typed_scan_pathology.apply_pre_rtc, config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_apply_pre_ptc"},
        options.scan_network_pathology_apply_pre_ptc,
        typed_scan_pathology.apply_pre_ptc, config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_apply_pre_mapmaking"},
        options.scan_network_pathology_apply_pre_mapmaking,
        typed_scan_pathology.apply_pre_mapmaking, config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_min_candidate_clusters"},
        options.scan_network_pathology_min_candidate_clusters,
        typed_scan_pathology.min_candidate_clusters, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_min_candidate_events"},
        options.scan_network_pathology_min_candidate_events,
        typed_scan_pathology.min_candidate_events, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_min_max_residual_z"},
        options.scan_network_pathology_min_max_residual_z,
        typed_scan_pathology.min_max_residual_z, config_diagnostics.missing_keys, config_diagnostics.invalid_keys,
        {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_severe_candidate_events"},
        options.scan_network_pathology_severe_candidate_events,
        typed_scan_pathology.severe_candidate_events, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_severe_max_residual_z"},
        options.scan_network_pathology_severe_max_residual_z,
        typed_scan_pathology.severe_max_residual_z, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys, {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_max_new_flagged_fraction"},
        options.scan_network_pathology_max_new_flagged_fraction,
        typed_scan_pathology.max_new_flagged_fraction, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys, {0.0});

    learning.configure(options);
    const bool map_contribution_diag =
        citlali::engine_detail::learning_map_contribution_diagnostics_enabled(
            learning.options);
    citlali::engine_detail::set_learning_map_contribution_diagnostics(
        map_contribution_diag, omb, cmb);
    citlali::engine_detail::log_reduction_learning_config(
        learning.options, logger);
}
