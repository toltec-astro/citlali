#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/learning_config_logging.h>
#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/engine/detail/learning_config_read.h>

template<typename CT>
void Engine::get_learning_config(CT &config) {
    ReductionLearningState::Options options;

    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","enabled"},
        options.enabled, typed_timestream_config.learning.enabled,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","diagnostics_enabled"},
        options.diagnostics_enabled,
        typed_timestream_config.learning.diagnostics_enabled,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","learn_iters"},
        options.learn_iters, typed_timestream_config.learning.learn_iters,
        missing_keys, invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","apply_start_iter"},
        options.apply_start_iter,
        typed_timestream_config.learning.apply_start_iter,
        missing_keys, invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","max_records_per_type"},
        options.max_records_per_type,
        typed_timestream_config.learning.max_records_per_type,
        missing_keys, invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","apply_sample_masks_enabled"},
        options.apply_sample_masks_enabled,
        typed_timestream_config.learning.apply_sample_masks_enabled,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","apply_max_new_flagged_fraction"},
        options.apply_max_new_flagged_fraction,
        typed_timestream_config.learning.apply_max_new_flagged_fraction,
        missing_keys, invalid_keys, {0.0});
    auto &typed_map_outlier =
        typed_timestream_config.learning.map_pixel_outlier;
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_diagnostics_enabled"},
        options.map_pixel_outlier_diagnostics_enabled,
        typed_map_outlier.diagnostics_enabled, missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_contributor_diagnostics_enabled"},
        options.map_pixel_outlier_contributor_diagnostics_enabled,
        typed_map_outlier.contributor_diagnostics_enabled, missing_keys,
        invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_diagnostics_enabled"},
        options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
        typed_map_outlier.targeted_contributor_diagnostics_enabled,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_enabled"},
        options.map_pixel_outlier_detector_exclusion_enabled,
        typed_map_outlier.detector_exclusion_enabled, missing_keys,
        invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_top_n"},
        options.map_pixel_outlier_top_n, typed_map_outlier.top_n,
        missing_keys, invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_max_pixels"},
        options.map_pixel_outlier_targeted_contributor_max_pixels,
        typed_map_outlier.targeted_contributor_max_pixels, missing_keys,
        invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_min_pixels"},
        options.map_pixel_outlier_detector_exclusion_min_pixels,
        typed_map_outlier.detector_exclusion_min_pixels, missing_keys,
        invalid_keys, {1});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_min_abs_z"},
        options.map_pixel_outlier_min_abs_z, typed_map_outlier.min_abs_z,
        missing_keys, invalid_keys, {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_min_n_eff"},
        options.map_pixel_outlier_min_n_eff, typed_map_outlier.min_n_eff,
        missing_keys, invalid_keys, {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","map_pixel_outlier_source_radius_arcsec"},
        options.map_pixel_outlier_source_radius_arcsec,
        typed_map_outlier.source_radius_arcsec, missing_keys, invalid_keys,
        {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","busy_detector_exclusion_enabled"},
        options.busy_detector_exclusion_enabled,
        typed_timestream_config.learning.busy_detector.exclusion_enabled,
        missing_keys, invalid_keys);
    auto &typed_scan_pathology =
        typed_timestream_config.learning.scan_network_pathology;
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_enabled"},
        options.scan_network_pathology_enabled, typed_scan_pathology.enabled,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_apply_pre_rtc"},
        options.scan_network_pathology_apply_pre_rtc,
        typed_scan_pathology.apply_pre_rtc, missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_apply_pre_ptc"},
        options.scan_network_pathology_apply_pre_ptc,
        typed_scan_pathology.apply_pre_ptc, missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_apply_pre_mapmaking"},
        options.scan_network_pathology_apply_pre_mapmaking,
        typed_scan_pathology.apply_pre_mapmaking, missing_keys, invalid_keys);
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_min_candidate_clusters"},
        options.scan_network_pathology_min_candidate_clusters,
        typed_scan_pathology.min_candidate_clusters, missing_keys,
        invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_min_candidate_events"},
        options.scan_network_pathology_min_candidate_events,
        typed_scan_pathology.min_candidate_events, missing_keys,
        invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_min_max_residual_z"},
        options.scan_network_pathology_min_max_residual_z,
        typed_scan_pathology.min_max_residual_z, missing_keys, invalid_keys,
        {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_severe_candidate_events"},
        options.scan_network_pathology_severe_candidate_events,
        typed_scan_pathology.severe_candidate_events, missing_keys,
        invalid_keys, {0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_severe_max_residual_z"},
        options.scan_network_pathology_severe_max_residual_z,
        typed_scan_pathology.severe_max_residual_z, missing_keys,
        invalid_keys, {0.0});
    citlali::engine_detail::read_optional_learning_config(
        config, std::tuple{"timestream","learning","scan_network_pathology_max_new_flagged_fraction"},
        options.scan_network_pathology_max_new_flagged_fraction,
        typed_scan_pathology.max_new_flagged_fraction, missing_keys,
        invalid_keys, {0.0});

    reduction_learning.configure(options);
    const bool map_contribution_diag =
        reduction_learning.options.enabled &&
        reduction_learning.options.diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_contributor_diagnostics_enabled;
    omb.contribution_diag_enabled = map_contribution_diag;
    cmb.contribution_diag_enabled = map_contribution_diag;
    citlali::engine_detail::log_reduction_learning_config(
        reduction_learning.options, logger);
}
