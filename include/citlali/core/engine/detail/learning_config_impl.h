#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/learning_config_logging.h>
#include <citlali/core/engine/detail/config_parse_tracking.h>

template<typename CT>
void Engine::get_learning_config(CT &config) {
    ReductionLearningState::Options options;

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return citlali::engine_detail::config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before);
    };
    auto mirror_if_parsed = [&](auto &target, const auto &source,
                                std::size_t missing_before,
                                std::size_t invalid_before) {
        citlali::engine_detail::mirror_if_config_parsed(
            target, source, missing_keys, invalid_keys, missing_before,
            invalid_before);
    };

    if (config.template has_typed<bool>(std::tuple{"timestream","learning","enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.enabled = options.enabled;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.diagnostics_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","diagnostics_enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.diagnostics_enabled =
                options.diagnostics_enabled;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","learn_iters"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.learn_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","learn_iters"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.learn_iters = options.learn_iters;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","apply_start_iter"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_start_iter, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_start_iter"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_start_iter =
                options.apply_start_iter;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","max_records_per_type"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.max_records_per_type, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","max_records_per_type"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.max_records_per_type =
                options.max_records_per_type;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","apply_sample_masks_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_sample_masks_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_sample_masks_enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_sample_masks_enabled =
                options.apply_sample_masks_enabled;
        }
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","apply_max_new_flagged_fraction"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_max_new_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_max_new_flagged_fraction"}, {}, {0.0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_max_new_flagged_fraction =
                options.apply_max_new_flagged_fraction;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_diagnostics_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.diagnostics_enabled,
            options.map_pixel_outlier_diagnostics_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_contributor_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_contributor_diagnostics_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_contributor_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.contributor_diagnostics_enabled,
            options.map_pixel_outlier_contributor_diagnostics_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.targeted_contributor_diagnostics_enabled,
            options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_detector_exclusion_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.detector_exclusion_enabled,
            options.map_pixel_outlier_detector_exclusion_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_top_n"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_top_n, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_top_n"}, {}, {0});
        mirror_if_parsed(typed_timestream_config.learning.map_pixel_outlier.top_n,
                         options.map_pixel_outlier_top_n, missing_before,
                         invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_max_pixels"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_targeted_contributor_max_pixels,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_max_pixels"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.targeted_contributor_max_pixels,
            options.map_pixel_outlier_targeted_contributor_max_pixels,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_min_pixels"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_detector_exclusion_min_pixels,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_min_pixels"},
                         {}, {1});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.detector_exclusion_min_pixels,
            options.map_pixel_outlier_detector_exclusion_min_pixels,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_min_abs_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_min_abs_z, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_min_abs_z"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.min_abs_z,
            options.map_pixel_outlier_min_abs_z, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_min_n_eff"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_min_n_eff, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_min_n_eff"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.min_n_eff,
            options.map_pixel_outlier_min_n_eff, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_source_radius_arcsec"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_source_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_source_radius_arcsec"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.source_radius_arcsec,
            options.map_pixel_outlier_source_radius_arcsec, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","busy_detector_exclusion_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.busy_detector_exclusion_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","busy_detector_exclusion_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.busy_detector.exclusion_enabled,
            options.busy_detector_exclusion_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.enabled,
            options.scan_network_pathology_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_rtc"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_rtc,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_rtc"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_rtc,
            options.scan_network_pathology_apply_pre_rtc, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_ptc"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_ptc,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_ptc"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_ptc,
            options.scan_network_pathology_apply_pre_ptc, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_mapmaking"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_mapmaking,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_mapmaking"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_mapmaking,
            options.scan_network_pathology_apply_pre_mapmaking, missing_before,
            invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_min_candidate_clusters"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_candidate_clusters,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_candidate_clusters"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_candidate_clusters,
            options.scan_network_pathology_min_candidate_clusters,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_min_candidate_events"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_candidate_events,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_candidate_events"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_candidate_events,
            options.scan_network_pathology_min_candidate_events, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_min_max_residual_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_max_residual_z,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_max_residual_z"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_max_residual_z,
            options.scan_network_pathology_min_max_residual_z, missing_before,
            invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_severe_candidate_events"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_severe_candidate_events,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_severe_candidate_events"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.severe_candidate_events,
            options.scan_network_pathology_severe_candidate_events,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_severe_max_residual_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_severe_max_residual_z,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_severe_max_residual_z"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.severe_max_residual_z,
            options.scan_network_pathology_severe_max_residual_z,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_max_new_flagged_fraction"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_max_new_flagged_fraction,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_max_new_flagged_fraction"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.max_new_flagged_fraction,
            options.scan_network_pathology_max_new_flagged_fraction,
            missing_before, invalid_before);
    }

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
