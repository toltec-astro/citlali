#pragma once

#include <citlali/core/pipeline/config_parse_tracking.h>

#include <tuple>
#include <type_traits>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Target, class Key, class Diagnostics>
void read_optional_learning_config(
    Config &config, const Key &key, Target &target, Diagnostics &diagnostics,
    std::vector<std::decay_t<Target>> min_val = {},
    std::vector<std::decay_t<Target>> max_val = {}) {
    citlali::pipeline::read_optional_config_value(
        config, target, diagnostics, key, {}, std::move(min_val),
        std::move(max_val));
}

template <class Config, class LearningConfig, class Diagnostics>
void read_learning_core_config(Config &config, LearningConfig &typed_config,
                               Diagnostics &diagnostics) {
    read_optional_learning_config(
        config, std::tuple{"timestream", "learning", "enabled"},
        typed_config.enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning", "diagnostics_enabled"},
        typed_config.diagnostics_enabled, diagnostics);
    read_optional_learning_config(
        config, std::tuple{"timestream", "learning", "learn_iters"},
        typed_config.learn_iters, diagnostics, {0});
    read_optional_learning_config(
        config, std::tuple{"timestream", "learning", "apply_start_iter"},
        typed_config.apply_start_iter, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning", "max_records_per_type"},
        typed_config.max_records_per_type, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning", "apply_sample_masks_enabled"},
        typed_config.apply_sample_masks_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "apply_max_new_flagged_fraction"},
        typed_config.apply_max_new_flagged_fraction, diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "busy_detector_exclusion_enabled"},
        typed_config.busy_detector.exclusion_enabled, diagnostics);
}

template <class Config, class LearningConfig, class Diagnostics>
void read_learning_map_pixel_outlier_config(
    Config &config, LearningConfig &typed_config, Diagnostics &diagnostics) {
    auto &outlier = typed_config.map_pixel_outlier;
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_diagnostics_enabled"},
        outlier.diagnostics_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_contributor_diagnostics_enabled"},
        outlier.contributor_diagnostics_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{
            "timestream", "learning",
            "map_pixel_outlier_targeted_contributor_diagnostics_enabled"},
        outlier.targeted_contributor_diagnostics_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_detector_exclusion_enabled"},
        outlier.detector_exclusion_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning", "map_pixel_outlier_top_n"},
        outlier.top_n, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{
            "timestream", "learning",
            "map_pixel_outlier_targeted_contributor_max_pixels"},
        outlier.targeted_contributor_max_pixels, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{
            "timestream", "learning",
            "map_pixel_outlier_detector_exclusion_min_pixels"},
        outlier.detector_exclusion_min_pixels, diagnostics, {1});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_min_abs_z"},
        outlier.min_abs_z, diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_min_n_eff"},
        outlier.min_n_eff, diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_source_radius_arcsec"},
        outlier.source_radius_arcsec, diagnostics, {0.0});
}

template <class Config, class LearningConfig, class Diagnostics>
void read_learning_scan_network_pathology_config(
    Config &config, LearningConfig &typed_config, Diagnostics &diagnostics) {
    auto &pathology = typed_config.scan_network_pathology;
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_enabled"},
        pathology.enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_apply_pre_rtc"},
        pathology.apply_pre_rtc, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_apply_pre_ptc"},
        pathology.apply_pre_ptc, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_apply_pre_mapmaking"},
        pathology.apply_pre_mapmaking, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_min_candidate_clusters"},
        pathology.min_candidate_clusters, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_min_candidate_events"},
        pathology.min_candidate_events, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_min_max_residual_z"},
        pathology.min_max_residual_z, diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_severe_candidate_events"},
        pathology.severe_candidate_events, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_severe_max_residual_z"},
        pathology.severe_max_residual_z, diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_max_new_flagged_fraction"},
        pathology.max_new_flagged_fraction, diagnostics, {0.0});
}

template <class Config, class LearningConfig, class Diagnostics>
void read_learning_config(Config &config, LearningConfig &typed_config,
                          Diagnostics &diagnostics) {
    read_learning_core_config(config, typed_config, diagnostics);
    read_learning_map_pixel_outlier_config(config, typed_config, diagnostics);
    read_learning_scan_network_pathology_config(
        config, typed_config, diagnostics);
}

}  // namespace citlali::pipeline
