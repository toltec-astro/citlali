#pragma once

#include <citlali/core/pipeline/config_parse_tracking.h>

#include <tuple>
#include <type_traits>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Param, class Target, class Key, class KeyVec>
void read_optional_learning_config(Config &config, const Key &key,
                                   Param &param, Target &target,
                                   KeyVec &missing_keys,
                                   KeyVec &invalid_keys,
                                   std::vector<std::decay_t<Param>> min_val = {},
                                   std::vector<std::decay_t<Param>> max_val = {}) {
    citlali::pipeline::read_optional_mirrored_config_value(
        config, key, param, target, missing_keys, invalid_keys,
        {}, std::move(min_val), std::move(max_val));
}

template <class Config, class Param, class Target, class Key,
          class Diagnostics>
void read_optional_learning_config(
    Config &config, const Key &key, Param &param, Target &target,
    Diagnostics &diagnostics,
    std::vector<std::decay_t<Param>> min_val = {},
    std::vector<std::decay_t<Param>> max_val = {}) {
    read_optional_learning_config(
        config, key, param, target, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths(), std::move(min_val),
        std::move(max_val));
}

template <class Config, class LearningOptions, class LearningConfig,
          class Diagnostics>
void read_learning_core_config(Config &config, LearningOptions &options,
                               LearningConfig &typed_config,
                               Diagnostics &diagnostics) {
    read_optional_learning_config(
        config, std::tuple{"timestream", "learning", "enabled"},
        options.enabled, typed_config.enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning", "diagnostics_enabled"},
        options.diagnostics_enabled, typed_config.diagnostics_enabled,
        diagnostics);
    read_optional_learning_config(
        config, std::tuple{"timestream", "learning", "learn_iters"},
        options.learn_iters, typed_config.learn_iters, diagnostics, {0});
    read_optional_learning_config(
        config, std::tuple{"timestream", "learning", "apply_start_iter"},
        options.apply_start_iter, typed_config.apply_start_iter, diagnostics,
        {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning", "max_records_per_type"},
        options.max_records_per_type, typed_config.max_records_per_type,
        diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning", "apply_sample_masks_enabled"},
        options.apply_sample_masks_enabled,
        typed_config.apply_sample_masks_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "apply_max_new_flagged_fraction"},
        options.apply_max_new_flagged_fraction,
        typed_config.apply_max_new_flagged_fraction, diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "busy_detector_exclusion_enabled"},
        options.busy_detector_exclusion_enabled,
        typed_config.busy_detector.exclusion_enabled, diagnostics);
}

template <class Config, class LearningOptions, class LearningConfig,
          class Diagnostics>
void read_learning_map_pixel_outlier_config(
    Config &config, LearningOptions &options, LearningConfig &typed_config,
    Diagnostics &diagnostics) {
    auto &typed_outlier = typed_config.map_pixel_outlier;
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_diagnostics_enabled"},
        options.map_pixel_outlier_diagnostics_enabled,
        typed_outlier.diagnostics_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_contributor_diagnostics_enabled"},
        options.map_pixel_outlier_contributor_diagnostics_enabled,
        typed_outlier.contributor_diagnostics_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{
            "timestream", "learning",
            "map_pixel_outlier_targeted_contributor_diagnostics_enabled"},
        options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
        typed_outlier.targeted_contributor_diagnostics_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_detector_exclusion_enabled"},
        options.map_pixel_outlier_detector_exclusion_enabled,
        typed_outlier.detector_exclusion_enabled, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning", "map_pixel_outlier_top_n"},
        options.map_pixel_outlier_top_n, typed_outlier.top_n, diagnostics,
        {0});
    read_optional_learning_config(
        config,
        std::tuple{
            "timestream", "learning",
            "map_pixel_outlier_targeted_contributor_max_pixels"},
        options.map_pixel_outlier_targeted_contributor_max_pixels,
        typed_outlier.targeted_contributor_max_pixels, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_detector_exclusion_min_pixels"},
        options.map_pixel_outlier_detector_exclusion_min_pixels,
        typed_outlier.detector_exclusion_min_pixels, diagnostics, {1});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_min_abs_z"},
        options.map_pixel_outlier_min_abs_z, typed_outlier.min_abs_z,
        diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_min_n_eff"},
        options.map_pixel_outlier_min_n_eff, typed_outlier.min_n_eff,
        diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "map_pixel_outlier_source_radius_arcsec"},
        options.map_pixel_outlier_source_radius_arcsec,
        typed_outlier.source_radius_arcsec, diagnostics, {0.0});
}

template <class Config, class LearningOptions, class LearningConfig,
          class Diagnostics>
void read_learning_scan_network_pathology_config(
    Config &config, LearningOptions &options, LearningConfig &typed_config,
    Diagnostics &diagnostics) {
    auto &typed_pathology = typed_config.scan_network_pathology;
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_enabled"},
        options.scan_network_pathology_enabled, typed_pathology.enabled,
        diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_apply_pre_rtc"},
        options.scan_network_pathology_apply_pre_rtc,
        typed_pathology.apply_pre_rtc, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_apply_pre_ptc"},
        options.scan_network_pathology_apply_pre_ptc,
        typed_pathology.apply_pre_ptc, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_apply_pre_mapmaking"},
        options.scan_network_pathology_apply_pre_mapmaking,
        typed_pathology.apply_pre_mapmaking, diagnostics);
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_min_candidate_clusters"},
        options.scan_network_pathology_min_candidate_clusters,
        typed_pathology.min_candidate_clusters, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_min_candidate_events"},
        options.scan_network_pathology_min_candidate_events,
        typed_pathology.min_candidate_events, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_min_max_residual_z"},
        options.scan_network_pathology_min_max_residual_z,
        typed_pathology.min_max_residual_z, diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_severe_candidate_events"},
        options.scan_network_pathology_severe_candidate_events,
        typed_pathology.severe_candidate_events, diagnostics, {0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_severe_max_residual_z"},
        options.scan_network_pathology_severe_max_residual_z,
        typed_pathology.severe_max_residual_z, diagnostics, {0.0});
    read_optional_learning_config(
        config,
        std::tuple{"timestream", "learning",
                   "scan_network_pathology_max_new_flagged_fraction"},
        options.scan_network_pathology_max_new_flagged_fraction,
        typed_pathology.max_new_flagged_fraction, diagnostics, {0.0});
}

template <class Config, class LearningOptions, class LearningConfig,
          class Diagnostics>
void read_learning_config(Config &config, LearningOptions &options,
                          LearningConfig &typed_config,
                          Diagnostics &diagnostics) {
    read_learning_core_config(config, options, typed_config, diagnostics);
    read_learning_map_pixel_outlier_config(
        config, options, typed_config, diagnostics);
    read_learning_scan_network_pathology_config(
        config, options, typed_config, diagnostics);
}

template <class LearningOptions>
bool learning_map_contribution_diagnostics_enabled(
    const LearningOptions &options) {
    return options.enabled && options.diagnostics_enabled &&
           options.map_pixel_outlier_diagnostics_enabled &&
           options.map_pixel_outlier_contributor_diagnostics_enabled;
}

template <class OutputMapBlock, class CoaddMapBlock>
void set_learning_map_contribution_diagnostics(bool enabled,
                                               OutputMapBlock &omb,
                                               CoaddMapBlock &cmb) {
    omb.contribution_diag_enabled = enabled;
    cmb.contribution_diag_enabled = enabled;
}

}  // namespace citlali::pipeline
