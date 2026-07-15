#pragma once

#include <citlali/core/config/timestream_config.h>

namespace citlali::pipeline {

template <class Options>
Options make_learning_options(
    const citlali::config::TimestreamLearningConfig &config) {
    Options options;
    options.enabled = config.enabled;
    options.diagnostics_enabled = config.diagnostics_enabled;
    options.learn_iters = config.learn_iters;
    options.apply_start_iter = config.apply_start_iter;
    options.max_records_per_type = config.max_records_per_type;
    options.apply_sample_masks_enabled = config.apply_sample_masks_enabled;
    options.apply_max_new_flagged_fraction =
        config.apply_max_new_flagged_fraction;

    const auto &outlier = config.map_pixel_outlier;
    options.map_pixel_outlier_diagnostics_enabled =
        outlier.diagnostics_enabled;
    options.map_pixel_outlier_contributor_diagnostics_enabled =
        outlier.contributor_diagnostics_enabled;
    options.map_pixel_outlier_targeted_contributor_diagnostics_enabled =
        outlier.targeted_contributor_diagnostics_enabled;
    options.map_pixel_outlier_detector_exclusion_enabled =
        outlier.detector_exclusion_enabled;
    options.map_pixel_outlier_top_n = outlier.top_n;
    options.map_pixel_outlier_targeted_contributor_max_pixels =
        outlier.targeted_contributor_max_pixels;
    options.map_pixel_outlier_detector_exclusion_min_pixels =
        outlier.detector_exclusion_min_pixels;
    options.map_pixel_outlier_min_abs_z = outlier.min_abs_z;
    options.map_pixel_outlier_min_n_eff = outlier.min_n_eff;
    options.map_pixel_outlier_source_radius_arcsec =
        outlier.source_radius_arcsec;
    options.busy_detector_exclusion_enabled =
        config.busy_detector.exclusion_enabled;

    const auto &pathology = config.scan_network_pathology;
    options.scan_network_pathology_enabled = pathology.enabled;
    options.scan_network_pathology_apply_pre_rtc = pathology.apply_pre_rtc;
    options.scan_network_pathology_apply_pre_ptc = pathology.apply_pre_ptc;
    options.scan_network_pathology_apply_pre_mapmaking =
        pathology.apply_pre_mapmaking;
    options.scan_network_pathology_min_candidate_clusters =
        pathology.min_candidate_clusters;
    options.scan_network_pathology_min_candidate_events =
        pathology.min_candidate_events;
    options.scan_network_pathology_min_max_residual_z =
        pathology.min_max_residual_z;
    options.scan_network_pathology_severe_candidate_events =
        pathology.severe_candidate_events;
    options.scan_network_pathology_severe_max_residual_z =
        pathology.severe_max_residual_z;
    options.scan_network_pathology_max_new_flagged_fraction =
        pathology.max_new_flagged_fraction;
    return options;
}

template <class LearningState>
void adapt_learning_config_one_way(
    const citlali::config::TimestreamLearningConfig &config,
    LearningState &learning) {
    learning.configure(
        make_learning_options<typename LearningState::Options>(config));
}

template <class LearningOptions>
bool learning_map_contribution_diagnostics_enabled(
    const LearningOptions &options) {
    return options.enabled && options.diagnostics_enabled &&
           options.map_pixel_outlier_diagnostics_enabled &&
           options.map_pixel_outlier_contributor_diagnostics_enabled;
}

template <class OutputMapBlock, class CoaddMapBlock>
void set_learning_map_contribution_diagnostics(
    bool enabled, OutputMapBlock &omb, CoaddMapBlock &cmb) {
    omb.contribution_diag_enabled = enabled;
    cmb.contribution_diag_enabled = enabled;
}

}  // namespace citlali::pipeline
