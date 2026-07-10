#pragma once

namespace citlali::pipeline {

template <class LearningOptions, class Logger>
void log_reduction_learning_config(const LearningOptions &options,
                                   const Logger &logger) {
    logger->info(
        "reduction learning state configured: enabled={} diagnostics_enabled={} "
        "learn_iters={} apply_start_iter={} max_records_per_type={} "
        "apply_sample_masks_enabled={} apply_max_new_flagged_fraction={:.4g} "
        "map_pixel_outliers(enabled={} contributors={} targeted_contributors={} detector_exclusion={} top_n={} target_max={} exclude_min_pixels={} min_abs_z={} min_n_eff={} source_radius_arcsec={}) "
        "busy_detector_exclusion_enabled={} scan_network_pathology(enabled={} pre_rtc={} pre_ptc={} pre_mapmaking={} min_clusters={} min_events={} min_resid_z={} severe_events={} severe_resid_z={} max_new_flagged_fraction={:.4g})",
        options.enabled, options.diagnostics_enabled, options.learn_iters,
        options.apply_start_iter, options.max_records_per_type,
        options.apply_sample_masks_enabled,
        options.apply_max_new_flagged_fraction,
        options.map_pixel_outlier_diagnostics_enabled,
        options.map_pixel_outlier_contributor_diagnostics_enabled,
        options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
        options.map_pixel_outlier_detector_exclusion_enabled,
        options.map_pixel_outlier_top_n,
        options.map_pixel_outlier_targeted_contributor_max_pixels,
        options.map_pixel_outlier_detector_exclusion_min_pixels,
        options.map_pixel_outlier_min_abs_z,
        options.map_pixel_outlier_min_n_eff,
        options.map_pixel_outlier_source_radius_arcsec,
        options.busy_detector_exclusion_enabled,
        options.scan_network_pathology_enabled,
        options.scan_network_pathology_apply_pre_rtc,
        options.scan_network_pathology_apply_pre_ptc,
        options.scan_network_pathology_apply_pre_mapmaking,
        options.scan_network_pathology_min_candidate_clusters,
        options.scan_network_pathology_min_candidate_events,
        options.scan_network_pathology_min_max_residual_z,
        options.scan_network_pathology_severe_candidate_events,
        options.scan_network_pathology_severe_max_residual_z,
        options.scan_network_pathology_max_new_flagged_fraction);
}

}  // namespace citlali::pipeline
