#pragma once

// Included by mapdiag_stats_outliers.h inside namespace citlali::pipeline.

inline bool mapdiag_dominance_meets_min_pixels(
    const MapdiagDetectorDominance &entry, int min_pixels) {
    return entry.count >= min_pixels;
}

template <class ArrayIds>
int mapdiag_array_id_or_default(Eigen::Index map_index,
                                const ArrayIds &array_ids,
                                int default_array_id) {
    if (map_index >= 0 &&
        map_index < static_cast<Eigen::Index>(array_ids.size())) {
        return array_ids[map_index];
    }
    return default_array_id;
}

inline int mapdiag_display_scan_index(int scan) {
    return scan + 1;
}

inline bool mapdiag_mask_pixel_is_selected(const Eigen::ArrayXXd &mask,
                                           Eigen::Index row,
                                           Eigen::Index col) {
    return mask(row, col) > 0.0;
}

template <class ReductionLearning>
bool mapdiag_outlier_diagnostics_enabled(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.is_enabled() &&
           reduction_learning.diagnostics_enabled() &&
           reduction_learning.options.map_pixel_outlier_diagnostics_enabled &&
           reduction_learning.options.map_pixel_outlier_top_n > 0;
}

template <class ReductionLearning>
double mapdiag_source_protect_radius_arcsec(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.options.map_pixel_outlier_source_radius_arcsec;
}

template <class ReductionLearning>
double mapdiag_min_effective_samples(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.options.map_pixel_outlier_min_n_eff;
}

template <class ReductionLearning>
double mapdiag_min_abs_z(const ReductionLearning &reduction_learning) {
    return reduction_learning.options.map_pixel_outlier_min_abs_z;
}

template <class ReductionLearning>
int mapdiag_candidate_top_n(const ReductionLearning &reduction_learning) {
    return reduction_learning.options.map_pixel_outlier_top_n;
}

template <class ReductionLearning>
bool mapdiag_detector_exclusion_enabled(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.options
        .map_pixel_outlier_detector_exclusion_enabled;
}

template <class ReductionLearning>
int mapdiag_detector_exclusion_min_pixels(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.options
        .map_pixel_outlier_detector_exclusion_min_pixels;
}

