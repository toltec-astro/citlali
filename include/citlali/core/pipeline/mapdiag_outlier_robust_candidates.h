#pragma once

// Included by mapdiag_stats_outliers.h inside namespace citlali::pipeline.

inline bool mapdiag_has_minimum_samples(std::size_t n_values,
                                        std::size_t min_values) {
    return n_values >= min_values;
}

inline bool mapdiag_has_enough_off_source_values(
    const std::vector<double> &values) {
    return mapdiag_has_minimum_samples(values.size(), 8);
}

inline std::vector<double> mapdiag_absolute_deviations(
    const std::vector<double> &values, double center) {
    std::vector<double> abs_dev;
    abs_dev.reserve(values.size());
    for (const auto &value : values) {
        abs_dev.push_back(std::abs(value - center));
    }
    return abs_dev;
}

inline double mapdiag_robust_sigma(
    const MapdiagStatsContext &stats, const std::vector<double> &values,
    double center) {
    return 1.4826 * stats.median(mapdiag_absolute_deviations(values, center));
}

inline MapdiagRobustCenterStats mapdiag_robust_center_stats(
    const MapdiagStatsContext &stats, const std::vector<double> &values) {
    const double center = stats.median(values);
    return {center, mapdiag_robust_sigma(stats, values, center)};
}

inline bool mapdiag_has_valid_robust_center_stats(
    const MapdiagRobustCenterStats &stats) {
    return std::isfinite(stats.center) && std::isfinite(stats.robust_sigma) &&
           stats.robust_sigma > std::numeric_limits<double>::epsilon();
}

inline bool mapdiag_is_valid_outlier_pixel_value(double value, double weight,
                                                 double sig2noise) {
    return std::isfinite(value) && std::isfinite(weight) && weight > 0.0 &&
           std::isfinite(sig2noise);
}

inline bool mapdiag_passes_min_effective_samples(double n_eff,
                                                 double min_n_eff) {
    return !std::isfinite(n_eff) || n_eff >= min_n_eff;
}

inline double mapdiag_robust_z(double sig2noise,
                               const MapdiagRobustCenterStats &stats) {
    return (sig2noise - stats.center) / stats.robust_sigma;
}

inline bool mapdiag_passes_min_abs_z(double z, double min_abs_z) {
    return std::isfinite(z) && std::abs(z) >= min_abs_z;
}

inline MapdiagMapPixelCandidate make_mapdiag_map_pixel_candidate(
    Eigen::Index row, Eigen::Index col, double value, double weight,
    double n_eff, double robust_z, double source_distance_arcsec,
    int fill_int, double fill_double) {
    return {static_cast<int>(row),
            static_cast<int>(col),
            fill_int,
            fill_int,
            fill_int,
            value,
            weight,
            n_eff,
            robust_z,
            robust_z,
            source_distance_arcsec,
            false,
            false};
}

inline std::vector<MapdiagMapPixelCandidate> make_mapdiag_pixel_candidates() {
    return {};
}

inline void append_mapdiag_pixel_candidate(
    std::vector<MapdiagMapPixelCandidate> &candidates,
    const MapdiagMapPixelCandidate &candidate) {
    candidates.push_back(candidate);
}

inline const MapdiagMapPixelCandidate &mapdiag_emitted_candidate(
    const std::vector<MapdiagMapPixelCandidate> &candidates,
    std::size_t index) {
    return candidates[index];
}

inline void assign_mapdiag_candidate_contributor(
    MapdiagMapPixelCandidate &candidate, int uid, int scan,
    long long sample) {
    candidate.has_contributor = true;
    candidate.uid = uid;
    candidate.scan = scan;
    candidate.sample = sample;
}

template <class ScanMatrix, class SampleMatrix>
void assign_mapdiag_candidate_contributor_from_products(
    MapdiagMapPixelCandidate &candidate, int uid,
    const ScanMatrix &scan_matrix, const SampleMatrix &sample_matrix,
    Eigen::Index row, Eigen::Index col) {
    assign_mapdiag_candidate_contributor(
        candidate, uid, mapdiag_matrix_value(scan_matrix, row, col),
        mapdiag_matrix_value(sample_matrix, row, col));
}

inline bool mapdiag_candidate_abs_z_greater(
    const MapdiagMapPixelCandidate &a,
    const MapdiagMapPixelCandidate &b) {
    return std::abs(a.robust_z) > std::abs(b.robust_z);
}

inline void sort_mapdiag_pixel_candidates(
    std::vector<MapdiagMapPixelCandidate> &candidates) {
    std::sort(
        candidates.begin(), candidates.end(),
        mapdiag_candidate_abs_z_greater);
}

inline std::size_t mapdiag_candidate_emit_count(std::size_t n_candidates,
                                                int top_n) {
    return std::min<std::size_t>(
        n_candidates, static_cast<std::size_t>(top_n));
}

