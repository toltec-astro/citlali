#pragma once

// Included by mapdiag_stats.h inside namespace citlali::pipeline.

inline MapdiagCoverageStats mapdiag_coverage_stats(
    const Eigen::MatrixXd &coverage, const Eigen::ArrayXXd &core_mask,
    double fill_value) {
    return {coverage.sum(),
            coverage.maxCoeff(),
            mapdiag_masked_median(coverage, core_mask, fill_value)};
}

inline void assign_mapdiag_coverage_stats(
    std::size_t idx, const MapdiagCoverageStats &stats,
    MapdiagCoverageRefs refs) {
    refs.coverage_sum[idx] = stats.sum;
    refs.coverage_max[idx] = stats.max;
    refs.coverage_median_core[idx] = stats.median_core;
}

inline void assign_mapdiag_coverage_stats(
    std::size_t idx, const Eigen::MatrixXd &coverage,
    const Eigen::ArrayXXd &core_mask, double fill_value,
    MapdiagCoverageRefs refs) {
    assign_mapdiag_coverage_stats(
        idx, mapdiag_coverage_stats(coverage, core_mask, fill_value), refs);
}

template <class CoverageList, class CoreMask>
void assign_mapdiag_coverage_stats_if_present(
    std::size_t idx, const CoverageList &coverage, Eigen::Index map_index,
    const CoreMask &core_mask, double fill_value,
    MapdiagCoverageRefs refs) {
    if (!mapdiag_has_coverage_map(coverage, map_index)) {
        return;
    }
    assign_mapdiag_coverage_stats(
        idx, coverage[map_index], core_mask, fill_value, refs);
}

inline void assign_mapdiag_coverage_stats(
    const Eigen::MatrixXd &coverage, const Eigen::ArrayXXd &core_mask,
    double fill_value, double &coverage_sum, double &coverage_max,
    double &coverage_median_core) {
    const auto stats =
        mapdiag_coverage_stats(coverage, core_mask, fill_value);
    coverage_sum = stats.sum;
    coverage_max = stats.max;
    coverage_median_core = stats.median_core;
}

template <class Matrix, class Mask>
double mapdiag_core_noise_rms(const Matrix &noise_matrix,
                              const Mask &valid_core,
                              double valid_core_count) {
    const double rms_sq =
        (valid_core.select(noise_matrix.array().square(), 0.0)).sum();
    return std::sqrt(rms_sq / valid_core_count);
}

template <class Matrix, class Mask>
void add_mapdiag_core_noise_rms_sample(
    MapdiagNoiseTailSamples &samples, const Matrix &noise_matrix,
    const Mask &valid_core, double valid_core_count) {
    if (mapdiag_has_positive_count(valid_core_count)) {
        samples.rms.push_back(
            mapdiag_core_noise_rms(noise_matrix, valid_core, valid_core_count));
    }
}

template <class Matrix>
void add_mapdiag_noise_tail_sample(
    MapdiagNoiseTailSamples &samples, const MapdiagStatsContext &stats,
    const Matrix &noise_matrix, const Eigen::ArrayXXd &core_mask) {
    const auto noise_values =
        stats.collect_masked_values(noise_matrix, core_mask);
    samples.add_tail_stats(stats.tail_stats(noise_values));
}

template <class Matrix, class Mask>
void add_mapdiag_noise_realization_samples(
    MapdiagNoiseTailSamples &samples, const MapdiagStatsContext &stats,
    const Matrix &noise_matrix, const Mask &valid_core,
    double valid_core_count, const Eigen::ArrayXXd &core_mask) {
    add_mapdiag_core_noise_rms_sample(
        samples, noise_matrix, valid_core, valid_core_count);
    add_mapdiag_noise_tail_sample(
        samples, stats, noise_matrix, core_mask);
}

template <class MapBuffer, class CoreMask>
void assign_mapdiag_noise_tail_for_map(
    std::size_t idx, const MapBuffer &mb, Eigen::Index map_index,
    const MapdiagStatsContext &stats, const CoreMask &core_mask,
    MapdiagNoiseTailRefs refs) {
    if (!mapdiag_has_noise_realizations(
            mb->noise, map_index, mb->n_noise)) {
        return;
    }

    auto noise_samples = make_mapdiag_noise_tail_samples(mb);
    const auto valid_core = mapdiag_valid_core_noise_mask(core_mask);
    const double valid_core_count =
        mapdiag_valid_core_noise_count(valid_core);
    const Eigen::Index n_noise_realizations =
        mapdiag_noise_realization_count(mb);
    for (Eigen::Index n = 0; n < n_noise_realizations; ++n) {
        const auto noise_matrix = mapdiag_noise_matrix(mb, map_index, n);
        add_mapdiag_noise_realization_samples(
            noise_samples, stats, noise_matrix, valid_core,
            valid_core_count, core_mask);
    }
    assign_mapdiag_noise_tail_samples(idx, stats, noise_samples, refs);
}

inline MapdiagTailStats mapdiag_tail_stats(const std::vector<double> &values,
                                           double fill_value) {
    MapdiagTailStats stats;
    if (values.size() < 8) {
        return stats;
    }
    const double center = mapdiag_vector_median(values, fill_value);
    if (!std::isfinite(center)) {
        return stats;
    }
    std::vector<double> abs_dev;
    abs_dev.reserve(values.size());
    for (const auto &value : values) {
        abs_dev.push_back(std::abs(value - center));
    }
    const double mad = mapdiag_vector_median(abs_dev, fill_value);
    const double robust_sigma = 1.4826 * mad;
    if (!std::isfinite(robust_sigma) ||
        robust_sigma <= std::numeric_limits<double>::epsilon()) {
        return stats;
    }

    std::size_t n_abs = 0;
    std::size_t n_pos = 0;
    std::size_t n_neg = 0;
    double skew_sum = 0.0;
    for (const auto &value : values) {
        const double z = (value - center) / robust_sigma;
        if (!std::isfinite(z)) {
            continue;
        }
        if (std::abs(z) >= 3.0) {
            ++n_abs;
        }
        if (z >= 3.0) {
            ++n_pos;
        }
        if (z <= -3.0) {
            ++n_neg;
        }
        skew_sum += z * z * z;
    }

    const double n = static_cast<double>(values.size());
    stats.frac_abs3 = static_cast<double>(n_abs) / n;
    stats.frac_pos3 = static_cast<double>(n_pos) / n;
    stats.frac_neg3 = static_cast<double>(n_neg) / n;
    constexpr double gauss_pos3 = 1.3498980316300959e-3;
    constexpr double gauss_abs3 = 2.6997960632601918e-3;
    stats.excess_abs3 = stats.frac_abs3 / gauss_abs3;
    stats.excess_pos3 = stats.frac_pos3 / gauss_pos3;
    stats.excess_neg3 = stats.frac_neg3 / gauss_pos3;
    stats.skew = skew_sum / n;
    return stats;
}

inline MapdiagTailStats MapdiagStatsContext::tail_stats(
    const std::vector<double> &values) const {
    return mapdiag_tail_stats(values, fill_value);
}

