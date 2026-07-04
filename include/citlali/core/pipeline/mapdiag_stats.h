#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <tula/algorithm/ei_stats.h>

namespace citlali::pipeline {

struct MapdiagTailStats {
    double frac_abs3 = std::numeric_limits<double>::quiet_NaN();
    double frac_pos3 = std::numeric_limits<double>::quiet_NaN();
    double frac_neg3 = std::numeric_limits<double>::quiet_NaN();
    double excess_abs3 = std::numeric_limits<double>::quiet_NaN();
    double excess_pos3 = std::numeric_limits<double>::quiet_NaN();
    double excess_neg3 = std::numeric_limits<double>::quiet_NaN();
    double skew = std::numeric_limits<double>::quiet_NaN();
};

struct MapdiagStatsContext {
    double fill_value;

    double median(const std::vector<double> &values) const;
    double quantile(std::vector<double> values, double q) const;
    std::vector<double> collect_masked_values(
        const Eigen::MatrixXd &matrix, const Eigen::ArrayXXd &mask) const;
    MapdiagTailStats tail_stats(const std::vector<double> &values) const;
};

inline void mapdiag_append_finite(std::vector<double> &values, double value);

struct MapdiagNoiseTailSamples {
    std::vector<double> rms;
    std::vector<double> tail_abs;
    std::vector<double> tail_pos;
    std::vector<double> tail_neg;
    std::vector<double> excess_abs;
    std::vector<double> excess_pos;
    std::vector<double> excess_neg;
    std::vector<double> skew;

    void reserve(std::size_t n_noise);
    void add_tail_stats(const MapdiagTailStats &stats);
};

struct MapdiagNoiseTailSummary {
    double rms_p16;
    double rms_p84;
    double tail_abs;
    double tail_pos;
    double tail_neg;
    double excess_abs;
    double excess_pos;
    double excess_neg;
    double skew;
};

struct MapdiagNoiseProductStats {
    double weight_median_ratio;
    double weight_scale;
    double s2n_sigma;
    double valid_pixels;
};

struct MapdiagWeightStats {
    int n_valid_pixels;
    int n_core_pixels;
    double weight_sum;
    double core_weight_sum;
};

struct MapdiagCoverageStats {
    double sum;
    double max;
    double median_core;
};

struct MapdiagPeakStats {
    double peak_abs_sig2noise;
    int peak_row;
    int peak_col;
    double core_peak_abs_sig2noise;
};

struct MapdiagPeakRefs {
    std::vector<double> &peak_abs_sig2noise;
    std::vector<double> &core_peak_abs_sig2noise;
    std::vector<int> &peak_row;
    std::vector<int> &peak_col;
};

struct MapdiagWeightRefs {
    std::vector<double> &weight_sum;
    std::vector<double> &core_weight_sum;
    std::vector<int> &n_valid_pixels;
    std::vector<int> &n_core_pixels;
};

struct MapdiagCoreTailRefs {
    std::vector<double> &frac_abs3;
    std::vector<double> &frac_pos3;
    std::vector<double> &frac_neg3;
    std::vector<double> &excess_abs3;
    std::vector<double> &excess_pos3;
    std::vector<double> &excess_neg3;
    std::vector<double> &skew;
};

inline void MapdiagNoiseTailSamples::reserve(std::size_t n_noise) {
    rms.reserve(n_noise);
    tail_abs.reserve(n_noise);
    tail_pos.reserve(n_noise);
    tail_neg.reserve(n_noise);
    excess_abs.reserve(n_noise);
    excess_pos.reserve(n_noise);
    excess_neg.reserve(n_noise);
    skew.reserve(n_noise);
}

inline void MapdiagNoiseTailSamples::add_tail_stats(
    const MapdiagTailStats &stats) {
    mapdiag_append_finite(tail_abs, stats.frac_abs3);
    mapdiag_append_finite(tail_pos, stats.frac_pos3);
    mapdiag_append_finite(tail_neg, stats.frac_neg3);
    mapdiag_append_finite(excess_abs, stats.excess_abs3);
    mapdiag_append_finite(excess_pos, stats.excess_pos3);
    mapdiag_append_finite(excess_neg, stats.excess_neg3);
    mapdiag_append_finite(skew, stats.skew);
}

inline MapdiagNoiseTailSummary summarize_mapdiag_noise_tail_samples(
    const MapdiagStatsContext &stats,
    const MapdiagNoiseTailSamples &samples) {
    return {stats.quantile(samples.rms, 0.16),
            stats.quantile(samples.rms, 0.84),
            stats.median(samples.tail_abs),
            stats.median(samples.tail_pos),
            stats.median(samples.tail_neg),
            stats.median(samples.excess_abs),
            stats.median(samples.excess_pos),
            stats.median(samples.excess_neg),
            stats.median(samples.skew)};
}

inline double mapdiag_vector_median(const std::vector<double> &values,
                                    double fill_value) {
    if (values.empty()) {
        return fill_value;
    }
    Eigen::Map<const Eigen::VectorXd> mapped(
        values.data(), static_cast<Eigen::Index>(values.size()));
    return tula::alg::median(mapped);
}

inline double MapdiagStatsContext::median(
    const std::vector<double> &values) const {
    return mapdiag_vector_median(values, fill_value);
}

inline double mapdiag_vector_quantile(std::vector<double> values, double q,
                                      double fill_value) {
    if (values.empty()) {
        return fill_value;
    }
    q = std::clamp(q, 0.0, 1.0);
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const std::size_t i0 = static_cast<std::size_t>(std::floor(pos));
    const std::size_t i1 = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(i0);
    return values[i0] * (1.0 - frac) + values[i1] * frac;
}

inline double MapdiagStatsContext::quantile(
    std::vector<double> values, double q) const {
    return mapdiag_vector_quantile(std::move(values), q, fill_value);
}

inline std::vector<double> mapdiag_collect_masked_values(
    const Eigen::MatrixXd &matrix, const Eigen::ArrayXXd &mask) {
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(mask.sum()));
    for (Eigen::Index r=0; r<matrix.rows(); ++r) {
        for (Eigen::Index c=0; c<matrix.cols(); ++c) {
            const double value = matrix(r, c);
            if (mask(r, c) > 0.0 && std::isfinite(value)) {
                values.push_back(value);
            }
        }
    }
    return values;
}

inline std::vector<double> MapdiagStatsContext::collect_masked_values(
    const Eigen::MatrixXd &matrix, const Eigen::ArrayXXd &mask) const {
    return mapdiag_collect_masked_values(matrix, mask);
}

inline double mapdiag_masked_median(const Eigen::MatrixXd &matrix,
                                    const Eigen::ArrayXXd &mask,
                                    double fill_value) {
    return mapdiag_vector_median(
        mapdiag_collect_masked_values(matrix, mask), fill_value);
}

inline double mapdiag_positive_sqrt_or_fill(double value, double fill_value) {
    if (std::isfinite(value) && value > std::numeric_limits<double>::epsilon()) {
        return std::sqrt(value);
    }
    return fill_value;
}

inline double mapdiag_positive_denominator_ratio_or_fill(double numerator,
                                                         double denominator,
                                                         double fill_value) {
    if (std::isfinite(numerator) && std::isfinite(denominator) &&
        denominator > std::numeric_limits<double>::epsilon()) {
        return numerator / denominator;
    }
    return fill_value;
}

inline double mapdiag_weight_threshold_or_zero(double weight_threshold) {
    if (std::isfinite(weight_threshold) && weight_threshold >= 0.0) {
        return weight_threshold;
    }
    return 0.0;
}

inline Eigen::ArrayXXd mapdiag_valid_weight_mask(
    const Eigen::ArrayXXd &weight) {
    return (weight > 0.0).template cast<double>();
}

inline Eigen::ArrayXXd mapdiag_core_weight_mask(
    const Eigen::ArrayXXd &weight, double weight_threshold) {
    return ((weight >= weight_threshold) && (weight > 0.0))
        .template cast<double>();
}

inline Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>
mapdiag_positive_mask(const Eigen::ArrayXXd &mask) {
    return mask > 0.0;
}

template <class Mask>
double mapdiag_mask_count_as_double(const Mask &mask) {
    return static_cast<double>(mask.count());
}

template <class Mask>
int mapdiag_mask_sum_as_int(const Mask &mask) {
    return static_cast<int>(mask.sum());
}

template <class Values, class Mask>
double mapdiag_weighted_mask_sum(const Values &values, const Mask &mask) {
    return (values * mask).sum();
}

template <class Values, class ValidMask, class CoreMask>
MapdiagWeightStats mapdiag_weight_stats(const Values &weight,
                                        const ValidMask &valid_mask,
                                        const CoreMask &core_mask) {
    return {mapdiag_mask_sum_as_int(valid_mask),
            mapdiag_mask_sum_as_int(core_mask),
            mapdiag_weighted_mask_sum(weight, valid_mask),
            mapdiag_weighted_mask_sum(weight, core_mask)};
}

inline bool mapdiag_has_matrix_samples(const Eigen::MatrixXd &matrix) {
    return matrix.size() > 0;
}

inline Eigen::MatrixXd mapdiag_sig2noise_image(
    const Eigen::MatrixXd &signal, const Eigen::MatrixXd &weight) {
    return signal.array() * weight.array().max(0.0).sqrt();
}

inline double mapdiag_peak_signal_or_fill(const Eigen::MatrixXd &signal,
                                          double fill_value) {
    return mapdiag_has_matrix_samples(signal) ? signal.maxCoeff() : fill_value;
}

inline double mapdiag_core_peak_abs_or_fill(const Eigen::MatrixXd &sig2noise,
                                            const Eigen::ArrayXXd &core_mask,
                                            int n_core_pixels,
                                            double fill_value) {
    if (n_core_pixels <= 0) {
        return fill_value;
    }
    const Eigen::MatrixXd core_sig2noise =
        (sig2noise.cwiseAbs().array() * core_mask).matrix();
    return core_sig2noise.maxCoeff();
}

inline MapdiagPeakStats mapdiag_peak_stats(
    const Eigen::MatrixXd &sig2noise, const Eigen::ArrayXXd &core_mask,
    int n_core_pixels, double fill_value) {
    Eigen::Index r_peak = 0;
    Eigen::Index c_peak = 0;
    const double peak_abs_sig2noise =
        sig2noise.cwiseAbs().maxCoeff(&r_peak, &c_peak);
    return {peak_abs_sig2noise,
            static_cast<int>(r_peak),
            static_cast<int>(c_peak),
            mapdiag_core_peak_abs_or_fill(
                sig2noise, core_mask, n_core_pixels, fill_value)};
}

inline void assign_mapdiag_peak_stats(std::size_t idx,
                                      const MapdiagPeakStats &stats,
                                      MapdiagPeakRefs refs) {
    refs.peak_abs_sig2noise[idx] = stats.peak_abs_sig2noise;
    refs.peak_row[idx] = stats.peak_row;
    refs.peak_col[idx] = stats.peak_col;
    refs.core_peak_abs_sig2noise[idx] = stats.core_peak_abs_sig2noise;
}

inline void assign_mapdiag_weight_stats(std::size_t idx,
                                        const MapdiagWeightStats &stats,
                                        MapdiagWeightRefs refs) {
    refs.n_valid_pixels[idx] = stats.n_valid_pixels;
    refs.n_core_pixels[idx] = stats.n_core_pixels;
    refs.weight_sum[idx] = stats.weight_sum;
    refs.core_weight_sum[idx] = stats.core_weight_sum;
}

inline void assign_mapdiag_core_tail_stats(
    std::size_t idx, const MapdiagTailStats &stats,
    MapdiagCoreTailRefs refs) {
    refs.frac_abs3[idx] = stats.frac_abs3;
    refs.frac_pos3[idx] = stats.frac_pos3;
    refs.frac_neg3[idx] = stats.frac_neg3;
    refs.excess_abs3[idx] = stats.excess_abs3;
    refs.excess_pos3[idx] = stats.excess_pos3;
    refs.excess_neg3[idx] = stats.excess_neg3;
    refs.skew[idx] = stats.skew;
}

inline void mapdiag_append_finite(std::vector<double> &values, double value) {
    if (std::isfinite(value)) {
        values.push_back(value);
    }
}

template <class Values>
bool mapdiag_has_value(const Values &values, Eigen::Index i) {
    return i >= 0 && i < static_cast<Eigen::Index>(values.size());
}

template <class Values>
double mapdiag_value_or_fill(const Values &values, Eigen::Index i,
                             double fill_value) {
    if (mapdiag_has_value(values, i)) {
        return values(i);
    }
    return fill_value;
}

template <class Values>
double mapdiag_positive_sqrt_value_or_fill(const Values &values,
                                           Eigen::Index i,
                                           double fill_value) {
    if (mapdiag_has_value(values, i)) {
        return mapdiag_positive_sqrt_or_fill(values(i), fill_value);
    }
    return fill_value;
}

template <class Values>
double mapdiag_finite_value_or_fill(const Values &values, Eigen::Index i,
                                    double fill_value) {
    if (mapdiag_has_value(values, i) && std::isfinite(values(i))) {
        return values(i);
    }
    return fill_value;
}

template <class Values>
MapdiagNoiseProductStats mapdiag_noise_product_stats_or_fill(
    const Values &weight_median_ratio, const Values &weight_scale,
    const Values &s2n_sigma, const Values &valid_pixels, Eigen::Index i,
    double fill_value) {
    return {mapdiag_value_or_fill(weight_median_ratio, i, fill_value),
            mapdiag_value_or_fill(weight_scale, i, fill_value),
            mapdiag_value_or_fill(s2n_sigma, i, fill_value),
            mapdiag_value_or_fill(valid_pixels, i, fill_value)};
}

template <class CoverageList>
bool mapdiag_has_coverage_map(const CoverageList &coverage, Eigen::Index i) {
    return !coverage.empty() && i >= 0 &&
           i < static_cast<Eigen::Index>(coverage.size());
}

template <class NoiseList>
bool mapdiag_has_noise_realizations(
    const NoiseList &noise, Eigen::Index i, Eigen::Index n_noise) {
    return !noise.empty() && i >= 0 &&
           i < static_cast<Eigen::Index>(noise.size()) && n_noise > 0;
}

inline Eigen::Index mapdiag_noise_realization_size(Eigen::Index n_rows,
                                                   Eigen::Index n_cols) {
    return n_rows * n_cols;
}

inline Eigen::Index mapdiag_noise_realization_offset(
    Eigen::Index realization_index, Eigen::Index n_rows, Eigen::Index n_cols) {
    return realization_index * mapdiag_noise_realization_size(n_rows, n_cols);
}

inline void assign_mapdiag_coverage_stats(
    const Eigen::MatrixXd &coverage, const Eigen::ArrayXXd &core_mask,
    double fill_value, double &coverage_sum, double &coverage_max,
    double &coverage_median_core) {
    coverage_sum = coverage.sum();
    coverage_max = coverage.maxCoeff();
    coverage_median_core =
        mapdiag_masked_median(coverage, core_mask, fill_value);
}

template <class Matrix, class Mask>
double mapdiag_core_noise_rms(const Matrix &noise_matrix,
                              const Mask &valid_core,
                              double valid_core_count) {
    const double rms_sq =
        (valid_core.select(noise_matrix.array().square(), 0.0)).sum();
    return std::sqrt(rms_sq / valid_core_count);
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

}  // namespace citlali::pipeline
