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

struct MapdiagFormalNoiseStats {
    double median_err;
    double median_rms;
    double empirical_to_formal_ratio;
};

struct MapdiagSourceDistanceContext {
    double center_row;
    double center_col;
    double pixel_size_arcsec;
    double fill_value;
};

struct MapdiagRobustCenterStats {
    double center;
    double robust_sigma;
};

struct MapdiagMapPixelCandidate {
    int row;
    int col;
    int uid;
    int scan;
    long long sample;
    double value;
    double weight;
    double n_eff;
    double robust_z;
    double leave_one_out_z;
    double source_distance_arcsec;
    bool source_protected;
    bool has_contributor;
};

struct MapdiagDetectorDominance {
    int uid;
    int scan;
    int count;
    double max_abs_value;
    double max_abs_leave_one_out_z;
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

struct MapdiagCoverageRefs {
    std::vector<double> &coverage_sum;
    std::vector<double> &coverage_max;
    std::vector<double> &coverage_median_core;
};

struct MapdiagNoiseProductRefs {
    std::vector<double> &weight_median_ratio;
    std::vector<double> &weight_scale;
    std::vector<double> &s2n_sigma;
    std::vector<double> &valid_pixels;
};

struct MapdiagFormalNoiseRefs {
    std::vector<double> &median_err;
    std::vector<double> &median_rms;
    std::vector<double> &empirical_to_formal_ratio;
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

struct MapdiagNoiseTailRefs {
    std::vector<double> &rms_p16;
    std::vector<double> &rms_p84;
    std::vector<double> &frac_abs3;
    std::vector<double> &frac_pos3;
    std::vector<double> &frac_neg3;
    std::vector<double> &excess_abs3;
    std::vector<double> &excess_pos3;
    std::vector<double> &excess_neg3;
    std::vector<double> &skew;
};

using MapdiagNoiseMatrix =
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;

inline std::size_t mapdiag_size_index(Eigen::Index map_index) {
    return static_cast<std::size_t>(map_index);
}

inline std::size_t mapdiag_contribution_map_index(Eigen::Index map_index) {
    return mapdiag_size_index(map_index);
}

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

inline MapdiagNoiseTailSamples make_mapdiag_noise_tail_samples(
    std::size_t n_noise) {
    MapdiagNoiseTailSamples samples;
    samples.reserve(n_noise);
    return samples;
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

inline Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>
mapdiag_valid_core_noise_mask(const Eigen::ArrayXXd &core_mask) {
    return mapdiag_positive_mask(core_mask);
}

template <class Mask>
double mapdiag_mask_count_as_double(const Mask &mask) {
    return static_cast<double>(mask.count());
}

template <class Mask>
double mapdiag_valid_core_noise_count(const Mask &valid_core_mask) {
    return mapdiag_mask_count_as_double(valid_core_mask);
}

inline bool mapdiag_has_positive_count(double count) {
    return count > 0.0;
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

inline void assign_mapdiag_noise_product_stats(
    std::size_t idx, const MapdiagNoiseProductStats &stats,
    MapdiagNoiseProductRefs refs) {
    refs.weight_median_ratio[idx] = stats.weight_median_ratio;
    refs.weight_scale[idx] = stats.weight_scale;
    refs.s2n_sigma[idx] = stats.s2n_sigma;
    refs.valid_pixels[idx] = stats.valid_pixels;
}

inline void assign_mapdiag_formal_noise_stats(
    std::size_t idx, const MapdiagFormalNoiseStats &stats,
    MapdiagFormalNoiseRefs refs) {
    refs.median_err[idx] = stats.median_err;
    refs.median_rms[idx] = stats.median_rms;
    refs.empirical_to_formal_ratio[idx] =
        stats.empirical_to_formal_ratio;
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

inline void assign_mapdiag_noise_tail_summary(
    std::size_t idx, const MapdiagNoiseTailSummary &summary,
    MapdiagNoiseTailRefs refs) {
    refs.rms_p16[idx] = summary.rms_p16;
    refs.rms_p84[idx] = summary.rms_p84;
    refs.frac_abs3[idx] = summary.tail_abs;
    refs.frac_pos3[idx] = summary.tail_pos;
    refs.frac_neg3[idx] = summary.tail_neg;
    refs.excess_abs3[idx] = summary.excess_abs;
    refs.excess_pos3[idx] = summary.excess_pos;
    refs.excess_neg3[idx] = summary.excess_neg;
    refs.skew[idx] = summary.skew;
}

inline void assign_mapdiag_noise_tail_samples(
    std::size_t idx, const MapdiagStatsContext &stats,
    const MapdiagNoiseTailSamples &samples, MapdiagNoiseTailRefs refs) {
    assign_mapdiag_noise_tail_summary(
        idx, summarize_mapdiag_noise_tail_samples(stats, samples), refs);
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

template <class MedianErrValues, class MedianRmsValues>
MapdiagFormalNoiseStats mapdiag_formal_noise_stats_or_fill(
    const MedianErrValues &median_err_values,
    const MedianRmsValues &median_rms_values, Eigen::Index i,
    double fill_value) {
    const double median_err = mapdiag_positive_sqrt_value_or_fill(
        median_err_values, i, fill_value);
    const double median_rms =
        mapdiag_finite_value_or_fill(median_rms_values, i, fill_value);
    return {median_err,
            median_rms,
            mapdiag_positive_denominator_ratio_or_fill(
                median_rms, median_err, fill_value)};
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

template <class CoverageList>
double mapdiag_effective_samples_or_fill(
    const CoverageList &coverage, Eigen::Index i, Eigen::Index row,
    Eigen::Index col, Eigen::Index n_rows, Eigen::Index n_cols,
    double ptc_fs_hz, double fill_value) {
    if (mapdiag_has_coverage_map(coverage, i) &&
        coverage[i].rows() == n_rows && coverage[i].cols() == n_cols &&
        std::isfinite(coverage[i](row, col)) && std::isfinite(ptc_fs_hz) &&
        ptc_fs_hz > 0.0) {
        return coverage[i](row, col) * ptc_fs_hz;
    }
    return fill_value;
}

template <class MapBuffer>
bool mapdiag_has_contribution_products(const MapBuffer &mb, Eigen::Index i) {
    return i < static_cast<Eigen::Index>(mb->contribution_uid.size()) &&
           i < static_cast<Eigen::Index>(mb->contribution_signal.size()) &&
           i < static_cast<Eigen::Index>(mb->contribution_weight.size()) &&
           i < static_cast<Eigen::Index>(
                   mb->contribution_variance_weight.size()) &&
           i < static_cast<Eigen::Index>(
                   mb->contribution_total_signal.size()) &&
           i < static_cast<Eigen::Index>(
                   mb->contribution_total_weight.size()) &&
           i < static_cast<Eigen::Index>(
                   mb->contribution_total_variance_weight.size()) &&
           i < static_cast<Eigen::Index>(mb->contribution_scan.size()) &&
           i < static_cast<Eigen::Index>(mb->contribution_sample.size()) &&
           mb->contribution_uid[static_cast<std::size_t>(i)].rows() ==
               mb->n_rows &&
           mb->contribution_uid[static_cast<std::size_t>(i)].cols() ==
               mb->n_cols;
}

inline bool mapdiag_has_valid_contributor(int uid, int fill_int,
                                          double contribution_signal) {
    return uid != fill_int && std::isfinite(contribution_signal);
}

inline bool mapdiag_has_full_leave_one_out_inputs(
    double total_signal, double total_weight, double contribution_weight,
    double contribution_variance_weight, double total_variance_weight,
    double remaining_weight) {
    return std::isfinite(total_signal) && std::isfinite(total_weight) &&
           std::isfinite(contribution_weight) &&
           std::isfinite(contribution_variance_weight) &&
           std::isfinite(total_variance_weight) &&
           contribution_weight >= 0.0 && contribution_variance_weight >= 0.0 &&
           remaining_weight > std::numeric_limits<double>::epsilon() &&
           total_variance_weight > contribution_variance_weight;
}

inline double mapdiag_remaining_contribution_weight(double total_weight,
                                                    double contribution_weight) {
    return total_weight - contribution_weight;
}

inline double mapdiag_full_leave_one_out_value(double total_signal,
                                               double contribution_signal,
                                               double remaining_weight) {
    return (total_signal - contribution_signal) / remaining_weight;
}

inline void mapdiag_assign_leave_one_out_z(double value, double weight,
                                           double leave_one_out_value,
                                           double &leave_one_out_z) {
    const double residual = value - leave_one_out_value;
    if (std::isfinite(residual) && std::isfinite(weight) && weight > 0.0) {
        leave_one_out_z = residual * std::sqrt(weight);
    }
}

inline bool mapdiag_has_fallback_leave_one_out_inputs(
    double weight, double contribution_weight) {
    return std::isfinite(contribution_weight) && contribution_weight >= 0.0 &&
           weight > contribution_weight &&
           (weight - contribution_weight) >
               std::numeric_limits<double>::epsilon();
}

inline double mapdiag_raw_weighted_signal(double value, double weight) {
    return value * weight;
}

inline double mapdiag_fallback_leave_one_out_value(
    double raw_weighted_signal, double contribution_signal, double weight,
    double contribution_weight) {
    return (raw_weighted_signal - contribution_signal) /
           (weight - contribution_weight);
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

inline MapdiagNoiseMatrix mapdiag_noise_matrix(
    double *noise_data, Eigen::Index realization_index, Eigen::Index n_rows,
    Eigen::Index n_cols) {
    return MapdiagNoiseMatrix(
        noise_data + mapdiag_noise_realization_offset(
                         realization_index, n_rows, n_cols),
        n_rows, n_cols);
}

inline double mapdiag_center_pixel_coordinate(Eigen::Index n_pixels) {
    return (static_cast<double>(n_pixels) - 1.0) / 2.0;
}

inline MapdiagSourceDistanceContext mapdiag_source_distance_context(
    Eigen::Index n_rows, Eigen::Index n_cols, double pixel_size_arcsec,
    double fill_value) {
    return {mapdiag_center_pixel_coordinate(n_rows),
            mapdiag_center_pixel_coordinate(n_cols),
            pixel_size_arcsec,
            fill_value};
}

inline double mapdiag_source_distance_arcsec(
    Eigen::Index row, Eigen::Index col, double center_row,
    double center_col, double pixel_size_arcsec, double fill_value) {
    if (!std::isfinite(pixel_size_arcsec) || pixel_size_arcsec <= 0.0) {
        return fill_value;
    }
    const double drow =
        (static_cast<double>(row) - center_row) * pixel_size_arcsec;
    const double dcol =
        (static_cast<double>(col) - center_col) * pixel_size_arcsec;
    return std::hypot(drow, dcol);
}

inline double mapdiag_source_distance_arcsec(
    Eigen::Index row, Eigen::Index col,
    const MapdiagSourceDistanceContext &context) {
    return mapdiag_source_distance_arcsec(
        row, col, context.center_row, context.center_col,
        context.pixel_size_arcsec, context.fill_value);
}

inline bool mapdiag_is_source_protected(double distance_arcsec,
                                        double protect_radius_arcsec) {
    return protect_radius_arcsec > 0.0 && std::isfinite(distance_arcsec) &&
           distance_arcsec <= protect_radius_arcsec;
}

inline void apply_mapdiag_source_protection_mask(
    Eigen::ArrayXXd &mask, const MapdiagSourceDistanceContext &context,
    double protect_radius_arcsec) {
    for (Eigen::Index r = 0; r < mask.rows(); ++r) {
        for (Eigen::Index c = 0; c < mask.cols(); ++c) {
            const double dist_arcsec =
                mapdiag_source_distance_arcsec(r, c, context);
            if (mapdiag_is_source_protected(
                    dist_arcsec, protect_radius_arcsec)) {
                mask(r, c) = 0.0;
            }
        }
    }
}

inline Eigen::ArrayXXd mapdiag_off_source_core_mask(
    const Eigen::ArrayXXd &core_mask,
    const MapdiagSourceDistanceContext &context,
    double protect_radius_arcsec) {
    Eigen::ArrayXXd off_source_mask = core_mask;
    apply_mapdiag_source_protection_mask(
        off_source_mask, context, protect_radius_arcsec);
    return off_source_mask;
}

inline bool mapdiag_has_minimum_samples(std::size_t n_values,
                                        std::size_t min_values) {
    return n_values >= min_values;
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

inline bool mapdiag_candidate_abs_z_greater(
    const MapdiagMapPixelCandidate &a,
    const MapdiagMapPixelCandidate &b) {
    return std::abs(a.robust_z) > std::abs(b.robust_z);
}

inline std::size_t mapdiag_candidate_emit_count(std::size_t n_candidates,
                                                int top_n) {
    return std::min<std::size_t>(
        n_candidates, static_cast<std::size_t>(top_n));
}

inline bool mapdiag_candidate_has_dominance_key(
    const MapdiagMapPixelCandidate &candidate, int fill_int) {
    return candidate.has_contributor && !candidate.source_protected &&
           candidate.uid != fill_int && candidate.scan != fill_int &&
           candidate.uid >= 0 && candidate.scan >= 0;
}

inline std::vector<MapdiagDetectorDominance>
make_mapdiag_detector_dominance_list() {
    return {};
}

inline bool mapdiag_dominance_matches_candidate(
    const MapdiagDetectorDominance &entry,
    const MapdiagMapPixelCandidate &candidate) {
    return entry.uid == candidate.uid && entry.scan == candidate.scan;
}

inline MapdiagDetectorDominance make_mapdiag_detector_dominance_entry(
    const MapdiagMapPixelCandidate &candidate) {
    return {candidate.uid, candidate.scan, 0, 0.0, 0.0};
}

inline void update_mapdiag_detector_dominance_stats(
    MapdiagDetectorDominance &entry,
    const MapdiagMapPixelCandidate &candidate) {
    ++entry.count;
    if (std::isfinite(candidate.value)) {
        entry.max_abs_value =
            std::max(entry.max_abs_value, std::abs(candidate.value));
    }
    if (std::isfinite(candidate.leave_one_out_z)) {
        entry.max_abs_leave_one_out_z =
            std::max(entry.max_abs_leave_one_out_z,
                     std::abs(candidate.leave_one_out_z));
    }
}

inline void update_mapdiag_detector_dominance(
    std::vector<MapdiagDetectorDominance> &dominance,
    const MapdiagMapPixelCandidate &candidate, int fill_int) {
    if (!mapdiag_candidate_has_dominance_key(candidate, fill_int)) {
        return;
    }
    auto it = std::find_if(
        dominance.begin(), dominance.end(),
        [&](const auto &entry) {
            return mapdiag_dominance_matches_candidate(entry, candidate);
        });
    if (it == dominance.end()) {
        dominance.push_back(make_mapdiag_detector_dominance_entry(candidate));
        it = dominance.end() - 1;
    }
    update_mapdiag_detector_dominance_stats(*it, candidate);
}

inline const char *mapdiag_map_pixel_outlier_reason(bool has_contributor,
                                                    bool targeted) {
    if (!has_contributor) {
        return "extreme_pixel_no_contributor";
    }
    return targeted ? "extreme_pixel_targeted_contributor"
                    : "extreme_pixel_contributor";
}

inline const char *mapdiag_detector_dominance_penalty_reason() {
    return "map_pixel_outlier_detector_dominance";
}

template <class Record, class Obsnum, class Producer, class Reason>
void assign_mapdiag_outlier_record_context(
    Record &record, const Obsnum &obsnum, const Producer &producer,
    const Reason &reason, int iter, int map_index) {
    record.obsnum = obsnum;
    record.producer = producer;
    record.reason = reason;
    record.iter = iter;
    record.map_index = map_index;
}

template <class Record>
void assign_mapdiag_outlier_record_candidate(
    Record &record, const MapdiagMapPixelCandidate &candidate) {
    record.scan = candidate.scan;
    record.uid = candidate.uid;
    record.row = candidate.row;
    record.col = candidate.col;
    record.sample = candidate.sample;
    record.value = candidate.value;
    record.weight = candidate.weight;
    record.n_eff = candidate.n_eff;
    record.leave_one_out_z = candidate.leave_one_out_z;
    record.source_distance_arcsec = candidate.source_distance_arcsec;
    record.source_protected = candidate.source_protected;
}

template <class Record, class Obsnum, class Producer, class Reason>
Record make_mapdiag_outlier_record(
    const Obsnum &obsnum, const Producer &producer, const Reason &reason,
    int iter, int map_index, const MapdiagMapPixelCandidate &candidate) {
    Record record;
    assign_mapdiag_outlier_record_context(
        record, obsnum, producer, reason, iter, map_index);
    assign_mapdiag_outlier_record_candidate(record, candidate);
    return record;
}

template <class Penalty, class Obsnum, class Producer, class Reason>
void assign_mapdiag_detector_penalty_context(
    Penalty &penalty, const Obsnum &obsnum, const Producer &producer,
    const Reason &reason, int iter, const MapdiagDetectorDominance &entry,
    int array_id) {
    penalty.obsnum = obsnum;
    penalty.producer = producer;
    penalty.reason = reason;
    penalty.iter = iter;
    penalty.scan = entry.scan;
    penalty.uid = entry.uid;
    penalty.nw = -1;
    penalty.array = array_id;
}

template <class Penalty>
void assign_mapdiag_detector_penalty_dominance(
    Penalty &penalty, const MapdiagDetectorDominance &entry) {
    penalty.factor = 0.0;
    penalty.score = static_cast<double>(entry.count);
    penalty.scan_local = true;
}

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
