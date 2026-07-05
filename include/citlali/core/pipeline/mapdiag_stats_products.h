#pragma once

// Included by mapdiag_stats.h inside namespace citlali::pipeline.

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

template <class MapBuffer>
void assign_mapdiag_formal_noise_stats_or_fill(
    std::size_t idx, const MapBuffer &mb, Eigen::Index map_index,
    double fill_value, MapdiagFormalNoiseRefs refs) {
    assign_mapdiag_formal_noise_stats(
        idx,
        mapdiag_formal_noise_stats_or_fill(
            mb->median_err, mb->median_rms, map_index, fill_value),
        refs);
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

template <class MapBuffer>
void assign_mapdiag_noise_product_stats_or_fill(
    std::size_t idx, const MapBuffer &mb, Eigen::Index map_index,
    double fill_value, MapdiagNoiseProductRefs refs) {
    assign_mapdiag_noise_product_stats(
        idx,
        mapdiag_noise_product_stats_or_fill(
            mb->noise_weight_median_ratio, mb->noise_weight_scale,
            mb->noise_s2n_sigma, mb->noise_valid_pixels, map_index,
            fill_value),
        refs);
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

template <class MapBuffer>
Eigen::Index mapdiag_n_rows(const MapBuffer &mb) {
    return mb->n_rows;
}

template <class MapBuffer>
Eigen::Index mapdiag_n_cols(const MapBuffer &mb) {
    return mb->n_cols;
}

template <class MapBuffer>
Eigen::Index mapdiag_noise_realization_count(const MapBuffer &mb) {
    return mb->n_noise;
}

