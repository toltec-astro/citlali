#pragma once

// Included by mapdiag_stats.h inside namespace citlali::pipeline.

inline std::size_t mapdiag_size_index(Eigen::Index map_index) {
    return static_cast<std::size_t>(map_index);
}

inline std::size_t mapdiag_contribution_map_index(Eigen::Index map_index) {
    return mapdiag_size_index(map_index);
}

template <class StageName>
std::string mapdiag_record_producer(const StageName &stage_name) {
    return "mapdiag:" + stage_name;
}

inline int mapdiag_record_map_index(Eigen::Index map_index) {
    return static_cast<int>(map_index);
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

template <class MapBuffer>
MapdiagNoiseTailSamples make_mapdiag_noise_tail_samples(
    const MapBuffer &mb) {
    return make_mapdiag_noise_tail_samples(
        static_cast<std::size_t>(mb->n_noise));
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

template <class MapBuffer>
double mapdiag_weight_threshold_for_map(const MapBuffer &mb,
                                        Eigen::Index map_index) {
    const auto cov_region = mb->calc_cov_region(map_index);
    return mapdiag_weight_threshold_or_zero(std::get<0>(cov_region));
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

inline bool mapdiag_has_signal_weight_samples(
    const Eigen::MatrixXd &signal, const Eigen::MatrixXd &weight) {
    return mapdiag_has_matrix_samples(signal) &&
           mapdiag_has_matrix_samples(weight);
}

template <class Matrix>
auto mapdiag_matrix_value(const Matrix &matrix, Eigen::Index row,
                          Eigen::Index col) {
    return matrix(row, col);
}

template <class Matrix>
double mapdiag_matrix_double_value(const Matrix &matrix, Eigen::Index row,
                                   Eigen::Index col) {
    return static_cast<double>(mapdiag_matrix_value(matrix, row, col));
}

inline Eigen::MatrixXd mapdiag_sig2noise_image(
    const Eigen::MatrixXd &signal, const Eigen::MatrixXd &weight) {
    return signal.array() * weight.array().max(0.0).sqrt();
}

inline double mapdiag_peak_signal_or_fill(const Eigen::MatrixXd &signal,
                                          double fill_value) {
    return mapdiag_has_matrix_samples(signal) ? signal.maxCoeff() : fill_value;
}

template <class SignalList>
void assign_mapdiag_peak_signal_or_fill(
    std::size_t idx, const SignalList &signals, Eigen::Index map_index,
    double fill_value, std::vector<double> &peak_signal) {
    peak_signal[idx] = mapdiag_peak_signal_or_fill(
        signals[map_index], fill_value);
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

inline Eigen::MatrixXd assign_mapdiag_signal_stats(
    std::size_t idx, const Eigen::MatrixXd &signal,
    const Eigen::MatrixXd &weight, const Eigen::ArrayXXd &core_mask,
    int n_core_pixels, double fill_value,
    const MapdiagStatsContext &stats, MapdiagPeakRefs peak_refs,
    MapdiagCoreTailRefs core_tail_refs) {
    const Eigen::MatrixXd sig2noise =
        mapdiag_sig2noise_image(signal, weight);
    assign_mapdiag_peak_stats(
        idx, mapdiag_peak_stats(
                 sig2noise, core_mask, n_core_pixels, fill_value),
        peak_refs);
    const auto core_values = stats.collect_masked_values(
        sig2noise, core_mask);
    assign_mapdiag_core_tail_stats(
        idx, stats.tail_stats(core_values), core_tail_refs);
    return sig2noise;
}

