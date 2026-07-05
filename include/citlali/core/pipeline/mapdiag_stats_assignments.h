#pragma once

// Included by mapdiag_stats_core.h inside namespace citlali::pipeline.

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

