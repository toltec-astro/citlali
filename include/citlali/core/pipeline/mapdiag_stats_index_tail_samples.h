#pragma once

// Included by mapdiag_stats_core.h inside namespace citlali::pipeline.

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

