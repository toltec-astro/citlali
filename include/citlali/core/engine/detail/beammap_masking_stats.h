#pragma once

// Beammap masking robust-statistics helpers.

namespace beammap_masking_stats {

struct RobustStats {
    double median = std::numeric_limits<double>::quiet_NaN();
    double sigma = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
};

inline RobustStats robust_stats(const std::vector<double> &values) {
    RobustStats stats;
    if (values.empty()) {
        return stats;
    }
    Eigen::Map<const Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
    stats.median = tula::alg::median(vec);
    Eigen::VectorXd abs_dev = (vec.array() - stats.median).abs().matrix();
    stats.sigma = 1.4826 * tula::alg::median(abs_dev);
    if (!std::isfinite(stats.sigma)) {
        stats.sigma = std::numeric_limits<double>::quiet_NaN();
    }
    stats.valid = std::isfinite(stats.median);
    return stats;
}

inline bool row_is_bad(double row_median,
                       double row_sigma,
                       double interior_median,
                       double interior_median_sigma,
                       double interior_row_sigma_median,
                       double median_sigma_threshold,
                       double sigma_ratio_threshold,
                       double eps) {
    bool bad = false;
    if (median_sigma_threshold > 0.0 &&
        std::isfinite(row_median) &&
        std::isfinite(interior_median) &&
        std::isfinite(interior_median_sigma) &&
        interior_median_sigma > eps) {
        bad = std::abs(row_median - interior_median) > median_sigma_threshold * interior_median_sigma;
    }
    if (!bad &&
        sigma_ratio_threshold > 0.0 &&
        std::isfinite(row_sigma) &&
        std::isfinite(interior_row_sigma_median) &&
        interior_row_sigma_median > eps) {
        bad = row_sigma > sigma_ratio_threshold * interior_row_sigma_median;
    }
    return bad;
}

inline void dilate_block_mask(std::vector<unsigned char> &bad_blocks,
                              int dilate_blocks) {
    if (dilate_blocks <= 0 || bad_blocks.empty()) {
        return;
    }
    const Eigen::Index n_blocks = static_cast<Eigen::Index>(bad_blocks.size());
    std::vector<unsigned char> dilated_blocks(bad_blocks.size(), 0);
    for (Eigen::Index b = 0; b < n_blocks; ++b) {
        if (!bad_blocks[static_cast<std::size_t>(b)]) {
            continue;
        }
        const Eigen::Index b0 = std::max<Eigen::Index>(0, b - dilate_blocks);
        const Eigen::Index b1 = std::min<Eigen::Index>(n_blocks - 1, b + dilate_blocks);
        for (Eigen::Index bb = b0; bb <= b1; ++bb) {
            dilated_blocks[static_cast<std::size_t>(bb)] = 1;
        }
    }
    bad_blocks.swap(dilated_blocks);
}

} // namespace beammap_masking_stats
