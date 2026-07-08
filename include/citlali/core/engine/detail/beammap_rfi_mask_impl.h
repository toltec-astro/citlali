#pragma once

// Beammap RFI sample masking implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_masking_stats.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

Beammap::RFIMaskScanSummary Beammap::apply_rfi_sample_mask(TCData<TCDataKind::PTC,Eigen::MatrixXd> &ptc) {
    RFIMaskScanSummary summary;
    const auto &rfi_config = typed_config.beammap.rfi_mask;
    if (!rfi_config.enabled) {
        return summary;
    }

    const Eigen::Index n_samples = ptc.scans.data.rows();
    const Eigen::Index n_dets = ptc.scans.data.cols();
    if (n_samples < 4 || n_dets <= 0 || ptc.flags.data.rows() != n_samples || ptc.flags.data.cols() != n_dets) {
        return summary;
    }

    const Eigen::Index block_size =
        std::max<Eigen::Index>(8, rfi_config.block_size_samples);
    const Eigen::Index min_good =
        std::max<Eigen::Index>(
            4, std::min<Eigen::Index>(
                   rfi_config.min_good_samples, block_size));
    const int dilate_blocks = std::max(0, rfi_config.dilate_blocks);
    const double sigma_threshold = std::max(1.0, rfi_config.sigma_threshold);
    const double sigma_floor = std::max(0.0, rfi_config.sigma_floor);
    const double max_flagged_fraction =
        std::clamp(rfi_config.max_flagged_fraction, 0.0, 1.0);
    const double eps = std::numeric_limits<double>::epsilon();

    const Eigen::Index n_blocks = (n_samples + block_size - 1) / block_size;
    std::vector<unsigned char> bad_blocks(static_cast<std::size_t>(n_blocks), 0);
    std::vector<double> diffs;
    std::vector<Eigen::Index> to_flag;

    Eigen::VectorXi local_samples = Eigen::VectorXi::Zero(n_dets);
    Eigen::VectorXi local_scans = Eigen::VectorXi::Zero(n_dets);

    for (Eigen::Index det = 0; det < n_dets; ++det) {
        Eigen::Index n_good_samples = 0;
        for (Eigen::Index t = 0; t < n_samples; ++t) {
            const double s = ptc.scans.data(t, det);
            if (!ptc.flags.data(t, det) && std::isfinite(s)) {
                n_good_samples++;
            }
        }
        if (n_good_samples < min_good + 1) {
            continue;
        }
        summary.n_det_candidates++;

        diffs.clear();
        diffs.reserve(static_cast<std::size_t>(n_good_samples));
        for (Eigen::Index t = 1; t < n_samples; ++t) {
            if (ptc.flags.data(t, det) || ptc.flags.data(t - 1, det)) {
                continue;
            }
            const double s0 = ptc.scans.data(t - 1, det);
            const double s1 = ptc.scans.data(t, det);
            if (!std::isfinite(s0) || !std::isfinite(s1)) {
                continue;
            }
            diffs.push_back(s1 - s0);
        }
        if (static_cast<Eigen::Index>(diffs.size()) < min_good - 1) {
            continue;
        }

        const double global_sigma = beammap_masking_stats::robust_stats(diffs).sigma;
        if (!std::isfinite(global_sigma) || global_sigma <= eps) {
            continue;
        }

        std::fill(bad_blocks.begin(), bad_blocks.end(), 0);
        bool any_bad = false;
        for (Eigen::Index b = 0; b < n_blocks; ++b) {
            const Eigen::Index b_start = b * block_size;
            const Eigen::Index b_end = std::min(b_start + block_size, n_samples);
            diffs.clear();
            diffs.reserve(static_cast<std::size_t>(b_end - b_start));
            for (Eigen::Index t = std::max<Eigen::Index>(b_start + 1, 1); t < b_end; ++t) {
                if (ptc.flags.data(t, det) || ptc.flags.data(t - 1, det)) {
                    continue;
                }
                const double s0 = ptc.scans.data(t - 1, det);
                const double s1 = ptc.scans.data(t, det);
                if (!std::isfinite(s0) || !std::isfinite(s1)) {
                    continue;
                }
                diffs.push_back(s1 - s0);
            }
            if (static_cast<Eigen::Index>(diffs.size()) < min_good - 1) {
                continue;
            }

            const double block_sigma = beammap_masking_stats::robust_stats(diffs).sigma;
            if (!std::isfinite(block_sigma) || block_sigma <= eps) {
                continue;
            }
            if (block_sigma >= sigma_floor && block_sigma > sigma_threshold * global_sigma) {
                bad_blocks[static_cast<std::size_t>(b)] = 1;
                any_bad = true;
            }
        }

        if (!any_bad) {
            continue;
        }

        beammap_masking_stats::dilate_block_mask(bad_blocks, dilate_blocks);

        to_flag.clear();
        for (Eigen::Index b = 0; b < n_blocks; ++b) {
            if (!bad_blocks[static_cast<std::size_t>(b)]) {
                continue;
            }
            const Eigen::Index b_start = b * block_size;
            const Eigen::Index b_end = std::min(b_start + block_size, n_samples);
            for (Eigen::Index t = b_start; t < b_end; ++t) {
                const double s = ptc.scans.data(t, det);
                if (!ptc.flags.data(t, det) && std::isfinite(s)) {
                    to_flag.push_back(t);
                }
            }
        }

        if (to_flag.empty()) {
            continue;
        }
        const double flagged_fraction =
            static_cast<double>(to_flag.size()) / static_cast<double>(std::max<Eigen::Index>(1, n_good_samples));
        if (max_flagged_fraction > 0.0 && flagged_fraction > max_flagged_fraction) {
            summary.n_det_rejected++;
            continue;
        }

        for (const auto t: to_flag) {
            ptc.flags.data(t, det) = true;
        }

        summary.n_det_flagged++;
        summary.n_samples_flagged += static_cast<Eigen::Index>(to_flag.size());
        local_samples(det) += static_cast<int>(to_flag.size());
        local_scans(det) = 1;
    }

    if (summary.n_samples_flagged > 0 &&
        rfi_mask_samples_flagged.size() == n_dets &&
        rfi_mask_scans_flagged.size() == n_dets) {
        if (!rfi_mask_diag_mutex) {
            rfi_mask_diag_mutex = std::make_shared<std::mutex>();
        }
        std::lock_guard<std::mutex> lock(*rfi_mask_diag_mutex);
        rfi_mask_samples_flagged += local_samples;
        rfi_mask_scans_flagged += local_scans;
    }

    return summary;
}
