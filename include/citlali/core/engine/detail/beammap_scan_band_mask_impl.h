#pragma once

// Beammap scan-band masking implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <citlali/core/engine/detail/beammap_scan_band_result_impl.h>
#include <citlali/core/engine/detail/beammap_scan_band_row_selection_impl.h>
#include <citlali/core/engine/detail/beammap_scan_band_sample_selection_impl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

Beammap::ScanBandMaskSummary Beammap::apply_scan_band_mask(mapmaking::MapBuffer &map_buffer) {
    ScanBandMaskSummary summary;
    const auto &scan_band_config =
        citlali::pipeline::beammap_config(*this).scan_band_mask;

    if (!scan_band_config.enabled ||
        citlali::pipeline::mapmaking_config(*this).grouping !=
            citlali::config::MapGrouping::detector) {
        return summary;
    }

    const Eigen::Index n_det_maps = std::min<Eigen::Index>(
        static_cast<Eigen::Index>(map_buffer.signal.size()), calib.n_dets);
    if (n_det_maps <= 0 || map_buffer.n_rows <= 0 || map_buffer.n_cols <= 0) {
        return summary;
    }

    const Eigen::Index search_rows = std::min<Eigen::Index>(
        std::max<Eigen::Index>(1, scan_band_config.edge_rows),
        map_buffer.n_rows / 2);
    if (search_rows <= 0) {
        return summary;
    }

    const Eigen::Index min_row_pixels =
        std::max<Eigen::Index>(1, scan_band_config.min_row_pixels);
    const Eigen::Index min_contiguous_rows =
        std::max<Eigen::Index>(1, scan_band_config.min_contiguous_rows);
    const double median_sigma_threshold =
        std::max(0.0, scan_band_config.row_median_sigma_threshold);
    const double sigma_ratio_threshold =
        std::max(0.0, scan_band_config.row_sigma_ratio_threshold);
    const double max_flagged_fraction =
        std::clamp(scan_band_config.max_flagged_fraction, 0.0, 1.0);
    const double eps = std::numeric_limits<double>::epsilon();
    const double row0 = static_cast<double>(map_buffer.n_rows - 1) / 2.0;

    for (Eigen::Index det = 0; det < n_det_maps; ++det) {
        const auto &sig = map_buffer.signal[det];
        const auto &wt = map_buffer.weight[det];
        if (sig.rows() != map_buffer.n_rows || sig.cols() != map_buffer.n_cols ||
            wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
            continue;
        }

        const auto row_stats = calculate_scan_band_row_stats(
            sig, wt, map_buffer.n_rows, map_buffer.n_cols, search_rows,
            min_row_pixels);
        const auto edge_rows = select_scan_band_edge_rows(
            row_stats, map_buffer.n_rows, search_rows, min_row_pixels,
            min_contiguous_rows, median_sigma_threshold,
            sigma_ratio_threshold, eps);
        if (edge_rows.top.empty() && edge_rows.bottom.empty()) {
            continue;
        }

        Eigen::Index n_bad_rows = 0;
        const auto bad_row_mask = make_scan_band_bad_row_mask(
            edge_rows, map_buffer.n_rows, n_bad_rows);
        if (n_bad_rows <= 0) {
            continue;
        }

        const auto proposed_flags = collect_scan_band_proposed_flags(
            det, map_buffer, bad_row_mask, row0);
        if (proposed_flags.samples.empty()) {
            continue;
        }

        const double flagged_fraction =
            static_cast<double>(proposed_flags.samples.size()) /
            static_cast<double>(
                std::max<Eigen::Index>(1, proposed_flags.n_good_samples));
        if (reject_scan_band_mask_candidate(
                det, n_bad_rows, proposed_flags.samples.size(),
                flagged_fraction,
                max_flagged_fraction, summary)) {
            continue;
        }

        apply_scan_band_mask_flags(det, proposed_flags.samples);
        record_scan_band_mask_success(
            det, n_bad_rows, proposed_flags.samples, edge_rows.top,
            edge_rows.bottom, flagged_fraction, summary);
    }

    return summary;
}
