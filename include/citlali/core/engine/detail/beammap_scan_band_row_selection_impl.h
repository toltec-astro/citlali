#pragma once

// Beammap scan-band row selection implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_masking_stats.h>

#include <cmath>
#include <limits>
#include <vector>

Beammap::ScanBandRowStats Beammap::calculate_scan_band_row_stats(
    const Eigen::MatrixXd &signal,
    const Eigen::MatrixXd &weight,
    Eigen::Index n_rows,
    Eigen::Index n_cols,
    Eigen::Index search_rows,
    Eigen::Index min_row_pixels) {
    ScanBandRowStats row_stats;
    row_stats.medians.assign(static_cast<std::size_t>(n_rows),
                             std::numeric_limits<double>::quiet_NaN());
    row_stats.sigmas.assign(static_cast<std::size_t>(n_rows),
                            std::numeric_limits<double>::quiet_NaN());
    row_stats.counts.assign(static_cast<std::size_t>(n_rows), 0);

    std::vector<double> row_values;
    for (Eigen::Index row = 0; row < n_rows; ++row) {
        row_values.clear();
        row_values.reserve(static_cast<std::size_t>(n_cols));
        for (Eigen::Index col = 0; col < n_cols; ++col) {
            const double w = weight(row, col);
            const double s = signal(row, col);
            if (!std::isfinite(w) || w <= 0.0 || !std::isfinite(s)) {
                continue;
            }
            row_values.push_back(s);
        }
        row_stats.counts[static_cast<std::size_t>(row)] =
            static_cast<Eigen::Index>(row_values.size());
        if (row_stats.counts[static_cast<std::size_t>(row)] <
            min_row_pixels) {
            continue;
        }
        const auto stats = beammap_masking_stats::robust_stats(row_values);
        if (!stats.valid) {
            continue;
        }
        row_stats.medians[static_cast<std::size_t>(row)] = stats.median;
        row_stats.sigmas[static_cast<std::size_t>(row)] = stats.sigma;
        if (row >= search_rows && row < n_rows - search_rows) {
            row_stats.interior_medians.push_back(stats.median);
            if (std::isfinite(stats.sigma)) {
                row_stats.interior_sigmas.push_back(stats.sigma);
            }
        }
    }

    return row_stats;
}

std::vector<Eigen::Index> Beammap::collect_scan_band_edge_rows(
    const ScanBandRowStats &row_stats,
    bool from_top,
    Eigen::Index n_rows,
    Eigen::Index search_rows,
    Eigen::Index min_row_pixels,
    Eigen::Index min_contiguous_rows,
    double interior_median,
    double interior_median_sigma,
    double interior_row_sigma_median,
    double median_sigma_threshold,
    double sigma_ratio_threshold,
    double eps) {
    std::vector<Eigen::Index> flagged_rows;
    bool saw_eligible_row = false;
    for (Eigen::Index edge_idx = 0; edge_idx < search_rows; ++edge_idx) {
        const Eigen::Index row =
            from_top ? edge_idx : (n_rows - 1 - edge_idx);
        if (row < 0 || row >= n_rows) {
            continue;
        }
        if (row_stats.counts[static_cast<std::size_t>(row)] <
                min_row_pixels ||
            !std::isfinite(row_stats.medians[static_cast<std::size_t>(row)])) {
            if (saw_eligible_row) {
                break;
            }
            continue;
        }
        saw_eligible_row = true;
        const bool bad = beammap_masking_stats::row_is_bad(
            row_stats.medians[static_cast<std::size_t>(row)],
            row_stats.sigmas[static_cast<std::size_t>(row)],
            interior_median,
            interior_median_sigma,
            interior_row_sigma_median,
            median_sigma_threshold,
            sigma_ratio_threshold,
            eps);
        if (!bad) {
            break;
        }
        flagged_rows.push_back(row);
    }
    if (flagged_rows.size() < static_cast<std::size_t>(min_contiguous_rows)) {
        flagged_rows.clear();
    }
    return flagged_rows;
}

Beammap::ScanBandEdgeRows Beammap::select_scan_band_edge_rows(
    const ScanBandRowStats &row_stats,
    Eigen::Index n_rows,
    Eigen::Index search_rows,
    Eigen::Index min_row_pixels,
    Eigen::Index min_contiguous_rows,
    double median_sigma_threshold,
    double sigma_ratio_threshold,
    double eps) {
    ScanBandEdgeRows edge_rows;
    if (row_stats.interior_medians.size() <
        static_cast<std::size_t>(min_contiguous_rows)) {
        return edge_rows;
    }

    const auto interior_stats =
        beammap_masking_stats::robust_stats(row_stats.interior_medians);
    if (!interior_stats.valid) {
        return edge_rows;
    }
    const double interior_median = interior_stats.median;
    const double interior_median_sigma = interior_stats.sigma;

    double interior_row_sigma_median =
        std::numeric_limits<double>::quiet_NaN();
    if (!row_stats.interior_sigmas.empty()) {
        interior_row_sigma_median =
            beammap_masking_stats::robust_stats(row_stats.interior_sigmas)
                .median;
    }

    edge_rows.top = collect_scan_band_edge_rows(
        row_stats, true, n_rows, search_rows, min_row_pixels,
        min_contiguous_rows, interior_median, interior_median_sigma,
        interior_row_sigma_median, median_sigma_threshold,
        sigma_ratio_threshold, eps);
    edge_rows.bottom = collect_scan_band_edge_rows(
        row_stats, false, n_rows, search_rows, min_row_pixels,
        min_contiguous_rows, interior_median, interior_median_sigma,
        interior_row_sigma_median, median_sigma_threshold,
        sigma_ratio_threshold, eps);
    return edge_rows;
}
