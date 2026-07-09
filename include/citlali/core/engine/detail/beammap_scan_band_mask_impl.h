#pragma once

// Beammap scan-band masking implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_masking_stats.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

bool Beammap::reject_scan_band_mask_candidate(
    Eigen::Index det,
    Eigen::Index n_bad_rows,
    std::size_t n_proposed_flags,
    double flagged_fraction,
    double max_flagged_fraction,
    Beammap::ScanBandMaskSummary &summary) {
    if (!(max_flagged_fraction > 0.0 && flagged_fraction > max_flagged_fraction)) {
        return false;
    }
    if (scan_band_mask_rejected.size() == calib.n_dets) {
        scan_band_mask_rejected(det) = 1;
    }
    summary.n_det_rejected++;
    logger->debug(
        "beammap scan-band mask det={} rejected: proposed rows={} samples={} flagged_fraction={:.4f} exceeds limit={:.4f}",
        det, n_bad_rows, n_proposed_flags, flagged_fraction, max_flagged_fraction);
    return true;
}

void Beammap::apply_scan_band_mask_flags(
    Eigen::Index det,
    const std::vector<std::pair<Eigen::Index, Eigen::Index>> &proposed_flags) {
    for (const auto &[chunk_idx, sample_idx] : proposed_flags) {
        ptcs[chunk_idx].flags.data(sample_idx, det) = true;
        if (chunk_idx < static_cast<Eigen::Index>(ptcs0.size()) &&
            sample_idx < ptcs0[chunk_idx].flags.data.rows() &&
            det < ptcs0[chunk_idx].flags.data.cols()) {
            ptcs0[chunk_idx].flags.data(sample_idx, det) = true;
        }
    }
}

void Beammap::record_scan_band_mask_success(
    Eigen::Index det,
    Eigen::Index n_bad_rows,
    const std::vector<std::pair<Eigen::Index, Eigen::Index>> &proposed_flags,
    const std::vector<Eigen::Index> &top_rows,
    const std::vector<Eigen::Index> &bottom_rows,
    double flagged_fraction,
    Beammap::ScanBandMaskSummary &summary) {
    summary.n_det_flagged++;
    summary.n_rows_flagged += n_bad_rows;
    summary.n_samples_flagged += static_cast<Eigen::Index>(proposed_flags.size());
    if (scan_band_mask_samples_flagged.size() == calib.n_dets) {
        scan_band_mask_samples_flagged(det) += static_cast<int>(proposed_flags.size());
    }
    if (scan_band_mask_rows_flagged.size() == calib.n_dets) {
        scan_band_mask_rows_flagged(det) += static_cast<int>(n_bad_rows);
    }
    if (scan_band_mask_edge_code.size() == calib.n_dets) {
        const int edge_code = (!top_rows.empty() ? 1 : 0) + (!bottom_rows.empty() ? 2 : 0);
        scan_band_mask_edge_code(det) = edge_code;
    }

    logger->info(
        "beammap scan-band mask det={} array={} nw={} rows={} samples={} flagged_fraction={:.4f} top_rows={} bottom_rows={}",
        det,
        static_cast<int>(calib.apt["array"](det)),
        static_cast<int>(calib.apt["nw"](det)),
        n_bad_rows,
        proposed_flags.size(),
        flagged_fraction,
        static_cast<int>(top_rows.size()),
        static_cast<int>(bottom_rows.size()));
}

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

        std::vector<double> row_medians(static_cast<std::size_t>(map_buffer.n_rows),
                                        std::numeric_limits<double>::quiet_NaN());
        std::vector<double> row_sigmas(static_cast<std::size_t>(map_buffer.n_rows),
                                       std::numeric_limits<double>::quiet_NaN());
        std::vector<Eigen::Index> row_counts(static_cast<std::size_t>(map_buffer.n_rows), 0);
        std::vector<double> row_values;
        std::vector<double> interior_row_medians;
        std::vector<double> interior_row_sigmas;

        for (Eigen::Index row = 0; row < map_buffer.n_rows; ++row) {
            row_values.clear();
            row_values.reserve(static_cast<std::size_t>(map_buffer.n_cols));
            for (Eigen::Index col = 0; col < map_buffer.n_cols; ++col) {
                const double w = wt(row, col);
                const double s = sig(row, col);
                if (!std::isfinite(w) || w <= 0.0 || !std::isfinite(s)) {
                    continue;
                }
                row_values.push_back(s);
            }
            row_counts[static_cast<std::size_t>(row)] = static_cast<Eigen::Index>(row_values.size());
            if (row_counts[static_cast<std::size_t>(row)] < min_row_pixels) {
                continue;
            }
            const auto row_stats = beammap_masking_stats::robust_stats(row_values);
            if (!row_stats.valid) {
                continue;
            }
            row_medians[static_cast<std::size_t>(row)] = row_stats.median;
            row_sigmas[static_cast<std::size_t>(row)] = row_stats.sigma;
            if (row >= search_rows && row < map_buffer.n_rows - search_rows) {
                interior_row_medians.push_back(row_stats.median);
                if (std::isfinite(row_stats.sigma)) {
                    interior_row_sigmas.push_back(row_stats.sigma);
                }
            }
        }

        if (interior_row_medians.size() < static_cast<std::size_t>(min_contiguous_rows)) {
            continue;
        }

        const auto interior_stats = beammap_masking_stats::robust_stats(interior_row_medians);
        if (!interior_stats.valid) {
            continue;
        }
        const double interior_median = interior_stats.median;
        const double interior_median_sigma = interior_stats.sigma;

        double interior_row_sigma_median = std::numeric_limits<double>::quiet_NaN();
        if (!interior_row_sigmas.empty()) {
            interior_row_sigma_median = beammap_masking_stats::robust_stats(interior_row_sigmas).median;
        }

        auto collect_edge_rows = [&](bool from_top) {
            std::vector<Eigen::Index> flagged_rows;
            bool saw_eligible_row = false;
            for (Eigen::Index edge_idx = 0; edge_idx < search_rows; ++edge_idx) {
                const Eigen::Index row = from_top ? edge_idx : (map_buffer.n_rows - 1 - edge_idx);
                if (row < 0 || row >= map_buffer.n_rows) {
                    continue;
                }
                if (row_counts[static_cast<std::size_t>(row)] < min_row_pixels ||
                    !std::isfinite(row_medians[static_cast<std::size_t>(row)])) {
                    if (saw_eligible_row) {
                        break;
                    }
                    continue;
                }
                saw_eligible_row = true;
                const bool bad = beammap_masking_stats::row_is_bad(
                    row_medians[static_cast<std::size_t>(row)],
                    row_sigmas[static_cast<std::size_t>(row)],
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
        };

        auto top_rows = collect_edge_rows(true);
        auto bottom_rows = collect_edge_rows(false);
        if (top_rows.empty() && bottom_rows.empty()) {
            continue;
        }

        std::vector<unsigned char> bad_row_mask(static_cast<std::size_t>(map_buffer.n_rows), 0);
        Eigen::Index n_bad_rows = 0;
        for (const auto row : top_rows) {
            if (!bad_row_mask[static_cast<std::size_t>(row)]) {
                bad_row_mask[static_cast<std::size_t>(row)] = 1;
                n_bad_rows++;
            }
        }
        for (const auto row : bottom_rows) {
            if (!bad_row_mask[static_cast<std::size_t>(row)]) {
                bad_row_mask[static_cast<std::size_t>(row)] = 1;
                n_bad_rows++;
            }
        }
        if (n_bad_rows <= 0) {
            continue;
        }

        std::vector<std::pair<Eigen::Index, Eigen::Index>> proposed_flags;
        Eigen::Index n_good_samples = 0;
        for (Eigen::Index chunk_idx = 0; chunk_idx < static_cast<Eigen::Index>(ptcs.size()); ++chunk_idx) {
            auto &ptc = ptcs[chunk_idx];
            if (det >= ptc.scans.data.cols() || det >= ptc.flags.data.cols()) {
                continue;
            }
            Eigen::VectorXd lat;
            auto lat_it = ptc.pointing.data.find("lat");
            if (lat_it != ptc.pointing.data.end() &&
                lat_it->second.rows() == ptc.scans.data.rows() &&
                det < lat_it->second.cols()) {
                lat = lat_it->second.col(det);
            }
            else {
                auto latlon = engine_utils::calc_det_pointing(
                    ptc.tel_data.data,
                    calib.apt["x_t"](det),
                    calib.apt["y_t"](det),
                    telescope.pixel_axes,
                    ptc.pointing_offsets_arcsec.data,
                    citlali::pipeline::mapmaking_config(*this).grouping);
                lat = std::get<0>(latlon);
            }
            if (lat.size() != ptc.scans.data.rows()) {
                continue;
            }
            for (Eigen::Index t = 0; t < ptc.scans.data.rows(); ++t) {
                const double s = ptc.scans.data(t, det);
                if (ptc.flags.data(t, det) || !std::isfinite(s)) {
                    continue;
                }
                n_good_samples++;
                const double lat_v = lat(t);
                if (!std::isfinite(lat_v)) {
                    continue;
                }
                const Eigen::Index row = static_cast<Eigen::Index>(std::llround(lat_v / map_buffer.pixel_size_rad + row0));
                if (row < 0 || row >= map_buffer.n_rows) {
                    continue;
                }
                if (bad_row_mask[static_cast<std::size_t>(row)]) {
                    proposed_flags.emplace_back(chunk_idx, t);
                }
            }
        }

        if (proposed_flags.empty()) {
            continue;
        }

        const double flagged_fraction =
            static_cast<double>(proposed_flags.size()) /
            static_cast<double>(std::max<Eigen::Index>(1, n_good_samples));
        if (reject_scan_band_mask_candidate(
                det, n_bad_rows, proposed_flags.size(), flagged_fraction,
                max_flagged_fraction, summary)) {
            continue;
        }

        apply_scan_band_mask_flags(det, proposed_flags);
        record_scan_band_mask_success(
            det, n_bad_rows, proposed_flags, top_rows, bottom_rows,
            flagged_fraction, summary);
    }

    return summary;
}
