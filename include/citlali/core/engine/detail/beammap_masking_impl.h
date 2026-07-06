#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_masking_stats.h>

void Beammap::log_beammap_masking_config() {
    const bool detector_grouping =
        typed_config.mapmaking.grouping ==
        citlali::config::MapGrouping::detector;

    if (beammap_rfi_mask_enabled && detector_grouping) {
        logger->info("beammap rfi mask enabled: block_size={} min_good={} sigma_threshold={:.4g} sigma_floor={:.4g} dilate_blocks={} max_flagged_fraction={:.4f}",
                     beammap_rfi_mask_block_size_samples,
                     beammap_rfi_mask_min_good_samples,
                     beammap_rfi_mask_sigma_threshold,
                     beammap_rfi_mask_sigma_floor,
                     beammap_rfi_mask_dilate_blocks,
                     beammap_rfi_mask_max_flagged_fraction);
    }
    if (beammap_scan_band_mask_enabled && detector_grouping) {
        logger->info(
            "beammap scan-band mask enabled: edge_rows={} min_row_pixels={} min_contiguous_rows={} row_median_sigma_threshold={:.4g} row_sigma_ratio_threshold={:.4g} max_flagged_fraction={:.4f}",
            beammap_scan_band_mask_edge_rows,
            beammap_scan_band_mask_min_row_pixels,
            beammap_scan_band_mask_min_contiguous_rows,
            beammap_scan_band_mask_row_median_sigma_threshold,
            beammap_scan_band_mask_row_sigma_ratio_threshold,
            beammap_scan_band_mask_max_flagged_fraction);
    }
}

Beammap::RFIMaskScanSummary Beammap::apply_rfi_sample_mask(TCData<TCDataKind::PTC,Eigen::MatrixXd> &ptc) {
    RFIMaskScanSummary summary;
    if (!beammap_rfi_mask_enabled) {
        return summary;
    }

    const Eigen::Index n_samples = ptc.scans.data.rows();
    const Eigen::Index n_dets = ptc.scans.data.cols();
    if (n_samples < 4 || n_dets <= 0 || ptc.flags.data.rows() != n_samples || ptc.flags.data.cols() != n_dets) {
        return summary;
    }

    const Eigen::Index block_size = std::max<Eigen::Index>(8, beammap_rfi_mask_block_size_samples);
    const Eigen::Index min_good = std::max<Eigen::Index>(4, std::min<Eigen::Index>(beammap_rfi_mask_min_good_samples, block_size));
    const int dilate_blocks = std::max(0, beammap_rfi_mask_dilate_blocks);
    const double sigma_threshold = std::max(1.0, beammap_rfi_mask_sigma_threshold);
    const double sigma_floor = std::max(0.0, beammap_rfi_mask_sigma_floor);
    const double max_flagged_fraction = std::clamp(beammap_rfi_mask_max_flagged_fraction, 0.0, 1.0);
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

Beammap::ScanBandMaskSummary Beammap::apply_scan_band_mask(mapmaking::MapBuffer &map_buffer) {
    ScanBandMaskSummary summary;

    if (!beammap_scan_band_mask_enabled ||
        typed_config.mapmaking.grouping !=
            citlali::config::MapGrouping::detector) {
        return summary;
    }

    const Eigen::Index n_det_maps = std::min<Eigen::Index>(
        static_cast<Eigen::Index>(map_buffer.signal.size()), calib.n_dets);
    if (n_det_maps <= 0 || map_buffer.n_rows <= 0 || map_buffer.n_cols <= 0) {
        return summary;
    }

    const Eigen::Index search_rows = std::min<Eigen::Index>(
        std::max<Eigen::Index>(1, beammap_scan_band_mask_edge_rows), map_buffer.n_rows / 2);
    if (search_rows <= 0) {
        return summary;
    }

    const Eigen::Index min_row_pixels = std::max<Eigen::Index>(1, beammap_scan_band_mask_min_row_pixels);
    const Eigen::Index min_contiguous_rows = std::max<Eigen::Index>(1, beammap_scan_band_mask_min_contiguous_rows);
    const double median_sigma_threshold = std::max(0.0, beammap_scan_band_mask_row_median_sigma_threshold);
    const double sigma_ratio_threshold = std::max(0.0, beammap_scan_band_mask_row_sigma_ratio_threshold);
    const double max_flagged_fraction = std::clamp(beammap_scan_band_mask_max_flagged_fraction, 0.0, 1.0);
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
                    typed_config.mapmaking.grouping);
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
        if (max_flagged_fraction > 0.0 && flagged_fraction > max_flagged_fraction) {
            if (scan_band_mask_rejected.size() == calib.n_dets) {
                scan_band_mask_rejected(det) = 1;
            }
            summary.n_det_rejected++;
            logger->debug(
                "beammap scan-band mask det={} rejected: proposed rows={} samples={} flagged_fraction={:.4f} exceeds limit={:.4f}",
                det, n_bad_rows, proposed_flags.size(), flagged_fraction, max_flagged_fraction);
            continue;
        }

        for (const auto &[chunk_idx, sample_idx] : proposed_flags) {
            ptcs[chunk_idx].flags.data(sample_idx, det) = true;
            if (chunk_idx < static_cast<Eigen::Index>(ptcs0.size()) &&
                sample_idx < ptcs0[chunk_idx].flags.data.rows() &&
                det < ptcs0[chunk_idx].flags.data.cols()) {
                ptcs0[chunk_idx].flags.data(sample_idx, det) = true;
            }
        }

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

    return summary;
}
