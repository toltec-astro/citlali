#pragma once

// Beammap scan-band result application implementation detail.
// Include this only after Beammap has been declared.

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
