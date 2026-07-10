#pragma once

// Beammap scan-band sample selection implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <cmath>
#include <utility>
#include <vector>

std::vector<unsigned char> Beammap::make_scan_band_bad_row_mask(
    const ScanBandEdgeRows &edge_rows,
    Eigen::Index n_rows,
    Eigen::Index &n_bad_rows) {
    std::vector<unsigned char> bad_row_mask(
        static_cast<std::size_t>(n_rows), 0);
    n_bad_rows = 0;
    for (const auto row : edge_rows.top) {
        if (!bad_row_mask[static_cast<std::size_t>(row)]) {
            bad_row_mask[static_cast<std::size_t>(row)] = 1;
            n_bad_rows++;
        }
    }
    for (const auto row : edge_rows.bottom) {
        if (!bad_row_mask[static_cast<std::size_t>(row)]) {
            bad_row_mask[static_cast<std::size_t>(row)] = 1;
            n_bad_rows++;
        }
    }
    return bad_row_mask;
}

Beammap::ScanBandProposedFlags Beammap::collect_scan_band_proposed_flags(
    Eigen::Index det,
    const mapmaking::MapBuffer &map_buffer,
    const std::vector<unsigned char> &bad_row_mask,
    double row0) {
    ScanBandProposedFlags proposed;
    for (Eigen::Index chunk_idx = 0;
         chunk_idx < static_cast<Eigen::Index>(ptcs.size()); ++chunk_idx) {
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
            proposed.n_good_samples++;
            const double lat_v = lat(t);
            if (!std::isfinite(lat_v)) {
                continue;
            }
            const Eigen::Index row = static_cast<Eigen::Index>(
                std::llround(lat_v / map_buffer.pixel_size_rad + row0));
            if (row < 0 || row >= map_buffer.n_rows) {
                continue;
            }
            if (bad_row_mask[static_cast<std::size_t>(row)]) {
                proposed.samples.emplace_back(chunk_idx, t);
            }
        }
    }
    return proposed;
}
