#pragma once

// Included by tod_output_selection.h inside namespace citlali::pipeline.

struct TodSourceCrossingScan {
    Eigen::Index scan_index = 0;
    double min_distance2 = std::numeric_limits<double>::infinity();
};

template <class PointingOffsets>
Eigen::VectorXd tod_output_pointing_offset_or_zero(
    const PointingOffsets &pointing_offsets_arcsec, const std::string &axis,
    Eigen::Index n_tel) {
    auto it = pointing_offsets_arcsec.find(axis);
    if (it != pointing_offsets_arcsec.end() && it->second.size() == n_tel) {
        return it->second;
    }
    return Eigen::VectorXd::Zero(n_tel);
}

template <class ScanIndices>
TodSourceCrossingScan nearest_source_crossing_scan(
    const Eigen::VectorXd &lat, const Eigen::VectorXd &lon,
    const ScanIndices &scan_indices) {
    const Eigen::Index n_scans = scan_indices.cols();
    TodSourceCrossingScan result{(n_scans - 1) / 2,
                                 std::numeric_limits<double>::infinity()};

    for (Eigen::Index scan_index = 0; scan_index < n_scans; ++scan_index) {
        const Eigen::Index start =
            std::max<Eigen::Index>(0, scan_indices(0, scan_index));
        const Eigen::Index end =
            std::min<Eigen::Index>(lat.size() - 1,
                                   scan_indices(1, scan_index));
        if (end < start || lon.size() <= end) {
            continue;
        }
        double scan_best_d2 = std::numeric_limits<double>::infinity();
        for (Eigen::Index sample = start; sample <= end; ++sample) {
            const double y = lat(sample);
            const double x = lon(sample);
            if (!std::isfinite(x) || !std::isfinite(y)) {
                continue;
            }
            const double d2 = x * x + y * y;
            if (d2 < scan_best_d2) {
                scan_best_d2 = d2;
            }
        }
        if (scan_best_d2 < result.min_distance2) {
            result.min_distance2 = scan_best_d2;
            result.scan_index = scan_index;
        }
    }

    return result;
}

template <class Telescope, class PointingOffsets, class MapGrouping>
TodSourceCrossingScan find_source_crossing_scan(
    const Telescope &telescope, const PointingOffsets &pointing_offsets_arcsec,
    const MapGrouping &map_grouping) {
    auto tel_data_copy = telescope.tel_data;
    Eigen::Index n_tel = 0;
    if (!tel_data_copy.empty()) {
        n_tel = tel_data_copy.begin()->second.size();
    }

    std::map<std::string, Eigen::VectorXd> pointing_offsets;
    pointing_offsets[citlali::config::pointing_axis_az()] =
        tod_output_pointing_offset_or_zero(
            pointing_offsets_arcsec, citlali::config::pointing_axis_az(),
            n_tel);
    pointing_offsets[citlali::config::pointing_axis_alt()] =
        tod_output_pointing_offset_or_zero(
            pointing_offsets_arcsec, citlali::config::pointing_axis_alt(),
            n_tel);

    auto [lat, lon] = ::engine_utils::calc_det_pointing(
        tel_data_copy, 0.0, 0.0, telescope.pixel_axes, pointing_offsets,
        map_grouping, true);

    return nearest_source_crossing_scan(
        lat, lon, telescope.scan_indices);
}
