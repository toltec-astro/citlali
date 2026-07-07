#pragma once

// Beammap detector-specific TOD scan selection helpers.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace beammap_detector_tod_selection {

inline std::vector<Eigen::Index> uniform_scan_indices(int n_uniform,
                                                      Eigen::Index n_scans) {
    std::vector<Eigen::Index> scans;
    scans.reserve(static_cast<std::size_t>(n_uniform));
    for (int i = 0; i < n_uniform; ++i) {
        Eigen::Index scan_index = (n_scans - 1) / 2;
        if (n_uniform > 1) {
            const double frac = static_cast<double>(i) /
                                static_cast<double>(n_uniform - 1);
            scan_index = static_cast<Eigen::Index>(std::llround(frac * (n_scans - 1)));
        }
        scans.push_back(std::clamp<Eigen::Index>(scan_index, 0, n_scans - 1));
    }
    return scans;
}

inline std::vector<Eigen::Index> dense_scan_window(Eigen::Index center_scan,
                                                   int n_dense,
                                                   Eigen::Index n_scans) {
    std::vector<Eigen::Index> scans;
    scans.reserve(static_cast<std::size_t>(n_dense));
    if (n_dense <= 0) {
        return scans;
    }
    if (n_dense > n_scans) {
        for (int i = 0; i < n_dense; ++i) {
            Eigen::Index scan_index = 0;
            if (n_dense > 1) {
                const double frac = static_cast<double>(i) /
                                    static_cast<double>(n_dense - 1);
                scan_index = static_cast<Eigen::Index>(std::llround(frac * (n_scans - 1)));
            }
            scans.push_back(std::clamp<Eigen::Index>(scan_index, 0, n_scans - 1));
        }
        return scans;
    }

    Eigen::Index first_dense =
        center_scan - static_cast<Eigen::Index>((n_dense - 1) / 2);
    first_dense = std::clamp<Eigen::Index>(
        first_dense, 0, std::max<Eigen::Index>(0, n_scans - static_cast<Eigen::Index>(n_dense)));
    for (int i = 0; i < n_dense; ++i) {
        scans.push_back(first_dense + static_cast<Eigen::Index>(i));
    }
    return scans;
}

inline std::size_t flat_detector_slot(Eigen::Index det,
                                      Eigen::Index slot,
                                      Eigen::Index n_slots) {
    return static_cast<std::size_t>(det) * static_cast<std::size_t>(n_slots) +
           static_cast<std::size_t>(slot);
}

inline std::string format_center_scan_counts(
    const std::map<Eigen::Index, Eigen::Index> &center_scan_counts,
    std::size_t max_entries = 8) {
    std::vector<std::pair<Eigen::Index, Eigen::Index>> center_hist(
        center_scan_counts.begin(), center_scan_counts.end());
    std::sort(center_hist.begin(), center_hist.end(),
              [](const auto &lhs, const auto &rhs) {
                  if (lhs.second != rhs.second) {
                      return lhs.second > rhs.second;
                  }
                  return lhs.first < rhs.first;
              });

    std::ostringstream center_os;
    center_os << "[";
    for (std::size_t i = 0;
         i < std::min<std::size_t>(max_entries, center_hist.size()); ++i) {
        if (i != 0) {
            center_os << ", ";
        }
        center_os << center_hist[i].first + 1 << ":" << center_hist[i].second;
    }
    center_os << "]";
    return center_os.str();
}

template <class ScanIndices, class Ptcs>
inline void fill_slot_scan_metadata(
    Eigen::Index det,
    Eigen::Index slot,
    Eigen::Index n_slots,
    Eigen::Index scan_index,
    Eigen::Index n_scans,
    int slot_kind_value,
    const ScanIndices &scan_indices,
    const Ptcs &ptcs,
    const std::vector<double> &distances_arcsec,
    std::vector<int> &slot_scan_index,
    std::vector<int> &slot_kind,
    std::vector<int> &slot_n_samples,
    std::vector<int> &slot_inner_start,
    std::vector<int> &slot_inner_end,
    std::vector<int> &slot_outer_start,
    std::vector<int> &slot_outer_end,
    std::vector<double> &slot_source_distance_arcsec) {
    const auto idx = flat_detector_slot(det, slot, n_slots);
    slot_scan_index[idx] = static_cast<int>(scan_index + 1);
    slot_kind[idx] = slot_kind_value;
    if (scan_index < 0 || scan_index >= n_scans) {
        return;
    }

    slot_inner_start[idx] = static_cast<int>(scan_indices(0, scan_index));
    slot_inner_end[idx] = static_cast<int>(scan_indices(1, scan_index));
    slot_outer_start[idx] = static_cast<int>(scan_indices(2, scan_index));
    slot_outer_end[idx] = static_cast<int>(scan_indices(3, scan_index));
    if (scan_index < static_cast<Eigen::Index>(ptcs.size())) {
        slot_n_samples[idx] = static_cast<int>(ptcs[scan_index].scans.data.rows());
    }
    if (scan_index < static_cast<Eigen::Index>(distances_arcsec.size())) {
        slot_source_distance_arcsec[idx] =
            distances_arcsec[static_cast<std::size_t>(scan_index)];
    }
}

template <class ScanIndices, class TelData>
inline std::pair<std::vector<Eigen::Index>, std::vector<Eigen::Index>>
sampled_scan_samples(const ScanIndices &scan_indices,
                     const TelData &tel_data,
                     Eigen::Index n_scans,
                     Eigen::Index max_samples_per_scan = 96) {
    std::vector<Eigen::Index> sampled_indices;
    std::vector<Eigen::Index> sampled_scan;
    for (Eigen::Index scan_index = 0; scan_index < n_scans; ++scan_index) {
        const Eigen::Index start = std::max<Eigen::Index>(0, scan_indices(0, scan_index));
        const Eigen::Index tel_end =
            tel_data.empty()
                ? scan_indices(1, scan_index)
                : static_cast<Eigen::Index>(tel_data.begin()->second.size() - 1);
        const Eigen::Index end =
            std::min<Eigen::Index>(scan_indices(1, scan_index), tel_end);
        if (end < start) {
            continue;
        }
        const Eigen::Index n_scan_pts = end - start + 1;
        const Eigen::Index stride =
            std::max<Eigen::Index>(1, n_scan_pts / max_samples_per_scan);
        Eigen::Index last_sample = -1;
        for (Eigen::Index sample = start; sample <= end; sample += stride) {
            sampled_indices.push_back(sample);
            sampled_scan.push_back(scan_index);
            last_sample = sample;
        }
        if (last_sample != end) {
            sampled_indices.push_back(end);
            sampled_scan.push_back(scan_index);
        }
    }
    return {sampled_indices, sampled_scan};
}

template <class TelData>
inline std::map<std::string, Eigen::VectorXd> sample_tel_data(
    const TelData &tel_data,
    const std::vector<Eigen::Index> &sampled_indices) {
    const Eigen::Index n_sampled =
        static_cast<Eigen::Index>(sampled_indices.size());
    std::map<std::string, Eigen::VectorXd> sampled_tel_data;
    for (const auto &[key, values] : tel_data) {
        Eigen::VectorXd sampled(n_sampled);
        for (Eigen::Index i = 0; i < n_sampled; ++i) {
            const Eigen::Index sample = sampled_indices[static_cast<std::size_t>(i)];
            sampled(i) = (sample >= 0 && sample < values.size())
                             ? values(sample)
                             : std::numeric_limits<double>::quiet_NaN();
        }
        sampled_tel_data[key] = std::move(sampled);
    }
    return sampled_tel_data;
}

template <class PointingOffsets>
inline Eigen::VectorXd sample_pointing_offset(
    const PointingOffsets &pointing_offsets_arcsec,
    const std::string &axis,
    const std::vector<Eigen::Index> &sampled_indices) {
    const Eigen::Index n_sampled =
        static_cast<Eigen::Index>(sampled_indices.size());
    Eigen::VectorXd sampled = Eigen::VectorXd::Zero(n_sampled);
    auto it = pointing_offsets_arcsec.find(axis);
    if (it == pointing_offsets_arcsec.end()) {
        return sampled;
    }
    for (Eigen::Index i = 0; i < n_sampled; ++i) {
        const Eigen::Index sample = sampled_indices[static_cast<std::size_t>(i)];
        if (sample >= 0 && sample < it->second.size()) {
            sampled(i) = it->second(sample);
        }
    }
    return sampled;
}

template <class SampledTelData, class PointingOffsets, class PixelAxes,
          class MapGrouping>
inline Eigen::Index scan_distances_for_detector_source(
    Eigen::Index det,
    double source_x_arcsec,
    double source_y_arcsec,
    Eigen::Index n_scans,
    Eigen::Index n_sampled,
    const std::vector<Eigen::Index> &sampled_scan,
    SampledTelData &sampled_tel_data,
    const Eigen::VectorXd &apt_x_t,
    const Eigen::VectorXd &apt_y_t,
    const PixelAxes &pixel_axes,
    PointingOffsets &pointing_offsets,
    MapGrouping map_grouping,
    std::vector<double> &distances_arcsec) {
    Eigen::Index best_scan = (n_scans - 1) / 2;
    distances_arcsec.assign(static_cast<std::size_t>(n_scans),
                            std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(source_x_arcsec) || !std::isfinite(source_y_arcsec) ||
        det < 0 || det >= apt_x_t.size() || det >= apt_y_t.size() ||
        !std::isfinite(apt_x_t(det)) || !std::isfinite(apt_y_t(det))) {
        return best_scan;
    }

    double best_d2 = std::numeric_limits<double>::infinity();
    const double source_x_rad = source_x_arcsec * ASEC_TO_RAD;
    const double source_y_rad = source_y_arcsec * ASEC_TO_RAD;
    // Use the detector pointing that built the map, then find where that
    // pointing passes closest to the fitted source location.
    auto [lat, lon] = engine_utils::calc_det_pointing(
        sampled_tel_data, apt_x_t(det), apt_y_t(det), pixel_axes,
        pointing_offsets, map_grouping, true);

    std::vector<double> best_d2_by_scan(static_cast<std::size_t>(n_scans),
                                        std::numeric_limits<double>::infinity());
    for (Eigen::Index sample_i = 0; sample_i < n_sampled; ++sample_i) {
        if (sample_i >= lat.size() || sample_i >= lon.size()) {
            continue;
        }
        const double y = lat(sample_i) - source_y_rad;
        const double x = lon(sample_i) - source_x_rad;
        if (!std::isfinite(x) || !std::isfinite(y)) {
            continue;
        }
        const auto scan_index = sampled_scan[static_cast<std::size_t>(sample_i)];
        if (scan_index < 0 || scan_index >= n_scans) {
            continue;
        }
        const double d2 = x * x + y * y;
        auto &scan_best = best_d2_by_scan[static_cast<std::size_t>(scan_index)];
        if (d2 < scan_best) {
            scan_best = d2;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best_scan = scan_index;
        }
    }
    for (Eigen::Index scan_index = 0; scan_index < n_scans; ++scan_index) {
        const double d2 = best_d2_by_scan[static_cast<std::size_t>(scan_index)];
        if (std::isfinite(d2)) {
            distances_arcsec[static_cast<std::size_t>(scan_index)] =
                std::sqrt(d2) * RAD_TO_ASEC;
        }
    }
    return best_scan;
}

template <class GoodFits>
inline std::pair<double, double> detector_source_position(
    Eigen::Index det,
    const GoodFits &good_fits,
    const Eigen::MatrixXd &params,
    const Eigen::VectorXd &apt_x_t,
    const Eigen::VectorXd &apt_y_t,
    double pixel_size_rad,
    Eigen::Index n_cols,
    Eigen::Index n_rows,
    bool &used_fit) {
    used_fit = false;
    double x_arcsec = std::numeric_limits<double>::quiet_NaN();
    double y_arcsec = std::numeric_limits<double>::quiet_NaN();
    const bool fit_ok = det < good_fits.size() && good_fits(det) &&
                        det < params.rows() && params.cols() > 2 &&
                        std::isfinite(params(det, 1)) &&
                        std::isfinite(params(det, 2));
    if (fit_ok) {
        x_arcsec = RAD_TO_ASEC * pixel_size_rad *
                   (params(det, 1) - (n_cols - 1) / 2.0);
        y_arcsec = RAD_TO_ASEC * pixel_size_rad *
                   (params(det, 2) - (n_rows - 1) / 2.0);
        used_fit = true;
    }
    if ((!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) &&
        det < apt_x_t.size() && det < apt_y_t.size()) {
        x_arcsec = apt_x_t(det);
        y_arcsec = apt_y_t(det);
    }
    return {x_arcsec, y_arcsec};
}

} // namespace beammap_detector_tod_selection
