#pragma once

// Beammap detector-specific TOD scan and slot selection helpers.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
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

} // namespace beammap_detector_tod_selection
