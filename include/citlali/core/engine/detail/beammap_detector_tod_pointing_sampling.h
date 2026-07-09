#pragma once

// Beammap detector-specific TOD pointing sampling helpers.

#include <Eigen/Core>

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace beammap_detector_tod_selection {

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

} // namespace beammap_detector_tod_selection
