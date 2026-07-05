#pragma once

#include <citlali/core/config/timestream_config.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

inline std::string tod_output_chunks_to_string(
    const std::vector<Eigen::Index> &values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << values[i];
    }
    os << "]";
    return os.str();
}

inline void add_uniform_tod_output_chunks(
    std::set<Eigen::Index> &selected_0based, Eigen::Index n_scans,
    int n_uniform) {
    n_uniform = std::max(0, n_uniform);
    if (n_scans <= 0 || n_uniform <= 0) {
        return;
    }
    if (n_uniform == 1) {
        selected_0based.insert((n_scans - 1) / 2);
        return;
    }
    for (int i = 0; i < n_uniform; ++i) {
        const double frac =
            static_cast<double>(i) / static_cast<double>(n_uniform - 1);
        Eigen::Index scan_index =
            static_cast<Eigen::Index>(std::lround(frac * (n_scans - 1)));
        scan_index = std::clamp<Eigen::Index>(scan_index, 0, n_scans - 1);
        selected_0based.insert(scan_index);
    }
}

inline void add_source_dense_tod_output_chunks(
    std::set<Eigen::Index> &selected_0based, Eigen::Index n_scans,
    Eigen::Index source_scan, int n_source_dense) {
    n_source_dense = std::max(0, n_source_dense);
    if (n_scans <= 0 || n_source_dense <= 0) {
        return;
    }
    Eigen::Index first_dense =
        source_scan - static_cast<Eigen::Index>((n_source_dense - 1) / 2);
    first_dense = std::clamp<Eigen::Index>(
        first_dense, 0, std::max<Eigen::Index>(0, n_scans - n_source_dense));
    const Eigen::Index last_dense =
        std::min<Eigen::Index>(
            n_scans - 1,
            first_dense + static_cast<Eigen::Index>(n_source_dense) - 1);
    for (Eigen::Index scan_index = first_dense; scan_index <= last_dense;
         ++scan_index) {
        selected_0based.insert(scan_index);
    }
}

inline std::vector<Eigen::Index> selected_tod_output_chunks_1based(
    const std::set<Eigen::Index> &selected_0based) {
    std::vector<Eigen::Index> selected_1based;
    selected_1based.reserve(selected_0based.size());
    for (const auto scan_index : selected_0based) {
        selected_1based.push_back(scan_index + 1);
    }
    return selected_1based;
}

inline std::vector<Eigen::Index> uniform_plus_source_tod_output_chunks(
    Eigen::Index n_scans, int n_uniform, int n_source_dense,
    Eigen::Index source_scan) {
    std::set<Eigen::Index> selected_0based;
    add_uniform_tod_output_chunks(selected_0based, n_scans, n_uniform);
    add_source_dense_tod_output_chunks(
        selected_0based, n_scans, source_scan, n_source_dense);
    return selected_tod_output_chunks_1based(selected_0based);
}

inline void mirror_tod_output_selection_config(
    const std::vector<Eigen::Index> &chunks_1based,
    bool chunk_select_enabled, const std::string &selection_mode,
    int n_uniform, int n_source_dense,
    citlali::config::TodStreamOutputConfig &target) {
    target.chunk_select_enabled = chunk_select_enabled;
    target.chunks_1based.clear();
    target.chunks_1based.reserve(chunks_1based.size());
    for (const auto chunk : chunks_1based) {
        target.chunks_1based.push_back(static_cast<int>(chunk));
    }
    if (auto parsed =
            citlali::config::parse_tod_output_selection_mode(selection_mode)) {
        target.selection_mode = *parsed;
    }
    target.selection_n_uniform = n_uniform;
    target.selection_n_source_dense = n_source_dense;
}

inline bool tod_output_chunk_is_valid(Eigen::Index chunk_1based,
                                      Eigen::Index n_scans) {
    return chunk_1based >= 1 && chunk_1based <= n_scans;
}

inline Eigen::Index assign_all_tod_output_rows(
    Eigen::VectorXI &scan_to_output, Eigen::Index n_scans) {
    scan_to_output.resize(n_scans);
    for (Eigen::Index i = 0; i < n_scans; ++i) {
        scan_to_output(i) = i;
    }
    return n_scans;
}

inline Eigen::Index assign_selected_tod_output_rows(
    Eigen::VectorXI &scan_to_output, Eigen::Index n_scans,
    const std::vector<Eigen::Index> &chunks_1based) {
    scan_to_output.resize(n_scans);
    scan_to_output.setConstant(-1);

    std::set<Eigen::Index> selected_chunks;
    for (const auto chunk_1based : chunks_1based) {
        selected_chunks.insert(chunk_1based - 1);
    }

    Eigen::Index out_index = 0;
    for (Eigen::Index i = 0; i < n_scans; ++i) {
        if (selected_chunks.count(i) > 0) {
            scan_to_output(i) = out_index;
            ++out_index;
        }
    }
    return out_index;
}

}  // namespace citlali::pipeline
