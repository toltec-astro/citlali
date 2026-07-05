#pragma once

// Included by tod_output_selection.h inside namespace citlali::pipeline.

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

