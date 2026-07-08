#pragma once

// Included by tod_output_selection.h inside namespace citlali::pipeline.

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

enum class TodOutputSelectionStatus {
    valid,
    invalid_mode,
    empty_uniform_source_selection
};

struct EffectiveTodOutputSelection {
    bool select_enabled = false;
    std::vector<Eigen::Index> chunks_1based;
    TodOutputSelectionStatus status = TodOutputSelectionStatus::valid;
};

template <class TodStreamOutputConfig>
EffectiveTodOutputSelection effective_tod_output_selection(
    const TodStreamOutputConfig &config,
    const std::vector<Eigen::Index> &uniform_source_chunks_1based) {
    EffectiveTodOutputSelection selection;
    selection.chunks_1based.reserve(config.chunks_1based.size());
    for (const auto chunk : config.chunks_1based) {
        selection.chunks_1based.push_back(static_cast<Eigen::Index>(chunk));
    }
    selection.select_enabled = config.chunk_select_enabled;

    if (citlali::config::is_all_tod_output_selection_mode(
            config.selection_mode)) {
        selection.select_enabled = false;
        selection.chunks_1based.clear();
    }
    else if (citlali::config::is_uniform_source_tod_output_selection_mode(
                 config.selection_mode)) {
        selection.select_enabled = true;
        selection.chunks_1based = uniform_source_chunks_1based;
        if (selection.chunks_1based.empty()) {
            selection.status =
                TodOutputSelectionStatus::empty_uniform_source_selection;
        }
    }
    else if (!citlali::config::is_indices_tod_output_selection_mode(
                 config.selection_mode)) {
        selection.status = TodOutputSelectionStatus::invalid_mode;
    }

    return selection;
}
