#pragma once

// Included by tod_output_selection.h inside namespace citlali::pipeline.

inline void mirror_tod_output_selection_config(
    const std::vector<Eigen::Index> &chunks_1based,
    bool chunk_select_enabled,
    citlali::config::TodOutputSelectionMode selection_mode,
    int n_uniform, int n_source_dense,
    citlali::config::TodStreamOutputConfig &target) {
    target.chunk_select_enabled = chunk_select_enabled;
    target.chunks_1based.clear();
    target.chunks_1based.reserve(chunks_1based.size());
    for (const auto chunk : chunks_1based) {
        target.chunks_1based.push_back(static_cast<int>(chunk));
    }
    target.selection_mode = selection_mode;
    target.selection_n_uniform = n_uniform;
    target.selection_n_source_dense = n_source_dense;
}

template <class OutputConfig>
void mirror_tod_output_selections_config(
    const std::vector<Eigen::Index> &raw_chunks_1based,
    bool raw_chunk_select_enabled,
    citlali::config::TodOutputSelectionMode raw_selection_mode,
    int raw_n_uniform, int raw_n_source_dense,
    const std::vector<Eigen::Index> &processed_chunks_1based,
    bool processed_chunk_select_enabled,
    citlali::config::TodOutputSelectionMode processed_selection_mode,
    int processed_n_uniform, int processed_n_source_dense,
    OutputConfig &target) {
    mirror_tod_output_selection_config(
        raw_chunks_1based, raw_chunk_select_enabled, raw_selection_mode,
        raw_n_uniform, raw_n_source_dense, target.raw_time_chunk);
    mirror_tod_output_selection_config(
        processed_chunks_1based, processed_chunk_select_enabled,
        processed_selection_mode, processed_n_uniform,
        processed_n_source_dense, target.processed_time_chunk);
}
