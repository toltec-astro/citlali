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

template <class TodStreamOutputConfig, class Logger>
void configure_tod_output_stream_selection(
    const std::string &stream_name, bool output_enabled,
    const TodStreamOutputConfig &config, Eigen::Index n_scans,
    const std::vector<Eigen::Index> &uniform_source_chunks_1based,
    Eigen::VectorXI &scan_to_output, Eigen::Index &n_output_scans,
    const Logger &logger) {
    scan_to_output.resize(n_scans);
    scan_to_output.setConstant(-1);
    n_output_scans = 0;

    if (!output_enabled) {
        logger->info("{} TOD output disabled", stream_name);
        return;
    }

    const auto selection = effective_tod_output_selection(
        config, uniform_source_chunks_1based);
    if (selection.status == TodOutputSelectionStatus::invalid_mode) {
        logger->error("{} TOD output selection mode '{}' is invalid",
                      stream_name,
                      citlali::config::to_string(config.selection_mode));
        throw citlali::error::invalid_config(
            stream_name + " TOD output selection mode is invalid");
    }
    if (selection.status ==
        TodOutputSelectionStatus::empty_uniform_source_selection) {
        logger->error(
            "{} TOD output selection mode uniform_plus_source_crossing selected no chunks",
            stream_name);
        throw citlali::error::runtime(
            stream_name +
            " TOD output uniform-plus-source selection produced no chunks");
    }

    if (!selection.select_enabled || selection.chunks_1based.empty()) {
        n_output_scans =
            assign_all_tod_output_rows(scan_to_output, n_scans);
        logger->info("{} TOD output chunk selection disabled: writing all {} chunks",
                     stream_name, n_output_scans);
        return;
    }

    for (const auto chunk_1based : selection.chunks_1based) {
        if (!tod_output_chunk_is_valid(chunk_1based, n_scans)) {
            logger->error("{} TOD output indices contain {} but valid scan range is [1, {}]",
                          stream_name, chunk_1based, n_scans);
            throw citlali::error::invalid_config(
                stream_name + " TOD output chunk index is outside scan range");
        }
    }

    n_output_scans = assign_selected_tod_output_rows(
        scan_to_output, n_scans, selection.chunks_1based);
    logger->info("{} TOD output chunk selection enabled: writing {} of {} chunks",
                 stream_name, n_output_scans, n_scans);
}
