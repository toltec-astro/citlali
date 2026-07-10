#pragma once

#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/timestream_config_read.h>
#include <citlali/core/pipeline/tod_output_selection.h>

#include <string>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

template <class Config, class TimestreamConfig, class RtcProc,
          class PtcProc, class Diagnostics>
void read_tod_output_runtime_config(
    Config &config, TimestreamConfig &timestream_config, RtcProc &rtcproc,
    PtcProc &ptcproc, bool &run_tod_output_rtc, bool &run_tod_output_ptc,
    bool &run_tod_output, Diagnostics &diagnostics) {
    run_tod_output_rtc = false;
    read_raw_tod_output_enabled_config(
        config, run_tod_output_rtc, timestream_config, diagnostics);
    rtcproc.tod_output_mini = false;
    rtcproc.tod_output_outer = false;
    rtcproc.tod_output_outer_context_samples = 0;
    std::string rtc_output_mode = "full";
    read_tod_stream_output_mode_config(
        config, std::tuple{"timestream", "raw_time_chunk", "output", "mode"},
        run_tod_output_rtc, {"full", "mini", "full_outer", "mini_outer"},
        rtc_output_mode, rtcproc.tod_output_mini, rtcproc.tod_output_outer,
        timestream_config.output.raw_time_chunk, diagnostics);
    read_tod_stream_outer_context_config(
        config,
        std::tuple{"timestream", "raw_time_chunk", "output",
                   "outer_context_samples"},
        run_tod_output_rtc, rtcproc.tod_output_outer_context_samples,
        timestream_config.output.raw_time_chunk, diagnostics);

    run_tod_output_ptc = false;
    read_processed_tod_output_enabled_config(
        config, run_tod_output_ptc, timestream_config, diagnostics);
    ptcproc.tod_output_mini = false;
    ptcproc.tod_output_outer = false;
    ptcproc.tod_output_outer_context_samples = 0;
    std::string ptc_output_mode = "full";
    read_tod_stream_output_mode_config(
        config,
        std::tuple{"timestream", "processed_time_chunk", "output", "mode"},
        run_tod_output_ptc, {"full", "mini"}, ptc_output_mode,
        ptcproc.tod_output_mini, ptcproc.tod_output_outer,
        timestream_config.output.processed_time_chunk, diagnostics);

    run_tod_output = false;
    sync_tod_output_type_config(
        run_tod_output_rtc, run_tod_output_ptc, run_tod_output,
        timestream_config);
}

template <class Config, class TimestreamConfig, class Diagnostics>
void read_timestream_output_metadata_config(
    Config &config, TimestreamConfig &timestream_config, bool &write_evals,
    Diagnostics &diagnostics) {
    std::string tod_output_subdir_name =
        timestream_config.output.subdir_name;
    read_mirrored_config_value(
        config, std::tuple{"timestream", "output", "subdir_name"},
        tod_output_subdir_name, timestream_config.output.subdir_name,
        diagnostics);
    read_mirrored_config_value(
        config, std::tuple{"timestream", "output", "stats", "eigenvalues"},
        write_evals, timestream_config.output.write_eigenvalues,
        diagnostics);
}

template <class Config, class TimestreamConfig, class Telescope,
          class Diagnostics>
void read_timestream_chunking_config(Config &config,
                                     TimestreamConfig &timestream_config,
                                     Telescope &telescope,
                                     Diagnostics &diagnostics) {
    read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "chunk_mode"},
        telescope.chunk_mode, timestream_config.chunking.mode,
        diagnostics);
    read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "value"},
        telescope.chunking_value, timestream_config.chunking.value,
        diagnostics);
    read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "force_chunking"},
        telescope.force_chunk, timestream_config.chunking.force,
        diagnostics);
}

template <class Config, class TimestreamOutputConfig, class Diagnostics,
          class Logger>
void read_tod_output_selection_config(
    Config &config, bool run_tod_output_rtc, bool run_tod_output_ptc,
    TimestreamOutputConfig &output_config, Diagnostics &diagnostics,
    const Logger &logger) {
    bool rtc_chunk_select_enabled = false;
    bool ptc_chunk_select_enabled = false;
    std::vector<Eigen::Index> rtc_output_chunks, ptc_output_chunks;
    auto rtc_selection_mode =
        citlali::config::TodOutputSelectionMode::indices;
    auto ptc_selection_mode =
        citlali::config::TodOutputSelectionMode::indices;
    int rtc_uniform_count = 10;
    int ptc_uniform_count = 10;
    int rtc_source_dense_count = 10;
    int ptc_source_dense_count = 10;

    parse_tod_output_indices_configs(
        config, run_tod_output_rtc, run_tod_output_ptc,
        rtc_chunk_select_enabled, rtc_output_chunks, ptc_chunk_select_enabled,
        ptc_output_chunks, logger);

    read_tod_selection_mode_config(
        config,
        std::tuple{"timestream", "raw_time_chunk", "output", "selection",
                   "mode"},
        std::tuple{"timestream", "raw_time_chunk", "output", "selection",
                   "n_uniform"},
        std::tuple{"timestream", "raw_time_chunk", "output", "selection",
                   "n_source_dense"},
        run_tod_output_rtc,
        "timestream.raw_time_chunk.output.selection.mode",
        "timestream.raw_time_chunk.output.selection.n_uniform",
        "timestream.raw_time_chunk.output.selection.n_source_dense",
        rtc_selection_mode, rtc_uniform_count, rtc_source_dense_count,
        diagnostics, logger);
    read_tod_selection_mode_config(
        config,
        std::tuple{"timestream", "processed_time_chunk", "output",
                   "selection", "mode"},
        std::tuple{"timestream", "processed_time_chunk", "output",
                   "selection", "n_uniform"},
        std::tuple{"timestream", "processed_time_chunk", "output",
                   "selection", "n_source_dense"},
        run_tod_output_ptc,
        "timestream.processed_time_chunk.output.selection.mode",
        "timestream.processed_time_chunk.output.selection.n_uniform",
        "timestream.processed_time_chunk.output.selection.n_source_dense",
        ptc_selection_mode, ptc_uniform_count, ptc_source_dense_count,
        diagnostics, logger);

    mirror_tod_output_selections_config(
        rtc_output_chunks, rtc_chunk_select_enabled,
        rtc_selection_mode, rtc_uniform_count, rtc_source_dense_count,
        ptc_output_chunks, ptc_chunk_select_enabled, ptc_selection_mode,
        ptc_uniform_count, ptc_source_dense_count, output_config);
}

}  // namespace citlali::pipeline
