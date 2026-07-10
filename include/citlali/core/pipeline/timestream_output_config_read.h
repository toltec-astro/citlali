#pragma once

#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/timestream_config_read.h>

#include <string>
#include <tuple>

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

}  // namespace citlali::pipeline
