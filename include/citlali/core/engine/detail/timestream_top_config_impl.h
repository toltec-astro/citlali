#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/timestream_config_read.h>

template<typename CT>
void Engine::get_timestream_config(CT &config) {
    logger->info("getting timestream config options");
    auto &timestream_config = typed_config.timestream;
    timestream_config = citlali::config::TimestreamConfig{};

    bool run_tod = timestream_config.enabled;
    citlali::engine_detail::read_timestream_enabled_config(
        config, run_tod, timestream_config, config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    if (!run_tod) {
        logger->error("timestream.enabled is false. This reduction requires TOD processing; set "
                      "low_level.timestream.enabled: true in your reduce config.");
        std::exit(EXIT_FAILURE);
    }
    std::string tod_type{
        std::string(citlali::config::to_string(timestream_config.type))};
    citlali::engine_detail::read_timestream_type_config(
        config, tod_type, timestream_config, config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_auxiliary_quadrature_channel_config(
        config, timestream_config, config_diagnostics.missing_keys, config_diagnostics.invalid_keys);

    // run rtc or ptc tod output?
    // output rtc
    bool run_tod_output_rtc = false;
    citlali::engine_detail::read_raw_tod_output_enabled_config(
        config, run_tod_output_rtc, timestream_config, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys);
    rtcproc.tod_output_mini = false;
    rtcproc.tod_output_outer = false;
    rtcproc.tod_output_outer_context_samples = 0;
    std::string rtc_output_mode = "full";
    citlali::engine_detail::read_tod_stream_output_mode_config(
        config, std::tuple{"timestream", "raw_time_chunk", "output", "mode"},
        run_tod_output_rtc, {"full", "mini", "full_outer", "mini_outer"},
        rtc_output_mode, rtcproc.tod_output_mini, rtcproc.tod_output_outer,
        timestream_config.output.raw_time_chunk, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys);
    citlali::engine_detail::read_tod_stream_outer_context_config(
        config,
        std::tuple{"timestream", "raw_time_chunk", "output",
                   "outer_context_samples"},
        run_tod_output_rtc, rtcproc.tod_output_outer_context_samples,
        timestream_config.output.raw_time_chunk, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys);
    // output ptc
    bool run_tod_output_ptc = false;
    citlali::engine_detail::read_processed_tod_output_enabled_config(
        config, run_tod_output_ptc, timestream_config, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys);
    ptcproc.tod_output_mini = false;
    ptcproc.tod_output_outer = false;
    ptcproc.tod_output_outer_context_samples = 0;
    std::string ptc_output_mode = "full";
    citlali::engine_detail::read_tod_stream_output_mode_config(
        config,
        std::tuple{"timestream", "processed_time_chunk", "output", "mode"},
        run_tod_output_ptc, {"full", "mini"}, ptc_output_mode,
        ptcproc.tod_output_mini, ptcproc.tod_output_outer,
        timestream_config.output.processed_time_chunk, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys);
    bool run_tod_output = false;
    citlali::engine_detail::sync_tod_output_type_config(
        run_tod_output_rtc, run_tod_output_ptc, run_tod_output,
        timestream_config);

    std::string tod_output_subdir_name =
        timestream_config.output.subdir_name;
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "output", "subdir_name"},
        tod_output_subdir_name, timestream_config.output.subdir_name,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "output", "stats", "eigenvalues"},
        diagnostics.write_evals,
        timestream_config.output.write_eigenvalues, config_diagnostics.missing_keys,
        config_diagnostics.invalid_keys);

    // optional selection of TOD chunks to write (1-based indices) under each output block.
    // default is "all" for both rtc and ptc outputs.
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

    citlali::pipeline::parse_tod_output_indices_configs(
        config, run_tod_output_rtc, run_tod_output_ptc,
        rtc_chunk_select_enabled, rtc_output_chunks, ptc_chunk_select_enabled,
        ptc_output_chunks, logger);

    citlali::pipeline::read_tod_selection_mode_config(
        config,
        std::tuple{"timestream","raw_time_chunk","output","selection","mode"},
        std::tuple{"timestream","raw_time_chunk","output","selection","n_uniform"},
        std::tuple{"timestream","raw_time_chunk","output","selection","n_source_dense"},
        run_tod_output_rtc,
        "timestream.raw_time_chunk.output.selection.mode",
        "timestream.raw_time_chunk.output.selection.n_uniform",
        "timestream.raw_time_chunk.output.selection.n_source_dense",
        rtc_selection_mode, rtc_uniform_count, rtc_source_dense_count,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys,
        logger);
    citlali::pipeline::read_tod_selection_mode_config(
        config,
        std::tuple{"timestream","processed_time_chunk","output","selection","mode"},
        std::tuple{"timestream","processed_time_chunk","output","selection","n_uniform"},
        std::tuple{"timestream","processed_time_chunk","output","selection","n_source_dense"},
        run_tod_output_ptc,
        "timestream.processed_time_chunk.output.selection.mode",
        "timestream.processed_time_chunk.output.selection.n_uniform",
        "timestream.processed_time_chunk.output.selection.n_source_dense",
        ptc_selection_mode, ptc_uniform_count, ptc_source_dense_count,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys,
        logger);

    citlali::pipeline::mirror_tod_output_selections_config(
        rtc_output_chunks, rtc_chunk_select_enabled,
        rtc_selection_mode, rtc_uniform_count, rtc_source_dense_count,
        ptc_output_chunks, ptc_chunk_select_enabled, ptc_selection_mode,
        ptc_uniform_count, ptc_source_dense_count,
        timestream_config.output);

    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "chunk_mode"},
        telescope.chunk_mode, timestream_config.chunking.mode,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "value"},
        telescope.chunking_value, timestream_config.chunking.value,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "force_chunking"},
        telescope.force_chunk, timestream_config.chunking.force,
        config_diagnostics.missing_keys, config_diagnostics.invalid_keys);

    /* get raw time chunk config */
    get_rtc_config(config);

    /* get processed time chunk config */
    get_ptc_config(config);

    /* get shared reduction-learning config */
    get_learning_config(config);
}
