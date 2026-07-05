#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/timestream_config_read.h>

template<typename CT>
void Engine::get_timestream_config(CT &config) {
    logger->info("getting timestream config options");
    typed_timestream_config = citlali::config::TimestreamConfig{};

    citlali::engine_detail::read_timestream_enabled_config(
        config, run_tod, typed_timestream_config, missing_keys, invalid_keys);
    if (!run_tod) {
        logger->error("timestream.enabled is false. This reduction requires TOD processing; set "
                      "low_level.timestream.enabled: true in your reduce config.");
        std::exit(EXIT_FAILURE);
    }
    citlali::engine_detail::read_timestream_type_config(
        config, tod_type, typed_timestream_config, missing_keys, invalid_keys);

    // run rtc or ptc tod output?
    // output rtc
    citlali::engine_detail::read_raw_tod_output_enabled_config(
        config, run_tod_output_rtc, typed_timestream_config, missing_keys,
        invalid_keys);
    rtcproc.tod_output_mini = false;
    rtcproc.tod_output_outer = false;
    rtcproc.tod_output_outer_context_samples = 0;
    std::string rtc_output_mode = "full";
    citlali::engine_detail::read_tod_stream_output_mode_config(
        config, std::tuple{"timestream", "raw_time_chunk", "output", "mode"},
        run_tod_output_rtc, {"full", "mini", "full_outer", "mini_outer"},
        rtc_output_mode, rtcproc.tod_output_mini, rtcproc.tod_output_outer,
        typed_timestream_config.output.raw_time_chunk, missing_keys,
        invalid_keys);
    citlali::engine_detail::read_tod_stream_outer_context_config(
        config,
        std::tuple{"timestream", "raw_time_chunk", "output",
                   "outer_context_samples"},
        run_tod_output_rtc, rtcproc.tod_output_outer_context_samples,
        typed_timestream_config.output.raw_time_chunk, missing_keys,
        invalid_keys);
    // output ptc
    citlali::engine_detail::read_processed_tod_output_enabled_config(
        config, run_tod_output_ptc, typed_timestream_config, missing_keys,
        invalid_keys);
    ptcproc.tod_output_mini = false;
    ptcproc.tod_output_outer = false;
    ptcproc.tod_output_outer_context_samples = 0;
    std::string ptc_output_mode = "full";
    citlali::engine_detail::read_tod_stream_output_mode_config(
        config,
        std::tuple{"timestream", "processed_time_chunk", "output", "mode"},
        run_tod_output_ptc, {"full", "mini"}, ptc_output_mode,
        ptcproc.tod_output_mini, ptcproc.tod_output_outer,
        typed_timestream_config.output.processed_time_chunk, missing_keys,
        invalid_keys);
    citlali::engine_detail::sync_tod_output_type_config(
        run_tod_output_rtc, run_tod_output_ptc, run_tod_output,
        tod_output_type, typed_timestream_config);

    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "output", "subdir_name"},
        tod_output_subdir_name, typed_timestream_config.output.subdir_name,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "output", "stats", "eigenvalues"},
        diagnostics.write_evals,
        typed_timestream_config.output.write_eigenvalues, missing_keys,
        invalid_keys);

    // optional selection of TOD chunks to write (1-based indices) under each output block.
    // default is "all" for both rtc and ptc outputs.
    bool rtc_chunk_select_enabled = false;
    bool ptc_chunk_select_enabled = false;
    std::vector<Eigen::Index> rtc_output_chunks, ptc_output_chunks;

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
        tod_output_selection_mode_rtc,
        tod_output_uniform_count_rtc,
        tod_output_source_dense_count_rtc, missing_keys, invalid_keys,
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
        tod_output_selection_mode_ptc,
        tod_output_uniform_count_ptc,
        tod_output_source_dense_count_ptc, missing_keys, invalid_keys,
        logger);

    citlali::pipeline::mirror_tod_output_selections_config(
        rtc_output_chunks, rtc_chunk_select_enabled,
        tod_output_selection_mode_rtc, tod_output_uniform_count_rtc,
        tod_output_source_dense_count_rtc, ptc_output_chunks,
        ptc_chunk_select_enabled, tod_output_selection_mode_ptc,
        tod_output_uniform_count_ptc, tod_output_source_dense_count_ptc,
        typed_timestream_config.output);
    citlali::engine_detail::sync_legacy_tod_output_selection_state(
        run_tod_output_rtc, run_tod_output_ptc, rtc_chunk_select_enabled,
        ptc_chunk_select_enabled, rtc_output_chunks, ptc_output_chunks,
        tod_output_chunk_select_enabled_rtc,
        tod_output_chunk_select_enabled_ptc, tod_output_chunks_rtc,
        tod_output_chunks_ptc, tod_output_chunk_select_enabled,
        tod_output_chunks);

    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "chunk_mode"},
        telescope.chunk_mode, typed_timestream_config.chunking.mode,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "value"},
        telescope.chunking_value, typed_timestream_config.chunking.value,
        missing_keys, invalid_keys);
    citlali::engine_detail::read_mirrored_config_value(
        config, std::tuple{"timestream", "chunking", "force_chunking"},
        telescope.force_chunk, typed_timestream_config.chunking.force,
        missing_keys, invalid_keys);

    /* get raw time chunk config */
    get_rtc_config(config);

    /* get processed time chunk config */
    get_ptc_config(config);

    /* get shared reduction-learning config */
    get_learning_config(config);
}
