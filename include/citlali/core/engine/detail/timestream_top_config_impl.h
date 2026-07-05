#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>

template<typename CT>
void Engine::get_timestream_config(CT &config) {
    logger->info("getting timestream config options");
    typed_timestream_config = citlali::config::TimestreamConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return citlali::engine_detail::config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before);
    };

    // run tod processing
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_tod, missing_keys, invalid_keys,
                         std::tuple{"timestream","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.enabled = run_tod;
        }
    }
    if (!run_tod) {
        logger->error("timestream.enabled is false. This reduction requires TOD processing; set "
                      "low_level.timestream.enabled: true in your reduce config.");
        std::exit(EXIT_FAILURE);
    }
    // tod type (xs, rs, is, qs)
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, tod_type, missing_keys, invalid_keys,
                         std::tuple{"timestream","type"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_tod_type(tod_type)) {
                typed_timestream_config.type = *parsed;
            }
        }
    }

    // run rtc or ptc tod output?
    // output rtc
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_tod_output_rtc, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.raw_time_chunk_enabled = run_tod_output_rtc;
            typed_timestream_config.output.raw_time_chunk.enabled = run_tod_output_rtc;
        }
    }
    rtcproc.tod_output_mini = false;
    rtcproc.tod_output_outer = false;
    rtcproc.tod_output_outer_context_samples = 0;
    std::string rtc_output_mode = "full";
    if (run_tod_output_rtc && config.has(std::tuple{"timestream","raw_time_chunk","output","mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, rtc_output_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","mode"},
                         {"full","mini","full_outer","mini_outer"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_tod_stream_output_mode(rtc_output_mode)) {
                typed_timestream_config.output.raw_time_chunk.mode = *parsed;
            }
        }
        rtcproc.tod_output_mini = (rtc_output_mode == "mini" || rtc_output_mode == "mini_outer");
        rtcproc.tod_output_outer = (rtc_output_mode == "full_outer" || rtc_output_mode == "mini_outer");
    }
    if (run_tod_output_rtc && config.has(std::tuple{"timestream","raw_time_chunk","output","outer_context_samples"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, rtcproc.tod_output_outer_context_samples, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","output","outer_context_samples"},
                         {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.raw_time_chunk.outer_context_samples =
                static_cast<int>(rtcproc.tod_output_outer_context_samples);
        }
    }
    // output ptc
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_tod_output_ptc, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","output","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.processed_time_chunk_enabled = run_tod_output_ptc;
            typed_timestream_config.output.processed_time_chunk.enabled = run_tod_output_ptc;
        }
    }
    ptcproc.tod_output_mini = false;
    ptcproc.tod_output_outer = false;
    ptcproc.tod_output_outer_context_samples = 0;
    std::string ptc_output_mode = "full";
    if (run_tod_output_ptc && config.has(std::tuple{"timestream","processed_time_chunk","output","mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, ptc_output_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","output","mode"}, {"full","mini"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_tod_stream_output_mode(ptc_output_mode)) {
                typed_timestream_config.output.processed_time_chunk.mode = *parsed;
            }
        }
        ptcproc.tod_output_mini = (ptc_output_mode == "mini");
    }
    // set tod output to false by default
    run_tod_output = false;

    if (auto requested_output_type =
            citlali::pipeline::requested_tod_output_type_name(
                run_tod_output_rtc, run_tod_output_ptc)) {
        run_tod_output = true;
        tod_output_type = *requested_output_type;
    }
    if (run_tod_output) {
        if (auto parsed = citlali::config::parse_tod_output_type(tod_output_type)) {
            typed_timestream_config.output.type = *parsed;
        }
    }

    // tod subdirectory name
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, tod_output_subdir_name, missing_keys, invalid_keys,
                         std::tuple{"timestream","output", "subdir_name"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.subdir_name = tod_output_subdir_name;
        }
    }
    // write eigenvalues to stats file
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, diagnostics.write_evals, missing_keys, invalid_keys,
                         std::tuple{"timestream","output", "stats","eigenvalues"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.output.write_eigenvalues = diagnostics.write_evals;
        }
    }

    // optional selection of TOD chunks to write (1-based indices) under each output block.
    // default is "all" for both rtc and ptc outputs.
    bool rtc_chunk_select_enabled = false;
    bool ptc_chunk_select_enabled = false;
    std::vector<Eigen::Index> rtc_output_chunks, ptc_output_chunks;

    citlali::pipeline::parse_tod_output_indices_config(
        config, std::tuple{"timestream","raw_time_chunk","output","indices"},
        run_tod_output_rtc, "timestream.raw_time_chunk.output.indices",
        rtc_chunk_select_enabled, rtc_output_chunks, logger);
    citlali::pipeline::parse_tod_output_indices_config(
        config,
        std::tuple{"timestream","processed_time_chunk","output","indices"},
        run_tod_output_ptc,
        "timestream.processed_time_chunk.output.indices",
        ptc_chunk_select_enabled, ptc_output_chunks, logger);

    auto parse_tod_selection_mode = [&](const auto &mode_key,
                                        const auto &n_uniform_key,
                                        const auto &n_source_dense_key,
                                        bool output_enabled,
                                        const std::string &mode_path,
                                        const std::string &n_uniform_path,
                                        const std::string &n_source_dense_path,
                                        std::string &mode,
                                        int &n_uniform,
                                        int &n_source_dense) {
        mode = "indices";
        n_uniform = 10;
        n_source_dense = 10;
        if (!output_enabled) {
            return;
        }
        if (config.has(mode_key)) {
            get_config_value(config, mode, missing_keys, invalid_keys, mode_key,
                             {"indices", "all", "uniform_plus_source_crossing"});
        }
        citlali::pipeline::read_tod_selection_count_config(
            config, n_uniform_key, n_uniform_path, n_uniform, logger);
        citlali::pipeline::read_tod_selection_count_config(
            config, n_source_dense_key, n_source_dense_path, n_source_dense,
            logger);
        if (mode == "uniform_plus_source_crossing" && n_uniform + n_source_dense <= 0) {
            logger->error("{} selects uniform_plus_source_crossing but {} + {} is zero",
                          mode_path, n_uniform_path, n_source_dense_path);
            std::exit(EXIT_FAILURE);
        }
    };

    parse_tod_selection_mode(
        std::tuple{"timestream","raw_time_chunk","output","selection","mode"},
        std::tuple{"timestream","raw_time_chunk","output","selection","n_uniform"},
        std::tuple{"timestream","raw_time_chunk","output","selection","n_source_dense"},
        run_tod_output_rtc,
        "timestream.raw_time_chunk.output.selection.mode",
        "timestream.raw_time_chunk.output.selection.n_uniform",
        "timestream.raw_time_chunk.output.selection.n_source_dense",
        tod_output_selection_mode_rtc,
        tod_output_uniform_count_rtc,
        tod_output_source_dense_count_rtc);
    parse_tod_selection_mode(
        std::tuple{"timestream","processed_time_chunk","output","selection","mode"},
        std::tuple{"timestream","processed_time_chunk","output","selection","n_uniform"},
        std::tuple{"timestream","processed_time_chunk","output","selection","n_source_dense"},
        run_tod_output_ptc,
        "timestream.processed_time_chunk.output.selection.mode",
        "timestream.processed_time_chunk.output.selection.n_uniform",
        "timestream.processed_time_chunk.output.selection.n_source_dense",
        tod_output_selection_mode_ptc,
        tod_output_uniform_count_ptc,
        tod_output_source_dense_count_ptc);

    citlali::pipeline::mirror_tod_output_selection_config(
        rtc_output_chunks, rtc_chunk_select_enabled,
        tod_output_selection_mode_rtc, tod_output_uniform_count_rtc,
        tod_output_source_dense_count_rtc,
        typed_timestream_config.output.raw_time_chunk);
    citlali::pipeline::mirror_tod_output_selection_config(
        ptc_output_chunks, ptc_chunk_select_enabled,
        tod_output_selection_mode_ptc, tod_output_uniform_count_ptc,
        tod_output_source_dense_count_ptc,
        typed_timestream_config.output.processed_time_chunk);

    tod_output_chunk_select_enabled_rtc = rtc_chunk_select_enabled;
    tod_output_chunk_select_enabled_ptc = ptc_chunk_select_enabled;
    tod_output_chunks_rtc = std::move(rtc_output_chunks);
    tod_output_chunks_ptc = std::move(ptc_output_chunks);

    // keep legacy shared fields aligned with rtc (or ptc if rtc is disabled)
    if (run_tod_output_rtc) {
        tod_output_chunk_select_enabled = tod_output_chunk_select_enabled_rtc;
        tod_output_chunks = tod_output_chunks_rtc;
    }
    else if (run_tod_output_ptc) {
        tod_output_chunk_select_enabled = tod_output_chunk_select_enabled_ptc;
        tod_output_chunks = tod_output_chunks_ptc;
    }
    else {
        tod_output_chunk_select_enabled = false;
        tod_output_chunks.clear();
    }

    // get time chunk size
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, telescope.chunk_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","chunking", "chunk_mode"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.chunking.mode = telescope.chunk_mode;
        }
    }
    // get time chunk size
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, telescope.chunking_value, missing_keys, invalid_keys,
                         std::tuple{"timestream","chunking", "value"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.chunking.value = telescope.chunking_value;
        }
    }
    // force chunking?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, telescope.force_chunk, missing_keys, invalid_keys,
                         std::tuple{"timestream","chunking", "force_chunking"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.chunking.force = telescope.force_chunk;
        }
    }

    /* get raw time chunk config */
    get_rtc_config(config);

    /* get processed time chunk config */
    get_ptc_config(config);

    /* get shared reduction-learning config */
    get_learning_config(config);
}
