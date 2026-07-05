#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

template<typename CT>
void Engine::get_timestream_config(CT &config) {
    logger->info("getting timestream config options");
    typed_timestream_config = citlali::config::TimestreamConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
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

    // check if rtc output is requested
    if (run_tod_output_rtc) {
        run_tod_output = true;
        tod_output_type = "rtc";
    }
    // if ptc output is requested
    if (run_tod_output_ptc) {
        // check if rtc output was requested
        if (run_tod_output == true) {
            tod_output_type = "both";
        }
        // else just output ptc
        else {
            run_tod_output = true;
            tod_output_type = "ptc";
        }
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
    auto parse_tod_output_indices = [&](const auto &indices_key, bool output_enabled, const std::string &config_path,
                                        bool &select_enabled, std::vector<Eigen::Index> &chunks_out) {
        select_enabled = false;
        chunks_out.clear();

        if (!output_enabled || !config.has(indices_key)) {
            return;
        }

        if (config.template has_typed<std::string>(indices_key)) {
            const auto indices_value = config.template get_typed<std::string>(indices_key);
            if (indices_value == "all") {
                return;
            }
            logger->error("{} must be \"all\" or a non-empty list of 1-based positive integers. Found \"{}\"",
                          config_path, indices_value);
            std::exit(EXIT_FAILURE);
        }

        if (config.template has_typed<std::vector<int>>(indices_key)) {
            const auto chunks = config.template get_typed<std::vector<int>>(indices_key);
            if (chunks.empty()) {
                logger->error("{} must be \"all\" or a non-empty list of 1-based positive integers", config_path);
                std::exit(EXIT_FAILURE);
            }
            select_enabled = true;
            for (const auto chunk_index : chunks) {
                if (chunk_index <= 0) {
                    logger->error("{} must be 1-based positive integers. Found {}", config_path, chunk_index);
                    std::exit(EXIT_FAILURE);
                }
                chunks_out.push_back(static_cast<Eigen::Index>(chunk_index));
            }
            return;
        }

        logger->error("{} must be \"all\" or a list of 1-based positive integers", config_path);
        std::exit(EXIT_FAILURE);
    };

    bool rtc_chunk_select_enabled = false;
    bool ptc_chunk_select_enabled = false;
    std::vector<Eigen::Index> rtc_output_chunks, ptc_output_chunks;

    parse_tod_output_indices(std::tuple{"timestream","raw_time_chunk","output","indices"}, run_tod_output_rtc,
                             "timestream.raw_time_chunk.output.indices", rtc_chunk_select_enabled, rtc_output_chunks);
    parse_tod_output_indices(std::tuple{"timestream","processed_time_chunk","output","indices"}, run_tod_output_ptc,
                             "timestream.processed_time_chunk.output.indices", ptc_chunk_select_enabled, ptc_output_chunks);

    auto read_tod_selection_count = [&](const auto &key, const std::string &config_path,
                                        int &value) {
        if (!config.template has_typed<int>(key)) {
            return;
        }
        value = config.template get_typed<int>(key);
        if (value < 0) {
            logger->error("{} must be non-negative. Found {}", config_path, value);
            std::exit(EXIT_FAILURE);
        }
    };

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
        read_tod_selection_count(n_uniform_key, n_uniform_path, n_uniform);
        read_tod_selection_count(n_source_dense_key, n_source_dense_path, n_source_dense);
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

    auto mirror_tod_output_selection = [](const std::vector<Eigen::Index> &chunks_1based,
                                          bool chunk_select_enabled,
                                          const std::string &selection_mode,
                                          int n_uniform,
                                          int n_source_dense,
                                          citlali::config::TodStreamOutputConfig &target) {
        target.chunk_select_enabled = chunk_select_enabled;
        target.chunks_1based.clear();
        target.chunks_1based.reserve(chunks_1based.size());
        for (const auto chunk : chunks_1based) {
            target.chunks_1based.push_back(static_cast<int>(chunk));
        }
        if (auto parsed = citlali::config::parse_tod_output_selection_mode(selection_mode)) {
            target.selection_mode = *parsed;
        }
        target.selection_n_uniform = n_uniform;
        target.selection_n_source_dense = n_source_dense;
    };

    mirror_tod_output_selection(rtc_output_chunks, rtc_chunk_select_enabled,
                                tod_output_selection_mode_rtc,
                                tod_output_uniform_count_rtc,
                                tod_output_source_dense_count_rtc,
                                typed_timestream_config.output.raw_time_chunk);
    mirror_tod_output_selection(ptc_output_chunks, ptc_chunk_select_enabled,
                                tod_output_selection_mode_ptc,
                                tod_output_uniform_count_ptc,
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

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    logger->info("getting mapmaking config options");
    typed_mapmaking_config = citlali::config::MapmakingConfig{};
    typed_coadd_config = citlali::config::CoaddConfig{};
    typed_noise_config = citlali::config::NoiseConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };

    // enable mapmaking?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_mapmaking, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_mapmaking_config.enabled = run_mapmaking;
        }
    }
    // map grouping
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, map_grouping, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","grouping"},{"auto","array","nw","detector","fg"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_map_grouping(map_grouping)) {
                typed_mapmaking_config.grouping = *parsed;
            }
        }
    }

    // optional expected sky regime for interpreting map diagnostics
    map_regime = "unknown";
    if (config.template has_typed<std::string>(std::tuple{"source", "map_regime"})) {
        map_regime = config.template get_typed<std::string>(std::tuple{"source", "map_regime"});
        check_allowed(map_regime, missing_keys, invalid_keys,
                      std::vector<std::string>{"source_dominant", "source_faint", "blank_field", "unknown"},
                      std::tuple{"source", "map_regime"});
    }

    // polarization is disabled for detector grouping
    if (rtcproc.run_polarization && ((redu_type=="beammap" && map_grouping=="auto") || map_grouping=="detector")) {
        logger->error("Detector grouping reductions do not currently support polarimetry mode");
        std::exit(EXIT_FAILURE);
    }

    // set rtcproc map_grouping
    rtcproc.kernel.map_grouping = map_grouping;
    ptcproc.active_map_grouping = map_grouping;

    // map_method
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, map_method, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","method"},{"naive","jinc","maximum_likelihood"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_map_method(map_method)) {
                typed_mapmaking_config.method = *parsed;
            }
        }
    }
    std::string fruit_interp_default = (map_method == "jinc") ? "jinc" : "bilinear";
    ptcproc.fruit_loops_interp_mode = fruit_interp_default;
    if (ptcproc.run_fruit_loops && ptcproc.fruit_loops_interp_mode_override != "auto") {
        ptcproc.fruit_loops_interp_mode = ptcproc.fruit_loops_interp_mode_override;
    }
    if (ptcproc.fruit_loops_interp_mode == "jinc" && map_method != "jinc") {
        logger->warn("fruit_loops.interp_mode_override='jinc' requires mapmaking.method='jinc'; using bilinear");
        ptcproc.fruit_loops_interp_mode = "bilinear";
    }
    logger->info("fruit loops interpolation mode: {} (default from mapmaking.method='{}' is {})",
                 ptcproc.fruit_loops_interp_mode, map_method, fruit_interp_default);
    logger->info("fruit loops center convention: {}",
                 ptcproc.fruit_loops_legacy_center ? "legacy n/2" : "current (n-1)/2");
    logger->info("fruit loops post-addback weight mode: {}",
                 ptcproc.fruit_loops_recompute_weights_after_addback
                     ? "recompute from add-back TOD"
                     : "keep source-subtracted");
    logger->info("fruit loops weight feedback: enabled={} reference={} relative=[{}, {}]",
                 ptcproc.fruit_loops_weight_feedback_enabled,
                 ptcproc.fruit_loops_weight_feedback_reference,
                 ptcproc.fruit_loops_weight_feedback_low_relative_weight,
                 ptcproc.fruit_loops_weight_feedback_high_relative_weight);
    ptcproc.fruit_loops_jinc_r_max = 0.0;
    ptcproc.fruit_loops_jinc_subpixel_n = 1;
    ptcproc.fruit_loops_jinc_shape_params.clear();

    // map reference frame (radec, altaz, galactic)
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, telescope.pixel_axes, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","pixel_axes"},{"radec","altaz", "galactic"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_mapmaking_config.pixel_axes = telescope.pixel_axes;
        }
    }
    if (redu_type == "beammap" && telescope.pixel_axes != "altaz") {
        logger->error(
            "beammap reductions require mapmaking.pixel_axes='altaz'; got '{}'",
            telescope.pixel_axes);
        std::exit(EXIT_FAILURE);
    }

    // get config for omb
    logger->info("getting omb config options");
    const auto omb_missing_before = missing_keys.size();
    const auto omb_invalid_before = invalid_keys.size();
    omb.get_config(config, missing_keys, invalid_keys, telescope.pixel_axes, redu_type);
    if (parsed_cleanly(omb_missing_before, omb_invalid_before)) {
        typed_mapmaking_config.coverage_cut = omb.cov_cut;
        typed_mapmaking_config.pixel_size_arcsec = omb.pixel_size_rad * RAD_TO_ASEC;
        typed_mapmaking_config.unit = omb.sig_unit;
        if (omb.wcs.naxis.size() >= 2) {
            typed_mapmaking_config.x_size_pix = static_cast<int>(omb.wcs.naxis[0]);
            typed_mapmaking_config.y_size_pix = static_cast<int>(omb.wcs.naxis[1]);
        }
        if (omb.wcs.crpix.size() >= 2) {
            typed_mapmaking_config.crpix1 = omb.wcs.crpix[0];
            typed_mapmaking_config.crpix2 = omb.wcs.crpix[1];
        }
        if (omb.crval_config.size() >= 2) {
            typed_mapmaking_config.crval1_j2000 = omb.crval_config[0];
            typed_mapmaking_config.crval2_j2000 = omb.crval_config[1];
        }
        typed_post_processing_config.map_histogram_n_bins = omb.hist_n_bins;
    }

    // run coaddition?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_coadd, missing_keys, invalid_keys,
                         std::tuple{"coadd","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_coadd_config.enabled = run_coadd;
        }
    }
    // re-run to get config for cmb
    if (run_coadd) {
        logger->info("getting cmb config options");
        cmb.get_config(config, missing_keys, invalid_keys, telescope.pixel_axes, redu_type);
    }

    // if flux calibration is not enabled, use tod type units (xs, rs, is, or qs)
    if (!rtcproc.run_calibrate) {
        omb.sig_unit = tod_type;
        cmb.sig_unit = tod_type;
    }

    // set parallelization for psd filter ffts (maintained with tod output/verbose mode)
    omb.parallel_policy = parallel_policy;
    cmb.parallel_policy = parallel_policy;
    jinc_mm.parallel_policy = parallel_policy;

    if (map_method=="jinc") {
        // maximum radius for jinc filter
        get_config_value(config, jinc_mm.r_max, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","jinc_filter","r_max"});
        // get jinc filter shape params
        for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
            auto jinc_shape_vec = config.template get_typed<std::vector<double>>(std::tuple{"mapmaking","jinc_filter","shape_params",arr_name});
            if (jinc_shape_vec.size() != 3) {
                invalid_keys.push_back({"mapmaking","jinc_filter","shape_params",arr_name});
                jinc_shape_vec.resize(3, 0.0);
            }
            jinc_mm.shape_params[arr_index] = Eigen::Map<Eigen::VectorXd>(jinc_shape_vec.data(),jinc_shape_vec.size());
        }
        // optional: sub-pixel sampling for jinc kernel
        if (config.template has_typed<int>(std::tuple{"mapmaking","jinc_filter","subpixel_n"})) {
            get_config_value(config, jinc_mm.subpixel_n, missing_keys, invalid_keys,
                             std::tuple{"mapmaking","jinc_filter","subpixel_n"},{},{1});
        }
        ptcproc.fruit_loops_jinc_r_max = jinc_mm.r_max;
        ptcproc.fruit_loops_jinc_subpixel_n = jinc_mm.subpixel_n;
        ptcproc.fruit_loops_jinc_shape_params = jinc_mm.shape_params;

        if (jinc_mm.mode=="matrix") {
            // allocate jinc matrix
            jinc_mm.allocate_jinc_matrix(omb.pixel_size_rad);
        }
        else if (jinc_mm.mode=="splines") {
            // precompute jinc spline
            jinc_mm.calculate_jinc_splines();
        }
    }

    else if (map_method=="maximum_likelihood") {
        get_config_value(config, ml_mm.tolerance, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","maximum_likelihood","tolerance"});
        get_config_value(config, ml_mm.max_iterations, missing_keys, invalid_keys,
                         std::tuple{"mapmaking","maximum_likelihood","max_iterations"});
    }

    // make noise maps?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_noise, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.enabled = run_noise;
        }
    }
    if (run_noise) {
        // number of noise maps
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.n_noise, missing_keys, invalid_keys,
                             std::tuple{"noise_maps","n_noise_maps"},{},{0},{});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_noise_config.n_noise_maps = static_cast<int>(omb.n_noise);
            }
        }
        // randomize noise maps on detector as well as time chunk
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.randomize_dets, missing_keys, invalid_keys,
                             std::tuple{"noise_maps","randomize_dets"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_noise_config.randomize_dets = omb.randomize_dets;
            }
        }

        if (run_coadd) {
            // copy omb number of noise maps to cmb
            cmb.n_noise = omb.n_noise;
            // copy randomize_dets to cmb
            cmb.randomize_dets = omb.randomize_dets;
        }
    }
    // otherwise set number of noise maps to zero
    else {
        omb.n_noise = 0;
        cmb.n_noise = 0;
        typed_noise_config.n_noise_maps = 0;
    }

    write_noise_realizations = false;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","write_realizations"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, write_noise_realizations, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","write_realizations"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.write_realizations = write_noise_realizations;
        }
    }
    run_noise_products = run_noise;
    typed_noise_config.products_enabled = run_noise_products;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","products","enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_noise_products, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","products","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.products_enabled = run_noise_products;
        }
    }
    apply_empirical_noise_weights = run_noise;
    typed_noise_config.apply_empirical_weights = apply_empirical_noise_weights;
    if (config.template has_typed<bool>(std::tuple{"noise_maps","products","apply_empirical_weights"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, apply_empirical_noise_weights, missing_keys, invalid_keys,
                         std::tuple{"noise_maps","products","apply_empirical_weights"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_noise_config.apply_empirical_weights = apply_empirical_noise_weights;
        }
    }

    // set mapmaker polarization
    naive_mm.run_polarization = rtcproc.run_polarization;
    jinc_mm.run_polarization = rtcproc.run_polarization;
}

template<typename CT>
void Engine::get_pointing_config(CT &config) {
    logger->info("getting pointing config options");
    typed_pointing_config = citlali::config::PointingConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };

    pointing_source_strategy = "standard";
    if (config.template has_typed<std::string>(std::tuple{"pointing","source_strategy","mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_source_strategy, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","mode"},
                         {"standard", "psf_preserve"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_pointing_source_strategy(
                    pointing_source_strategy)) {
                typed_pointing_config.source_strategy = *parsed;
            }
        }
    }

    pointing_fit_gaussian_enabled = (pointing_source_strategy == "standard");
    typed_pointing_config.fit_gaussian = pointing_fit_gaussian_enabled;
    if (config.template has_typed<bool>(std::tuple{"pointing","source_strategy","fit_gaussian"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_fit_gaussian_enabled, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","fit_gaussian"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_pointing_config.fit_gaussian = pointing_fit_gaussian_enabled;
        }
    }

    pointing_fruitloops_center_mode =
        (pointing_source_strategy == "psf_preserve") ? "map_center" : "auto";
    if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
            pointing_fruitloops_center_mode)) {
        typed_pointing_config.fruitloops_center_mode = *parsed;
    }
    if (config.template has_typed<std::string>(std::tuple{"pointing","source_strategy","fruitloops_center_mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_fruitloops_center_mode, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","fruitloops_center_mode"},
                         {"auto", "header", "peak", "map_center"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
                    pointing_fruitloops_center_mode)) {
                typed_pointing_config.fruitloops_center_mode = *parsed;
            }
        }
    }

    pointing_header_center_max_radius_arcsec = 0.0;
    if (pointing_source_strategy == "standard" &&
        std::isfinite(map_fitter.fitting_region_pix) && map_fitter.fitting_region_pix > 0.0 &&
        std::isfinite(omb.pixel_size_rad) && omb.pixel_size_rad > 0.0) {
        pointing_header_center_max_radius_arcsec =
            map_fitter.fitting_region_pix * omb.pixel_size_rad * RAD_TO_ASEC;
    }
    typed_pointing_config.header_max_radius_arcsec =
        pointing_header_center_max_radius_arcsec;
    if (config.template has_typed<double>(std::tuple{"pointing","source_strategy","header_max_radius_arcsec"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_header_center_max_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","header_max_radius_arcsec"},
                         {}, {0.0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_pointing_config.header_max_radius_arcsec =
                pointing_header_center_max_radius_arcsec;
        }
    }

    pointing_header_center_require_coverage = true;
    typed_pointing_config.header_require_coverage =
        pointing_header_center_require_coverage;
    if (config.template has_typed<bool>(std::tuple{"pointing","source_strategy","header_require_coverage"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_header_center_require_coverage, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","header_require_coverage"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_pointing_config.header_require_coverage =
                pointing_header_center_require_coverage;
        }
    }

    ptcproc.fruit_loops_source_center_mode = pointing_fruitloops_center_mode;
    ptcproc.fruit_loops_header_center_max_radius_arcsec =
        pointing_header_center_max_radius_arcsec;
    ptcproc.fruit_loops_header_center_require_coverage =
        pointing_header_center_require_coverage;

    logger->info("pointing source strategy: mode={} fit_gaussian={} fruitloops_center_mode={} "
                 "header_max_radius_arcsec={} header_require_coverage={}",
                 pointing_source_strategy, pointing_fit_gaussian_enabled,
                 pointing_fruitloops_center_mode,
                 pointing_header_center_max_radius_arcsec,
                 pointing_header_center_require_coverage);

    if (!ptcproc.run_fruit_loops) {
        logger->warn("pointing source strategy is configured but timestream.fruit_loops.enabled=false");
    }
    else if (ptcproc.fruit_loops_iters < 2) {
        logger->warn("pointing source-aware fruit loops uses previous maps; max_iters={} will not run a measurement iteration",
                     ptcproc.fruit_loops_iters);
    }

    if (pointing_source_strategy == "psf_preserve" && pointing_fit_gaussian_enabled) {
        logger->warn("pointing.source_strategy.mode=psf_preserve with fit_gaussian=true; "
                     "Gaussian fits remain diagnostics only and do not constrain fruit loops");
    }
    if (pointing_source_strategy == "psf_preserve" &&
        pointing_fruitloops_center_mode == "peak") {
        logger->warn("pointing.source_strategy.mode=psf_preserve with fruitloops_center_mode=peak; "
                     "messy out-of-focus maps may bias the fruit loops source support");
    }
    if (!pointing_fit_gaussian_enabled &&
        (pointing_fruitloops_center_mode == "header" ||
         pointing_fruitloops_center_mode == "auto")) {
        logger->warn("pointing Gaussian fitting is disabled; later fruit loops iterations will not "
                     "get new valid POINTING header centers from this run");
    }
}

template<typename CT>
void Engine::get_beammap_config(CT &config) {
    logger->info("getting beammap config options");
    // max beammap iteration
    get_config_value(config, beammap_iter_max, missing_keys, invalid_keys,
                     std::tuple{"beammap","iter_max"});
    // beammap iteration tolerance
    get_config_value(config, beammap_iter_tolerance, missing_keys, invalid_keys,
                     std::tuple{"beammap","iter_tolerance"});
    beammap_convergence_radius_arcsec = 10.0;
    if (config.template has_typed<double>(std::tuple{"beammap","convergence_radius_arcsec"})) {
        get_config_value(config, beammap_convergence_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"beammap","convergence_radius_arcsec"},
                         {}, {0.0});
    }

    beammap_phase_split_enabled = true;
    beammap_locator_iter = 0;
    beammap_measurement_start_iter = 1;
    if (config.template has_typed<bool>(std::tuple{"beammap","phase_strategy","enabled"})) {
        get_config_value(config, beammap_phase_split_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","phase_strategy","locator_iter"})) {
        get_config_value(config, beammap_locator_iter, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","locator_iter"},
                         {}, {0});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","phase_strategy","measurement_start_iter"})) {
        get_config_value(config, beammap_measurement_start_iter, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","measurement_start_iter"},
                         {}, {1});
    }
    if (beammap_locator_iter != 0) {
        logger->warn(
            "beammap.phase_strategy.locator_iter={} requested, but the locator pass must be iter 0; using 0",
            beammap_locator_iter);
        beammap_locator_iter = 0;
    }
    if (beammap_measurement_start_iter <= beammap_locator_iter) {
        logger->warn(
            "beammap.phase_strategy.measurement_start_iter={} must be after locator_iter={}; using {}",
            beammap_measurement_start_iter, beammap_locator_iter, beammap_locator_iter + 1);
        beammap_measurement_start_iter = beammap_locator_iter + 1;
    }
    if (beammap_iter_max <= beammap_measurement_start_iter) {
        logger->warn(
            "beammap.iter_max={} will not run a measurement pass with measurement_start_iter={}",
            beammap_iter_max, beammap_measurement_start_iter);
    }

    // beammap reference detector
    get_config_value(config, beammap_reference_det, missing_keys, invalid_keys,
                     std::tuple{"beammap","reference_det"});
    // subtract reference detector?
    get_config_value(config, beammap_subtract_reference, missing_keys, invalid_keys,
                     std::tuple{"beammap","subtract_reference_det"});
    // derotate apt?
    get_config_value(config, beammap_derotate, missing_keys, invalid_keys,
                     std::tuple{"beammap","derotate"});

    // optional robust sample-level RFI masking (detector grouping)
    beammap_rfi_mask_enabled = false;
    beammap_rfi_mask_block_size_samples = 64;
    beammap_rfi_mask_min_good_samples = 32;
    beammap_rfi_mask_dilate_blocks = 1;
    beammap_rfi_mask_sigma_threshold = 6.0;
    beammap_rfi_mask_sigma_floor = 0.0;
    beammap_rfi_mask_max_flagged_fraction = 0.35;

    if (config.template has_typed<bool>(std::tuple{"beammap","rfi_mask","enabled"})) {
        get_config_value(config, beammap_rfi_mask_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","block_size_samples"})) {
        get_config_value(config, beammap_rfi_mask_block_size_samples, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","block_size_samples"},
                         {}, {8});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","min_good_samples"})) {
        get_config_value(config, beammap_rfi_mask_min_good_samples, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","min_good_samples"},
                         {}, {4});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","dilate_blocks"})) {
        get_config_value(config, beammap_rfi_mask_dilate_blocks, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","dilate_blocks"},
                         {}, {0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","sigma_threshold"})) {
        get_config_value(config, beammap_rfi_mask_sigma_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","sigma_threshold"},
                         {}, {1.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","sigma_floor"})) {
        get_config_value(config, beammap_rfi_mask_sigma_floor, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","sigma_floor"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","max_flagged_fraction"})) {
        get_config_value(config, beammap_rfi_mask_max_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","max_flagged_fraction"},
                         {}, {0.0}, {1.0});
    }

    beammap_detector_weighting_mode = "const";
    if (config.template has_typed<std::string>(std::tuple{"beammap","detector_weighting","mode"})) {
        get_config_value(config, beammap_detector_weighting_mode, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_weighting","mode"},
                         {"const", "ptc", "ptc_after_iter0"});
    }

    beammap_fit_radius_fwhm = 0.0;
    if (config.template has_typed<double>(std::tuple{"beammap","fitting","fit_radius_fwhm"})) {
        get_config_value(config, beammap_fit_radius_fwhm, missing_keys, invalid_keys,
                         std::tuple{"beammap","fitting","fit_radius_fwhm"},
                         {}, {0.0});
    }
    map_fitter.beammap_fit_radius_fwhm = beammap_fit_radius_fwhm;

    // optional detector-map edge-band masking for coherent bad scan legs
    beammap_scan_band_mask_enabled = false;
    beammap_scan_band_mask_edge_rows = 24;
    beammap_scan_band_mask_min_row_pixels = 8;
    beammap_scan_band_mask_min_contiguous_rows = 2;
    beammap_scan_band_mask_row_median_sigma_threshold = 4.0;
    beammap_scan_band_mask_row_sigma_ratio_threshold = 2.5;
    beammap_scan_band_mask_max_flagged_fraction = 0.30;

    if (config.template has_typed<bool>(std::tuple{"beammap","scan_band_mask","enabled"})) {
        get_config_value(config, beammap_scan_band_mask_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","edge_rows"})) {
        get_config_value(config, beammap_scan_band_mask_edge_rows, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","edge_rows"},
                         {}, {2});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","min_row_pixels"})) {
        get_config_value(config, beammap_scan_band_mask_min_row_pixels, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","min_row_pixels"},
                         {}, {1});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","min_contiguous_rows"})) {
        get_config_value(config, beammap_scan_band_mask_min_contiguous_rows, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","min_contiguous_rows"},
                         {}, {1});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","row_median_sigma_threshold"})) {
        get_config_value(config, beammap_scan_band_mask_row_median_sigma_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","row_median_sigma_threshold"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","row_sigma_ratio_threshold"})) {
        get_config_value(config, beammap_scan_band_mask_row_sigma_ratio_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","row_sigma_ratio_threshold"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","max_flagged_fraction"})) {
        get_config_value(config, beammap_scan_band_mask_max_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","max_flagged_fraction"},
                         {}, {0.0}, {1.0});
    }

    // optional split output detector-map FITS files by detector quality flag
    beammap_split_fits_by_flag = false;
    beammap_split_flag_values = {0, 1};
    if (config.template has_typed<bool>(std::tuple{"beammap","split_fits_by_flag","enabled"})) {
        get_config_value(config, beammap_split_fits_by_flag, missing_keys, invalid_keys,
                         std::tuple{"beammap","split_fits_by_flag","enabled"});
    }
    if (config.template has_typed<std::vector<int>>(std::tuple{"beammap","split_fits_by_flag","flag_values"})) {
        auto values = config.template get_typed<std::vector<int>>(
            std::tuple{"beammap","split_fits_by_flag","flag_values"});
        if (values.empty()) {
            logger->warn("beammap.split_fits_by_flag.flag_values is empty; using defaults [0, 1]");
        }
        else {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            beammap_split_flag_values = std::move(values);
        }
    }

    // optional soft priors for beammap peak initialization
    beammap_priors_enabled = false;
    beammap_priors_filepath = "null";
    beammap_priors_candidate_top_n = 64;
    beammap_priors_min_snr = 0.0;
    beammap_priors_max_d2 = 25.0;
    beammap_priors_max_d2_iter0 = 25.0;
    beammap_priors_max_d2_after_iter0 = 25.0;
    beammap_priors_score_lambda = 2.0;
    beammap_priors_score_lambda_iter0 = 2.0;
    beammap_priors_score_lambda_after_iter0 = 2.0;
    beammap_priors_fallback_blind = true;
    beammap_priors_align_after_iter0 = true;
    beammap_priors_alignment_scope = "array";
    beammap_priors_alignment_common_support = "all";
    beammap_priors_alignment_common_support_quantile = 0.02;
    beammap_priors_alignment_min_matches = 30;
    beammap_priors_alignment_max_d2 = 25.0;
    beammap_priors_alignment_fit_rotation = true;
    beammap_priors_alignment_max_rotation_deg = 8.0;

    if (config.template has_typed<bool>(std::tuple{"beammap","priors","enabled"})) {
        get_config_value(config, beammap_priors_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","enabled"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","filepath"})) {
        get_config_value(config, beammap_priors_filepath, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","filepath"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","priors","candidate_top_n"})) {
        get_config_value(config, beammap_priors_candidate_top_n, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","candidate_top_n"},
                         {}, {1});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","min_snr"})) {
        get_config_value(config, beammap_priors_min_snr, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","min_snr"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2"})) {
        get_config_value(config, beammap_priors_max_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2"},
                         {}, {0.0});
    }
    beammap_priors_max_d2_iter0 = beammap_priors_max_d2;
    beammap_priors_max_d2_after_iter0 = beammap_priors_max_d2;
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda"})) {
        get_config_value(config, beammap_priors_score_lambda, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda"},
                         {}, {0.0});
    }
    beammap_priors_score_lambda_iter0 = beammap_priors_score_lambda;
    beammap_priors_score_lambda_after_iter0 = beammap_priors_score_lambda;
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2_iter0"})) {
        get_config_value(config, beammap_priors_max_d2_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2_after_iter0"})) {
        get_config_value(config, beammap_priors_max_d2_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2_after_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda_iter0"})) {
        get_config_value(config, beammap_priors_score_lambda_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda_after_iter0"})) {
        get_config_value(config, beammap_priors_score_lambda_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda_after_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","fallback_blind"})) {
        get_config_value(config, beammap_priors_fallback_blind, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","fallback_blind"});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","align_after_iter0"})) {
        get_config_value(config, beammap_priors_align_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","align_after_iter0"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","alignment_scope"})) {
        get_config_value(config, beammap_priors_alignment_scope, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_scope"},
                         {"array", "common"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","alignment_common_support"})) {
        get_config_value(config, beammap_priors_alignment_common_support, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_common_support"},
                         {"all", "overlap_box"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_common_support_quantile"})) {
        get_config_value(config, beammap_priors_alignment_common_support_quantile, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_common_support_quantile"},
                         {}, {0.0}, {0.45});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","priors","alignment_min_matches"})) {
        get_config_value(config, beammap_priors_alignment_min_matches, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_min_matches"},
                         {}, {3});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_max_d2"})) {
        get_config_value(config, beammap_priors_alignment_max_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_max_d2"},
                         {}, {0.0});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","alignment_fit_rotation"})) {
        get_config_value(config, beammap_priors_alignment_fit_rotation, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_fit_rotation"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_max_rotation_deg"})) {
        get_config_value(config, beammap_priors_alignment_max_rotation_deg, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_max_rotation_deg"},
                         {}, {0.0});
    }
    if (beammap_priors_enabled && beammap_priors_filepath == "null") {
        logger->warn("beammap.priors.enabled=true but beammap.priors.filepath is null; disabling priors");
        beammap_priors_enabled = false;
    }

    auto get_fixed_beammap_vector = [&](const std::vector<std::string> &path,
                                        std::size_t expected_size) {
        std::vector<double> values;
        if (path.size() == 2) {
            values = config.template get_typed<std::vector<double>>(std::make_tuple(path[0], path[1]));
        }
        else {
            values = config.template get_typed<std::vector<double>>(std::make_tuple(path[0], path[1], path[2]));
        }
        if (values.size() != expected_size) {
            invalid_keys.push_back(path);
            values.resize(expected_size, 0.0);
        }
        return values;
    };

    const std::size_t n_toltec_arrays = toltec_io.array_name_map.size();
    // lower fwhm limit
    auto lower_fwhm_arcsec_vec = get_fixed_beammap_vector({"beammap","flagging","array_lower_fwhm_arcsec"},
                                                          n_toltec_arrays);
    // upper fwhm limit
    auto upper_fwhm_arcsec_vec = get_fixed_beammap_vector({"beammap","flagging","array_upper_fwhm_arcsec"},
                                                          n_toltec_arrays);
    // lower signal-to-noise limit
    auto lower_sig2noise_vec = get_fixed_beammap_vector({"beammap","flagging","array_lower_sig2noise"},
                                                        n_toltec_arrays);
    // upper signal-to-noise limit
    auto upper_sig2noise_vec = get_fixed_beammap_vector({"beammap","flagging","array_upper_sig2noise"},
                                                        n_toltec_arrays);
    // maximum allowed distance limit
    auto max_dist_arcsec_vec = get_fixed_beammap_vector({"beammap","flagging","array_max_dist_arcsec"},
                                                        n_toltec_arrays);
    // per-array post-derotation network geometry cut
    auto network_robust_z_vec = get_fixed_beammap_vector({"beammap","flagging","array_network_robust_z"},
                                                         n_toltec_arrays);
    beammap_flag_max_prior_d2 = 0.0;
    if (config.template has_typed<double>(std::tuple{"beammap","flagging","max_prior_d2"})) {
        get_config_value(config, beammap_flag_max_prior_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","flagging","max_prior_d2"},
                         {}, {0.0});
    }

    // add params to respective array values
    Eigen::Index i = 0;
    for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
        // lower fwhm limit
        lower_fwhm_arcsec[arr_name] = lower_fwhm_arcsec_vec[i];
        // upper fwhm limit
        upper_fwhm_arcsec[arr_name] = upper_fwhm_arcsec_vec[i];
        // lower signal-to-noise limit
        lower_sig2noise[arr_name] = lower_sig2noise_vec[i];
        // upper signal-to-noise limit
        upper_sig2noise[arr_name] = upper_sig2noise_vec[i];
        // maximum allowed distance limit
        max_dist_arcsec[arr_name] = max_dist_arcsec_vec[i];
        // post-process per-network robust-z limit
        network_robust_z[arr_name] = network_robust_z_vec[i];
        i++;
    }

    // sensitivity factors
    auto sens_factors_vec = get_fixed_beammap_vector({"beammap","flagging","sens_factors"}, 2);
    lower_sens_factor = sens_factors_vec[0];
    upper_sens_factor = sens_factors_vec[1];

    // upper and lower frequencies over which to calculate sensitivity
    sens_psd_limits_Hz.resize(2);
    // get psd limits for sens from config
    auto sens_psd_limits_Hz_vec = get_fixed_beammap_vector({"beammap","sens_psd_limits_Hz"}, 2);
    // map sens limits back to Eigen vector
    sens_psd_limits_Hz = (Eigen::Map<Eigen::VectorXd>(sens_psd_limits_Hz_vec.data(), sens_psd_limits_Hz_vec.size()));

    // Beammap PTC TOD/diagnostics are written after the convergence decision.
    // The default is the actual last attempted iteration, including early
    // convergence, so the saved PTC reflects the final cleaning state.
    beammap_tod_output_iter = -1;

    beammap_detector_tod_output_enabled = false;
    beammap_detector_tod_output_subdir_name = "source_crossing_tod";
    beammap_detector_tod_output_n_uniform = 10;
    beammap_detector_tod_output_n_source_dense = 10;
    if (config.template has_typed<bool>(std::tuple{"beammap","detector_tod_output","enabled"})) {
        get_config_value(config, beammap_detector_tod_output_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","enabled"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","detector_tod_output","subdir_name"})) {
        get_config_value(config, beammap_detector_tod_output_subdir_name, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","subdir_name"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","detector_tod_output","n_uniform"})) {
        get_config_value(config, beammap_detector_tod_output_n_uniform, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","n_uniform"},
                         {}, {0});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","detector_tod_output","n_source_dense"})) {
        get_config_value(config, beammap_detector_tod_output_n_source_dense, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","n_source_dense"},
                         {}, {0});
    }

    typed_beammap_config = citlali::config::BeammapConfig{};
    typed_beammap_config.iteration.max_iterations = beammap_iter_max;
    typed_beammap_config.iteration.tolerance = beammap_iter_tolerance;
    typed_beammap_config.iteration.convergence_radius_arcsec =
        beammap_convergence_radius_arcsec;
    typed_beammap_config.phase_strategy.enabled = beammap_phase_split_enabled;
    typed_beammap_config.phase_strategy.locator_iter = beammap_locator_iter;
    typed_beammap_config.phase_strategy.measurement_start_iter =
        beammap_measurement_start_iter;
    typed_beammap_config.reference.subtract_reference_detector =
        beammap_subtract_reference;
    typed_beammap_config.reference.reference_detector =
        static_cast<long>(beammap_reference_det);
    typed_beammap_config.reference.derotate = beammap_derotate;
    typed_beammap_config.rfi_mask.enabled = beammap_rfi_mask_enabled;
    typed_beammap_config.rfi_mask.block_size_samples =
        beammap_rfi_mask_block_size_samples;
    typed_beammap_config.rfi_mask.min_good_samples =
        beammap_rfi_mask_min_good_samples;
    typed_beammap_config.rfi_mask.dilate_blocks = beammap_rfi_mask_dilate_blocks;
    typed_beammap_config.rfi_mask.sigma_threshold =
        beammap_rfi_mask_sigma_threshold;
    typed_beammap_config.rfi_mask.sigma_floor = beammap_rfi_mask_sigma_floor;
    typed_beammap_config.rfi_mask.max_flagged_fraction =
        beammap_rfi_mask_max_flagged_fraction;
    if (auto parsed = citlali::config::parse_beammap_detector_weighting_mode(
            beammap_detector_weighting_mode)) {
        typed_beammap_config.detector_weighting_mode = *parsed;
    }
    typed_beammap_config.fitting.fit_radius_fwhm = beammap_fit_radius_fwhm;
    typed_beammap_config.scan_band_mask.enabled = beammap_scan_band_mask_enabled;
    typed_beammap_config.scan_band_mask.edge_rows = beammap_scan_band_mask_edge_rows;
    typed_beammap_config.scan_band_mask.min_row_pixels =
        beammap_scan_band_mask_min_row_pixels;
    typed_beammap_config.scan_band_mask.min_contiguous_rows =
        beammap_scan_band_mask_min_contiguous_rows;
    typed_beammap_config.scan_band_mask.row_median_sigma_threshold =
        beammap_scan_band_mask_row_median_sigma_threshold;
    typed_beammap_config.scan_band_mask.row_sigma_ratio_threshold =
        beammap_scan_band_mask_row_sigma_ratio_threshold;
    typed_beammap_config.scan_band_mask.max_flagged_fraction =
        beammap_scan_band_mask_max_flagged_fraction;
    typed_beammap_config.split_fits_by_flag.enabled = beammap_split_fits_by_flag;
    typed_beammap_config.split_fits_by_flag.flag_values = beammap_split_flag_values;
    typed_beammap_config.priors.enabled = beammap_priors_enabled;
    typed_beammap_config.priors.filepath = beammap_priors_filepath;
    typed_beammap_config.priors.candidate_top_n =
        beammap_priors_candidate_top_n;
    typed_beammap_config.priors.min_snr = beammap_priors_min_snr;
    typed_beammap_config.priors.max_d2 = beammap_priors_max_d2;
    typed_beammap_config.priors.max_d2_iter0 = beammap_priors_max_d2_iter0;
    typed_beammap_config.priors.max_d2_after_iter0 =
        beammap_priors_max_d2_after_iter0;
    typed_beammap_config.priors.score_lambda = beammap_priors_score_lambda;
    typed_beammap_config.priors.score_lambda_iter0 =
        beammap_priors_score_lambda_iter0;
    typed_beammap_config.priors.score_lambda_after_iter0 =
        beammap_priors_score_lambda_after_iter0;
    typed_beammap_config.priors.fallback_blind = beammap_priors_fallback_blind;
    typed_beammap_config.priors.align_after_iter0 =
        beammap_priors_align_after_iter0;
    typed_beammap_config.priors.alignment_scope =
        beammap_priors_alignment_scope;
    typed_beammap_config.priors.alignment_common_support =
        beammap_priors_alignment_common_support;
    typed_beammap_config.priors.alignment_common_support_quantile =
        beammap_priors_alignment_common_support_quantile;
    typed_beammap_config.priors.alignment_min_matches =
        beammap_priors_alignment_min_matches;
    typed_beammap_config.priors.alignment_max_d2 =
        beammap_priors_alignment_max_d2;
    typed_beammap_config.priors.alignment_fit_rotation =
        beammap_priors_alignment_fit_rotation;
    typed_beammap_config.priors.alignment_max_rotation_deg =
        beammap_priors_alignment_max_rotation_deg;
    typed_beammap_config.detector_tod_output.enabled =
        beammap_detector_tod_output_enabled;
    typed_beammap_config.detector_tod_output.subdir_name =
        beammap_detector_tod_output_subdir_name;
    typed_beammap_config.detector_tod_output.n_uniform =
        beammap_detector_tod_output_n_uniform;
    typed_beammap_config.detector_tod_output.n_source_dense =
        beammap_detector_tod_output_n_source_dense;
    typed_beammap_config.flagging.array_lower_fwhm_arcsec =
        lower_fwhm_arcsec_vec;
    typed_beammap_config.flagging.array_upper_fwhm_arcsec =
        upper_fwhm_arcsec_vec;
    typed_beammap_config.flagging.array_lower_sig2noise =
        lower_sig2noise_vec;
    typed_beammap_config.flagging.array_upper_sig2noise =
        upper_sig2noise_vec;
    typed_beammap_config.flagging.array_max_dist_arcsec =
        max_dist_arcsec_vec;
    typed_beammap_config.flagging.array_network_robust_z =
        network_robust_z_vec;
    typed_beammap_config.flagging.sens_factors = sens_factors_vec;
    typed_beammap_config.flagging.sens_psd_limits_hz = sens_psd_limits_Hz_vec;
    typed_beammap_config.flagging.max_prior_d2 = beammap_flag_max_prior_d2;
}

template<typename CT>
void Engine::get_map_filter_config(CT &config) {
    logger->info("getting map filtering config options");
    // get wiener filter config options
    wiener_filter.get_config(config, missing_keys, invalid_keys);

    auto &typed_map_filter = typed_post_processing_config.map_filtering;
    typed_map_filter.enabled = run_map_filter;
    if (auto parsed = citlali::config::parse_map_filter_type(wiener_filter.filter_type)) {
        typed_map_filter.type = *parsed;
    }
    if (auto parsed = citlali::config::parse_map_filter_template_type(
            wiener_filter.template_type)) {
        typed_map_filter.template_type = *parsed;
    }
    typed_map_filter.lowpass_only = wiener_filter.run_lowpass;
    typed_map_filter.normalize_errors = wiener_filter.normalize_error;
    typed_map_filter.edge_guard.enabled = wiener_filter.edge_guard_enabled;
    typed_map_filter.edge_guard.weight_threshold_mode =
        wiener_filter.edge_weight_threshold_mode;
    typed_map_filter.edge_guard.hits_threshold_mode =
        wiener_filter.edge_hits_threshold_mode;
    typed_map_filter.edge_guard.hits_core_fraction =
        wiener_filter.edge_hits_core_fraction;
    typed_map_filter.edge_guard.guard_radius_fwhm =
        wiener_filter.edge_guard_radius_fwhm;
    typed_map_filter.edge_guard.fill_mode = wiener_filter.edge_fill_mode;
    if (auto parsed = citlali::config::parse_map_filter_edge_taper_mode(
            wiener_filter.edge_taper_mode)) {
        typed_map_filter.edge_guard.taper_mode = *parsed;
    }
    typed_map_filter.edge_guard.taper_min_fraction =
        wiener_filter.edge_taper_min_fraction;
    typed_map_filter.denom_rel_tol = wiener_filter.denom_rel_tol;
    typed_map_filter.tail_frac_tol = wiener_filter.tail_frac_tol;
    typed_map_filter.max_loops = wiener_filter.max_loops;
    typed_map_filter.denom_check_iters = wiener_filter.denom_check_iters;
    typed_map_filter.max_denom_iters = wiener_filter.max_denom_iters;
    typed_map_filter.template_fwhm_arcsec.clear();
    for (const auto &[array_name, fwhm_rad] : wiener_filter.template_fwhm_rad) {
        typed_map_filter.template_fwhm_arcsec[array_name] =
            fwhm_rad * RAD_TO_ASEC;
    }

    // if in science mode, write filtered maps as they complete
    if (redu_type=="science") {
        write_filtered_maps_partial = true;
    }
    // otherwise write at end
    else {
        write_filtered_maps_partial = false;
    }
    // check if kernel is enabled
    if (wiener_filter.template_type=="kernel") {
        if (!rtcproc.run_kernel) {
            logger->error("wiener filter kernel template requires kernel");
            std::exit(EXIT_FAILURE);
        }
        // copy the map fitter
        else {
            wiener_filter.map_fitter = map_fitter;
        }
    }
    // make sure noise maps were enabled
    if (!run_noise && (!wiener_filter.run_lowpass && wiener_filter.filter_type=="wiener_filter")) {
        logger->error("wiener filter requires noise maps");
        std::exit(EXIT_FAILURE);
    }

    // set parallelization for ffts (maintained with tod output/verbose mode)
    wiener_filter.parallel_policy = parallel_policy;
}

template<typename CT>
citlali::config::RuntimeConfig Engine::get_runtime_config(CT &config) {
    citlali::config::RuntimeConfig runtime_config;

    // verbose mode?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, verbose_mode, missing_keys, invalid_keys,
                         std::tuple{"runtime","verbose"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.verbose = verbose_mode;
        }
    }
    // output directory
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, output_dir, missing_keys, invalid_keys,
                         std::tuple{"runtime","output_dir"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.output_dir = output_dir;
        }
    }
    // number of threads to use
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, n_threads, missing_keys, invalid_keys,
                         std::tuple{"runtime","n_threads"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.n_threads = n_threads;
        }
    }
    // overall parallel policy
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, parallel_policy, missing_keys, invalid_keys,
                         std::tuple{"runtime","parallel_policy"},{"seq","omp"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            if (auto parsed = citlali::config::parse_parallel_policy(parallel_policy)) {
                runtime_config.parallel_policy = *parsed;
            }
        }
    }
    // reduction type (science, pointing, beammap)
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, redu_type, missing_keys, invalid_keys,
                         std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            if (auto parsed = citlali::config::parse_reduction_type(redu_type)) {
                runtime_config.reduction_type = *parsed;
            }
        }
    }
    // create redu00, redu01... subdirectories
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, use_subdir, missing_keys, invalid_keys,
                         std::tuple{"runtime","use_subdir"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.use_subdir = use_subdir;
        }
    }
    // interp over gaps in align_timestream
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, interp_over_gaps, missing_keys, invalid_keys,
                         std::tuple{"runtime","interp_over_gaps"});
        if (missing_keys.size() == missing_before && invalid_keys.size() == invalid_before) {
            runtime_config.interp_over_gaps = interp_over_gaps;
        }
    }
    return runtime_config;
}

template<typename CT>
void Engine::get_citlali_config(CT &config) {
    // interface key names
    const std::vector<std::string> interface_keys = {
        "toltec0",
        "toltec1",
        "toltec2",
        "toltec3",
        "toltec4",
        "toltec5",
        "toltec6",
        "toltec7",
        "toltec8",
        "toltec9",
        "toltec10",
        "toltec11",
        "toltec12",
        "hwpr"
    };
    // initialize all offsets explicitly to zero
    for (const auto &key : interface_keys) {
        interface_sync_offset[key] = 0.0;
    }

    //  get interface offsets
    if (config.has(std::tuple{"interface_sync_offset"})) {
        auto interface_node = config.get_node(std::tuple{"interface_sync_offset"});
        std::set<std::string> configured_keys;
        // parse each list entry by key name so YAML order does not matter
        for (Eigen::Index i=0; i<interface_node.size(); ++i) {
            bool found_key = false;
            for (const auto &key : interface_keys) {
                if (config.has(std::tuple{"interface_sync_offset", i, key})) {
                    auto offset = config.template get_typed<double>(std::tuple{"interface_sync_offset", i, key});
                    if (configured_keys.find(key) != configured_keys.end()) {
                        logger->warn("interface_sync_offset for {} specified multiple times; using last value", key);
                    }
                    interface_sync_offset[key] = offset;
                    configured_keys.insert(key);
                    found_key = true;
                }
            }
            if (!found_key) {
                logger->warn("interface_sync_offset entry {} does not contain a recognized interface key; ignoring entry", i);
            }
        }
        for (const auto &key : interface_keys) {
            if (configured_keys.find(key) == configured_keys.end()) {
                logger->warn("interface_sync_offset missing {}; using 0.0 s", key);
            }
        }
    }

    typed_runtime_config = get_runtime_config(config);
    if (!typed_runtime_config.interp_over_gaps) {
        logger->error("runtime.interp_over_gaps=false is unsupported; set runtime.interp_over_gaps: true");
        std::exit(EXIT_FAILURE);
    }

    /* get timestream config */
    get_timestream_config(config);
    {
        // The pointing pipeline also covers PSF-preserving focus and holography-style reductions.
        const bool source_aware_reduction = (redu_type == "pointing");
        rtcproc.despiker.source_protection_enabled =
            rtcproc.run_despike &&
            rtcproc.despike_source_protection_config_enabled &&
            source_aware_reduction;
        ptcproc.second_pass_local.source_protection_enabled =
            ptcproc.second_pass_local.enabled &&
            ptcproc.second_pass_local.source_protection_config_enabled &&
            source_aware_reduction;
        typed_timestream_config.raw_time_chunk.despike.source_protection.active =
            rtcproc.despiker.source_protection_enabled;
        typed_timestream_config.processed_time_chunk.flagging.second_pass_local
            .source_protection.active =
            ptcproc.second_pass_local.source_protection_enabled;
        if (rtcproc.run_despike && rtcproc.despike_source_protection_config_enabled) {
            logger->info(
                "raw_time_chunk.despike source protection active={} reduction_type={} radius_arcsec={:.4g}",
                rtcproc.despiker.source_protection_enabled, redu_type,
                rtcproc.despiker.source_protection_radius_arcsec);
        }
        if (ptcproc.second_pass_local.enabled &&
            ptcproc.second_pass_local.source_protection_config_enabled) {
            logger->info(
                "processed_time_chunk.flagging.second_pass_local source protection active={} reduction_type={} radius_arcsec={:.4g}",
                ptcproc.second_pass_local.source_protection_enabled, redu_type,
                ptcproc.second_pass_local.source_protection_radius_arcsec);
        }
    }

    /* get mapmaking config */
    typed_post_processing_config = citlali::config::PostProcessingConfig{};
    get_mapmaking_config(config);

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };

    // run map filter?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_map_filter, missing_keys, invalid_keys,
                         std::tuple{"post_processing","map_filtering","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_post_processing_config.map_filtering_enabled = run_map_filter;
            typed_post_processing_config.map_filtering.enabled = run_map_filter;
        }
    }

    // run source finder?
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, run_source_finder, missing_keys, invalid_keys,
                         std::tuple{"post_processing","source_finding","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_post_processing_config.source_finding_enabled = run_source_finder;
            typed_post_processing_config.source_finding.enabled = run_source_finder;
        }
    }

    // map fitter options if in pointing or beammap mode or if map filtering or source finding are enabled
    if (redu_type=="pointing" || redu_type=="beammap" || run_map_filter || run_source_finder) {
        typed_post_processing_config.source_fitting.active = true;
        // size of region around found source to fit
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, map_fitter.bounding_box_pix, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_fitting","bounding_box_arcsec"},{},{0});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_fitting.bounding_box_arcsec =
                    map_fitter.bounding_box_pix;
            }
        }
        // radius around center of map to find source within
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, map_fitter.fitting_region_pix, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_fitting","fitting_radius_arcsec"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_fitting.fitting_radius_arcsec =
                    map_fitter.fitting_region_pix;
            }
        }
        // fit 2d gaussian rotation angle
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, map_fitter.fit_angle, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_fitting", "gauss_model","fit_rotation_angle"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_fitting.fit_rotation_angle =
                    map_fitter.fit_angle;
            }
        }

        // convert bounding box and fitting region to pixels
        map_fitter.bounding_box_pix = ASEC_TO_RAD*map_fitter.bounding_box_pix/omb.pixel_size_rad;
        map_fitter.fitting_region_pix = ASEC_TO_RAD*map_fitter.fitting_region_pix/omb.pixel_size_rad;

        // fitter flux and fwhm limits
        map_fitter.flux_limits.resize(2);
        map_fitter.fwhm_limits.resize(2);
        for (Eigen::Index i=0; i<map_fitter.flux_limits.size(); ++i) {
            // flux limit
            map_fitter.flux_limits(i) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting",
                                                                                     "gauss_model","amp_limit_factors",i});
            typed_post_processing_config.source_fitting.amp_limit_factors[static_cast<std::size_t>(i)] =
                map_fitter.flux_limits(i);
            // fwhm limit
            map_fitter.fwhm_limits(i) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting",
                                                                                     "gauss_model","fwhm_limit_factors",i});
            typed_post_processing_config.source_fitting.fwhm_limit_factors[static_cast<std::size_t>(i)] =
                map_fitter.fwhm_limits(i);
        }

        // flux lower factor
        if (map_fitter.flux_limits(0) > 0) {
            map_fitter.flux_low = map_fitter.flux_limits(0);
        }
        // flux lower factor
        if (map_fitter.flux_limits(1) > 0) {
            map_fitter.flux_high = map_fitter.flux_limits(1);
        }
        // fwhm lower factor
        if (map_fitter.fwhm_limits(0) > 0) {
            map_fitter.fwhm_low = map_fitter.fwhm_limits(0);
        }
        // fwhm upper factor
        if (map_fitter.fwhm_limits(1) > 0) {
            map_fitter.fwhm_high = map_fitter.fwhm_limits(1);
        }
    }

    /* get wiener filter config */
    if (run_map_filter) {
        // needs map fitter config
        get_map_filter_config(config);
    }

    // get source finder config options
    if (run_source_finder) {
        // minimum found source sigma
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_sigma, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","source_sigma"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.source_sigma =
                    omb.source_sigma;
            }
        }
        // window around source to exclude other sources
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_window_rad, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","source_window_arcsec"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.source_window_arcsec =
                    omb.source_window_rad;
            }
        }
        // search map, negative of map, or both
        {
            const auto missing_before = missing_keys.size();
            const auto invalid_before = invalid_keys.size();
            get_config_value(config, omb.source_finder_mode, missing_keys, invalid_keys,
                             std::tuple{"post_processing","source_finding","mode"});
            if (parsed_cleanly(missing_before, invalid_before)) {
                typed_post_processing_config.source_finding.mode =
                    omb.source_finder_mode;
            }
        }

        // convert source window to radians
        omb.source_window_rad = omb.source_window_rad*ASEC_TO_RAD;

        if (run_coadd) {
            // copy omb source sigma to cmb
            cmb.source_sigma = omb.source_sigma;
            // copy omb source_window_rad to cmb
            cmb.source_window_rad = omb.source_window_rad;
            // copy omb source_finder_mode to cmb
            cmb.source_finder_mode = omb.source_finder_mode;
        }
    }

    /* get pointing config */
    if (redu_type=="pointing") {
        get_pointing_config(config);
    }

    /* get beammap config */
    if (redu_type=="beammap") {
        // needs redu_type config
        get_beammap_config(config);
    }

    // disable map related keys if map-making is disabled
    if (!run_mapmaking) {
        run_coadd = false;
        run_noise = false;
        run_map_filter = false;
        run_source_finder = false;
        typed_coadd_config.enabled = false;
        typed_noise_config.enabled = false;
        typed_post_processing_config.map_filtering_enabled = false;
        typed_post_processing_config.map_filtering.enabled = false;
        typed_post_processing_config.source_finding_enabled = false;
        typed_post_processing_config.source_finding.enabled = false;
        typed_post_processing_config.source_fitting.active = false;
        // we don't need to do iterations if no maps are made
        beammap_iter_max = 1;
        typed_beammap_config.iteration.max_iterations = 1;
    }
}

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    typed_beammap_config.source = citlali::config::BeammapSourceConfig{};

    // beammap source name
    get_config_value(config, beammap_source_name, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","name"});
    typed_beammap_config.source.name = beammap_source_name;
    // beammap source ra
    get_config_value(config, beammap_ra_rad, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","ra_deg"});
    typed_beammap_config.source.ra_deg = beammap_ra_rad;
    // convert ra to radians
    beammap_ra_rad = beammap_ra_rad*DEG_TO_RAD;

    // beammap source dec
    get_config_value(config, beammap_dec_rad, missing_keys, invalid_keys,
                     std::tuple{"beammap_source","dec_deg"});
    typed_beammap_config.source.dec_deg = beammap_dec_rad;
    // convert dec to radians
    beammap_dec_rad = beammap_dec_rad*DEG_TO_RAD;

    // number of fluxes
    Eigen::Index n_fluxes = config.get_node(std::tuple{"beammap_source","fluxes"}).size();

    // get source fluxes
    for (Eigen::Index i=0; i<n_fluxes; ++i) {
        auto array = config.get_str(std::tuple{"beammap_source","fluxes",i,"array_name"});
        // source flux in mJy/beam
        auto flux = config.template get_typed<double>(std::tuple{"beammap_source","fluxes",i,"value_mJy"});
        // source flux uncertainty in mJy/beam
        auto uncertainty_mJy = config.template get_typed<double>(std::tuple{"beammap_source","fluxes",i,"uncertainty_mJy"});

        // copy flux and uncertainty
        beammap_fluxes_mJy_beam[array] = flux;
        beammap_err_mJy_beam[array] = uncertainty_mJy;
        typed_beammap_config.source.fluxes.push_back(
            citlali::config::BeammapSourceFluxConfig{array, flux, uncertainty_mJy});
    }

    if (redu_type == "beammap") {
        bool valid_flux_config = true;
        for (auto const& entry : toltec_io.array_name_map) {
            const auto &arr_name = entry.second;
            auto flux_it = beammap_fluxes_mJy_beam.find(arr_name);
            if (flux_it == beammap_fluxes_mJy_beam.end()) {
                logger->error(
                    "beammap reductions require a positive source flux for {}; no beammap_source.fluxes entry was found",
                    arr_name);
                valid_flux_config = false;
                continue;
            }
            const double flux = flux_it->second;
            if (!std::isfinite(flux) || flux <= 0.0) {
                logger->error(
                    "beammap reductions require positive finite source fluxes; {} value_mJy={}",
                    arr_name, flux);
                valid_flux_config = false;
            }
        }
        if (!valid_flux_config) {
            std::exit(EXIT_FAILURE);
        }
    }
}

template<typename CT>
void Engine::get_astrometry_config(CT &config) {
    typed_astrometry_config = citlali::config::AstrometryConfig{};

    // check if config file has pointing_offsets
    if (config.has("pointing_offsets")) {
        // reset for each observation
        pointing_offsets_arcsec.clear();
        pointing_offsets_modified_julian_date.setZero(2);

        auto pointing_node = config.get_node(std::tuple{"pointing_offsets"});
        bool has_az = false;
        bool has_alt = false;
        bool has_mjd = false;
        std::vector<double> mjd_values;

        for (Eigen::Index i = 0; i < pointing_node.size(); ++i) {
            if (config.has(std::tuple{"pointing_offsets", i, "axes_name"})) {
                auto axis = config.get_str(std::tuple{"pointing_offsets", i, "axes_name"});
                std::transform(axis.begin(), axis.end(), axis.begin(),
                               [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
                if (axis == "az" || axis == "alt") {
                    auto offset = config.template get_typed<std::vector<double>>(
                        std::tuple{"pointing_offsets", i, "value_arcsec"});
                    if (offset.empty()) {
                        logger->error("pointing_offsets {} has empty value_arcsec", axis);
                        std::exit(EXIT_FAILURE);
                    }
                    if (pointing_offsets_arcsec.find(axis) != pointing_offsets_arcsec.end()) {
                        logger->warn("pointing_offsets {} specified multiple times; using last value", axis);
                    }
                    pointing_offsets_arcsec[axis] =
                        Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
                    if (axis == "az") {
                        has_az = true;
                    }
                    else {
                        has_alt = true;
                    }
                }
                else {
                    logger->warn("unknown pointing_offsets axis_name '{}' at entry {}", axis, i);
                }
            }
            else if (config.has(std::tuple{"pointing_offsets", i, "modified_julian_date"})) {
                mjd_values = config.template get_typed<std::vector<double>>(
                    std::tuple{"pointing_offsets", i, "modified_julian_date"});
                has_mjd = true;
            }
            else {
                logger->warn("unrecognized pointing_offsets entry {}. expected axes_name/value_arcsec or modified_julian_date", i);
            }
        }

        // backward-compatible fallback for positional configs
        if (!has_az && config.has(std::tuple{"pointing_offsets", 0, "value_arcsec"})) {
            auto offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 0, "value_arcsec"});
            if (offset.empty()) {
                logger->error("pointing_offsets az has empty value_arcsec");
                std::exit(EXIT_FAILURE);
            }
            logger->warn("pointing_offsets az parsed by positional index; consider setting axes_name: az");
            pointing_offsets_arcsec["az"] = Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
            has_az = true;
        }
        if (!has_alt && config.has(std::tuple{"pointing_offsets", 1, "value_arcsec"})) {
            auto offset = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 1, "value_arcsec"});
            if (offset.empty()) {
                logger->error("pointing_offsets alt has empty value_arcsec");
                std::exit(EXIT_FAILURE);
            }
            logger->warn("pointing_offsets alt parsed by positional index; consider setting axes_name: alt");
            pointing_offsets_arcsec["alt"] = Eigen::Map<Eigen::VectorXd>(offset.data(), offset.size());
            has_alt = true;
        }
        if (!has_mjd && config.has(std::tuple{"pointing_offsets", 2, "modified_julian_date"})) {
            mjd_values = config.template get_typed<std::vector<double>>(std::tuple{"pointing_offsets", 2, "modified_julian_date"});
            has_mjd = true;
        }

        if (!has_az || !has_alt) {
            logger->error("pointing_offsets must include both az and alt entries");
            std::exit(EXIT_FAILURE);
        }

        const auto n_az = pointing_offsets_arcsec["az"].size();
        const auto n_alt = pointing_offsets_arcsec["alt"].size();
        if (n_az != n_alt) {
            logger->error("pointing_offsets az/alt lengths differ (az={} alt={})", n_az, n_alt);
            std::exit(EXIT_FAILURE);
        }
        if (n_az != 1 && n_az != 2) {
            logger->error("pointing_offsets supports only one or two values per axis (got {})", n_az);
            std::exit(EXIT_FAILURE);
        }

        if (has_mjd) {
            if (mjd_values.size() == 2) {
                pointing_offsets_modified_julian_date =
                    Eigen::Map<Eigen::VectorXd>(mjd_values.data(), mjd_values.size());
            }
            else if (!mjd_values.empty() &&
                     std::all_of(mjd_values.begin(), mjd_values.end(), [](double v){ return v <= 0.0; })) {
                // non-positive sentinel means "not specified"
                pointing_offsets_modified_julian_date.setZero(2);
            }
            else if (mjd_values.size() == 1 && n_az == 1) {
                logger->warn(
                    "ignoring single pointing_offsets.modified_julian_date for single pointing offset; using a constant offset across the observation");
                pointing_offsets_modified_julian_date.setZero(2);
            }
            else {
                logger->error(
                    "pointing_offsets.modified_julian_date must contain 2 values when interpolating two offsets, or non-positive sentinels");
                std::exit(EXIT_FAILURE);
            }
        }

        auto &typed_offsets = typed_astrometry_config.pointing_offsets;
        typed_offsets.enabled = true;
        const auto &az_offsets = pointing_offsets_arcsec["az"];
        typed_offsets.az_arcsec.assign(
            az_offsets.data(), az_offsets.data() + az_offsets.size());
        const auto &alt_offsets = pointing_offsets_arcsec["alt"];
        typed_offsets.alt_arcsec.assign(
            alt_offsets.data(), alt_offsets.data() + alt_offsets.size());
        typed_offsets.modified_julian_date.assign(
            pointing_offsets_modified_julian_date.data(),
            pointing_offsets_modified_julian_date.data() +
                pointing_offsets_modified_julian_date.size());
    }
    else {
        logger->error("pointing_offsets not found in config");
        std::exit(EXIT_FAILURE);
    }
}
