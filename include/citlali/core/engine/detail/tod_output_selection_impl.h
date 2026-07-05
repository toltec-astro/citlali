#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

void Engine::setup_tod_output_chunk_selection() {
    const Eigen::Index n_scans = telescope.scan_indices.cols();

    auto build_uniform_plus_source_crossing_chunks =
        [&](const std::string &stream_name, int n_uniform, int n_source_dense) {
            if (n_scans <= 0) {
                return std::vector<Eigen::Index>{};
            }

            n_uniform = std::max(0, n_uniform);
            n_source_dense = std::max(0, n_source_dense);

            Eigen::Index source_scan = (n_scans - 1) / 2;
            double best_scan_d2 = std::numeric_limits<double>::infinity();
            try {
                auto tel_data_copy = telescope.tel_data;
                std::map<std::string, Eigen::VectorXd> pointing_offsets;
                Eigen::Index n_tel = 0;
                if (!tel_data_copy.empty()) {
                    n_tel = tel_data_copy.begin()->second.size();
                }
                auto make_offset = [&](const std::string &axis) -> Eigen::VectorXd {
                    auto it = pointing_offsets_arcsec.find(axis);
                    if (it != pointing_offsets_arcsec.end() && it->second.size() == n_tel) {
                        return it->second;
                    }
                    return Eigen::VectorXd::Zero(n_tel);
                };
                pointing_offsets["az"] = make_offset("az");
                pointing_offsets["alt"] = make_offset("alt");

                auto [lat, lon] = engine_utils::calc_det_pointing(
                    tel_data_copy, 0.0, 0.0, telescope.pixel_axes, pointing_offsets,
                    map_grouping, true);

                for (Eigen::Index scan_index = 0; scan_index < n_scans; ++scan_index) {
                    const Eigen::Index start =
                        std::max<Eigen::Index>(0, telescope.scan_indices(0, scan_index));
                    const Eigen::Index end =
                        std::min<Eigen::Index>(lat.size() - 1, telescope.scan_indices(1, scan_index));
                    if (end < start || lon.size() <= end) {
                        continue;
                    }
                    double scan_best_d2 = std::numeric_limits<double>::infinity();
                    for (Eigen::Index sample = start; sample <= end; ++sample) {
                        const double y = lat(sample);
                        const double x = lon(sample);
                        if (!std::isfinite(x) || !std::isfinite(y)) {
                            continue;
                        }
                        const double d2 = x * x + y * y;
                        if (d2 < scan_best_d2) {
                            scan_best_d2 = d2;
                        }
                    }
                    if (scan_best_d2 < best_scan_d2) {
                        best_scan_d2 = scan_best_d2;
                        source_scan = scan_index;
                    }
                }
            }
            catch (const std::exception &e) {
                logger->warn(
                    "{} TOD uniform_plus_source_crossing selection could not calculate source-crossing scan ({}); using scan {}",
                    stream_name, e.what(), source_scan + 1);
            }

            const auto selected_1based =
                citlali::pipeline::uniform_plus_source_tod_output_chunks(
                    n_scans, n_uniform, n_source_dense, source_scan);

            logger->info(
                "{} TOD output selection mode uniform_plus_source_crossing: n_uniform={} n_source_dense={} source_scan={} source_min_distance_arcsec={:.3f} selected={}",
                stream_name,
                n_uniform,
                n_source_dense,
                source_scan + 1,
                std::isfinite(best_scan_d2) ? std::sqrt(best_scan_d2) * RAD_TO_ASEC
                                            : std::numeric_limits<double>::quiet_NaN(),
                citlali::pipeline::tod_output_chunks_to_string(
                    selected_1based));
            return selected_1based;
        };

    auto setup_one = [&](const std::string &stream_name,
                         bool output_enabled,
                         bool select_enabled,
                         const std::vector<Eigen::Index> &chunks_1based,
                         const std::string &selection_mode,
                         int uniform_count,
                         int source_dense_count,
                         Eigen::VectorXI &scan_to_output,
                         Eigen::Index &n_output_scans) {
        scan_to_output.resize(n_scans);
        scan_to_output.setConstant(-1);
        n_output_scans = 0;

        if (!output_enabled) {
            logger->info("{} TOD output disabled", stream_name);
            return;
        }

        std::vector<Eigen::Index> effective_chunks = chunks_1based;
        bool effective_select_enabled = select_enabled;
        if (selection_mode == "all") {
            effective_select_enabled = false;
            effective_chunks.clear();
        }
        else if (selection_mode == "uniform_plus_source_crossing") {
            effective_select_enabled = true;
            effective_chunks = build_uniform_plus_source_crossing_chunks(
                stream_name, uniform_count, source_dense_count);
            if (effective_chunks.empty()) {
                logger->error("{} TOD output selection mode uniform_plus_source_crossing selected no chunks",
                              stream_name);
                std::exit(EXIT_FAILURE);
            }
        }
        else if (selection_mode != "indices") {
            logger->error("{} TOD output selection mode '{}' is invalid", stream_name, selection_mode);
            std::exit(EXIT_FAILURE);
        }

        if (!effective_select_enabled || effective_chunks.empty()) {
            n_output_scans =
                citlali::pipeline::assign_all_tod_output_rows(
                    scan_to_output, n_scans);
            logger->info("{} TOD output chunk selection disabled: writing all {} chunks",
                         stream_name, n_output_scans);
            return;
        }

        for (const auto chunk_1based : effective_chunks) {
            if (!citlali::pipeline::tod_output_chunk_is_valid(
                    chunk_1based, n_scans)) {
                logger->error("{} TOD output indices contain {} but valid scan range is [1, {}]",
                              stream_name, chunk_1based, n_scans);
                std::exit(EXIT_FAILURE);
            }
        }

        n_output_scans =
            citlali::pipeline::assign_selected_tod_output_rows(
                scan_to_output, n_scans, effective_chunks);
        logger->info("{} TOD output chunk selection enabled: writing {} of {} chunks",
                     stream_name, n_output_scans, n_scans);
    };

    if (!run_tod_output) {
        tod_scan_to_output_scan_rtc.resize(0);
        tod_scan_to_output_scan_ptc.resize(0);
        n_tod_output_scans_rtc = 0;
        n_tod_output_scans_ptc = 0;
    }
    else {
        setup_one("RTC", run_tod_output_rtc, tod_output_chunk_select_enabled_rtc, tod_output_chunks_rtc,
                  tod_output_selection_mode_rtc, tod_output_uniform_count_rtc, tod_output_source_dense_count_rtc,
                  tod_scan_to_output_scan_rtc, n_tod_output_scans_rtc);
        setup_one("PTC", run_tod_output_ptc, tod_output_chunk_select_enabled_ptc, tod_output_chunks_ptc,
                  tod_output_selection_mode_ptc, tod_output_uniform_count_ptc, tod_output_source_dense_count_ptc,
                  tod_scan_to_output_scan_ptc, n_tod_output_scans_ptc);
    }

    // keep legacy shared fields for backwards compatibility with call sites that
    // do not specify stream type explicitly.
    if (run_tod_output_rtc) {
        tod_scan_to_output_scan = tod_scan_to_output_scan_rtc;
        n_tod_output_scans = n_tod_output_scans_rtc;
    }
    else if (run_tod_output_ptc) {
        tod_scan_to_output_scan = tod_scan_to_output_scan_ptc;
        n_tod_output_scans = n_tod_output_scans_ptc;
    }
    else {
        tod_scan_to_output_scan.resize(0);
        n_tod_output_scans = 0;
    }
}

bool Engine::should_write_tod_chunk(Eigen::Index scan_index) const {
    return tod_output_scan_row(scan_index) >= 0;
}

Eigen::Index Engine::tod_output_scan_row(Eigen::Index scan_index) const {
    if (run_tod_output_rtc) {
        return tod_output_scan_row(scan_index, "rtc");
    }
    if (run_tod_output_ptc) {
        return tod_output_scan_row(scan_index, "ptc");
    }
    return -1;
}

Eigen::Index Engine::tod_output_scan_row(Eigen::Index scan_index, const std::string &stream_name) const {
    const Eigen::VectorXI *scan_to_output = nullptr;
    if (stream_name == "rtc") {
        scan_to_output = &tod_scan_to_output_scan_rtc;
    }
    else if (stream_name == "ptc") {
        scan_to_output = &tod_scan_to_output_scan_ptc;
    }
    else {
        logger->error("invalid TOD stream name '{}' for output row lookup", stream_name);
        return -1;
    }

    if (scan_index < 0 || scan_index >= scan_to_output->size()) {
        return -1;
    }
    return (*scan_to_output)(scan_index);
}
