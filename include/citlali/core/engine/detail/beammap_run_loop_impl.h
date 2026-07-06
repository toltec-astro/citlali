#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

template <class KidsProc, class RawObs>
void Beammap::run_loop(KidsProc &kidsproc, RawObs &rawobs) {
    // variable to control iteration
    bool keep_going = true;

    // declare random number generator
    boost::random::mt19937 eng;

    // boost random number generator (0,1)
    boost::random::uniform_int_distribution<> rands{0,1};

    if (beammap_rfi_mask_enabled && map_grouping == "detector") {
        logger->info("beammap rfi mask enabled: block_size={} min_good={} sigma_threshold={:.4g} sigma_floor={:.4g} dilate_blocks={} max_flagged_fraction={:.4f}",
                     beammap_rfi_mask_block_size_samples,
                     beammap_rfi_mask_min_good_samples,
                     beammap_rfi_mask_sigma_threshold,
                     beammap_rfi_mask_sigma_floor,
                     beammap_rfi_mask_dilate_blocks,
                     beammap_rfi_mask_max_flagged_fraction);
    }
    if (beammap_scan_band_mask_enabled && map_grouping == "detector") {
        logger->info(
            "beammap scan-band mask enabled: edge_rows={} min_row_pixels={} min_contiguous_rows={} row_median_sigma_threshold={:.4g} row_sigma_ratio_threshold={:.4g} max_flagged_fraction={:.4f}",
            beammap_scan_band_mask_edge_rows,
            beammap_scan_band_mask_min_row_pixels,
            beammap_scan_band_mask_min_contiguous_rows,
            beammap_scan_band_mask_row_median_sigma_threshold,
            beammap_scan_band_mask_row_sigma_ratio_threshold,
            beammap_scan_band_mask_max_flagged_fraction);
    }

    // iterative loop
    while (keep_going) {
        const bool locator_iter = is_beammap_locator_iter(current_iter);
        const bool measurement_iter = is_beammap_measurement_iter(current_iter);
        const bool first_measurement_iter = is_beammap_first_measurement_iter(current_iter);
        logger->info(
            "starting iter {} phase={} locator_iter={} measurement_start_iter={}",
            current_iter, beammap_iter_phase_name(current_iter),
            beammap_locator_iter, beammap_measurement_start_iter);

        configure_detector_source_centers_from_previous_fit();
        const bool detector_kernel_source_centers_active =
            map_grouping == "detector" &&
            rtcproc.run_kernel &&
            rtcproc.kernel.has_source_centers();
        const bool rerun_source_aware_rtc =
            first_measurement_iter && detector_kernel_source_centers_active;
        if (rerun_source_aware_rtc) {
            logger->info(
                "beammap iter {} rerunning RTC with previous-fit detector source centers; regular RTC TOD output disabled for this internal pass",
                current_iter);
            timestream_pipeline(kidsproc, rawobs, false);
        }

        // copy ptcs
        ptcs = ptcs0;
        // copy calibs
        calib_scans = calib_scans0;
        if (beammap_rfi_mask_enabled && map_grouping == "detector" &&
            rfi_mask_samples_flagged.size() == calib.n_dets &&
            rfi_mask_scans_flagged.size() == calib.n_dets) {
            rfi_mask_samples_flagged.setZero();
            rfi_mask_scans_flagged.setZero();
        }
        const bool skip_centered_kernel_map_feedback =
            rerun_source_aware_rtc;
        ptcproc.fruit_loops_kernel_feedback_enabled = !skip_centered_kernel_map_feedback;
        if (skip_centered_kernel_map_feedback) {
            logger->info(
                "beammap detector kernel map feedback disabled on iter {} while building the first source-aware kernel map",
                current_iter);
        }

        // copy previous-iteration maps for source-aperture convergence tests
        if (run_mapmaking && beammap_iter_tolerance > 0.0 && measurement_iter) {
            omb_copy.signal = omb.signal;
            omb_copy.weight = omb.weight;
        }

        if (ptcproc.run_fruit_loops) {
            // calc mean rms
            if (first_measurement_iter) {
                // use obs map buffer
                if (!omb.noise.empty()) {
                    omb.calc_median_rms();
                }
            }
            if (measurement_iter) {
                ptcproc.configure_fruit_loops_adaptive_gate(omb, calib, map_grouping, false);
            }
        }

        // progress bar
        tula::logging::progressbar pb(
            [&](const auto &msg) { logger->info("{}", msg); }, 100, "PTC progress ");


        auto ptc_line_audit_mutex = std::make_shared<std::mutex>();

        // cleaning (separate from mapmaking loop due to jinc mapmaking parallelization)
        grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
            bool model_subtracted_for_ptc_line_audit = false;
            if (run_mapmaking) {
                if (measurement_iter) {
                    if (!ptcproc.run_fruit_loops) {
                        // if not running fruit loops use source fit
                        logger->info("subtracting gaussian from tod");
                        // subtract gaussian
                        ptcproc.add_gaussian<timestream::TCProc::SourceType::NegativeGaussian>(ptcs[i], params, telescope.pixel_axes, map_grouping,
                                                                                               calib.apt,omb.pixel_size_rad, omb.n_rows, omb.n_cols);
                    }
                    else {
                        logger->info("subtracting map from tod");
                        // subtract map
                        ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(omb, ptcs[i], calib, ptcs[i].map_indices.data, telescope.pixel_axes,
                                                                                        map_grouping);
                        model_subtracted_for_ptc_line_audit = true;
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(*ptc_line_audit_mutex);
                apply_model_protected_ptc_line_audit(
                    ptcs[i], calib_scans[i], model_subtracted_for_ptc_line_audit);
            }

            // clean the maps
            logger->info("processed time chunk processing for scan {}", i + 1);
            ptcproc.run(ptcs[i], ptcs[i], calib_scans[i], telescope.pixel_axes, map_grouping);

            if (run_mapmaking) {
                if (measurement_iter) {
                    // if not running fruit loops use source fit
                    if (!ptcproc.run_fruit_loops) {
                        logger->info("adding gaussian to tod");
                        // add gaussian back
                        ptcproc.add_gaussian<timestream::TCProc::SourceType::Gaussian>(ptcs[i], params, telescope.pixel_axes, map_grouping, calib.apt,
                                                                                       omb.pixel_size_rad,omb.n_rows, omb.n_cols);
                    }
                    else {
                        logger->info("adding map to tod");
                        // add map back
                        ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(omb, ptcs[i], calib, ptcs[i].map_indices.data, telescope.pixel_axes,
                                                                                map_grouping);
                    }
                }
            }

            // For detector-grouped beammaps, keep the locator pass permissive so
            // bright-source scans are less likely to be rejected before we have
            // any source-location estimate to feed back into later iterations.
            if (map_grouping == "detector" && locator_iter) {
                logger->info("skipping remove_bad_dets on beammap locator iter {} for detector scan {}",
                             current_iter, ptcs[i].index.data + 1);
            }
            else {
                // remove outliers after clean
                calib_scans[i] = ptcproc.remove_bad_dets(ptcs[i], calib_scans[i], map_grouping);
            }

            if (map_grouping == "detector") {
                auto rfi_summary = apply_rfi_sample_mask(ptcs[i]);
                if (beammap_rfi_mask_enabled) {
                    if (rfi_summary.n_samples_flagged > 0 || rfi_summary.n_det_rejected > 0) {
                        logger->info("beammap rfi mask scan {}: masked {} samples across {}/{} detectors ({} rejected by max_flagged_fraction={:.4f})",
                                     ptcs[i].index.data + 1,
                                     rfi_summary.n_samples_flagged,
                                     rfi_summary.n_det_flagged,
                                     rfi_summary.n_det_candidates,
                                     rfi_summary.n_det_rejected,
                                     beammap_rfi_mask_max_flagged_fraction);
                    }
                    else {
                        logger->debug("beammap rfi mask scan {}: no samples masked", ptcs[i].index.data + 1);
                    }
                }
                const bool use_ptc_weights =
                    beammap_detector_weighting_mode == "ptc" ||
                    (beammap_detector_weighting_mode == "ptc_after_iter0" && measurement_iter);
                if (use_ptc_weights) {
                    logger->info("calculating detector-mode PTC weights for scan {} (mode={})",
                                 ptcs[i].index.data + 1, beammap_detector_weighting_mode);
                    ptcproc.calc_weights(ptcs[i], calib_scans[i].apt, telescope);
                    calib_scans[i] = ptcproc.reset_weights(ptcs[i], calib_scans[i], map_grouping);
                }
                else {
                    // Constant weights remain the safest default for bright beammaps.
                    ptcs[i].weights.data.resize(ptcs[i].scans.data.cols());
                    ptcs[i].weights.data.setOnes();
                }
            }
            else {
                // calculate weights
                logger->info("calculating weights for scan {}", ptcs[i].index.data + 1);
                ptcproc.calc_weights(ptcs[i], calib_scans[i].apt, telescope);

                // reset weights to median
                calib_scans[i] = ptcproc.reset_weights(ptcs[i], calib_scans[i], map_grouping);
            }

            // calc stats
            logger->debug("calculating stats");
            diagnostics.calc_stats(ptcs[i]);

            return 0;
        });

        auto clear_beammap_ptc_diagnostics = [&]() {
            for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
            }
        };


        auto write_beammap_ptc_products = [&](int output_iter) {
            if (verbose_mode) {
                logger->debug("writing chunk summaries for beammap PTC iteration {}", output_iter);
                for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                    write_chunk_summary(ptcs[i]);
                }
            }
            if (!ptcdiag_filename.empty()) {
                logger->info("writing ptc diagnostics sidecar chunks for beammap iteration {}", output_iter);
                for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                    ptcproc.append_diag_to_netcdf(ptcs[i], ptcdiag_filename, calib_scans[i], ptcs[i].index.data);
                    if (!(run_tod_output && run_tod_output_ptc &&
                          !tod_filename.empty())) {
                        ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
                    }
                }
            }
            if (run_tod_output && run_tod_output_ptc && !tod_filename.empty()) {
                    logger->info("writing processed time chunk for beammap iteration {}", output_iter);
                    auto ptc_filename_it = tod_filename.find("ptc");
                    if (ptc_filename_it != tod_filename.end() && !ptc_filename_it->second.empty()) {
                        try {
                            netCDF::NcFile ptc_tod_file(ptc_filename_it->second, netCDF::NcFile::write);
                            netCDF::NcVar fruit_iter_var = ptc_tod_file.getVar("FRUITLOOPS_ITER");
                            if (!fruit_iter_var.isNull()) {
                                fruit_iter_var.putVar(&output_iter);
                            }
                            else {
                                logger->warn("PTC TOD file {} has no FRUITLOOPS_ITER variable",
                                             ptc_filename_it->second);
                            }
                        } catch (const std::exception &e) {
                            logger->warn("failed to update PTC TOD FRUITLOOPS_ITER in {}: {}",
                                         ptc_filename_it->second, e.what());
                        }
                    }
                    for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                        const auto ptc_scan_row = tod_output_scan_row(i, "ptc");
                        if (ptc_scan_row < 0) {
                            continue;
                        }
                        ptcproc.append_to_netcdf(ptcs[i], tod_filename["ptc"], map_grouping, telescope.pixel_axes,
                                                 ptcs[i].pointing_offsets_arcsec.data, calib_scans[i], true, ptc_scan_row);
                        ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
                    }
            }
            write_detector_specific_ptc_tod(output_iter);
        };

        logger->info("starting mapmaking");

        if (run_mapmaking) {
            const auto mapmaking_grouping = typed_config.mapmaking.grouping;
            const auto mapmaking_method = typed_config.mapmaking.method;
            auto run_mapmaking_pass = [&](bool update_progress) {
                Eigen::Matrix<bool, Eigen::Dynamic, 1> active_maps;
                const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps_ptr = nullptr;
                Eigen::Index n_active_maps = n_maps;
                if (mapmaking_grouping ==
                        citlali::config::MapGrouping::detector &&
                    converged.size() == n_maps) {
                    const Eigen::Index n_converged = (converged.array() == true).count();
                    if (n_converged > 0 && n_converged < n_maps) {
                        active_maps.resize(n_maps);
                        n_active_maps = 0;
                        for (Eigen::Index i = 0; i < n_maps; ++i) {
                            active_maps(i) = !converged(i);
                            if (active_maps(i)) {
                                ++n_active_maps;
                            }
                        }
                        active_maps_ptr = &active_maps;
                        logger->info("beammap detector mapmaking: remaking {}/{} unconverged maps",
                                     n_active_maps, n_maps);
                    }
                }

                if (mapmaking_method == citlali::config::MapMethod::jinc &&
                    static_cast<Eigen::Index>(omb.grid_weight.size()) != n_maps) {
                    logger->info("allocating jinc grid_weight maps: current={} expected={}",
                                 omb.grid_weight.size(), n_maps);
                    omb.grid_weight.assign(
                        static_cast<size_t>(n_maps),
                        Eigen::MatrixXd::Zero(omb.n_rows, omb.n_cols));
                }

                // set maps to zero for each pass
                omb.clear_contribution_diag();
                for (Eigen::Index i = 0; i < n_maps; ++i) {
                    if (active_maps_ptr != nullptr && !(*active_maps_ptr)(i)) {
                        continue;
                    }
                    omb.signal[i].setZero();
                    omb.weight[i].setZero();
                    if (!omb.grid_weight.empty()) {
                        omb.grid_weight[i].setZero();
                    }

                    if (!omb.coverage.empty()) {
                        omb.coverage[i].setZero();
                    }
                    if (rtcproc.run_kernel) {
                        omb.kernel[i].setZero();
                    }
                    if (!omb.noise.empty()) {
                        omb.noise[i].setZero();
                    }

                    if (run_noise) {
                        for (auto &ptcdata : ptcs) {
                            if (omb.randomize_dets) {
                                ptcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                                                         omb.n_noise, calib.n_dets)
                                                         .unaryExpr([&](int dummy) { return 2 * rands(eng) - 1; });
                            }
                            else {
                                ptcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(omb.n_noise)
                                                         .unaryExpr([&](int dummy) { return 2 * rands(eng) - 1; });
                            }
                        }
                    }
                }

                logger->info("running mapmaking");

                if (mapmaking_grouping ==
                    citlali::config::MapGrouping::detector) {
                    bool run_omb = true;
                    for (std::size_t scan_vec_idx = 0; scan_vec_idx < ptcs.size(); ++scan_vec_idx) {
                        auto &ptc = ptcs[scan_vec_idx];
                        auto &scan_apt = calib_scans[scan_vec_idx].apt;
                        if (mapmaking_method ==
                            citlali::config::MapMethod::naive) {
                            naive_mm.populate_maps_naive_parallel(ptc, omb, cmb, ptc.map_indices.data,
                                                                  telescope.pixel_axes, scan_apt,
                                                                  telescope.d_fsmp, run_omb, run_noise,
                                                                  active_maps_ptr);
                        }
                        else if (mapmaking_method ==
                                 citlali::config::MapMethod::jinc) {
                            std::array<Eigen::Index, 3> array_counts = {0, 0, 0};
                            for (Eigen::Index det = 0; det < ptc.scans.data.cols(); ++det) {
                                auto array_index = static_cast<int>(calib.apt["array"](det));
                                if (array_index >= 0 && array_index < static_cast<int>(array_counts.size())) {
                                    array_counts[static_cast<size_t>(array_index)]++;
                                }
                            }
                            Eigen::Index map_min = -1;
                            Eigen::Index map_max = -1;
                            if (ptc.map_indices.data.size() > 0) {
                                map_min = ptc.map_indices.data.minCoeff();
                                map_max = ptc.map_indices.data.maxCoeff();
                            }
                            std::ostringstream kernel_dims;
                            for (int array_index = 0; array_index < 3; ++array_index) {
                                auto it = jinc_mm.jinc_weights_mat.find(array_index);
                                if (it == jinc_mm.jinc_weights_mat.end()) {
                                    continue;
                                }
                                if (kernel_dims.tellp() > 0) {
                                    kernel_dims << ", ";
                                }
                                kernel_dims << "a" << array_index << "="
                                            << it->second.rows() << "x" << it->second.cols();
                            }
                            logger->info(
                                "beammap jinc preflight: n_dets={} n_pts={} n_maps={} map_index_range=[{}, {}] "
                                "subpixel_n={} kernel_dims=[{}] array_counts=[{},{},{}]",
                                ptc.scans.data.cols(),
                                ptc.scans.data.rows(),
                                omb.signal.size(),
                                map_min,
                                map_max,
                                jinc_mm.subpixel_n,
                                kernel_dims.str(),
                                array_counts[0],
                                array_counts[1],
                                array_counts[2]);
                            jinc_mm.populate_maps_jinc_parallel(ptc, omb, cmb, ptc.map_indices.data,
                                                                telescope.pixel_axes, scan_apt,
                                                                telescope.d_fsmp, run_omb, run_noise,
                                                                active_maps_ptr);
                        }
                        if (update_progress) {
                            pb.count(telescope.scan_indices.cols(), 1);
                        }
                    }
                }
                else {
                    grppi::map(tula::grppi_utils::dyn_ex(map_parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
                        bool run_omb = true;
                        if (mapmaking_method ==
                            citlali::config::MapMethod::naive) {
                            naive_mm.populate_maps_naive(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                                        telescope.pixel_axes, calib_scans[i].apt, telescope.d_fsmp,
                                                        run_omb, run_noise);
                        }
                        else if (mapmaking_method ==
                                 citlali::config::MapMethod::jinc) {
                            jinc_mm.populate_maps_jinc(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                                       telescope.pixel_axes, calib_scans[i].apt, telescope.d_fsmp,
                                                       run_omb, run_noise);
                        }
                        if (update_progress) {
                            pb.count(telescope.scan_indices.cols(), 1);
                        }
                        return 0;
                    });
                }

                logger->info("normalizing maps");
                if (rtcproc.run_kernel && !omb.grid_weight.empty()) {
                    timestream::log_kernel_map_diag(
                        logger,
                        "beammap iter " + std::to_string(current_iter) + " before normalize",
                        omb.kernel,
                        active_maps_ptr,
                        &omb.grid_weight);
                }
                omb.normalize_maps(active_maps_ptr);
                if (rtcproc.run_kernel) {
                    timestream::log_kernel_map_diag(
                        logger,
                        "beammap iter " + std::to_string(current_iter) + " after normalize",
                        omb.kernel,
                        active_maps_ptr);
                }
                if (!omb.normalize_support_diag.empty()) {
                    Eigen::Index n_diag_maps = 0;
                    Eigen::Index total_masked = 0;
                    Eigen::Index total_no_accum_weight = 0;
                    Eigen::Index total_bad_grid_weight = 0;
                    Eigen::Index total_support_threshold = 0;
                    Eigen::Index total_raw_signal_nonzero = 0;
                    Eigen::Index total_adjacent_support = 0;
                    std::vector<Eigen::Index> suspicious_maps;

                    for (Eigen::Index map_index = 0;
                         map_index < static_cast<Eigen::Index>(omb.normalize_support_diag.size());
                         ++map_index) {
                        const auto &diag = omb.normalize_support_diag[map_index];
                        if (diag.map_index < 0) {
                            continue;
                        }
                        n_diag_maps++;
                        total_masked += diag.n_masked;
                        total_no_accum_weight += diag.n_masked_no_accum_weight;
                        total_bad_grid_weight += diag.n_masked_bad_grid_weight_with_accum_weight;
                        total_support_threshold += diag.n_masked_by_support_threshold;
                        total_raw_signal_nonzero += diag.n_masked_raw_signal_nonzero;
                        total_adjacent_support += diag.n_masked_adjacent_support;
                        if (diag.n_masked_bad_grid_weight_with_accum_weight > 0 ||
                            diag.n_masked_by_support_threshold > 0 ||
                            diag.n_masked_adjacent_support > 0 ||
                            diag.n_masked_raw_signal_nonzero > 0) {
                            suspicious_maps.push_back(map_index);
                        }
                    }

                    logger->info(
                        "beammap normalize support summary iter={} maps={} masked={} no_accum_weight={} bad_grid_weight_with_accum_weight={} support_threshold={} raw_signal_nonzero={} adjacent_support_holes={}",
                        current_iter,
                        n_diag_maps,
                        total_masked,
                        total_no_accum_weight,
                        total_bad_grid_weight,
                        total_support_threshold,
                        total_raw_signal_nonzero,
                        total_adjacent_support);

                    auto support_diag_score = [&](Eigen::Index map_index) {
                        const auto &diag = omb.normalize_support_diag[map_index];
                        return diag.n_masked_adjacent_support +
                               diag.n_masked_bad_grid_weight_with_accum_weight +
                               diag.n_masked_by_support_threshold +
                               diag.n_masked_raw_signal_nonzero;
                    };
                    std::sort(suspicious_maps.begin(), suspicious_maps.end(),
                              [&](Eigen::Index lhs, Eigen::Index rhs) {
                                  const auto lhs_score = support_diag_score(lhs);
                                  const auto rhs_score = support_diag_score(rhs);
                                  if (lhs_score != rhs_score) {
                                      return lhs_score > rhs_score;
                                  }
                                  const double lhs_neighbor =
                                      omb.normalize_support_diag[lhs].max_masked_neighbor_weight;
                                  const double rhs_neighbor =
                                      omb.normalize_support_diag[rhs].max_masked_neighbor_weight;
                                  return std::isfinite(lhs_neighbor) && std::isfinite(rhs_neighbor)
                                             ? lhs_neighbor > rhs_neighbor
                                             : std::isfinite(lhs_neighbor);
                              });

                    auto cause_name = [](int cause) {
                        switch (cause) {
                        case 1:
                            return "no_accum_weight";
                        case 2:
                            return "bad_grid_weight";
                        case 3:
                            return "support_threshold";
                        default:
                            return "unknown";
                        }
                    };

                    const Eigen::Index n_log =
                        std::min<Eigen::Index>(10, static_cast<Eigen::Index>(suspicious_maps.size()));
                    for (Eigen::Index rank = 0; rank < n_log; ++rank) {
                        const Eigen::Index map_index = suspicious_maps[rank];
                        const auto &diag = omb.normalize_support_diag[map_index];
                        const int uid = (map_index < calib.apt["uid"].size())
                                            ? static_cast<int>(std::lround(calib.apt["uid"](map_index)))
                                            : -1;
                        const int array = (map_index < calib.apt["array"].size())
                                              ? static_cast<int>(std::lround(calib.apt["array"](map_index)))
                                              : -1;
                        const int nw = (map_index < calib.apt["nw"].size())
                                           ? static_cast<int>(std::lround(calib.apt["nw"](map_index)))
                                           : -1;
                        const double x_t = (map_index < calib.apt["x_t"].size())
                                               ? calib.apt["x_t"](map_index)
                                               : std::numeric_limits<double>::quiet_NaN();
                        const double y_t = (map_index < calib.apt["y_t"].size())
                                               ? calib.apt["y_t"](map_index)
                                               : std::numeric_limits<double>::quiet_NaN();
                        logger->info(
                            "beammap normalize support detail iter={} rank={} map={} uid={} array={} nw={} x_t={:.3f} y_t={:.3f} masked={} no_accum={} bad_grid_with_accum={} threshold={} raw_signal_nonzero={} adjacent_holes={} support_threshold={:.4g} max_raw_signal={:.4g} max_neighbor_weight={:.4g} max_neighbor_rc=({}, {}) max_neighbor_cause={}",
                            current_iter,
                            rank + 1,
                            map_index,
                            uid,
                            array,
                            nw,
                            x_t,
                            y_t,
                            diag.n_masked,
                            diag.n_masked_no_accum_weight,
                            diag.n_masked_bad_grid_weight_with_accum_weight,
                            diag.n_masked_by_support_threshold,
                            diag.n_masked_raw_signal_nonzero,
                            diag.n_masked_adjacent_support,
                            diag.support_weight_threshold,
                            diag.max_masked_abs_raw_signal,
                            diag.max_masked_neighbor_weight,
                            diag.max_neighbor_row,
                            diag.max_neighbor_col,
                            cause_name(diag.max_neighbor_cause));
                    }
                }
            };

            run_mapmaking_pass(true);

            if (beammap_scan_band_mask_enabled && map_grouping == "detector" && locator_iter) {
                auto scan_band_summary = apply_scan_band_mask(omb);
                if (scan_band_summary.n_samples_flagged > 0) {
                    logger->info(
                        "beammap scan-band mask summary: flagged {} samples in {} rows across {} detectors ({} rejected by max_flagged_fraction={:.4f}); rebuilding maps",
                        scan_band_summary.n_samples_flagged,
                        scan_band_summary.n_rows_flagged,
                        scan_band_summary.n_det_flagged,
                        scan_band_summary.n_det_rejected,
                        beammap_scan_band_mask_max_flagged_fraction);
                    run_mapmaking_pass(false);
                }
                else {
                    logger->info(
                        "beammap scan-band mask summary: no edge bands flagged ({} detectors rejected by max_flagged_fraction={:.4f})",
                        scan_band_summary.n_det_rejected,
                        beammap_scan_band_mask_max_flagged_fraction);
                }
            }

            Eigen::VectorXi iter_bound_low = Eigen::VectorXi::Zero(map_fitter.n_params);
            Eigen::VectorXi iter_bound_high = Eigen::VectorXi::Zero(map_fitter.n_params);
            Eigen::Index iter_bound_any = 0;
            Eigen::Index iter_init_prev = 0;
            Eigen::Index iter_init_prior = 0;
            Eigen::Index iter_init_blind = 0;
            Eigen::Index iter_init_skip = 0;
            Eigen::Index iter_attempt_prev = 0;
            Eigen::Index iter_attempt_prior = 0;
            Eigen::Index iter_attempt_blind = 0;
            Eigen::Index iter_fail_prev = 0;
            Eigen::Index iter_fail_prior = 0;
            Eigen::Index iter_fail_blind = 0;
            Eigen::Index iter_prev_rejected_by_peak = 0;
            Eigen::Index iter_init_amp_zero_prev = 0;
            Eigen::Index iter_init_amp_zero_prior = 0;
            Eigen::Index iter_init_amp_zero_blind = 0;
            Eigen::Index iter_amp_bounds_zero_prev = 0;
            Eigen::Index iter_amp_bounds_zero_prior = 0;
            Eigen::Index iter_amp_bounds_zero_blind = 0;

            logger->info("fitting maps");
            logger->info("beammap fit diagnostics enabled");
            if (beammap_priors_enabled && beammap_soft_priors_loaded && map_grouping == "detector") {
                update_prior_frame_estimates();
            }
            // Run beammap fits sequentially. This avoids allocator/covariance instability
            // observed with parallel Ceres fits on some systems.
            for (Eigen::Index i = 0; i < n_maps; ++i) {
                logger->debug("beammap fit checkpoint: map={} begin converged={}", i, converged(i));

                if (omb.signal[i].rows() != omb.n_rows || omb.signal[i].cols() != omb.n_cols ||
                    omb.weight[i].rows() != omb.n_rows || omb.weight[i].cols() != omb.n_cols) {
                    logger->error("beammap fit map={} geometry mismatch: signal={}x{} weight={}x{} expected={}x{}",
                                  i, omb.signal[i].rows(), omb.signal[i].cols(),
                                  omb.weight[i].rows(), omb.weight[i].cols(),
                                  omb.n_rows, omb.n_cols);
                    std::exit(EXIT_FAILURE);
                }

                const auto &sig = omb.signal[i];
                const auto &wt = omb.weight[i];
                const Eigen::Index n_pix = sig.size();
                const Eigen::Index sig_finite = sig.array().isFinite().count();
                const Eigen::Index wt_finite = wt.array().isFinite().count();
                const Eigen::Index wt_pos = (wt.array() > 0.0).count();
                logger->debug("beammap fit map={} stats: sig_finite={}/{} wt_finite={}/{} wt_pos={}/{} sig[min,max]=({:.6g}, {:.6g}) wt[min,max]=({:.6g}, {:.6g})",
                              i, sig_finite, n_pix, wt_finite, n_pix, wt_pos, n_pix,
                              sig.minCoeff(), sig.maxCoeff(), wt.minCoeff(), wt.maxCoeff());

                // only fit if not converged
                if (!converged(i)) {
                    if (prior_diag_values.rows() == n_maps && prior_diag_values.cols() == n_prior_diag_cols) {
                        prior_diag_values.row(i).setConstant(std::numeric_limits<double>::quiet_NaN());
                        prior_diag_values(i, prior_init_mode_col) = -1.0;
                        prior_diag_values(i, prior_used_col) = 0.0;
                        prior_diag_values(i, prior_fallback_blind_col) = 0.0;
                        prior_diag_values(i, prior_no_candidate_reason_col) = 0.0;
                        prior_diag_values(i, prior_slot_index_col) = -1.0;
                    }

                    const Eigen::Index n_weight_pos = (omb.weight[i].array() > 0.0).count();
                    if (n_weight_pos < map_fitter.n_params) {
                        logger->warn("beammap fit map={} skipped: insufficient weighted pixels ({})", i, n_weight_pos);
                        params.row(i).setZero();
                        perrors.row(i).setZero();
                        fit_diag_init_params.row(i).setZero();
                        fit_diag_lower_limits.row(i).setZero();
                        fit_diag_upper_limits.row(i).setZero();
                        fit_diag_hit_lower.row(i).setZero();
                        fit_diag_hit_upper.row(i).setZero();
                        fit_diag_bound_code(i) = 0;
                        fit_diag_bound_nhit(i) = 0;
                        good_fits(i) = false;
                        continue;
                    }

                    // get array number
                    auto array = maps_to_arrays(i);
                    // get initial guess fwhm from theoretical fwhms for the arrays
                    double init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/omb.pixel_size_rad;
                    // choose fit initialization
                    double init_row = -99.0;
                    double init_col = -99.0;
                    bool init_from_prev = false;
                    bool init_from_prior = false;
                    enum class FitInitMode { Blind, Previous, Prior };
                    auto init_mode = FitInitMode::Blind;
                    const bool can_try_prior =
                        beammap_priors_enabled && beammap_soft_priors_loaded && map_grouping == "detector";
                    if (measurement_iter &&
                        good_fits(i) &&
                        p0.cols() > 2 &&
                        std::isfinite(p0(i,0)) && p0(i,0) > 0.0 &&
                        std::isfinite(p0(i,1)) && std::isfinite(p0(i,2))) {
                        const double prev_col = p0(i,1);
                        const double prev_row = p0(i,2);
                        Eigen::Index prev_row_i = static_cast<Eigen::Index>(std::llround(prev_row));
                        Eigen::Index prev_col_i = static_cast<Eigen::Index>(std::llround(prev_col));
                        bool prev_seed_valid = false;
                        if (prev_row_i >= 0 && prev_row_i < omb.signal[i].rows() &&
                            prev_col_i >= 0 && prev_col_i < omb.signal[i].cols()) {
                            const double seed_w = omb.weight[i](prev_row_i, prev_col_i);
                            const double seed_s = omb.signal[i](prev_row_i, prev_col_i);
                            prev_seed_valid = std::isfinite(seed_w) && seed_w > 0.0 &&
                                              std::isfinite(seed_s) && seed_s > 0.0;
                            if (prev_seed_valid) {
                                Eigen::Index peak_row = -1;
                                Eigen::Index peak_col = -1;
                                double peak_snr = -std::numeric_limits<double>::infinity();
                                if (find_map_weighted_peak(i, peak_row, peak_col, peak_snr) &&
                                    peak_row >= 0 && peak_col >= 0 && std::isfinite(peak_snr)) {
                                    const double prev_snr = seed_s * std::sqrt(seed_w);
                                    const double dr = static_cast<double>(peak_row) - prev_row;
                                    const double dc = static_cast<double>(peak_col) - prev_col;
                                    const double dist_pix = std::sqrt(dr * dr + dc * dc);
                                    const double min_switch_dist_pix = std::max(1.0, init_fwhm);
                                    constexpr double min_switch_snr_ratio = 1.25;
                                    bool prior_allows_switch = true;
                                    if (can_try_prior) {
                                        const int array_int = static_cast<int>(maps_to_arrays(i));
                                        const int nw_int = static_cast<int>(std::lround(calib.apt["nw"](i)));
                                        const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
                                        const double col0 = static_cast<double>(omb.n_cols - 1) / 2.0;
                                        const double row0 = static_cast<double>(omb.n_rows - 1) / 2.0;
                                        const double derot_elev_rad = get_prior_derot_elev_rad();
                                        const double prior_max_d2 = effective_prior_max_d2();

                                        auto prior_compatible = [&](double row, double col, double &d2_out) {
                                            const double x_raw = pix_to_arcsec * (col - col0);
                                            const double y_raw = pix_to_arcsec * (row - row0);
                                            double x_prior = std::numeric_limits<double>::quiet_NaN();
                                            double y_prior = std::numeric_limits<double>::quiet_NaN();
                                            d2_out = std::numeric_limits<double>::infinity();
                                            int slot_index = -1;
                                            if (!observed_to_prior_frame(array_int, x_raw, y_raw, derot_elev_rad,
                                                                         x_prior, y_prior, nullptr, nullptr, true)) {
                                                return false;
                                            }
                                            if (!match_prior_slot(array_int, nw_int, x_prior, y_prior,
                                                                  d2_out, slot_index)) {
                                                return false;
                                            }
                                            static_cast<void>(slot_index);
                                            return prior_max_d2 <= 0.0 || d2_out <= prior_max_d2;
                                        };

                                        double prev_prior_d2 = std::numeric_limits<double>::infinity();
                                        double peak_prior_d2 = std::numeric_limits<double>::infinity();
                                        const bool prev_prior_ok = prior_compatible(prev_row, prev_col, prev_prior_d2);
                                        const bool peak_prior_ok = prior_compatible(
                                            static_cast<double>(peak_row), static_cast<double>(peak_col), peak_prior_d2);
                                        prior_allows_switch = peak_prior_ok || !prev_prior_ok;
                                        if (!prior_allows_switch) {
                                            logger->debug(
                                                "beammap fit map={} kept previous init over stronger weighted peak because prior d2 prev={} peak={} max_d2={}",
                                                i, prev_prior_d2, peak_prior_d2, prior_max_d2);
                                        }
                                    }
                                    if (std::isfinite(prev_snr) &&
                                        peak_snr > min_switch_snr_ratio * prev_snr &&
                                        dist_pix > min_switch_dist_pix &&
                                        prior_allows_switch) {
                                        prev_seed_valid = false;
                                        iter_prev_rejected_by_peak++;
                                        logger->debug(
                                            "beammap fit map={} rejected previous init: current weighted peak row={} col={} snr={} is {} pix from previous row={} col={} snr={}",
                                            i, peak_row, peak_col, peak_snr, dist_pix,
                                            prev_row, prev_col, prev_snr);
                                    }
                                }
                            }
                        }
                        if (prev_seed_valid) {
                            init_col = prev_col;
                            init_row = prev_row;
                            init_from_prev = true;
                            init_mode = FitInitMode::Previous;
                            iter_init_prev++;
                        }
                        else {
                            logger->debug(
                                "beammap fit map={} rejected previous init at row={} col={} due to invalid/no-weight/non-positive seed pixel",
                                i, prev_row, prev_col);
                        }
                    }
                    if (!init_from_prev && can_try_prior) {
                        if (choose_prior_guided_init(i, init_row, init_col)) {
                            init_from_prior = true;
                            init_mode = FitInitMode::Prior;
                            iter_init_prior++;
                        }
                        else if (!beammap_priors_fallback_blind) {
                            if (prior_diag_values.rows() == n_maps && prior_diag_values.cols() == n_prior_diag_cols) {
                                prior_diag_values(i, prior_init_mode_col) = -1.0;
                            }
                            logger->warn("beammap fit map={} skipped: no prior-guided init candidate and fallback_blind=false", i);
                            params.row(i).setZero();
                            perrors.row(i).setZero();
                            fit_diag_init_params.row(i).setZero();
                            fit_diag_lower_limits.row(i).setZero();
                            fit_diag_upper_limits.row(i).setZero();
                            fit_diag_hit_lower.row(i).setZero();
                            fit_diag_hit_upper.row(i).setZero();
                            fit_diag_bound_code(i) = 0;
                            fit_diag_bound_nhit(i) = 0;
                            good_fits(i) = false;
                            iter_init_skip++;
                            continue;
                        }
                        else if (prior_diag_values.rows() == n_maps && prior_diag_values.cols() == n_prior_diag_cols) {
                            prior_diag_values(i, prior_fallback_blind_col) = 1.0;
                        }
                    }
                    if (!init_from_prev && !init_from_prior) {
                        iter_init_blind++;
                    }
                    if (prior_diag_values.rows() == n_maps && prior_diag_values.cols() == n_prior_diag_cols) {
                        if (init_from_prev) {
                            prior_diag_values(i, prior_init_mode_col) = 1.0;
                        }
                        else if (init_from_prior) {
                            prior_diag_values(i, prior_init_mode_col) = 2.0;
                        }
                        else {
                            prior_diag_values(i, prior_init_mode_col) = 0.0;
                        }
                    }
                    logger->debug("beammap fit map={} init mode={} row={:.3f} col={:.3f}",
                                  i, init_from_prev ? "previous" : (init_from_prior ? "prior" : "blind"),
                                  init_row, init_col);
                    // fit the maps
                    logger->debug("beammap fit checkpoint: map={} call fit_to_gaussian", i);
                    engine_utils::mapFitter::FitDiagnostics fit_diag;
                    auto [det_params, det_perror, good_fit] =
                        map_fitter.fit_to_gaussian<engine_utils::mapFitter::beammap>(omb.signal[i], omb.weight[i],
                                                                                     init_fwhm, init_row, init_col, &fit_diag);
                    logger->debug("beammap fit checkpoint: map={} fit_to_gaussian returned good_fit={}", i, good_fit);

                    if (!(det_params.array().isFinite().all() && det_perror.array().isFinite().all())) {
                        det_params.setZero();
                        det_perror.setZero();
                        good_fit = false;
                    }

                    params.row(i) = det_params;
                    perrors.row(i) = det_perror;
                    good_fits(i) = good_fit;

                    bool init_amp_zero = false;
                    bool amp_bounds_zero = false;
                    if (fit_diag.valid &&
                        fit_diag.init_params.size() > 0 &&
                        fit_diag.lower_limits.size() > 0 &&
                        fit_diag.upper_limits.size() > 0) {
                        const double init_amp = fit_diag.init_params(0);
                        const double amp_low = fit_diag.lower_limits(0);
                        const double amp_high = fit_diag.upper_limits(0);
                        init_amp_zero = std::isfinite(init_amp) && std::abs(init_amp) <= 1e-12;
                        amp_bounds_zero =
                            std::isfinite(amp_low) && std::isfinite(amp_high) &&
                            std::abs(amp_high - amp_low) <= 1e-12;
                    }
                    switch (init_mode) {
                        case FitInitMode::Previous:
                            iter_attempt_prev++;
                            if (!good_fit) {
                                iter_fail_prev++;
                            }
                            if (init_amp_zero) {
                                iter_init_amp_zero_prev++;
                            }
                            if (amp_bounds_zero) {
                                iter_amp_bounds_zero_prev++;
                            }
                            break;
                        case FitInitMode::Prior:
                            iter_attempt_prior++;
                            if (!good_fit) {
                                iter_fail_prior++;
                            }
                            if (init_amp_zero) {
                                iter_init_amp_zero_prior++;
                            }
                            if (amp_bounds_zero) {
                                iter_amp_bounds_zero_prior++;
                            }
                            break;
                        case FitInitMode::Blind:
                            iter_attempt_blind++;
                            if (!good_fit) {
                                iter_fail_blind++;
                            }
                            if (init_amp_zero) {
                                iter_init_amp_zero_blind++;
                            }
                            if (amp_bounds_zero) {
                                iter_amp_bounds_zero_blind++;
                            }
                            break;
                    }

                    if (fit_diag.valid &&
                        fit_diag.init_params.size() == map_fitter.n_params &&
                        fit_diag.lower_limits.size() == map_fitter.n_params &&
                        fit_diag.upper_limits.size() == map_fitter.n_params &&
                        fit_diag.hit_lower.size() == map_fitter.n_params &&
                        fit_diag.hit_upper.size() == map_fitter.n_params) {
                        fit_diag_init_params.row(i) = fit_diag.init_params.transpose();
                        fit_diag_lower_limits.row(i) = fit_diag.lower_limits.transpose();
                        fit_diag_upper_limits.row(i) = fit_diag.upper_limits.transpose();
                        fit_diag_hit_lower.row(i) = fit_diag.hit_lower.transpose();
                        fit_diag_hit_upper.row(i) = fit_diag.hit_upper.transpose();

                        int bound_code = 0;
                        int bound_nhit = 0;
                        for (int p = 0; p < map_fitter.n_params; ++p) {
                            const bool hit_low = fit_diag.hit_lower(p) != 0;
                            const bool hit_high = fit_diag.hit_upper(p) != 0;
                            if (hit_low) {
                                bound_code |= (1 << (2 * p));
                                iter_bound_low(p)++;
                                bound_nhit++;
                            }
                            if (hit_high) {
                                bound_code |= (1 << (2 * p + 1));
                                iter_bound_high(p)++;
                                bound_nhit++;
                            }
                        }
                        fit_diag_bound_code(i) = bound_code;
                        fit_diag_bound_nhit(i) = bound_nhit;
                        if (bound_nhit > 0) {
                            iter_bound_any++;
                        }
                    }
                    else {
                        fit_diag_init_params.row(i).setZero();
                        fit_diag_lower_limits.row(i).setZero();
                        fit_diag_upper_limits.row(i).setZero();
                        fit_diag_hit_lower.row(i).setZero();
                        fit_diag_hit_upper.row(i).setZero();
                        fit_diag_bound_code(i) = 0;
                        fit_diag_bound_nhit(i) = 0;
                    }
                }
                // otherwise keep value from previous iteration
                else {
                    params.row(i) = p0.row(i);
                    perrors.row(i) = perror0.row(i);
                }

                logger->debug("beammap fit checkpoint: map={} end good_fit={}", i, good_fits(i));
            }

            logger->info("beammap init summary (iter {}): previous={} prior={} blind={} skipped={} prev_rejected_by_peak={}",
                         current_iter, iter_init_prev, iter_init_prior, iter_init_blind, iter_init_skip,
                         iter_prev_rejected_by_peak);
            logger->info(
                "beammap fit diagnostics (iter {}): prev fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{} | "
                "prior fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{} | "
                "blind fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{}",
                current_iter,
                iter_fail_prev, iter_attempt_prev, iter_init_amp_zero_prev, iter_attempt_prev,
                iter_amp_bounds_zero_prev, iter_attempt_prev,
                iter_fail_prior, iter_attempt_prior, iter_init_amp_zero_prior, iter_attempt_prior,
                iter_amp_bounds_zero_prior, iter_attempt_prior,
                iter_fail_blind, iter_attempt_blind, iter_init_amp_zero_blind, iter_attempt_blind,
                iter_amp_bounds_zero_blind, iter_attempt_blind);

            if (map_fitter.n_params >= 6) {
                logger->info(
                    "beammap fit bound summary (iter {}): any_hit={}/{} amp(lo/hi)={}/{} x(lo/hi)={}/{} y(lo/hi)={}/{} a(lo/hi)={}/{} b(lo/hi)={}/{} angle(lo/hi)={}/{}",
                    current_iter, iter_bound_any, n_maps,
                    iter_bound_low(0), iter_bound_high(0),
                    iter_bound_low(1), iter_bound_high(1),
                    iter_bound_low(2), iter_bound_high(2),
                    iter_bound_low(3), iter_bound_high(3),
                    iter_bound_low(4), iter_bound_high(4),
                    iter_bound_low(5), iter_bound_high(5));
            }
            else {
                logger->info("beammap fit bound summary (iter {}): any_hit={}/{}",
                             current_iter, iter_bound_any, n_maps);
            }
            logger->info("number of good fits {}/{}", static_cast<long long>(good_fits.cast<int>().sum()), n_maps);
        }

        const int completed_iter = current_iter;

        // increment loop iteration
        current_iter++;

        if (current_iter < beammap_iter_max) {
            // check if all detectors are converged
            if ((converged.array() == true).all()) {
                logger->info("all maps converged");
                keep_going = false;
            }
            else if (has_completed_beammap_measurement_iter(current_iter)) {
                // only do convergence test if tolerance is above zero, otherwise run all iterations
                if (run_mapmaking && beammap_iter_tolerance > 0) {
                    // loop through maps and check if it is converged
                    logger->info("checking convergence in fitted-source aperture radius={:.3f} arcsec",
                                 beammap_convergence_radius_arcsec);
                    Eigen::VectorXd convergence_delta =
                        Eigen::VectorXd::Constant(n_maps, std::numeric_limits<double>::quiet_NaN());
                    grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
                        if (!converged(i)) {
                            const double delta = calc_beammap_convergence_delta(i);
                            convergence_delta(i) = delta;
                            if (std::isfinite(delta) && delta <= beammap_iter_tolerance) {
                                // set as converged
                                converged(i) = true;
                                // set convergence iteration
                                converge_iter(i) = current_iter;
                            }
                        }
                        return 0;
                    });

                    Eigen::Index n_delta_finite = 0;
                    Eigen::Index n_delta_invalid = 0;
                    double max_delta = 0.0;
                    for (Eigen::Index i = 0; i < convergence_delta.size(); ++i) {
                        if (std::isfinite(convergence_delta(i))) {
                            n_delta_finite++;
                            max_delta = std::max(max_delta, convergence_delta(i));
                        }
                        else if (!converged(i)) {
                            n_delta_invalid++;
                        }
                    }

                    logger->info(
                        "{} maps converged on iter {} (finite_metrics={} invalid_metrics={} max_delta={})",
                        (converged.array() == true).count(), current_iter,
                        n_delta_finite, n_delta_invalid, max_delta);

                    // stop if all maps converged
                    if ((converged.array() == true).all()) {
                        logger->info("all maps converged");
                        keep_going = false;
                    }
                }
                else {
                    logger->info("bypassing convergence check");
                }
            }

            // set previous iteration fits to current iteration fits
            p0 = params;
            perror0 = perrors;
        }
        else {
            logger->info("max iteration reached");
            keep_going = false;
        }

        const bool beammap_iter_is_final = !keep_going;
        const bool write_beammap_ptc_this_iter =
            (beammap_tod_output_iter < 0 && beammap_iter_is_final) ||
            (beammap_tod_output_iter >= 0 && completed_iter == beammap_tod_output_iter);
        if (write_beammap_ptc_this_iter) {
            write_beammap_ptc_products(completed_iter);
        }
        else {
            clear_beammap_ptc_diagnostics();
        }
    }
}
