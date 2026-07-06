#pragma once

// Implementation detail included by pointing.h.

#include <citlali/core/pipeline/ordered_writer.h>
#include <citlali/core/pipeline/output_policy.h>

template <class KidsProc>
auto Pointing::run(KidsProc &kidsproc) {
    auto scans_done_mutex = std::make_shared<std::mutex>();
    auto ptc_line_audit_mutex = std::make_shared<std::mutex>();
    const auto mapmaking_method = typed_config.mapmaking.method;
    const bool make_maps = citlali::pipeline::mapmaking_enabled(*this);
    const bool make_noise_maps = citlali::pipeline::noise_maps_enabled(*this);

    const bool write_rtc =
        run_tod_output && run_tod_output_rtc && !tod_filename.empty();
    const bool write_ptc =
        run_tod_output && run_tod_output_ptc && !tod_filename.empty();
    const bool write_rtcdiag = !rtcdiag_filename.empty();
    const bool write_ptcdiag = !ptcdiag_filename.empty();

    auto rtc_writer =
        write_rtc ? std::make_shared<citlali::pipeline::OrderedWriter>()
                  : nullptr;
    auto ptc_writer =
        write_ptc ? std::make_shared<citlali::pipeline::OrderedWriter>()
                  : nullptr;
    auto rtcdiag_writer =
        write_rtcdiag ? std::make_shared<citlali::pipeline::OrderedWriter>()
                      : nullptr;
    auto ptcdiag_writer =
        write_ptcdiag ? std::make_shared<citlali::pipeline::OrderedWriter>()
                      : nullptr;

    auto farm = grppi::farm(n_threads,[&, scans_done_mutex, ptc_line_audit_mutex,
                                       rtc_writer, ptc_writer,
                                       rtcdiag_writer, ptcdiag_writer,
                                       mapmaking_method, make_maps,
                                       make_noise_maps, write_rtc, write_ptc,
                                       write_rtcdiag, write_ptcdiag](auto &rtcdata) {

        // starting index for scan
        Eigen::Index si = rtcdata.scan_indices.data(2);

        // current length of outer scans
        Eigen::Index sl = rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

        // copy scan's telescope vectors
        for (auto const& x: telescope.tel_data) {
            rtcdata.tel_data.data[x.first] = telescope.tel_data[x.first].segment(si,sl);
        }

        // copy pointing offsets
        for (auto const& [axis,offset]: pointing_offsets_arcsec) {
            rtcdata.pointing_offsets_arcsec.data[axis] = offset.segment(si,sl);
        }

        // get hwpr
        if (rtcproc.run_polarization) {
            if (calib.run_hwpr) {
                rtcdata.hwpr_angle.data = calib.hwpr_angle.segment(si + hwpr_start_indices, sl);
            }
        }

        // set up flags
        rtcdata.flags.data.resize(rtcdata.scans.data.rows(), rtcdata.scans.data.cols());
        rtcdata.flags.data.setConstant(0);

        if (interp_over_gaps) {
            for (auto const& [key, val] : calib.nw_limits) {
                auto mask_it = nw_masks.find(key);
                if (mask_it == nw_masks.end()) {
                    logger->error("missing gap mask for nw {}; cannot apply gap flagging", key);
                    std::exit(EXIT_FAILURE);
                }
                auto& mask = mask_it->second;

                Eigen::Index start = std::get<0>(calib.nw_limits[key]);
                Eigen::Index end = std::get<1>(calib.nw_limits[key]) - 1;

                for (int j = 0; j < rtcdata.flags.data.rows(); ++j) {
                    int start_index = j;
                    int size = 1;
                    if (rtcproc.filter_edge_guard.context_samples > 0) {
                        const int context = static_cast<int>(rtcproc.filter_edge_guard.context_samples);
                        start_index = std::max(0, j - context);
                        int end_index = std::min(j + context, static_cast<int>(rtcdata.flags.data.rows() - 1));
                        size = end_index - start_index + 1;
                    }
                    if (mask(j + si) == 0) {
                        rtcdata.flags.data.block(start_index, start, size, end - start + 1).setOnes();
                    }
                }
                logger->debug("{}/{} gaps flagged", rtcdata.flags.data.col(start).template cast<int>().sum(), rtcdata.flags.data.rows());
            }
        }

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;
        TCData<TCDataKind::RTC,Eigen::MatrixXd> rtc_outer_output;
        const auto rtc_scan_row = tod_output_scan_row(rtcdata.index.data, "rtc");
        const bool write_this_rtc = write_rtc && rtc_scan_row >= 0;
        auto *rtc_outer_output_ptr =
            (write_this_rtc && rtcproc.tod_output_outer) ? &rtc_outer_output : nullptr;

        {
            std::lock_guard<std::mutex> lk(*scans_done_mutex);
            logger->info("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done,
                         telescope.scan_indices.cols());
        }

        // run rtcproc
        logger->info("raw time chunk processing for scan {}", rtcdata.index.data + 1);
        apply_learned_rtc_sample_masks(rtcdata, calib);
        auto map_indices = rtcproc.run(rtcdata, ptcdata, calib, telescope, omb.pixel_size_rad, map_grouping,
                                       rtc_outer_output_ptr);
        const auto rtc_detector_summary =
            rtcproc.snapshot_detector_diag_summary(ptcdata.index.data);

        // remove flagged detectors
        rtcproc.remove_flagged_dets(ptcdata, calib.apt);

        // remove outliers before cleaning
        auto calib_scan = rtcproc.remove_bad_dets(ptcdata, calib, map_grouping);

        // remove duplicate tones
        if (!telescope.sim_obs) {
            calib_scan = rtcproc.remove_nearby_tones(ptcdata, calib_scan, map_grouping);
        }

        collect_rtc_learning_diagnostics(rtcdata, ptcdata, calib_scan, rtc_detector_summary);

        if (write_rtcdiag) {
            rtcdiag_writer->wait_turn(ptcdata.index.data);
            logger->info("writing rtc diagnostics sidecar chunk");
            rtcproc.append_diag_to_netcdf(ptcdata, rtcdiag_filename, calib_scan, ptcdata.index.data);
            rtcdiag_writer->advance();
        }

        // write rtc timestreams
        if (write_this_rtc) {
            rtc_writer->wait_turn(rtc_scan_row);
            if (rtcproc.tod_output_outer) {
                logger->info("writing outer raw time chunk");
                rtcproc.append_to_netcdf(rtc_outer_output, tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         rtc_outer_output.pointing_offsets_arcsec.data, calib, false, rtc_scan_row);
            }
            else {
                logger->info("writing raw time chunk");
                rtcproc.append_to_netcdf(ptcdata, tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         ptcdata.pointing_offsets_arcsec.data, calib, false, rtc_scan_row);
            }
            rtc_writer->advance();
        }
        if (write_rtc || write_rtcdiag) {
            rtcproc.clear_cached_diagnostics(ptcdata.index.data);
        }

        apply_learned_ptc_sample_masks(ptcdata, calib_scan);
        apply_learned_ptc_detector_exclusions(ptcdata, calib_scan);

        const bool use_fruit_noise_weights =
            ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty();
        const bool keep_source_subtracted_weights =
            use_fruit_noise_weights && !ptcproc.fruit_loops_recompute_weights_after_addback;

        // if running fruit loops and a map has been read in
        if (use_fruit_noise_weights) {
            timestream::log_kernel_matrix_diag(
                logger, "ptc before fruitloops map subtraction", ptcdata.kernel.data, ptcdata.index.data);
            logger->info("subtracting map from tod");
            // subtract map
            ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(ptcproc.tod_mb, ptcdata, calib_scan,
                                                                            map_indices, telescope.pixel_axes,
                                                                            map_grouping);
            timestream::log_kernel_matrix_diag(
                logger, "ptc after fruitloops map subtraction", ptcdata.kernel.data, ptcdata.index.data);
        }

        ptcproc.accumulate_weight_validation_atmosphere(ptcdata, calib_scan.apt);

        {
            std::lock_guard<std::mutex> lock(*ptc_line_audit_mutex);
            apply_model_protected_ptc_line_audit(ptcdata, calib_scan, use_fruit_noise_weights);
        }

        // run cleaning
        logger->info("processed time chunk processing for scan {}", ptcdata.index.data + 1);
        ptcproc.run(ptcdata, ptcdata, calib_scan, telescope.pixel_axes, map_grouping);
        timestream::log_kernel_matrix_diag(
            logger, "ptc after processed time chunk cleaning", ptcdata.kernel.data, ptcdata.index.data);
        const auto ptc_second_pass_summary =
            ptcproc.snapshot_second_pass_summary(ptcdata.index.data);

        // if running fruit loops and a map has been read in
        if (use_fruit_noise_weights) {
            // calculate weights
            logger->info("calculating weights for scan {} (fruit loops noise-only pass)",
                         ptcdata.index.data + 1);
            ptcproc.calc_weights(ptcdata, calib_scan.apt, telescope, true);

            // reset weights to median
            calib_scan = ptcproc.reset_weights(ptcdata, calib_scan, map_grouping);

            // populate maps
            if (make_maps) {
                bool run_omb = false;
                logger->info("populating noise maps");
                if (mapmaking_method == citlali::config::MapMethod::naive) {
                    naive_mm.populate_maps_naive(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                                 calib_scan.apt, telescope.d_fsmp, run_omb, make_noise_maps);
                }
                else if (mapmaking_method == citlali::config::MapMethod::jinc) {
                    jinc_mm.populate_maps_jinc(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                               calib_scan.apt, telescope.d_fsmp, run_omb, make_noise_maps);
                }
            }
            logger->info("adding map to tod");
            // add map back
            ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(ptcproc.tod_mb, ptcdata, calib_scan,
                                                                    map_indices, telescope.pixel_axes,
                                                                    map_grouping);
            timestream::log_kernel_matrix_diag(
                logger, "ptc after fruitloops map addback", ptcdata.kernel.data, ptcdata.index.data);
        }

        // remove outliers after cleaning
        calib_scan = ptcproc.remove_bad_dets(ptcdata, calib_scan, map_grouping);

        if (keep_source_subtracted_weights) {
            logger->info("keeping source-subtracted weights for scan {}", ptcdata.index.data + 1);
        }
        else {
            // calculate weights
            if (use_fruit_noise_weights) {
                logger->info("recomputing weights after fruit loops add-back for scan {}",
                             ptcdata.index.data + 1);
            }
            else {
                logger->info("calculating weights for scan {}", ptcdata.index.data + 1);
            }
            ptcproc.calc_weights(ptcdata, calib_scan.apt, telescope);

            // reset weights to median
            calib_scan = ptcproc.reset_weights(ptcdata, calib_scan, map_grouping);
        }

        const auto ptc_high_weight_summary =
            ptcproc.snapshot_high_weight_summary(ptcdata.index.data);
        collect_ptc_learning_diagnostics(
            ptcdata, calib_scan, ptc_second_pass_summary, ptc_high_weight_summary);

        // write ptc timestreams
        if (write_ptcdiag) {
            ptcdiag_writer->wait_turn(ptcdata.index.data);
            logger->info("writing ptc diagnostics sidecar chunk");
            ptcproc.append_diag_to_netcdf(ptcdata, ptcdiag_filename, calib_scan, ptcdata.index.data);
            ptcdiag_writer->advance();
        }

        const auto ptc_scan_row = tod_output_scan_row(ptcdata.index.data, "ptc");
        if (write_ptc && ptc_scan_row >= 0) {
            ptc_writer->wait_turn(ptc_scan_row);
            logger->info("writing processed time chunk");
            ptcproc.append_to_netcdf(ptcdata, tod_filename["ptc"], map_grouping, telescope.pixel_axes,
                                     ptcdata.pointing_offsets_arcsec.data, calib_scan, false, ptc_scan_row);
            ptc_writer->advance();
        }
        if (write_ptc || write_ptcdiag) {
            ptcproc.clear_cached_diagnostics(ptcdata.index.data);
        }

        // write out chunk summary
        if (verbose_mode) {
            write_chunk_summary(ptcdata);
        }

        // calc stats
        logger->debug("calculating stats");
        diagnostics.calc_stats(ptcdata);

        // populate maps
        if (make_maps) {
            bool run_omb = true;
            bool run_noise_fruit;

            // if running fruit loops, noise maps are made on source
            // subtracted timestreams so don't make them here unless
            // on first iteration
            if (ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty()) {
                run_noise_fruit = false;
            }
            else {
                run_noise_fruit = make_noise_maps;
            }
            apply_learned_mapmaking_detector_exclusions(ptcdata, calib_scan);
            logger->info("populating maps");
            if (mapmaking_method == citlali::config::MapMethod::naive) {
                naive_mm.populate_maps_naive(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                             calib_scan.apt, telescope.d_fsmp, run_omb, run_noise_fruit);
            }
            else if (mapmaking_method == citlali::config::MapMethod::jinc) {
                jinc_mm.populate_maps_jinc(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                           calib_scan.apt, telescope.d_fsmp, run_omb, run_noise_fruit);
            }
        }
        // increment number of completed scans
        {
            std::lock_guard<std::mutex> lk(*scans_done_mutex);
            n_scans_done++;
            logger->info("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done,
                         telescope.scan_indices.cols());
        }

    });

    return farm;
}
