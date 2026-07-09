#pragma once

// Implementation detail included by lali.h.

#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/timestream_output_context.h>
#include <citlali/core/pipeline/timestream_run_context.h>
#include <citlali/core/pipeline/timestream_scan_context.h>

auto Lali::run() -> run_stage_t {
    auto scans_done_mutex = std::make_shared<std::mutex>();
    auto scans_done_count = std::make_shared<int>(0);
    auto ptc_line_audit_mutex = std::make_shared<std::mutex>();
    const auto mapmaking_method = typed_config.mapmaking.method;
    const bool make_maps = citlali::pipeline::mapmaking_enabled(*this);
    const bool make_noise_maps = citlali::pipeline::noise_maps_enabled(*this);

    const auto output_flags =
        citlali::pipeline::standard_timestream_output_flags(*this);
    const auto output_writers =
        citlali::pipeline::make_timestream_output_writers(output_flags);
    auto map_grouping_ptr = std::make_shared<std::string>(
        citlali::pipeline::active_map_grouping_name(*this));

    auto farm_fn = std::function<void(input_t &)>{[&, scans_done_mutex,
                                                   scans_done_count,
                                                   ptc_line_audit_mutex,
                                                   output_writers,
                                                   mapmaking_method, make_maps,
                                                   make_noise_maps,
                                                   output_flags,
                                                   map_grouping_ptr](input_t &rtcdata) {
        auto &map_grouping = *map_grouping_ptr;
        const auto scan_window = citlali::pipeline::copy_rtc_scan_context(
            rtcdata, telescope, pointing_offsets_arcsec);
        citlali::pipeline::copy_hwpr_angle_if_enabled(
            rtcdata, calib, rtcproc.run_polarization, calib.run_hwpr,
            alignment.hwpr_start_index, scan_window.start, scan_window.length);
        citlali::pipeline::initialize_rtc_flags(rtcdata);
        if (citlali::config::timing_gap_interpolation_active(
                typed_config.runtime)) {
            citlali::pipeline::apply_gap_masks_to_rtc_flags(
                rtcdata, calib, alignment.network_masks, scan_window.start,
                rtcproc.filter_edge_guard.context_samples, logger);
        }

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;
        TCData<TCDataKind::RTC,Eigen::MatrixXd> rtc_outer_output;
        const auto rtc_scan_row = tod_output_scan_row(
            rtcdata.index.data, citlali::config::TodOutputStream::rtc);
        const bool write_this_rtc =
            output_flags.write_rtc && rtc_scan_row >= 0;
        auto *rtc_outer_output_ptr =
            (write_this_rtc && rtcproc.tod_output_outer) ? &rtc_outer_output : nullptr;

        citlali::pipeline::log_scan_start(
            scans_done_mutex, logger, rtcdata.index.data, *scans_done_count,
            telescope);

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

        if (output_flags.write_rtcdiag) {
            output_writers.rtcdiag->wait_turn(ptcdata.index.data);
            logger->info("writing rtc diagnostics sidecar chunk");
            rtcproc.append_diag_to_netcdf(ptcdata, output_paths.rtcdiag_filename, calib_scan, ptcdata.index.data);
            output_writers.rtcdiag->advance();
        }

        // write rtc timestreams
        if (write_this_rtc) {
            output_writers.rtc->wait_turn(rtc_scan_row);
            if (rtcproc.tod_output_outer) {
                logger->info("writing outer raw time chunk");
                rtcproc.append_to_netcdf(rtc_outer_output, output_paths.tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         rtc_outer_output.pointing_offsets_arcsec.data, calib, false, rtc_scan_row);
            }
            else {
                logger->info("writing raw time chunk");
                rtcproc.append_to_netcdf(ptcdata, output_paths.tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         ptcdata.pointing_offsets_arcsec.data, calib, false, rtc_scan_row);
            }
            output_writers.rtc->advance();
        }
        if (output_flags.write_rtc || output_flags.write_rtcdiag) {
            rtcproc.clear_cached_diagnostics(ptcdata.index.data);
        }

        apply_learned_ptc_sample_masks(ptcdata, calib_scan);
        apply_learned_ptc_detector_exclusions(ptcdata, calib_scan);

        const auto fruit_weight_policy =
            citlali::pipeline::fruit_loop_weight_policy(ptcproc);

        // if running fruit loops and a map has been read in
        if (fruit_weight_policy.use_noise_weights) {
            logger->info("subtracting map from tod");
            // subtract map
            ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(ptcproc.tod_mb, ptcdata, calib_scan,
                                                                            map_indices, telescope.pixel_axes,
                                                                            map_grouping);
        }

        ptcproc.accumulate_weight_validation_atmosphere(ptcdata, calib_scan.apt);

        {
            std::lock_guard<std::mutex> lock(*ptc_line_audit_mutex);
            apply_model_protected_ptc_line_audit(
                ptcdata, calib_scan, fruit_weight_policy.use_noise_weights);
        }

        // run cleaning
        logger->info("processed time chunk processing for scan {}", ptcdata.index.data + 1);
        ptcproc.run(ptcdata, ptcdata, calib_scan, telescope.pixel_axes, map_grouping);
        const auto ptc_second_pass_summary =
            ptcproc.snapshot_second_pass_summary(ptcdata.index.data);

        // if running fruit loops and a map has been read in
        if (fruit_weight_policy.use_noise_weights) {
            // calculate weights
            logger->info("calculating weights for scan {} (fruit loops noise-only pass)",
                         ptcdata.index.data + 1);
            ptcproc.calc_weights(ptcdata, calib_scan.apt, telescope, true);

            // reset weights to median
            calib_scan = ptcproc.reset_weights(ptcdata, calib_scan, map_grouping);

            if (make_maps && make_noise_maps) {
                // populate noise maps only
                bool run_omb = false;
                logger->info("populating noise maps");
                citlali::pipeline::populate_naive_or_jinc_maps(
                    mapmaking_method, naive_mm, jinc_mm, ptcdata, omb, cmb,
                    map_indices, telescope.pixel_axes, calib_scan.apt,
                    telescope.d_fsmp, run_omb, make_noise_maps);
            }
            logger->info("adding map to tod");
            // add map back
            ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(ptcproc.tod_mb, ptcdata, calib_scan,
                                                                    map_indices, telescope.pixel_axes,
                                                                    map_grouping);
        }

        // remove outliers after cleaning
        calib_scan = ptcproc.remove_bad_dets(ptcdata, calib_scan, map_grouping);

        if (fruit_weight_policy.keep_source_subtracted_weights) {
            logger->info("keeping source-subtracted weights for scan {}", ptcdata.index.data + 1);
        }
        else {
            // calculate weights
            if (fruit_weight_policy.use_noise_weights) {
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
        if (output_flags.write_ptcdiag) {
            output_writers.ptcdiag->wait_turn(ptcdata.index.data);
            logger->info("writing ptc diagnostics sidecar chunk");
            ptcproc.append_diag_to_netcdf(ptcdata, output_paths.ptcdiag_filename, calib_scan, ptcdata.index.data);
            output_writers.ptcdiag->advance();
        }

        const auto ptc_scan_row = tod_output_scan_row(
            ptcdata.index.data, citlali::config::TodOutputStream::ptc);
        if (output_flags.write_ptc && ptc_scan_row >= 0) {
            output_writers.ptc->wait_turn(ptc_scan_row);
            logger->info("writing processed time chunk");
            ptcproc.append_to_netcdf(ptcdata, output_paths.tod_filename["ptc"], map_grouping, telescope.pixel_axes,
                                     ptcdata.pointing_offsets_arcsec.data, calib_scan, false, ptc_scan_row);
            output_writers.ptc->advance();
        }
        if (output_flags.write_ptc || output_flags.write_ptcdiag) {
            ptcproc.clear_cached_diagnostics(ptcdata.index.data);
        }

        // write out chunk summary
        if (citlali::pipeline::verbose_runtime_enabled(*this)) {
            write_chunk_summary(ptcdata);
        }

        // write stats
        logger->debug("calculating stats");
        diagnostics.calc_stats(ptcdata);

        // populate maps
        if (make_maps) {
            // make signal, weight, kernel, and coverage maps
            bool run_omb = true;
            const bool run_noise_fruit =
                citlali::pipeline::should_populate_final_noise_maps(
                    make_noise_maps, ptcproc.run_fruit_loops,
                    !ptcproc.tod_mb.signal.empty());

            apply_learned_mapmaking_detector_exclusions(ptcdata, calib_scan);
            // populate maps with current time chunk
            logger->info("populating maps");
            citlali::pipeline::populate_lali_maps(
                mapmaking_method, naive_mm, jinc_mm, ml_mm, ptcdata, omb,
                cmb, map_indices, telescope.pixel_axes, calib_scan,
                telescope.d_fsmp, run_omb, run_noise_fruit);
        }

        // increment number of completed scans
        citlali::pipeline::log_scan_done(
            scans_done_mutex, logger, ptcdata.index.data, *scans_done_count,
            telescope);
    }};
    auto farm = grppi::farm(
        citlali::pipeline::runtime_thread_count(*this), std::move(farm_fn));

    return farm;
}
