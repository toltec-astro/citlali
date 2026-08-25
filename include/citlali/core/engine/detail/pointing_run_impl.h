#pragma once

// Implementation detail included by pointing.h.

#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/jinc_processing_provenance.h>
#include <citlali/core/pipeline/native_consumer_execution.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_run_context.h>
#include <citlali/core/pipeline/timestream_scan_context.h>

template <class KidsProc>
auto Pointing::run(
    KidsProc &kidsproc,
    const citlali::pipeline::TimestreamOutputFlags &output_flags,
    const citlali::pipeline::TimestreamOutputWriters &output_writers,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    ptcproc.fruit_loops_diagnostic_iteration = iteration.fruit_iter;
    auto scans_done_mutex = std::make_shared<std::mutex>();
    auto scans_done_count = std::make_shared<int>(0);
    auto ptc_line_audit_mutex = std::make_shared<std::mutex>();
    const auto mapmaking_method =
        citlali::pipeline::mapmaking_config(*this).method;
    const bool make_maps = citlali::pipeline::mapmaking_enabled(*this);
    const bool make_noise_maps = citlali::pipeline::noise_maps_enabled(*this);

    auto map_grouping_ptr = std::make_shared<std::string>(
        citlali::pipeline::active_map_grouping_name(*this));

    auto farm = grppi::farm(
        citlali::pipeline::runtime_thread_count(*this),
        [&, scans_done_mutex, scans_done_count, ptc_line_audit_mutex,
         output_writers, mapmaking_method, make_maps, make_noise_maps,
         output_flags, map_grouping_ptr](
            auto &rtcdata) {
        auto &map_grouping = *map_grouping_ptr;

        citlali::pipeline::prepare_standard_rtc_scan_context(*this, rtcdata);

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;
        TCData<TCDataKind::RTC,Eigen::MatrixXd> rtc_outer_output;
        const auto rtc_scan_row = tod_output_scan_row(
            rtcdata.index.data, citlali::config::TodOutputStream::rtc);
        const bool write_this_rtc =
            output_flags.write_rtc && rtc_scan_row >= 0;
        auto *rtc_outer_output_ptr =
            (write_this_rtc &&
             citlali::pipeline::raw_tod_outer_output(*this))
                ? &rtc_outer_output
                : nullptr;

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

        citlali::pipeline::record_jinc_rtc_scan_state_if_available(
            *this, ptcdata, calib_scan.apt, map_indices);

        write_pointing_rtc_outputs(
            rtcdata, ptcdata, rtc_outer_output, calib_scan, output_flags,
            output_writers, stage_profile, rtc_scan_row, write_this_rtc,
            map_grouping);
        if (output_writers.failed()) {
            return;
        }

        apply_learned_ptc_sample_masks(ptcdata, calib_scan);
        apply_learned_ptc_detector_exclusions(ptcdata, calib_scan);

        const auto fruit_weight_policy =
            citlali::pipeline::fruit_loop_weight_policy(*this);

        maybe_inject_pointing_fruitloop_test_source(
            ptcdata, calib_scan);
        maybe_subtract_pointing_fruitloop_model(
            ptcdata, calib_scan, map_indices, map_grouping,
            fruit_weight_policy);

        ptcproc.accumulate_weight_validation_atmosphere(ptcdata, calib_scan.apt);

        {
            std::lock_guard<std::mutex> lock(*ptc_line_audit_mutex);
            apply_model_protected_ptc_line_audit(
                ptcdata, calib_scan, fruit_weight_policy.use_noise_weights);
        }

        // run cleaning
        logger->info("processed time chunk processing for scan {}", ptcdata.index.data + 1);
        ptcproc.run(ptcdata, ptcdata, calib_scan, telescope.pixel_axes, map_grouping);
        citlali::pipeline::log_fruit_loop_tod_stage(
            logger, ptcproc.fruit_loops_diagnostics_enabled &&
                        fruit_weight_policy.use_noise_weights,
            "after_cleaning", iteration.fruit_iter, ptcdata, calib_scan);
        timestream::log_kernel_matrix_diag(
            logger, "ptc after processed time chunk cleaning", ptcdata.kernel.data, ptcdata.index.data);
        const auto ptc_second_pass_summary =
            ptcproc.snapshot_second_pass_summary(ptcdata.index.data);

        run_pointing_fruitloop_noise_pass(
            ptcdata, calib_scan, map_indices, map_grouping,
            mapmaking_method, make_maps, make_noise_maps,
            fruit_weight_policy);

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
        citlali::pipeline::log_fruit_loop_detector_weights(
            logger, ptcproc.fruit_loops_diagnostics_enabled &&
                        fruit_weight_policy.use_noise_weights,
            "final", iteration.fruit_iter, ptcdata, calib_scan);

        const auto ptc_high_weight_summary =
            ptcproc.snapshot_high_weight_summary(ptcdata.index.data);
        collect_ptc_learning_diagnostics(
            ptcdata, calib_scan, ptc_second_pass_summary, ptc_high_weight_summary);

        citlali::pipeline::record_jinc_ptc_scan_state_if_available(
            *this, ptcdata, calib_scan.apt, map_indices);

        write_pointing_ptc_outputs(
            ptcdata, calib_scan, output_flags, output_writers, stage_profile,
            map_grouping);
        if (output_writers.failed()) {
            return;
        }

        // write out chunk summary
        if (citlali::pipeline::verbose_runtime_enabled(*this)) {
            write_chunk_summary(ptcdata);
        }

        // calc stats
        logger->debug("calculating stats");
        diagnostics.calc_stats(ptcdata);

        if (rtcdata.native_runtime) {
            auto native =
                citlali::pipeline::prepare_native_consumer_map_scan(
                    *this, rtcdata, map_indices);
            auto &raw_plan =
                citlali::pipeline::raw_timestream_plan(*this);
            if (!raw_plan.observation ||
                !raw_plan.observation->native_cohort_lineage) {
                throw std::logic_error(
                    "native Pointing scan lacks observation-owned lineage");
            }
            auto map_publication =
                citlali::pipeline::make_native_map_publication_request_v3(
                    *this, mapmaking_method, make_maps, *native.runtime,
                    native.ptcdata.weights.data);
            auto record =
                citlali::pipeline::make_native_cohort_scan_provenance_v3(
                    raw_plan.observation->native_cohort_lineage->binding(),
                    native.runtime->ledger(), *native.runtime->rtc,
                    *native.runtime->ptc_prepared,
                    *native.runtime->science_projection,
                    *native.runtime->ptc_flags,
                    native.ptcdata.flags.data,
                    std::move(map_publication));
            auto reservation =
                raw_plan.observation->native_cohort_lineage->reserve(
                    std::move(record));
            populate_pointing_final_maps(
                native.ptcdata, calib, native.map_indices, map_grouping,
                mapmaking_method, make_maps, make_noise_maps,
                &*native.runtime->science_projection);
            citlali::pipeline::publish_native_jinc_processing_trace_if_active(
                *this, ptcdata.index.data,
                *native.runtime->jinc_processing_trace);
            reservation.commit();
        }
        else {
            populate_pointing_final_maps(
                ptcdata, calib_scan, map_indices, map_grouping,
                mapmaking_method, make_maps, make_noise_maps);
        }
        // increment number of completed scans
        citlali::pipeline::log_scan_done(
            scans_done_mutex, logger, ptcdata.index.data, *scans_done_count,
            telescope);

    });

    return farm;
}
