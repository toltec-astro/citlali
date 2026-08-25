#pragma once

// Implementation detail included by lali.h.

#include <citlali/core/pipeline/mapmaking_dispatch.h>
#include <citlali/core/pipeline/map_group_indexing.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/jinc_processing_provenance.h>
#include <citlali/core/pipeline/native_consumer_execution.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_run_context.h>
#include <citlali/core/pipeline/timestream_scan_context.h>

auto Lali::run(
    const citlali::pipeline::TimestreamOutputFlags &output_flags,
    const citlali::pipeline::TimestreamOutputWriters &output_writers)
    -> run_stage_t {
    auto scans_done_mutex = std::make_shared<std::mutex>();
    auto scans_done_count = std::make_shared<int>(0);
    auto ptc_line_audit_mutex = std::make_shared<std::mutex>();
    const auto mapmaking_method =
        citlali::pipeline::mapmaking_config(*this).method;
    const bool make_maps = citlali::pipeline::mapmaking_enabled(*this);
    const bool make_noise_maps = citlali::pipeline::noise_maps_enabled(*this);
    auto jinc_merge_order =
        make_maps && citlali::config::is_jinc_map_method(mapmaking_method)
            ? std::make_shared<citlali::pipeline::OrderedWriter>()
            : nullptr;
    auto jinc_noise_merge_order =
        jinc_merge_order != nullptr
            ? std::make_shared<citlali::pipeline::OrderedWriter>()
            : nullptr;

    auto map_grouping_ptr = std::make_shared<std::string>(
        citlali::pipeline::active_map_grouping_name(*this));

    auto farm_fn = std::function<void(input_t &)>{[&, scans_done_mutex,
                                                   scans_done_count,
                                                   ptc_line_audit_mutex,
                                                   output_writers,
                                                   mapmaking_method, make_maps,
                                                   make_noise_maps,
                                                   output_flags,
                                                   map_grouping_ptr,
                                                   jinc_merge_order,
                                                   jinc_noise_merge_order](input_t &rtcdata) {
        auto &map_grouping = *map_grouping_ptr;
        try {
        if (rtcdata.native_runtime) {
            citlali::pipeline::log_scan_start(
                scans_done_mutex, logger, rtcdata.index.data,
                *scans_done_count, telescope);
            auto map_indices =
                citlali::pipeline::detector_map_indices_for_grouping(
                    citlali::pipeline::mapmaking_config(*this).grouping,
                    calib);
            auto native =
                citlali::pipeline::prepare_native_consumer_map_scan(
                    *this, rtcdata, map_indices);
            auto &raw_plan =
                citlali::pipeline::raw_timestream_plan(*this);
            if (!raw_plan.observation ||
                !raw_plan.observation->native_cohort_lineage) {
                throw std::logic_error(
                    "native Lali scan lacks observation-owned lineage");
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
                    *native.runtime->ptc_preclean_flags,
                    *native.runtime->ptc_flags,
                    native.ptcdata.flags.data,
                    std::move(map_publication));
            auto reservation =
                raw_plan.observation->native_cohort_lineage->reserve(
                    std::move(record));
            if (!native.runtime->map_projection) {
                throw std::logic_error(
                    "native Lali scan lacks its final map projection");
            }
            populate_lali_final_maps(
                native.ptcdata, calib, native.map_indices, map_grouping,
                mapmaking_method, make_maps, make_noise_maps,
                &*native.runtime->map_projection, jinc_merge_order.get());
            citlali::pipeline::publish_native_jinc_processing_trace_if_active(
                *this, rtcdata.index.data,
                *native.runtime->jinc_processing_trace);
            reservation.commit();
            citlali::pipeline::log_scan_done(
                scans_done_mutex, logger, rtcdata.index.data,
                *scans_done_count, telescope);
            return;
        }
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

        write_lali_rtc_outputs(
            rtcdata, ptcdata, rtc_outer_output, calib_scan, output_flags,
            output_writers, rtc_scan_row, write_this_rtc, map_grouping);
        if (output_writers.failed()) {
            if (jinc_merge_order != nullptr) {
                const auto error = std::make_exception_ptr(
                    std::runtime_error(
                        "ordered JINC accumulation cancelled after required output failure"));
                jinc_merge_order->cancel(error);
                jinc_noise_merge_order->cancel(error);
            }
            return;
        }

        apply_learned_ptc_sample_masks(ptcdata, calib_scan);
        apply_learned_ptc_detector_exclusions(ptcdata, calib_scan);

        const auto fruit_weight_policy =
            citlali::pipeline::fruit_loop_weight_policy(*this);

        maybe_subtract_lali_fruitloop_model(
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
        const auto ptc_second_pass_summary =
            ptcproc.snapshot_second_pass_summary(ptcdata.index.data);

        run_lali_fruitloop_noise_pass(
            ptcdata, calib_scan, map_indices, map_grouping,
            mapmaking_method, make_maps, make_noise_maps,
            fruit_weight_policy, jinc_noise_merge_order.get());

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

        citlali::pipeline::record_jinc_ptc_scan_state_if_available(
            *this, ptcdata, calib_scan.apt, map_indices);

        write_lali_ptc_outputs(
            ptcdata, calib_scan, output_flags, output_writers, map_grouping);
        if (output_writers.failed()) {
            if (jinc_merge_order != nullptr) {
                const auto error = std::make_exception_ptr(
                    std::runtime_error(
                        "ordered JINC accumulation cancelled after required output failure"));
                jinc_merge_order->cancel(error);
                jinc_noise_merge_order->cancel(error);
            }
            return;
        }

        // write out chunk summary
        if (citlali::pipeline::verbose_runtime_enabled(*this)) {
            write_chunk_summary(ptcdata);
        }

        // write stats
        logger->debug("calculating stats");
        diagnostics.calc_stats(ptcdata);

        populate_lali_final_maps(
            ptcdata, calib_scan, map_indices, map_grouping,
            mapmaking_method, make_maps, make_noise_maps, nullptr,
            jinc_merge_order.get());

        // increment number of completed scans
        citlali::pipeline::log_scan_done(
            scans_done_mutex, logger, ptcdata.index.data, *scans_done_count,
            telescope);
        }
        catch (...) {
            if (jinc_merge_order != nullptr) {
                jinc_merge_order->cancel(std::current_exception());
                jinc_noise_merge_order->cancel(std::current_exception());
            }
            throw;
        }
    }};
    auto farm = grppi::farm(
        citlali::pipeline::runtime_thread_count(*this), std::move(farm_fn));

    return farm;
}
