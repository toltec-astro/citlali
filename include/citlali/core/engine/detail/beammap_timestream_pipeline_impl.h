#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/timestream_output_context.h>
#include <citlali/core/pipeline/map_grouping_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/runtime_policy.h>
#include <citlali/core/pipeline/timestream_run_context.h>
#include <citlali/core/pipeline/timestream_scan_context.h>
#include <citlali/core/pipeline/timestream_scan_generation.h>

template <class KidsProc, class RawObs>
void Beammap::timestream_pipeline(KidsProc &kidsproc, RawObs &rawobs, bool write_outputs) {
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
    const auto output_flags =
        citlali::pipeline::beammap_timestream_output_flags(
            *this, write_outputs);
    const auto output_writers =
        citlali::pipeline::make_timestream_output_writers(output_flags);
    const auto output_expectations =
        citlali::pipeline::beammap_timestream_output_expectations(
            *this, output_flags);
    // progress bar
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "RTC progress ");

    // grppi generator function. gets time chunk data from files sequentially and passes them to grppi::farm
    grppi::pipeline(
        tula::grppi_utils::dyn_ex(
            citlali::pipeline::runtime_parallel_policy_name(*this)),
        [&]() -> std::optional<input_t> {

            // variable to hold current scan
            static int scan = 0;
            // loop through scans
            while (scan < telescope.scan_indices.cols()) {
                // update progress bar
                pb.count(telescope.scan_indices.cols(), 1);

                TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
                const Eigen::Index scan_length =
                    citlali::pipeline::initialize_rtc_scan(
                        rtcdata, telescope, scan);

                citlali::pipeline::populate_rtc_scan_samples(
                    rtcdata, kidsproc, rawobs, scan, telescope, alignment.start_indices,
                    alignment.end_indices, alignment.common_time, alignment.network_times, alignment.masks,
                    citlali::config::timing_gap_interpolation_active(
                        citlali::pipeline::runtime_config(*this)),
                    scan_length, calib.n_dets,
                    citlali::pipeline::timestream_config(*this).type);

                // increment scan
                scan++;
                // return rtcdata
                return rtcdata;
            }
            // reset scan to zero for each obs
            scan = 0;
            return {};
        },
        // run the raw time chunk processing
        run_timestream(kidsproc, output_flags, output_writers));

    output_writers.rethrow_if_failed();
    output_writers.verify_complete(output_expectations);
}

template <class KidsProc>
auto Beammap::run_timestream(
    KidsProc &kidsproc,
    const citlali::pipeline::TimestreamOutputFlags &output_flags,
    const citlali::pipeline::TimestreamOutputWriters &output_writers) {
    auto scans_done_mutex = std::make_shared<std::mutex>();
    auto scans_done_count = std::make_shared<int>(0);

    auto map_grouping_ptr = std::make_shared<std::string>(
        citlali::pipeline::active_map_grouping_name(*this));

    auto farm = grppi::farm(
        citlali::pipeline::runtime_thread_count(*this),
        [&, scans_done_mutex, scans_done_count, output_writers, output_flags,
         map_grouping_ptr](auto &rtcdata)
                       -> TCData<TCDataKind::PTC, Eigen::MatrixXd> {
        auto &map_grouping = *map_grouping_ptr;

        // allocate up bitwise timestream flags
        rtcdata.flags2.data.setConstant(timestream::TimestreamFlags::Good);

        const auto scan_window = citlali::pipeline::copy_rtc_scan_context(
            rtcdata, telescope, pointing_offsets.arcsec);
        citlali::pipeline::copy_hwpr_angle_if_enabled(
            rtcdata, calib, rtcproc.run_polarization, true,
            alignment.hwpr_start_index, scan_window.start, scan_window.length);
        citlali::pipeline::initialize_rtc_flags(rtcdata);
        if (citlali::config::timing_gap_interpolation_active(
                citlali::pipeline::runtime_config(*this))) {
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
        auto map_indices = rtcproc.run(rtcdata, ptcdata, calib, telescope, omb.pixel_size_rad, map_grouping,
                                       rtc_outer_output_ptr);

        if (citlali::pipeline::mapmaking_config(*this).grouping !=
            citlali::config::MapGrouping::detector) {
            // remove flagged detectors
            rtcproc.remove_flagged_dets(ptcdata, calib.apt);
        }

        // remove outliers before cleaning
        auto calib_scan = rtcproc.remove_bad_dets(ptcdata, calib, map_grouping);

        // remove duplicate tones
        if (!telescope.sim_obs) {
            calib_scan = rtcproc.remove_nearby_tones(ptcdata, calib_scan, map_grouping);
        }

        if (output_flags.write_rtcdiag) {
            if (!output_writers.write_when_ready(
                output_writers.rtcdiag, ptcdata.index.data, [&] {
                    logger->info("writing rtc diagnostics sidecar chunk");
                    rtcproc.append_diag_to_netcdf(
                        ptcdata, output_paths.rtcdiag_filename, calib_scan,
                        ptcdata.index.data);
                })) {
                return {};
            }
        }

        // write rtc timestreams
        if (write_this_rtc) {
            if (!output_writers.write_when_ready(
                output_writers.rtc, rtc_scan_row, [&] {
                    if (rtcproc.tod_output_outer) {
                        logger->info("writing outer raw time chunk");
                        rtcproc.append_to_netcdf(
                            rtc_outer_output, output_paths.tod_filename["rtc"],
                            map_grouping, telescope.pixel_axes,
                            rtc_outer_output.pointing_offsets_arcsec.data, calib,
                            true, rtc_scan_row);
                    }
                    else {
                        logger->info("writing raw time chunk");
                        rtcproc.append_to_netcdf(
                            ptcdata, output_paths.tod_filename["rtc"],
                            map_grouping, telescope.pixel_axes,
                            ptcdata.pointing_offsets_arcsec.data, calib_scan,
                            true, rtc_scan_row);
                    }
                })) {
                return {};
            }
        }
        rtcproc.clear_cached_diagnostics(ptcdata.index.data);

        // store indices for each ptcdata
        ptcdata.map_indices.data = std::move(map_indices);

        // move out ptcdata the PTCData vector at corresponding index
        ptcs0.at(ptcdata.index.data) = std::move(ptcdata);
        calib_scans0.at(ptcdata.index.data) = std::move(calib_scan);

        // increment number of completed scans
        citlali::pipeline::log_scan_done(
            scans_done_mutex, logger, ptcdata.index.data, *scans_done_count,
            telescope);

        return ptcdata;
    });

    return farm;
}
