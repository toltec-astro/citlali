#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/ordered_writer.h>

template <class KidsProc, class RawObs>
void Beammap::timestream_pipeline(KidsProc &kidsproc, RawObs &rawobs, bool write_outputs) {
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
    // initialize number of completed scans
    n_scans_done = 0;

    // progress bar
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "RTC progress ");

    // grppi generator function. gets time chunk data from files sequentially and passes them to grppi::farm
    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<input_t> {

            // variable to hold current scan
            static int scan = 0;
            // loop through scans
            while (scan < telescope.scan_indices.cols()) {
                // update progress bar
                pb.count(telescope.scan_indices.cols(), 1);

                // create rtcdata
                TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
                // get scan indices
                rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
                // current scan
                rtcdata.index.data = scan;

                // current length of outer scans
                Eigen::Index sl = rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

                // get raw tod from files
                if (!interp_over_gaps) {
                    rtcdata.scans.data = kidsproc.populate_rtc_from_rawobs(rawobs, scan, telescope.scan_indices,
                                                                           start_indices, end_indices,
                                                                           sl, calib.n_dets, tod_type);
                }
                else {
                    auto scan_rawobs = kidsproc.load_rawobs_gaps(rawobs, scan, telescope.scan_indices, start_indices,
                                                                 t_common, nw_times, 1 / (2 * telescope.fsmp));
                    rtcdata.scans.data = kidsproc.populate_rtc_gaps(scan_rawobs, t_common, nw_times, masks, scan, 1 / (2 * telescope.fsmp),
                                                                telescope.scan_indices, sl, calib.n_dets, tod_type);
                    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>().swap(scan_rawobs);
                }

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
        run_timestream(kidsproc, write_outputs));
}

template <class KidsProc>
auto Beammap::run_timestream(KidsProc &kidsproc, bool write_outputs) {
    auto scans_done_mutex = std::make_shared<std::mutex>();

    const bool write_rtc =
        write_outputs && run_tod_output && run_tod_output_rtc &&
        !tod_filename.empty();
    const bool write_rtcdiag = write_outputs && !rtcdiag_filename.empty();
    auto rtc_writer =
        write_rtc ? std::make_shared<citlali::pipeline::OrderedWriter>()
                  : nullptr;
    auto rtcdiag_writer =
        write_rtcdiag ? std::make_shared<citlali::pipeline::OrderedWriter>()
                      : nullptr;

    auto farm = grppi::farm(n_threads,[&, scans_done_mutex, rtc_writer, rtcdiag_writer,
                                       write_rtc, write_rtcdiag](auto &rtcdata) -> TCData<TCDataKind::PTC,Eigen::MatrixXd> {

        // allocate up bitwise timestream flags
        rtcdata.flags2.data.setConstant(timestream::TimestreamFlags::Good);

        // starting index for scan (outer scan)
        Eigen::Index si = rtcdata.scan_indices.data(2);

        // current length of outer scans
        Eigen::Index sl = rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

        // copy scan's telescope vectors
        for (const auto& x: telescope.tel_data) {
            rtcdata.tel_data.data[x.first] = telescope.tel_data[x.first].segment(si,sl);
        }

        // copy pointing offsets
        for (const auto& [axis,offset]: pointing_offsets_arcsec) {
            rtcdata.pointing_offsets_arcsec.data[axis] = offset.segment(si,sl);
        }

        // get hwpr
        if (rtcproc.run_polarization) {
            rtcdata.hwpr_angle.data = calib.hwpr_angle.segment(si + hwpr_start_indices, sl);
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
        auto map_indices = rtcproc.run(rtcdata, ptcdata, calib, telescope, omb.pixel_size_rad, map_grouping,
                                       rtc_outer_output_ptr);

        if (typed_config.mapmaking.grouping !=
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
                                         rtc_outer_output.pointing_offsets_arcsec.data, calib, true, rtc_scan_row);
            }
            else {
                logger->info("writing raw time chunk");
                rtcproc.append_to_netcdf(ptcdata, tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         ptcdata.pointing_offsets_arcsec.data, calib_scan, true, rtc_scan_row);
            }
            rtc_writer->advance();
        }
        rtcproc.clear_cached_diagnostics(ptcdata.index.data);

        // store indices for each ptcdata
        ptcdata.map_indices.data = std::move(map_indices);

        // move out ptcdata the PTCData vector at corresponding index
        ptcs0.at(ptcdata.index.data) = std::move(ptcdata);
        calib_scans0.at(ptcdata.index.data) = std::move(calib_scan);

        // increment number of completed scans
        {
            std::lock_guard<std::mutex> lk(*scans_done_mutex);
            n_scans_done++;
            logger->info("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done,
                         telescope.scan_indices.cols());
        }

        return ptcdata;
    });

    return farm;
}
