#pragma once

#include <functional>
#include <mutex>
#include <condition_variable>

#include <citlali/core/engine/engine.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Lali: public Engine {
public:
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
    using run_stage_t = decltype(grppi::farm(
        std::declval<int>(), std::declval<std::function<void(input_t &)>>() ));

    // initial setup for each obs
    void setup();

    // main grppi pipeline
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // run the reduction for the obs
    auto run() -> run_stage_t;

    // output files
    template <mapmaking::MapType map_type>
    void output();
};

void Lali::setup() {
    // run obsnum setup
    obsnum_setup();
}

template <class KidsProc, class RawObs>
void Lali::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    using tuple_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;

    // initialize number of completed scans
    n_scans_done = 0;

    // declare random number generator
    boost::random::mt19937 eng;

    // boost random number generator (0,1)
    boost::random::uniform_int_distribution<> rands{0,1};

    // progress bar
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "citlali progress ");

    // grppi generator function. gets time chunk data from files sequentially and passes them to grppi::farm
    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<tuple_t> {
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

                // populate noise matrix (do outside of parallelized region for thread safety)
                if (run_noise) {
                    if (omb.randomize_dets) {
                        // n_noise x n_dets
                        rtcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(omb.n_noise, calib.n_dets)
                                    .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                    } else {
                        // n_noise
                        rtcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(omb.n_noise)
                                    .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                    }
                }
                // current length of outer scans
                Eigen::Index sl = rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

                // get raw tod from files
                if (!interp_over_gaps) {
                    rtcdata.scans.data = kidsproc.populate_rtc_from_rawobs(rawobs, scan, telescope.scan_indices,
                                                                           start_indices, end_indices,
                                                                           sl, calib.n_dets, tod_type);
                }
                else {
                    // vector to store kids data
                    auto scan_rawobs = kidsproc.load_rawobs_gaps(rawobs, scan, telescope.scan_indices, start_indices,
                                                                 t_common, nw_times, 1 / (2 * telescope.fsmp));
                    rtcdata.scans.data = kidsproc.populate_rtc_gaps(scan_rawobs, t_common, nw_times, masks, scan, 1 / (2 * telescope.fsmp),
                                                                telescope.scan_indices, sl, calib.n_dets, tod_type);
                    // try and clear input vector
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

        // run the farm
        run());

    if (run_mapmaking) {
        // normalize maps
        logger->info("normalizing maps");
        if (map_method != "maximum_likelihood") {
            if (rtcproc.run_polarization) {
                omb.normalize_polarized_maps();
            }
            else {
                omb.normalize_maps();
            }
        }
        // calculate map psds
        logger->info("calculating map psd");
        omb.calc_map_psd();
        // calculate map histograms
        logger->info("calculating map histogram");
        omb.calc_map_hist();
        // calculate mean error
        omb.calc_median_err();
        // calculate mean rms
        omb.calc_median_rms();

        // write map summary
        if (verbose_mode) {
            write_map_summary(omb);
        }
    }
}

auto Lali::run() -> run_stage_t {
    auto scans_done_mutex = std::make_shared<std::mutex>();
    auto ptc_line_audit_mutex = std::make_shared<std::mutex>();

    struct OrderedWriter {
        std::mutex mutex;
        std::condition_variable cv;
        Eigen::Index next = 0;
        void wait_turn(Eigen::Index idx) {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [&] { return idx == next; });
        }
        void advance() {
            std::lock_guard<std::mutex> lk(mutex);
            ++next;
            cv.notify_all();
        }
    };

    const bool write_rtc = run_tod_output && !tod_filename.empty() &&
        (tod_output_type == "rtc" || tod_output_type == "both");
    const bool write_ptc = run_tod_output && !tod_filename.empty() &&
        (tod_output_type == "ptc" || tod_output_type == "both");
    const bool write_rtcdiag = !rtcdiag_filename.empty();
    const bool write_ptcdiag = !ptcdiag_filename.empty();

    auto rtc_writer = write_rtc ? std::make_shared<OrderedWriter>() : nullptr;
    auto ptc_writer = write_ptc ? std::make_shared<OrderedWriter>() : nullptr;
    auto rtcdiag_writer = write_rtcdiag ? std::make_shared<OrderedWriter>() : nullptr;
    auto ptcdiag_writer = write_ptcdiag ? std::make_shared<OrderedWriter>() : nullptr;

    auto farm_fn = std::function<void(input_t &)>{[&, scans_done_mutex, ptc_line_audit_mutex,
                                                   rtc_writer, ptc_writer, rtcdiag_writer, ptcdiag_writer,
                                                   write_rtc, write_ptc, write_rtcdiag, write_ptcdiag](input_t &rtcdata) {
        // starting index for scan
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
        auto map_indices = rtcproc.run(rtcdata, ptcdata, calib, telescope, omb.pixel_size_rad, map_grouping,
                                       rtc_outer_output_ptr);

        // remove flagged detectors
        rtcproc.remove_flagged_dets(ptcdata, calib.apt);

        // remove outliers before cleaning
        auto calib_scan = rtcproc.remove_bad_dets(ptcdata, calib, map_grouping);

        // remove duplicate tones
        if (!telescope.sim_obs) {
            calib_scan = rtcproc.remove_nearby_tones(ptcdata, calib_scan, map_grouping);
        }

        collect_rtc_learning_diagnostics(rtcdata, ptcdata, calib_scan);

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

        const bool use_fruit_noise_weights =
            ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty();
        const bool keep_source_subtracted_weights =
            use_fruit_noise_weights && !ptcproc.fruit_loops_recompute_weights_after_addback;

        // if running fruit loops and a map has been read in
        if (use_fruit_noise_weights) {
            logger->info("subtracting map from tod");
            // subtract map
            ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(ptcproc.tod_mb, ptcdata, calib_scan,
                                                                            map_indices, telescope.pixel_axes,
                                                                            map_grouping);
        }

        ptcproc.accumulate_weight_validation_atmosphere(ptcdata, calib_scan.apt);

        {
            std::lock_guard<std::mutex> lock(*ptc_line_audit_mutex);
            apply_model_protected_ptc_line_audit(ptcdata, calib_scan, use_fruit_noise_weights);
        }

        // run cleaning
        logger->info("processed time chunk processing for scan {}", ptcdata.index.data + 1);
        ptcproc.run(ptcdata, ptcdata, calib_scan, telescope.pixel_axes, map_grouping);
        collect_ptc_learning_diagnostics(ptcdata, calib_scan);

        // if running fruit loops and a map has been read in
        if (use_fruit_noise_weights) {
            // calculate weights
            logger->info("calculating weights for scan {} (fruit loops noise-only pass)",
                         ptcdata.index.data + 1);
            ptcproc.calc_weights(ptcdata, calib_scan.apt, telescope, true);

            // reset weights to median
            calib_scan = ptcproc.reset_weights(ptcdata, calib_scan, map_grouping);

            if (run_mapmaking && run_noise) {
                // populate noise maps only
                bool run_omb = false;
                logger->info("populating noise maps");
                if (map_method=="naive") {
                    naive_mm.populate_maps_naive(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                                 calib_scan.apt, telescope.d_fsmp, run_omb, run_noise);
                }
                else if (map_method=="jinc") {
                    jinc_mm.populate_maps_jinc(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                               calib_scan.apt, telescope.d_fsmp, run_omb, run_noise);
                }
            }
            logger->info("adding map to tod");
            // add map back
            ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(ptcproc.tod_mb, ptcdata, calib_scan,
                                                                    map_indices, telescope.pixel_axes,
                                                                    map_grouping);
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

        // write stats
        logger->debug("calculating stats");
        diagnostics.calc_stats(ptcdata);

        // populate maps
        if (run_mapmaking) {
            // make signal, weight, kernel, and coverage maps
            bool run_omb = true;
            bool run_noise_fruit = run_noise;

            // if running fruit loops, noise maps are made on source
            // subtracted timestreams so don't make them here
            if (ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty()) {
                run_noise_fruit = false;
            }

            // populate maps with current time chunk
            logger->info("populating maps");
            if (map_method=="naive") {
                naive_mm.populate_maps_naive(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                             calib_scan.apt, telescope.d_fsmp, run_omb, run_noise_fruit);
            }
            else if (map_method=="jinc") {
                jinc_mm.populate_maps_jinc(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                           calib_scan.apt, telescope.d_fsmp, run_omb, run_noise_fruit);
            }
            else if (map_method=="maximum_likelihood") {
                ml_mm.populate_maps_ml(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                       calib_scan, telescope.d_fsmp, run_omb, run_noise_fruit);
            }
        }

        // increment number of completed scans
        {
            std::lock_guard<std::mutex> lk(*scans_done_mutex);
            n_scans_done++;
            logger->info("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done,
                         telescope.scan_indices.cols());
        }
    }};
    auto farm = grppi::farm(n_threads, std::move(farm_fn));

    return farm;
}

template <mapmaking::MapType map_type>
void Lali::output() {
    // pointer to map buffer
    mapmaking::MapBuffer* mb = nullptr;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;

    // directory name
    std::string dir_name;

    // set common variables depending on map_type
    if constexpr (map_type == mapmaking::RawObs || map_type == mapmaking::FilteredObs) {
        mb = &omb;
        dir_name = obsnum_dir_name + (map_type == mapmaking::RawObs ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawObs) ? &fits_io_vec : &filtered_fits_io_vec;
        n_io = (map_type == mapmaking::RawObs) ? &noise_fits_io_vec : &filtered_noise_fits_io_vec;

        if constexpr (map_type == mapmaking::RawObs) {
            // write stats file
            write_stats();
            if (run_tod_output && !tod_filename.empty()) {
                // add tod header information
                add_tod_header(mb);
            }
        }
    }
    else if constexpr (map_type == mapmaking::RawCoadd || map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        dir_name = coadd_dir_name + (map_type == mapmaking::RawCoadd ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawCoadd) ? &coadd_fits_io_vec : &filtered_coadd_fits_io_vec;
        n_io = (map_type == mapmaking::RawCoadd) ? &coadd_noise_fits_io_vec : &filtered_coadd_noise_fits_io_vec;
    }

    if (run_mapmaking) {
        // wiener filtered maps write before this and are deleted from the vector.
        if (!f_io->empty()) {
            {
                // progress bar
                tula::logging::progressbar pb(
                    [&](const auto &msg) { logger->info("{}", msg); }, 100, "output progress ");

                for (Eigen::Index i=0; i<f_io->size(); ++i) {
                    // get the array for the given map
                    // add primary hdu
                    logger->debug("adding primary header to file {}",i);
                    add_phdu(f_io, mb, i);

                    // add primary hdu to noise maps
                    if (!mb->noise.empty() && !n_io->empty()) {
                        logger->debug("adding primary header to noise file {}",i);
                        add_phdu(n_io, mb, i);
                    }
                }

                logger->debug("done adding primary headers");

                // write the maps
                for (Eigen::Index i=0; i<n_maps; ++i) {
                    // update progress bar
                    pb.count(n_maps, 1);
                    write_maps(f_io,n_io,mb,i);
                }
            }

            logger->info("maps have been written to:");
            for (Eigen::Index i=0; i<f_io->size(); ++i) {
                logger->info("{}.fits",f_io->at(i).filepath);
            }
        }

        std::vector<std::string> data_filepaths;
        data_filepaths.reserve(f_io->size());
        for (const auto &fio : *f_io) {
            data_filepaths.push_back(fio.filepath + ".fits");
        }
        std::vector<std::string> noise_filepaths;
        noise_filepaths.reserve(n_io->size());
        for (const auto &nio : *n_io) {
            noise_filepaths.push_back(nio.filepath + ".fits");
        }
        auto join_filepaths = [](const std::vector<std::string> &paths) {
            std::string joined;
            for (std::size_t idx = 0; idx < paths.size(); ++idx) {
                if (idx > 0) {
                    joined += ", ";
                }
                joined += paths[idx];
            }
            return joined;
        };

        // clear fits file vectors to ensure its closed.
        try {
            std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(*f_io);
            std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(*n_io);
        } catch (const CCfits::FitsError &e) {
            throw std::runtime_error(
                fmt::format("failed while finalizing FITS outputs data_files=[{}] noise_files=[{}]: {}",
                            join_filepaths(data_filepaths),
                            join_filepaths(noise_filepaths),
                            e.message()));
        } catch (const std::exception &e) {
            throw std::runtime_error(
                fmt::format("failed while finalizing FITS outputs data_files=[{}] noise_files=[{}]: {}",
                            join_filepaths(data_filepaths),
                            join_filepaths(noise_filepaths),
                            e.what()));
        }

        // write psd and histogram files
        logger->debug("writing psds");
        write_psd<map_type>(mb, dir_name);
        logger->debug("writing histograms");
        write_hist<map_type>(mb, dir_name);
        logger->debug("writing map diagnostics");
        write_mapdiag<map_type>(mb, dir_name);

        // write source table
        if (run_source_finder) {
            logger->debug("writing source table");
            write_sources<map_type>(mb, dir_name);
        }
    }
}
