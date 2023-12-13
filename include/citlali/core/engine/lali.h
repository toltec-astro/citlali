#pragma once

#include <citlali/core/engine/engine.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Lali: public Engine {
public:
    // initial setup for each obs
    void setup(int);

    // run the reduction for the obs
    auto run();

    // main grppi pipeline
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // output files
    template <mapmaking::MapType map_type>
    void output();
};

void Lali::setup(int fruit_iter) {
    // run obsnum setup
    obsnum_setup(fruit_iter);

    // use per detector parallelization for jinc mapmaking
    if (map_method == "jinc") {
        parallel_policy = "seq";
    }
}

auto Lali::run() {
    auto farm = grppi::farm(n_threads,[&](auto &input_tuple) -> TCData<TCDataKind::PTC,Eigen::MatrixXd> {
        // RTCData input
        auto rtcdata = std::get<0>(input_tuple);
        // kidsproc
        auto kidsproc = std::get<1>(input_tuple);
        // start index input
        auto scan_rawobs = std::get<2>(input_tuple);

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

        // get raw tod from files
        rtcdata.scans.data = kidsproc.populate_rtc(scan_rawobs,rtcdata.scan_indices.data, sl, calib.n_dets, tod_type);

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;

        logger->info("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done,
                    telescope.scan_indices.cols());

        // run rtcproc
        logger->info("raw time chunk processing for scan {}", rtcdata.index.data + 1);
        auto [map_indices, array_indices, nw_indices, det_indices] = rtcproc.run(rtcdata, ptcdata, telescope.pixel_axes, redu_type,
                                                                                 calib, telescope, omb.pixel_size_rad, map_grouping);

        // remove flagged detectors
        rtcproc.remove_flagged_dets(ptcdata, calib.apt, det_indices);

        // remove outliers before cleaning
        auto calib_scan = rtcproc.remove_bad_dets(ptcdata, calib, det_indices, map_grouping);

        // remove duplicate tones
        if (!telescope.sim_obs) {
            calib_scan = rtcproc.remove_nearby_tones(ptcdata, calib, det_indices, map_grouping);
        }

        // write rtc timestreams
        if (run_tod_output && !tod_filename.empty()) {
            if (tod_output_type == "rtc" || tod_output_type == "both") {
                logger->info("writing raw time chunk");
                rtcproc.append_to_netcdf(ptcdata, tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         ptcdata.pointing_offsets_arcsec.data, det_indices, calib);
            }
        }

        // if running fruit loops and a map has been read in
        if (ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty()) {
            logger->info("subtracting map from tod");
            // subtract map
            ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(ptcproc.tod_mb, ptcdata, calib, det_indices,
                                                                            map_indices, telescope.pixel_axes,
                                                                            map_grouping);
        }

        // run cleaning
        logger->info("processed time chunk processing for scan {}", ptcdata.index.data + 1);
        ptcproc.run(ptcdata, ptcdata, calib, det_indices, telescope.pixel_axes, map_grouping);

        // if running fruit loops and a map has been read in
        if (ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty()) {
            // calculate weights
            logger->info("calculating weights");
            ptcproc.calc_weights(ptcdata, calib.apt, telescope, det_indices);

            // reset weights to median
            ptcproc.reset_weights(ptcdata, calib, det_indices);

            // populate maps
            if (run_mapmaking) {
                bool run_omb = false;
                logger->info("populating noise maps");
                if (map_method=="naive") {
                    naive_mm.populate_maps_naive(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
                                                 calib.apt, telescope.d_fsmp, run_omb, run_noise);
                }
                else if (map_method=="jinc") {
                    jinc_mm.populate_maps_jinc(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
                                               calib.apt, telescope.d_fsmp, run_omb, run_noise);
                }
            }
            logger->info("adding map to tod");
            // add map back
            ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(ptcproc.tod_mb, ptcdata, calib, det_indices,
                                                                    map_indices, telescope.pixel_axes,
                                                                    map_grouping);
        }

        // remove outliers after cleaning
        calib_scan = ptcproc.remove_bad_dets(ptcdata, calib, det_indices, map_grouping);

        // calculate weights
        logger->info("calculating weights");
        ptcproc.calc_weights(ptcdata, calib.apt, telescope, det_indices);

        // reset weights to median
        ptcproc.reset_weights(ptcdata, calib, det_indices);

        // write ptc timestreams
        if (run_tod_output && !tod_filename.empty()) {
            if (tod_output_type == "ptc" || tod_output_type == "both") {
                logger->info("writing processed time chunk");
                ptcproc.append_to_netcdf(ptcdata, tod_filename["ptc"], map_grouping, telescope.pixel_axes,
                                         ptcdata.pointing_offsets_arcsec.data, det_indices, calib);
            }
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
            bool run_noise_fruit;

            // if running fruit loops, noise maps are made on source
            // subtracted timestreams so don't make them here unless
            // on first iteration
            if (ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty()) {
                run_noise_fruit = false;
            }
            else {
                run_noise_fruit = run_noise;
            }

            logger->info("populating maps");
            if (map_method=="naive") {
                naive_mm.populate_maps_naive(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
                                             calib.apt, telescope.d_fsmp, run_omb, run_noise_fruit);
            }
            else if (map_method=="jinc") {
                jinc_mm.populate_maps_jinc(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
                                           calib.apt, telescope.d_fsmp, run_omb, run_noise_fruit);
            }
        }

        // increment number of completed scans
        n_scans_done++;
        logger->info("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

        return ptcdata;
    });

    return farm;
}

template <class KidsProc, class RawObs>
void Lali::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // initialize number of completed scans
    n_scans_done = 0;

    // progress bar
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "citlali progress ");

    // grppi generator function. gets time chunk data from files sequentially and passes them to grppi::farm
    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                          std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>>> {
            // variable to hold current scan
            static auto scan = 0;
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

                if (run_noise) {
                    // declare random number generator
                    thread_local boost::random::mt19937 eng;

                    // boost random number generator (0,1)
                    boost::random::uniform_int_distribution<> rands{0,1};

                    if (omb.randomize_dets) {
                        rtcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(omb.n_noise, calib.n_dets)
                                    .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                    } else {
                        rtcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(omb.n_noise)
                                    .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                    }
                }

                // vector to store kids data
                std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> scan_rawobs;
                // get kids data
                scan_rawobs = kidsproc.load_rawobs(rawobs, scan, telescope.scan_indices, start_indices, end_indices);

                // increment scan
                scan++;
                // return rtcdata, kidsproc, and raw data
                return std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                  std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>> (std::move(rtcdata), kidsproc,
                                                                                                   std::move(scan_rawobs));
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
        omb.normalize_maps();
        // calculate map psds
        logger->info("calculating map psd");
        omb.calc_map_psd();
        // calculate map histograms
        logger->info("calculating map histogram");
        omb.calc_map_hist();

        // calculate mean error
        omb.calc_mean_err();
        // calculate mean rms
        omb.calc_mean_rms();

        // write map summary
        if (verbose_mode) {
            write_map_summary(omb);
        }
    }
}

template <mapmaking::MapType map_type>
void Lali::output() {
    // pointer to map buffer
    mapmaking::ObsMapBuffer* mb = nullptr;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;

    // directory name
    std::string dir_name;

    // set common variables depending on map_type
    if constexpr (map_type == mapmaking::RawObs || map_type == mapmaking::FilteredObs) {
        // create output map files
        if (run_mapmaking) {
            create_obs_map_files();
        }

        mb = &omb;
        dir_name = obsnum_dir_name + (map_type == mapmaking::RawObs ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawObs) ? &fits_io_vec : &filtered_fits_io_vec;
        n_io = (map_type == mapmaking::RawObs) ? &noise_fits_io_vec : &filtered_noise_fits_io_vec;

        if constexpr (map_type == mapmaking::RawObs) {
            // write stats file
            write_stats();
            if (run_tod_output && !tod_filename.empty()) {
                // add tod header information
                add_tod_header();
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
                    add_phdu(f_io, mb, i);

                    if (!mb->noise.empty()) {
                        add_phdu(n_io, mb, i);
                    }
                }

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

        // clear fits file vectors to ensure its closed.
        f_io->clear();
        n_io->clear();

        // write psd and histogram files
        logger->debug("writing psds");
        write_psd<map_type>(mb, dir_name);
        logger->debug("writing histograms");
        write_hist<map_type>(mb, dir_name);

        // write source table
        if (run_source_finder) {
            logger->debug("writing source table");
            write_sources<map_type>(mb, dir_name);
        }
    }
}
