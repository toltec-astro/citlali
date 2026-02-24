#pragma once

#include <mutex>
#include <condition_variable>

#include <citlali/core/engine/engine.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Pointing: public Engine {
public:
    // fit parameters
    Eigen::MatrixXd params, perrors;

    // meta information for ppt table
    YAML::Node ppt_meta;

    // ppt header information
    std::vector<std::string> ppt_header = {
        "array",
        "amp",
        "amp_err",
        "x_t",
        "x_t_err",
        "y_t",
        "y_t_err",
        "a_fwhm",
        "a_fwhm_err",
        "b_fwhm",
        "b_fwhm_err",
        "angle",
        "angle_err",
        "sig2noise"
    };

    // ppt header units
    std::map<std::string,std::string> ppt_header_units;

    // initial setup for each obs
    void setup();

    // main grppi pipeline
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // run the reduction for the obs
    template <class KidsProc>
    auto run(KidsProc &);

    // fit the maps
    void fit_maps();

    // output files
    template <mapmaking::MapType map_type>
    void output();
};

void Pointing::setup() {
    // run obsnum setup
    obsnum_setup();

    // resize the current fit matrix
    params.setZero(n_maps, map_fitter.n_params);
    perrors.setZero(n_maps, map_fitter.n_params);

    // units for positions
    std::string pos_units = (telescope.pixel_axes == "radec") ? "deg" : "arcsec";

    // units for ppt header
    ppt_header_units = {
        {"array","N/A"},
        {"amp", omb.sig_unit},
        {"amp_err", omb.sig_unit},
        {"x_t", pos_units},
        {"x_t_err", pos_units},
        {"y_t", pos_units},
        {"y_t_err", pos_units},
        {"a_fwhm", "arcsec"},
        {"a_fwhm_err", "arcsec"},
        {"b_fwhm", "arcsec"},
        {"b_fwhm_err", "arcsec"},
        {"angle", "rad"},
        {"angle_err", "rad"},
        {"sig2noise", "N/A"}
    };

    /* populate ppt meta information */
    ppt_meta.reset();

    // add obsnum to meta data
    ppt_meta["obsnum"] = obsnum;

    // add source name
    ppt_meta["source"] = telescope.source_name;

    // add project id to meta data
    ppt_meta["project_id"] = telescope.project_id;

    // add date of file creation
    ppt_meta["creation_date"] = engine_utils::current_date_time();

    // add observation date
    ppt_meta["date"] = date_obs.back();

    // mean Modified Julian Date
    ppt_meta["mjd"] = engine_utils::unix_to_modified_julian_date(telescope.tel_data["TelTime"].mean());

    // reference frame
    ppt_meta["Radesys"] = telescope.pixel_axes;

    // add array mapping
    for (const auto &[arr_index,arr_name]: toltec_io.array_name_map) {
        ppt_meta["array_order"].push_back(std::to_string(arr_index) + ": " + arr_name);
    }

    // populate ppt meta information
    for (const auto &[param,unit]: ppt_header_units) {
        ppt_meta[param].push_back("units: " + unit);
        // description from apt
        auto description = calib.apt_header_description[unit];
        ppt_meta[param].push_back(description);
    }

    // add point model variables from telescope file
    for (const auto &val: telescope.tel_header) {
        std::size_t found = val.first.find("PointModel");
        if (found!=std::string::npos) {
            ppt_meta[val.first] = val.second(0);
        }
    }
    // add m2 z position
    ppt_meta["Header.M2.ZReq"] = telescope.tel_header["Header.M2.ZReq"](0);
    // add first m1 zernike coefficient
    ppt_meta["Header.M1.ZernikeC"] = telescope.tel_header["Header.M1.ZernikeC"](0);

    for (int i=0; i< telescope.tel_header["Header.M1.ActPos"].size(); ++i) {
        ppt_meta["Header.M1.ActPos"].push_back(telescope.tel_header["Header.M1.ActPos"](i));
        ppt_meta["Header.M1.CmdPos"].push_back(telescope.tel_header["Header.M1.CmdPos"](i));
    }
}

template <class KidsProc, class RawObs>
void Pointing::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
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

                // populate noise matrix
                if (run_noise) {
                    if (omb.randomize_dets) {
                        rtcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(omb.n_noise, calib.n_dets)
                                                 .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                    } else {
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
        run(kidsproc));

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
        omb.calc_median_err();
        // calculate mean rms
        omb.calc_median_rms();

        // fit maps
        fit_maps();
    }
}

template <class KidsProc>
auto Pointing::run(KidsProc &kidsproc) {
    auto scans_done_mutex = std::make_shared<std::mutex>();

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

    auto rtc_writer = write_rtc ? std::make_shared<OrderedWriter>() : nullptr;
    auto ptc_writer = write_ptc ? std::make_shared<OrderedWriter>() : nullptr;

    auto farm = grppi::farm(n_threads,[&, scans_done_mutex, rtc_writer, ptc_writer, write_rtc, write_ptc](auto &rtcdata) {

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
                    if (rtcproc.run_tod_filter) {
                        start_index = std::max(0, static_cast<int>(j - rtcproc.filter.n_terms));
                        int end_index = std::min(j + rtcproc.filter.n_terms, rtcdata.flags.data.rows() - 1);
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

        {
            std::lock_guard<std::mutex> lk(*scans_done_mutex);
            logger->info("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done,
                         telescope.scan_indices.cols());
        }

        // run rtcproc
        logger->info("raw time chunk processing for scan {}", rtcdata.index.data + 1);
        auto map_indices = rtcproc.run(rtcdata, ptcdata, calib, telescope, omb.pixel_size_rad, map_grouping);

        // remove flagged detectors
        rtcproc.remove_flagged_dets(ptcdata, calib.apt);

        // remove outliers before cleaning
        auto calib_scan = rtcproc.remove_bad_dets(ptcdata, calib, map_grouping);

        // remove duplicate tones
        if (!telescope.sim_obs) {
            calib_scan = rtcproc.remove_nearby_tones(ptcdata, calib, map_grouping);
        }

        // write rtc timestreams
        const auto rtc_scan_row = tod_output_scan_row(rtcdata.index.data, "rtc");
        if (write_rtc && rtc_scan_row >= 0) {
            rtc_writer->wait_turn(rtc_scan_row);
            logger->info("writing raw time chunk");
            rtcproc.append_to_netcdf(ptcdata, tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                     ptcdata.pointing_offsets_arcsec.data, calib, false, rtc_scan_row);
            rtc_writer->advance();
        }

        // if running fruit loops and a map has been read in
        if (ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty()) {
            logger->info("subtracting map from tod");
            // subtract map
            ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(ptcproc.tod_mb, ptcdata, calib,
                                                                            map_indices, telescope.pixel_axes,
                                                                            map_grouping);
        }

        // run cleaning
        logger->info("processed time chunk processing for scan {}", ptcdata.index.data + 1);
        ptcproc.run(ptcdata, ptcdata, calib, telescope.pixel_axes, map_grouping);

        // if running fruit loops and a map has been read in
        if (ptcproc.run_fruit_loops && !ptcproc.tod_mb.signal.empty()) {
            // calculate weights
            logger->info("calculating weights for scan {} (fruit loops noise-only pass)",
                         ptcdata.index.data + 1);
            ptcproc.calc_weights(ptcdata, calib.apt, telescope);

            // reset weights to median
            auto calib_scans = ptcproc.reset_weights(ptcdata, calib, map_grouping);

            // populate maps
            if (run_mapmaking) {
                bool run_omb = false;
                logger->info("populating noise maps");
                if (map_method=="naive") {
                    naive_mm.populate_maps_naive(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                                 calib.apt, telescope.d_fsmp, run_omb, run_noise);
                }
                else if (map_method=="jinc") {
                    jinc_mm.populate_maps_jinc(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                               calib.apt, telescope.d_fsmp, run_omb, run_noise);
                }
            }
            logger->info("adding map to tod");
            // add map back
            ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(ptcproc.tod_mb, ptcdata, calib,
                                                                    map_indices, telescope.pixel_axes,
                                                                    map_grouping);
        }

        // remove outliers after cleaning
        calib_scan = ptcproc.remove_bad_dets(ptcdata, calib_scan, map_grouping);

        // calculate weights
        logger->info("calculating weights for scan {}", ptcdata.index.data + 1);
        ptcproc.calc_weights(ptcdata, calib.apt, telescope);

        // reset weights to median
        calib_scan = ptcproc.reset_weights(ptcdata, calib, map_grouping);

        // write ptc timestreams
        const auto ptc_scan_row = tod_output_scan_row(ptcdata.index.data, "ptc");
        if (write_ptc && ptc_scan_row >= 0) {
            ptc_writer->wait_turn(ptc_scan_row);
            logger->info("writing processed time chunk");
            ptcproc.append_to_netcdf(ptcdata, tod_filename["ptc"], map_grouping, telescope.pixel_axes,
                                     ptcdata.pointing_offsets_arcsec.data, calib, false, ptc_scan_row);
            ptc_writer->advance();
        }

        // write out chunk summary
        if (verbose_mode) {
            write_chunk_summary(ptcdata);
        }

        // calc stats
        logger->debug("calculating stats");
        diagnostics.calc_stats(ptcdata);

        // populate maps
        if (run_mapmaking) {
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
                naive_mm.populate_maps_naive(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                             calib.apt, telescope.d_fsmp, run_omb, run_noise_fruit);
            }
            else if (map_method=="jinc") {
                jinc_mm.populate_maps_jinc(ptcdata, omb, cmb, map_indices, telescope.pixel_axes,
                                           calib.apt, telescope.d_fsmp, run_omb, run_noise_fruit);
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

void Pointing::fit_maps() {
    // fit maps
    logger->info("fitting maps");
    // placeholder vectors for grppi map
    std::vector<int> map_in_vec, map_out_vec;

    map_in_vec.resize(n_maps);
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(n_maps);

    double init_row = -99;
    double init_col = -99;

    // loop through maps
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
        auto array = maps_to_arrays(i);
        // init fwhm in pixels
        double init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/omb.pixel_size_rad;
        auto [map_params, map_perror, good_fit] =
            map_fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(omb.signal[i], omb.weight[i], init_fwhm, init_row, init_col);
        params.row(i) = map_params;
        perrors.row(i) = map_perror;

        if (good_fit) {
            // rescale fit params from pixel to on-sky units
            params(i,1) = RAD_TO_ASEC*omb.pixel_size_rad*(params(i,1) - (omb.n_cols - 1)/2.0);
            params(i,2) = RAD_TO_ASEC*omb.pixel_size_rad*(params(i,2) - (omb.n_rows - 1)/2.0);
            params(i,3) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params(i,3));
            params(i,4) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params(i,4));

            // rescale fit errors from pixel to on-sky units
            perrors(i,1) = RAD_TO_ASEC*omb.pixel_size_rad*(perrors(i,1));
            perrors(i,2) = RAD_TO_ASEC*omb.pixel_size_rad*(perrors(i,2));
            perrors(i,3) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors(i,3));
            perrors(i,4) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors(i,4));

            // if in radec calculate absolute pointing
            if (telescope.pixel_axes=="radec") {
                Eigen::VectorXd lat(1), lon(1);
                lat << params(i,2)*ASEC_TO_RAD;
                lon << params(i,1)*ASEC_TO_RAD;

                auto [adec, ara] = engine_utils::tangent_to_abs(lat, lon, omb.wcs.crval[0]*DEG_TO_RAD, omb.wcs.crval[1]*DEG_TO_RAD);

                params(i,1) = ara(0)*RAD_TO_DEG;
                params(i,2) = adec(0)*RAD_TO_DEG;

                perrors(i,1) = perrors(i,1)*ASEC_TO_DEG;
                perrors(i,2) = perrors(i,2)*ASEC_TO_DEG;
            }
        }
        return 0;
    });
}

template <mapmaking::MapType map_type>
void Pointing::output() {
    // pointer to map buffer
    mapmaking::MapBuffer* mb = nullptr;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;

    // directory name
    std::string dir_name;

    // matrix to hold pointing fit values and errors (n_params + 2 for array and S/N)
    Eigen::MatrixXf ppt_table(n_maps, 2 * map_fitter.n_params + 2);

    // determine pointers and directory name based on map_type
    if constexpr (map_type == mapmaking::RawObs || map_type == mapmaking::FilteredObs) {
        mb = &omb;
        dir_name = obsnum_dir_name + (map_type == mapmaking::RawObs ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawObs) ? &fits_io_vec : &filtered_fits_io_vec;
        n_io = (map_type == mapmaking::RawObs) ? &noise_fits_io_vec : &filtered_noise_fits_io_vec;

        // filename for ppt table
        auto ppt_filename = toltec_io.create_filename<engine_utils::toltecIO::ppt, engine_utils::toltecIO::map,
                                                      (map_type == mapmaking::RawObs ? engine_utils::toltecIO::raw : engine_utils::toltecIO::filtered)>
                            (dir_name, redu_type, "", obsnum, telescope.sim_obs);

        // add array and S/N to ppt
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            ppt_table(i, 0) = maps_to_arrays(i);
            double map_std_dev = engine_utils::calc_std_dev(mb->signal[i]);
            ppt_table(i, 2 * map_fitter.n_params + 1) = params(i, 0) / map_std_dev;
        }

        Eigen::Index j = 0;
        // populate ppt with fitted parameters and errors
        for (Eigen::Index i = 1; i < 2 * map_fitter.n_params; i += 2) {
            ppt_table.col(i) = params.col(j).cast<float>();
            ppt_table.col(i + 1) = perrors.col(j).cast<float>();
            j++;
        }

        // write ppt
        to_ecsv_from_matrix(ppt_filename, ppt_table, ppt_header, ppt_meta);

        if constexpr (map_type == mapmaking::RawObs) {
            // write stats file
            write_stats();
            if (run_tod_output && !tod_filename.empty()) {
                // add tod header information
                add_tod_header(mb);
            }
        }
    } else if constexpr (map_type == mapmaking::RawCoadd || map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        dir_name = coadd_dir_name + (map_type == mapmaking::RawCoadd ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawCoadd) ? &coadd_fits_io_vec : &filtered_coadd_fits_io_vec;
        n_io = (map_type == mapmaking::RawCoadd) ? &coadd_noise_fits_io_vec : &filtered_coadd_noise_fits_io_vec;
    }

    if (run_mapmaking) {
        if (!f_io->empty()) {
            {
                // progress bar
                tula::logging::progressbar pb(
                    [&](const auto &msg) { logger->info("{}", msg); }, 100, "output progress ");

                for (Eigen::Index i=0; i<f_io->size(); i++) {
                    // add primary hdu
                    add_phdu(f_io, mb, i);

                    if (!mb->noise.empty()) {
                        add_phdu(n_io, mb, i);
                    }
                }

                Eigen::Index k = 0;

                for (Eigen::Index i=0; i<n_maps; i++) {
                    // update progress bar
                    pb.count(n_maps, 1);
                    write_maps(f_io,n_io,mb,i);

                    Eigen::Index map_index = arrays_to_maps(i);

                    // check if we move from one file to the next
                    // if so go back to first hdu layer
                    if (i>0) {
                        if (map_index > arrays_to_maps(i-1)) {
                            k = 0;
                        }
                    }
                    // get current hdu extension name
                    std::string extname = f_io->at(map_index).hdus.at(k)->name();
                    // see if this is a signal extension
                    std::size_t found = extname.find("signal");

                    // find next signal extension
                    while (found==std::string::npos && k<f_io->at(map_index).hdus.size()) {
                        k = k + 1;
                        // get current hdu extension name
                        extname = f_io->at(map_index).hdus.at(k)->name();
                        // see if this is a signal extension
                        found = extname.find("signal");
                    }

                    // add ppt table
                    for (Eigen::Index j = 0; j < ppt_header.size(); ++j) {
                        const auto& key = ppt_header[j];
                        try {
                            f_io->at(map_index).hdus.at(k)->addKey("POINTING." + key, ppt_table(i, j), key + " (" + ppt_header_units[key] + ")");
                        } catch (...) {
                            f_io->at(map_index).hdus.at(k)->addKey("POINTING." + key, 0, key + " (" + ppt_header_units[key] + ")");
                        }
                    }
                    ++k; // Move to next extension
                }
            }

            logger->info("maps have been written to:");
            for (const auto& file: *f_io) {
                logger->info("{}.fits", file.filepath);
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
