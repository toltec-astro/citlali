#pragma once

#include <citlali/core/engine/engine.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Pointing: public Engine {
public:
    // number of parameters for map fitting
    Eigen::Index n_params;

    // fit parameters
    Eigen::MatrixXd params;

    // fit errors
    Eigen::MatrixXd perrors;

    // meta information for ppt table
    YAML::Node ppt_meta;

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

    std::map<std::string,std::string> ppt_header_units;

    void setup();
    auto run();

    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    void fit_maps();

    template <mapmaking::MapType map_type>
    void output();
};

void Pointing::setup() {
    if (!telescope.sim_obs) {
        // check tau calculation
        Eigen::VectorXd tau_el(1);
        tau_el << telescope.tel_data["TelElAct"].mean();
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);

        for (auto const& [key, val] : tau_freq) {
            if (val[0] < 0) {
                SPDLOG_ERROR("calculated mean {} tau {} < 0",toltec_io.array_name_map[key], val[0]);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    // set number of parameters for map fitting
    n_params = 6;

    // resize the current fit matrix
    params.setZero(n_maps, n_params);
    perrors.setZero(n_maps, n_params);

    // set despiker sample rate
    rtcproc.despiker.fsmp = telescope.fsmp;

    // setup kernel
    if (rtcproc.run_kernel) {
        rtcproc.kernel.setup(n_maps);
    }

    // if filter is requested, make it here
    if (rtcproc.run_tod_filter) {
        rtcproc.filter.make_filter(telescope.fsmp);
    }

    // create output map files
    if (run_mapmaking) {
        create_map_files();
    }
    // create timestream files
    if (run_tod_output) {
        // create tod output subdirectory if requested
        if (tod_output_subdir_name!="null") {
            fs::create_directories(obsnum_dir_name + "/raw/" + tod_output_subdir_name);
        }
        // make rtc tod output file
        if (tod_output_type == "rtc" || tod_output_type=="both") {
            create_tod_files<engine_utils::toltecIO::rtc_timestream>();
        }
        // make ptc tod output file
        if (tod_output_type == "ptc" || tod_output_type=="both") {
            create_tod_files<engine_utils::toltecIO::ptc_timestream>();
        }
    }
    else if (!diagnostics.write_evals) {
        ptcproc.cleaner.n_calc = 0;
    }

    // tod output mode require sequential policy so set explicitly
    if (run_tod_output || verbose_mode) {
        SPDLOG_WARN("tod output mode require sequential policy");
        parallel_policy = "seq";
    }

    // use per detector parallelization for jinc mapmaking
    if (map_method == "jinc") {
        parallel_policy = "seq";
    }

    // set center pointing
    if (telescope.pixel_axes == "icrs") {
        omb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0)*RAD_TO_DEG;
        omb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0)*RAD_TO_DEG;

        cmb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0)*RAD_TO_DEG;
        cmb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0)*RAD_TO_DEG;
    }

    std::string pos_units;

    if (telescope.pixel_axes=="icrs") {
        pos_units = "deg";
    }
    else {
        pos_units = "arcsec";
    }

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
    ppt_meta["Source"] = telescope.source_name;

    // add date
    ppt_meta["Date"] = engine_utils::current_date_time();

    // reference frame
    ppt_meta["Radesys"] = telescope.pixel_axes;

    ppt_meta["array"].push_back("units: N/A");
    ppt_meta["array"].push_back("array");

    ppt_meta["amp"].push_back("units: " + omb.sig_unit);
    ppt_meta["amp"].push_back("fitted amplitude");

    ppt_meta["amp_err"].push_back("units: " + omb.sig_unit);
    ppt_meta["amp_err"].push_back("fitted amplitude error");

    ppt_meta["x_t"].push_back("units: " + pos_units);
    ppt_meta["x_t"].push_back("fitted azimuthal offset");

    ppt_meta["x_t_err"].push_back("units: " + pos_units);
    ppt_meta["x_t_err"].push_back("fitted azimuthal offset error");

    ppt_meta["y_t"].push_back("units: " + pos_units);
    ppt_meta["y_t"].push_back("fitted altitude offset");

    ppt_meta["y_t_err"].push_back("units: " + pos_units);
    ppt_meta["y_t_err"].push_back("fitted altitude offset error");

    ppt_meta["a_fwhm"].push_back("units: arcsec");
    ppt_meta["a_fwhm"].push_back("fitted azimuthal FWHM");

    ppt_meta["a_fwhm_err"].push_back("units: arcsec");
    ppt_meta["a_fwhm_err"].push_back("fitted azimuthal FWHM error");

    ppt_meta["b_fwhm"].push_back("units: arcsec");
    ppt_meta["b_fwhm"].push_back("fitted altitude FWMH");

    ppt_meta["b_fwhm_err"].push_back("units: arcsec");
    ppt_meta["b_fwhm_err"].push_back("fitted altitude FWMH error");

    ppt_meta["angle"].push_back("units: radians");
    ppt_meta["angle"].push_back("fitted rotation angle");

    ppt_meta["angle_err"].push_back("units: radians");
    ppt_meta["angle_err"].push_back("fitted rotation angle error");

    ppt_meta["sig2noise"].push_back("units: N/A");
    ppt_meta["sig2noise"].push_back("signal to noise");

    // add point model variables from telescope file
    for (const auto &val: telescope.tel_header) {
        std::size_t found = val.first.find("PointModel");
        if (found!=std::string::npos) {
            ppt_meta[val.first] = val.second(0);
        }
    }

    ppt_meta["Header.M2.ZReq"] = telescope.tel_header["Header.M2.ZReq"](0);
    ppt_meta["Header.M1.ZernikeC"] = telescope.tel_header["Header.M1.ZernikeC"](0);

    // print basic info for obs reduction
    print_summary();

    for (const auto &stat: diagnostics.det_stats_header) {
        diagnostics.stats[stat].setZero(calib.n_dets, telescope.scan_indices.cols());
    }

    for (const auto &stat: diagnostics.grp_stats_header) {
        diagnostics.stats[stat].setZero(calib.n_arrays, telescope.scan_indices.cols());
    }

    // clear stored eigenvalues
    diagnostics.evals.clear();
}

auto Pointing::run() {
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
        rtcdata.pointing_offsets_arcsec.data["az"] = pointing_offsets_arcsec["az"].segment(si,sl);
        rtcdata.pointing_offsets_arcsec.data["alt"] = pointing_offsets_arcsec["alt"].segment(si,sl);

        // get hwp
        if (rtcproc.run_polarization) {
            if (calib.run_hwp) {
                rtcdata.hwp_angle.data = calib.hwp_angle.segment(si + hwpr_start_indices, sl);
            }
        }

        // get raw tod from files
        {
            rtcdata.scans.data = kidsproc.populate_rtc(scan_rawobs,rtcdata.scan_indices.data, sl, calib.n_dets, tod_type);
        }

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;

        // loop through polarizations
        for (const auto &[stokes_index,stokes_param]: rtcproc.polarization.stokes_params) {
            SPDLOG_INFO("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

            // run rtcproc
            SPDLOG_INFO("raw time chunk processing");
            auto [map_indices, array_indices, nw_indices, det_indices] = rtcproc.run(rtcdata, ptcdata, telescope.pixel_axes, redu_type,
                                                                                     calib, telescope, omb.pixel_size_rad, stokes_param,
                                                                                     map_grouping);
            // write rtc timestreams
            if (run_tod_output) {
                if (tod_output_type == "rtc" || tod_output_type=="both") {
                    SPDLOG_INFO("writing raw time chunk");
                    ptcproc.append_to_netcdf(ptcdata, tod_filename["rtc_" + stokes_param], redu_type, telescope.pixel_axes,
                                             ptcdata.pointing_offsets_arcsec.data, det_indices, calib.apt, tod_output_type, verbose_mode,
                                             telescope.d_fsmp, calib.run_hwp);
                }
            }

            // subtract scan means
            SPDLOG_INFO("subtracting detector means");
            ptcproc.subtract_mean(ptcdata);

            // remove flagged dets
            SPDLOG_INFO("removing flagged dets");
            ptcproc.remove_flagged_dets(ptcdata, calib.apt, det_indices);

            // remove outliers
            auto calib_scan = rtcproc.remove_bad_dets(ptcdata, calib, det_indices, nw_indices, array_indices, redu_type, map_grouping);

            // remove duplicate tones
            if (!telescope.sim_obs) {
                calib_scan = rtcproc.remove_nearby_tones(ptcdata, calib, det_indices, nw_indices, array_indices, redu_type, map_grouping);
            }

            // run cleaning
            SPDLOG_INFO("processed time chunk processing");
            ptcproc.run(ptcdata, ptcdata, calib, det_indices, stokes_param);

            // remove outliers after clean
            calib_scan = ptcproc.remove_bad_dets(ptcdata, calib_scan, det_indices, nw_indices, array_indices, redu_type, map_grouping);

            // calculate weights
            SPDLOG_INFO("calculating weights");
            ptcproc.calc_weights(ptcdata, calib.apt, telescope, det_indices);

            // reset weights to median
            if (ptcproc.med_weight_factor >= 1) {
                ptcproc.reset_weights(ptcdata, calib, det_indices);
            }

            if (verbose_mode) {
                write_chunk_summary(ptcdata);
            }

            // write ptc timestreams
            if (run_tod_output) {
                if (tod_output_type == "ptc" || tod_output_type == "both") {
                    SPDLOG_INFO("writing processed time chunk");
                    ptcproc.append_to_netcdf(ptcdata, tod_filename["ptc_" + stokes_param], redu_type, telescope.pixel_axes,
                                             ptcdata.pointing_offsets_arcsec.data, det_indices, calib.apt, "ptc", verbose_mode,
                                             telescope.d_fsmp, calib.run_hwp);
                }
            }

            // calc stats
            if (stokes_param=="I") {
                diagnostics.calc_stats(ptcdata);
            }

            // populate maps
            if (run_mapmaking) {
                SPDLOG_INFO("populating maps");
                if (map_method=="naive") {
                    mapmaking::populate_maps_naive(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
                                                   redu_type, calib.apt, ptcdata.pointing_offsets_arcsec.data, telescope.d_fsmp, run_noise);
                }
                else if (map_method=="jinc") {
                    mapmaking::populate_maps_jinc(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
                                                  redu_type, calib.apt, ptcdata.pointing_offsets_arcsec.data, telescope.d_fsmp, run_noise,
                                                  jinc_r_max, jinc_shape_params, omb.parallel_policy);
                }
            }
        }

        n_scans_done++;
        SPDLOG_INFO("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

        return ptcdata;
    });

    return farm;
}

template <class KidsProc, class RawObs>
void Pointing::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // initialize number of completed scans
    n_scans_done = 0;

    // progress bar
    tula::logging::progressbar pb(
        [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 100, "citlali progress ");

    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                          std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>>> {
            // variable to hold current scan
            static auto scan = 0;
            while (scan < telescope.scan_indices.cols()) {

                // update progress bar
                pb.count(telescope.scan_indices.cols(), 1);

                // create rtcdata
                TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
                // get scan indices
                rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
                // current scan
                rtcdata.index.data = scan;

                // vector to store kids data
                std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> scan_rawobs;
                {
                    tula::logging::scoped_loglevel<spdlog::level::off> _0;
                    // get kids data
                    scan_rawobs = kidsproc.load_rawobs(rawobs, scan, telescope.scan_indices, start_indices, end_indices);
                }

                // increment scan
                scan++;
                return std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                  std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>> (std::move(rtcdata), kidsproc,
                                                                                                  std::move(scan_rawobs));
            }
            // reset scan to zero for each obs
            scan = 0;
            return {};
        },

        run());

    if (run_mapmaking) {
        // normalize maps
        SPDLOG_INFO("normalizing maps");
        omb.normalize_maps();
        // calculate map psds
        SPDLOG_INFO("calculating map psd");
        omb.calc_map_psd();
        // calculate map histograms
        SPDLOG_INFO("calculating map histogram");
        omb.calc_map_hist();

        // fit maps
        fit_maps();
    }
}

void Pointing::fit_maps() {
    // fit maps
    SPDLOG_INFO("fitting maps");
    // placeholder vectors for grppi map
    std::vector<int> map_in_vec, map_out_vec;

    map_in_vec.resize(n_maps);
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(n_maps);

    double init_row = -99;
    double init_col = -99;

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
            params(i,1) = RAD_TO_ASEC*omb.pixel_size_rad*(params(i,1) - (omb.n_cols)/2);
            params(i,2) = RAD_TO_ASEC*omb.pixel_size_rad*(params(i,2) - (omb.n_rows)/2);
            params(i,3) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params(i,3));
            params(i,4) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params(i,4));

            // rescale fit errors from pixel to on-sky units
            perrors(i,1) = RAD_TO_ASEC*omb.pixel_size_rad*(perrors(i,1));
            perrors(i,2) = RAD_TO_ASEC*omb.pixel_size_rad*(perrors(i,2));
            perrors(i,3) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors(i,3));
            perrors(i,4) = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors(i,4));

            if (telescope.pixel_axes=="icrs") {
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
    mapmaking::ObsMapBuffer* mb = NULL;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = NULL;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = NULL;

    // directory name
    std::string dir_name;

    // matrix to hold pointing fit values and errors
    Eigen::MatrixXf ppt_table(n_maps, 2*n_params + 2);

    // raw obs maps
    if constexpr (map_type == mapmaking::RawObs) {
        mb = &omb;
        f_io = &fits_io_vec;
        n_io = &noise_fits_io_vec;
        dir_name = obsnum_dir_name + "raw/";

        auto ppt_filename = toltec_io.create_filename<engine_utils::toltecIO::ppt, engine_utils::toltecIO::map,
                                                      engine_utils::toltecIO::raw>
                            (dir_name, redu_type, "", obsnum, telescope.sim_obs);

        // loop through params and add arrays
        for (Eigen::Index i=0; i<n_maps; i++) {
            ppt_table(i,0) = maps_to_arrays(i);

            // calculate map standard deviation
            double map_std_dev = engine_utils::calc_std_dev(mb->signal[i]);
            // set signal to noise
            ppt_table(i,2*n_params + 1) = params(i,0)/map_std_dev;
        }

        // populate table
        Eigen::Index j = 0;
        for (Eigen::Index i=1; i<2*n_params; i=i+2) {
            ppt_table.col(i) = params.col(j).cast <float> ();
            ppt_table.col(i+1) = perrors.col(j).cast <float> ();
            j++;
        }

        // write table
        to_ecsv_from_matrix(ppt_filename, ppt_table, ppt_header, ppt_meta);

        // write stats file
        write_stats();
    }

    // filtered obs maps
    else if constexpr (map_type == mapmaking::FilteredObs) {
        mb = &omb;
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        dir_name = obsnum_dir_name + "filtered/";

        auto ppt_filename = toltec_io.create_filename<engine_utils::toltecIO::ppt, engine_utils::toltecIO::map,
                                                      engine_utils::toltecIO::filtered>
                            (dir_name, redu_type, "", obsnum, telescope.sim_obs);

        // loop through params and add arrays
        for (Eigen::Index i=0; i<n_maps; i++) {
            ppt_table(i,0) = maps_to_arrays(i);

            // calculate map standard deviation
            double map_std_dev = engine_utils::calc_std_dev(mb->signal[i]);
            // set signal to noise
            ppt_table(i,2*n_params + 1) = params(i,0)/map_std_dev;
        }

        // populate table
        Eigen::Index j = 0;
        for (Eigen::Index i=1; i<2*n_params; i=i+2) {
            ppt_table.col(i) = params.col(j).cast <float> ();
            ppt_table.col(i+1) = perrors.col(j).cast <float> ();
            j++;
        }

        // write table
        to_ecsv_from_matrix(ppt_filename, ppt_table, ppt_header, ppt_meta);
    }

    // raw coadded maps
    else if constexpr (map_type == mapmaking::RawCoadd) {
        mb = &cmb;
        f_io = &coadd_fits_io_vec;
        n_io = &coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "raw/";
    }

    // filtered coadded maps
    else if constexpr (map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        f_io = &filtered_coadd_fits_io_vec;
        n_io = &filtered_coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "filtered/";
    }

    if (!f_io->empty()) {
        // progress bar
        tula::logging::progressbar pb(
            [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 100, "output progress ");

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

            //if constexpr (map_type == mapmaking::RawObs) {
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
            Eigen::Index j = 0;
            for (auto const& key: ppt_header) {
                try {
                    f_io->at(map_index).hdus.at(k)->addKey("POINTING." + key, ppt_table(i,j), key + " (" + ppt_header_units[key] + ")");
                }
                catch(...) {
                    f_io->at(map_index).hdus.at(k)->addKey("POINTING." + key, 0, key + " (" + ppt_header_units[key] + ")");
                }
                j++;
            }
            k++;
            //}
        }

        SPDLOG_INFO("files have been written to:");
        for (Eigen::Index i=0; i<f_io->size(); i++) {
            SPDLOG_INFO("{}.fits",f_io->at(i).filepath);
        }
    }

    // clear fits file vectors to ensure its closed.
    f_io->clear();
    n_io->clear();

    // write source table
    if (run_source_finder) {
        write_sources<map_type>(mb, dir_name);
    }

    // write psd and histogram files
    write_psd<map_type>(mb, dir_name);
    write_hist<map_type>(mb, dir_name);
}
