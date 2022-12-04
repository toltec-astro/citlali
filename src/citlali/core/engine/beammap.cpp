#include <citlali_config/gitversion.h>

#include <citlali/core/engine/beammap.h>
#include <citlali/core/engine/kidsproc.h>

void Beammap::setup() {
    // set number of parameters for map fitting
    n_params = 6;

    // resize the PTCData vector to number of scans
    ptcs0.resize(telescope.scan_indices.cols());

    // resize the initial fit matrix
    p0.resize(calib.n_dets, n_params);
    // set initial fit to nan to pass first cutoff test
    p0.setConstant(std::nan(""));
    // resize the initial fit error matrix
    perror0.setZero(calib.n_dets, n_params);
    // resize the current fit matrix
    params.setZero(calib.n_dets, n_params);
    perrors.setZero(calib.n_dets, n_params);

    // initially all detectors are unconverged
    converged.setZero(calib.n_dets);
    // convergence iteration
    converge_iter.resize(calib.n_dets);
    converge_iter.setConstant(1);
    // set the initial iteration
    current_iter = 0;

    // derotation elevation
    derot_elev.resize(calib.n_dets);

    // setup kernel
    if (rtcproc.run_kernel) {
        rtcproc.kernel.setup(n_maps);
    }

    // if filter is requested, make it here
    if (rtcproc.run_tod_filter) {
        rtcproc.filter.make_filter(telescope.fsmp);
    }

    // create output map files
    create_map_files();
    // create timestream files
    if (run_tod_output) {
        create_tod_files();
    }
}

auto Beammap::run_timestream() {
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

        // get hwp
        if (rtcproc.run_polarization) {
            rtcdata.hwp_angle.data = calib.hwp_angle.segment(si, sl);
        }

        // get raw tod from files
        {
            tula::logging::scoped_loglevel<spdlog::level::off> _0;
            rtcdata.scans.data = kidsproc.populate_rtc(scan_rawobs,rtcdata.scan_indices.data, sl, calib.n_dets, tod_type);
        }

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;

        // loop through polarizations
        for (const auto &stokes_param: rtcproc.polarization.stokes_params) {
            SPDLOG_INFO("reducing {} timestream",stokes_param);
            // create a new rtcdata for each polarization
            TCData<TCDataKind::RTC,Eigen::MatrixXd> rtcdata_pol;
            // demodulate
            SPDLOG_INFO("demodulating polarization");
            auto [array_indices, nw_indices, det_indices] = rtcproc.polarization.demodulate_timestream(rtcdata, rtcdata_pol,
                                                                                                       stokes_param,
                                                                                                       redu_type, calib);
            // get indices for maps
            SPDLOG_INFO("calculating map indices");
            auto map_indices = calc_map_indices(det_indices, nw_indices, array_indices, stokes_param);

            // run rtcproc
            SPDLOG_INFO("rtcproc");
            rtcproc.run(rtcdata_pol, ptcdata, telescope.pixel_axes, redu_type, calib, pointing_offsets_arcsec, det_indices, array_indices,
                        map_indices, omb.pixel_size_rad);

            // write rtc timestreams
            if (run_tod_output) {
                SPDLOG_INFO("writing rtcdata");
                if (tod_output_type == "rtc") {
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], det_indices, calib.apt);
                }
            }

            // store indices for each ptcdata
            ptcdata.det_indices.data = std::move(det_indices);
            ptcdata.nw_indices.data = std::move(nw_indices);
            ptcdata.array_indices.data = std::move(array_indices);
            ptcdata.map_indices.data = std::move(map_indices);

            // move out ptcdata the PTCData vector at corresponding index
            ptcs0.at(ptcdata.index.data) = std::move(ptcdata);
        }

        return ptcdata;
    });

    return farm;
}

auto Beammap::run_loop() {
    bool keep_going = true;

    while (keep_going) {
        ptcs = ptcs0;
        // set maps to zero for each iteration
        for (Eigen::Index i=0; i<n_maps; i++) {
            omb.signal[i].setZero();
            omb.weight[i].setZero();
            omb.coverage[i].setZero();

            if (rtcproc.run_kernel) {
                omb.kernel[i].setZero();
            }
        }

        // cleaning
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
            // run cleaning
            SPDLOG_INFO("ptcproc");
            ptcproc.run(ptcs[i], ptcs[i], calib);
            return 0;
        });

        // sensitivity
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            Eigen::MatrixXd det_sens, noise_flux;
            calc_sensitivity(ptcs, det_sens, noise_flux, telescope.d_fsmp, i);
            calib.apt["sens"](i) = tula::alg::median(det_sens);
            return 0;
        });

        // calculate weights, populate maps
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
            // calculate weights
            SPDLOG_INFO("calculating weights");
            ptcproc.calc_weights(ptcs[i], calib, telescope);

            // write ptc timestreams
            if (run_tod_output) {
                SPDLOG_INFO("writing ptcdata");
                if (tod_output_type == "ptc") {
                    ptcproc.append_to_netcdf(ptcs[i], tod_filename["I"], ptcs[i].det_indices.data,
                                             calib.apt);
                }
            }

            // populate maps
            SPDLOG_INFO("populating maps");
            mapmaking::populate_maps_naive(ptcs[i], omb, ptcs[i].map_indices.data,
                                           ptcs[i].det_indices.data, telescope.pixel_axes,
                                           redu_type, calib.apt, pointing_offsets_arcsec, telescope.d_fsmp);
            return 0;
        });

        // normalize maps
        SPDLOG_INFO("normalizing maps");
        omb.normalize_maps();

        SPDLOG_INFO("fitting maps");
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            auto array_index = ptcs[0].array_indices.data(i);
            auto init_fwhm = toltec_io.array_fwhm_arcsec[array_index]*ASEC_TO_RAD/omb.pixel_size_rad;
            auto [det_params, det_perror, good_fit] =
                map_fitter.fit_to_gaussian<engine_utils::mapFitter::centerValue>(omb.signal[i], omb.weight[i], init_fwhm);

            params.row(i) = det_params;
            perrors.row(i) = det_perror;

            SPDLOG_INFO("det_params {}, det_perror {} {}",det_params,det_perror,good_fit);
            return 0;});


        // increment loop iteration
        current_iter++;

        if (current_iter < beammap_iter_max) {
            // check if all detectors are converged
            if ((converged.array() == true).all()) {
                SPDLOG_INFO("all detectors converged");
                keep_going = false;
            }
            else {
                // loop through detectors and check if it is converged
                SPDLOG_INFO("checking convergennce");
                grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
                    if (!converged(i)) {
                        // get relative change from last iteration
                        auto diff = abs((params.col(i).array() - p0.col(i).array())/p0.col(i).array());
                        if ((diff.array() <= beammap_iter_tolerance).all()) {
                            // set as converged
                            converged(i) = true;
                            // set convergence iteration
                            converge_iter(i) = current_iter;
                        }
                    }
                    return 0;
                });

                SPDLOG_INFO("{} detectors convergennce", (converged.array() == true).count());

                // set previous iteration fits to current iteration fits
                p0 = params;
                perror0 = perrors;
            }
        }
        else {
            SPDLOG_INFO("max iteration reached");
            keep_going = false;
        }
    }
}

template <class KidsProc, class RawObs>
auto Beammap::timestream_pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                          std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>>> {

            // variable to hold current scan
            static auto scan = 0;
            while (scan < telescope.scan_indices.cols()) {
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

        run_timestream());
}

auto Beammap::loop_pipeline() {
    // run iterative stage
    run_loop();

    // empty initial ptcdata vector to save memory
    ptcs0.clear();

    // rescale fit params from pixel to on-sky units
    calib.apt["x_t"] = omb.pixel_size_rad*(params.col(1).array() - (omb.n_cols)/2)/ASEC_TO_RAD;
    calib.apt["y_t"] = omb.pixel_size_rad*(params.col(2).array() - (omb.n_rows)/2)/ASEC_TO_RAD;
    calib.apt["a_fwhm"] = STD_TO_FWHM*omb.pixel_size_rad*(params.col(3))/ASEC_TO_RAD;
    calib.apt["b_fwhm"] = STD_TO_FWHM*omb.pixel_size_rad*(params.col(4))/ASEC_TO_RAD;

    // rescale fit errors from pixel to on-sky units
    /*calib.apt["x_t_err"] = omb.pixel_size_rad*(perrors.col(1))/ASEC_TO_RAD;
    calib.apt["y_t_err"] = omb.pixel_size_rad*(perrors.col(2))/ASEC_TO_RAD;
    calib.apt["a_fwhm_err"] = STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(3))/ASEC_TO_RAD;
    calib.apt["b_fwhm_err"] = STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(4))/ASEC_TO_RAD;
    calib.apt["angle_err"] = STD_TO_FWHM*omb.pixel_size_rad*(params.col(4))/ASEC_TO_RAD;*/

    // array indices for current polarization
    auto array_indices = ptcs[0].array_indices.data;

    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        auto array_index = array_indices(i);
        std::string array_name = toltec_io.array_name_map[calib.apt["array"](array_index)];

        // calculate map standard deviation
        double map_std_dev = engine_utils::calc_std_dev(omb.signal[i]);

        // flag detectors with outler a_fwhm values
        if (params(i,3) < lower_fwhm_arcsec[array_name] || params(i,3) > upper_fwhm_arcsec[array_name]) {
            calib.apt["flag"](i) = 0;
        }
        // flag detectors with outler b_fwhm values
        if (params(i,4) < lower_fwhm_arcsec[array_name] || params(i,4) > upper_fwhm_arcsec[array_name]) {
            calib.apt["flag"](i) = 0;
        }

        // flag detectors with outler S/N values
        if (params(i,0)/map_std_dev < lower_sig2noise[array_name]) {
            calib.apt["flag"](i) = 0;
        }

        // get detector pointing
        Eigen::VectorXd lat = -(params(i,2)*ASEC_TO_RAD) + telescope.tel_data["TelElAct"].array();
        Eigen::VectorXd lon = -(params(i,1)*ASEC_TO_RAD) + telescope.tel_data["TelAzAct"].array();

        // index of minimum elevation distance
        Eigen::Index min_index;

        // minimum cartesian distance from source
        double min_dist = ((telescope.tel_data["SourceEl"] - lat).array().pow(2) +
                           (telescope.tel_data["SourceAz"] - lon).array().pow(2)).minCoeff(&min_index);

        // elevation at which we are on source
        derot_elev(i) = telescope.tel_data["TelElAct"](min_index);

        // rotation az/el angles
        double rot_azoff = cos(-derot_elev(i))*params(i,1) - sin(-derot_elev(i))*params(i,2);
        double rot_eloff = sin(-derot_elev(i))*params(i,1) + cos(-derot_elev(i))*params(i,2);

        // overwrite x_t and y_t
        params(i,1) = -rot_azoff;
        params(i,2) = -rot_eloff;

        return 0;
    });
}

template <class KidsProc, class RawObs>
void Beammap::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // run timestream pipeline
    timestream_pipeline(kidsproc, rawobs);

    // placeholder vectors of size nscans for grppi maps
    scan_in_vec.resize(ptcs0.size());
    std::iota(scan_in_vec.begin(), scan_in_vec.end(), 0);
    scan_out_vec.resize(ptcs0.size());

    // placeholder vectors of size ndet for grppi maps
    det_in_vec.resize(calib.n_dets);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    det_out_vec.resize(calib.n_dets);

    // run iterative pipeline
    loop_pipeline();
}

void Beammap::output() {
    SPDLOG_INFO("writing apt table");
    auto apt_filename = toltec_io.create_filename<engine_utils::toltecIO::apt, engine_utils::toltecIO::map>
                        (obsnum_dir_name, redu_type, "", obsnum, telescope.sim_obs);

    Eigen::MatrixXf apt_table(calib.n_dets, calib.apt_header_keys.size());

    Eigen::Index i = 0;
    for (auto const& x: calib.apt_header_keys) {
        apt_table.col(i) = calib.apt[x].cast<float> ();
        i++;
    }

    to_ecsv_from_matrix(apt_filename, apt_table, calib.apt_header_keys, calib.apt_header);

    SPDLOG_INFO("writing maps");
    for (Eigen::Index i=0; i<rtcproc.polarization.stokes_params.size(); i++) {
        for (Eigen::Index j=0; j<calib.n_arrays; j++) {
            auto array = calib.arrays[j];
            for (Eigen::Index k=0; k<calib.n_dets; k++) {
                if (calib.apt["array"](k) == array) {
                    fits_io_vec[array].add_hdu("signal_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i], omb.signal[k]);
                    fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);

                    fits_io_vec[array].add_hdu("weight_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i], omb.weight[k]);
                    fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);

                    if (rtcproc.run_kernel) {
                        fits_io_vec[array].add_hdu("kernel_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i],
                                                       omb.kernel[k]);
                        fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);
                    }

                    fits_io_vec[array].add_hdu("coverage_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i],
                                                   omb.coverage[k]);
                    fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);
                }
            }

            // add telescope file header information
            for (auto const& [key, val] : telescope.tel_header) {
                fits_io_vec[array].pfits->pHDU().addKey(key, val(0), key);
            }

            fits_io_vec[array].pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
            fits_io_vec[array].pfits->pHDU().addKey("CITLALI_GIT_VERSION", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");
        }
    }
}
