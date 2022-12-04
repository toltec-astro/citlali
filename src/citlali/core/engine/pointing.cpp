#include <citlali/core/engine/pointing.h>
#include <citlali/core/engine/kidsproc.h>

void Pointing::setup() {
    // populate ppt meta information

    ppt_meta["array"].push_back("units: N/A");
    ppt_meta["array"].push_back("array");

    ppt_meta["amp"].push_back("units: " + omb.sig_unit);
    ppt_meta["amp"].push_back("fitted amplitude");

    ppt_meta["amp_err"].push_back("units: " + omb.sig_unit);
    ppt_meta["amp_err"].push_back("fitted amplitude error");

    ppt_meta["x_t"].push_back("units: arcsec");
    ppt_meta["x_t"].push_back("fitted azimuthal offset");

    ppt_meta["x_t_err"].push_back("units: arcsec");
    ppt_meta["x_t_err"].push_back("fitted azimuthal offset error");

    ppt_meta["y_t"].push_back("units: arcsec");
    ppt_meta["y_t"].push_back("fitted altitude offset");

    ppt_meta["y_t_err"].push_back("units: arcsec");
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

    // set number of parameters for map fitting
    n_params = 6;

    // resize the current fit matrix
    params.setZero(n_maps, n_params);
    perrors.setZero(n_maps, n_params);

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

            // remove flagged dets
            SPDLOG_INFO("removing flagged dets");
            ptcproc.remove_flagged_dets(ptcdata, calib.apt, det_indices);

            // run cleaning
            if (stokes_param == "I") {
                SPDLOG_INFO("ptcproc");
                ptcproc.run(ptcdata, ptcdata, calib);
            }
            // calculate weights
            SPDLOG_INFO("calculating weights");
            ptcproc.calc_weights(ptcdata, calib, telescope);

            // remove outliers
            SPDLOG_INFO("removing outlier weights");
            //ptcproc.remove_bad_dets(ptcdata, calib.apt, det_indices);

            // write ptc timestreams
            if (run_tod_output) {
                SPDLOG_INFO("writing ptcdata");
                if (tod_output_type == "ptc") {
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], det_indices, calib.apt);
                }
            }

            // populate maps
            SPDLOG_INFO("populating maps");
            mapmaking::populate_maps_naive(ptcdata, omb, map_indices, det_indices, telescope.pixel_axes,
                                           redu_type, calib.apt, pointing_offsets_arcsec, telescope.d_fsmp);

        }

        return ptcdata;
    });

    return farm;
}

template <class KidsProc, class RawObs>
void Pointing::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
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

        run());

    // normalize maps
    omb.normalize_maps();

    // fit maps
    for (Eigen::Index i=0; i<n_maps; i++) {
        //auto array_index = ptcs[stokes_param][0].array_indices.data(i);
        auto init_fwhm = toltec_io.array_fwhm_arcsec[i]*ASEC_TO_RAD/omb.pixel_size_rad;
        auto [det_params, det_perror, good_fit] =
            map_fitter.fit_to_gaussian<engine_utils::mapFitter::centerValue>(omb.signal[i], omb.weight[i], init_fwhm);
        params.row(i) = det_params;
        perrors.row(i) = det_perror;

        SPDLOG_INFO("params {} perrors {}", params, perrors);

        if (good_fit) {
            // rescale fit params from pixel to on-sky units
            params(i,1) = omb.pixel_size_rad*(params(i,1) - (omb.n_cols)/2)*RAD_TO_ASEC;
            params(i,2) = omb.pixel_size_rad*(params(i,2) - (omb.n_rows)/2)*RAD_TO_ASEC;
            params(i,3) = STD_TO_FWHM*omb.pixel_size_rad*(params(i,3))*RAD_TO_ASEC;
            params(i,4) = STD_TO_FWHM*omb.pixel_size_rad*(params(i,4))*RAD_TO_ASEC;

            // rescale fit errors from pixel to on-sky units
            perrors(i,1) = omb.pixel_size_rad*(perrors(i,1))*RAD_TO_ASEC;
            perrors(i,2) = omb.pixel_size_rad*(perrors(i,2))*RAD_TO_ASEC;
            perrors(i,3) = STD_TO_FWHM*omb.pixel_size_rad*(perrors(i,3))*RAD_TO_ASEC;
            perrors(i,4) = STD_TO_FWHM*omb.pixel_size_rad*(perrors(i,4))*RAD_TO_ASEC;
        }
    }
}

void Pointing::output() {
    SPDLOG_INFO("writing ppt table");
    auto ppt_filename = toltec_io.create_filename<engine_utils::toltecIO::ppt, engine_utils::toltecIO::map>
                        (obsnum_dir_name, redu_type, "", obsnum, telescope.sim_obs);

    Eigen::MatrixXf ppt_table(n_maps, 2*n_params + 1);

    for (const auto &stokes_param: rtcproc.polarization.stokes_params) {
        for (Eigen::Index i=0; i<calib.arrays.size(); i++) {
            ppt_table(i,0) = calib.arrays(i);
        }
    }

    Eigen::Index j = 0;
    for (Eigen::Index i=1; i<2*n_params; i=i+2) {
        ppt_table.col(i) = params.col(j).cast <float> ();
        ppt_table.col(i+1) = perrors.col(j).cast <float> ();
        j++;
    }

    to_ecsv_from_matrix(ppt_filename, ppt_table, ppt_header, ppt_meta);

    SPDLOG_INFO("writing maps");
    Eigen::Index k = 0;
    for (Eigen::Index i=0; i<rtcproc.polarization.stokes_params.size(); i++) {
        for (Eigen::Index j=0; j<n_maps/rtcproc.polarization.stokes_params.size(); j++) {
            write_maps(i,j,k);
            k++;
        }
    }
}
