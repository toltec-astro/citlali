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
        "angle_err"
    };

    void setup();
    auto run();

    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    template <mapmaking::MapType map_type>
    void output();
};

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
    create_map_files();
    // create timestream files
    if (run_tod_output) {
        create_tod_files();
    }

    // set center pointing
    if (telescope.pixel_axes == "ircs") {
        omb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0);
        omb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0);

        cmb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0);
        cmb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0);
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
            SPDLOG_INFO("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

            SPDLOG_INFO("reducing {} timestream",stokes_param);
            // create a new rtcdata for each polarization
            TCData<TCDataKind::RTC,Eigen::MatrixXd> rtcdata_pol;
            // demodulate
            SPDLOG_INFO("demodulating polarization");
            SPDLOG_INFO("array_limits {}", calib.array_limits);

            auto [array_indices, nw_indices, det_indices] = rtcproc.polarization.demodulate_timestream(rtcdata, rtcdata_pol,
                                                                                                       stokes_param,
                                                                                                       redu_type, calib);
            // get indices for maps
            SPDLOG_INFO("calculating map indices");
            SPDLOG_INFO("array_limits {}", calib.array_limits);

            auto map_indices = calc_map_indices(det_indices, nw_indices, array_indices, stokes_param);

            // run rtcproc
            SPDLOG_INFO("rtcproc");
            SPDLOG_INFO("array_limits {}", calib.array_limits);

            rtcproc.run(rtcdata_pol, ptcdata, telescope.pixel_axes, redu_type, calib, telescope, pointing_offsets_arcsec,
                        det_indices, array_indices, map_indices, omb.pixel_size_rad);

            SPDLOG_INFO("scans before clean {}", ptcdata.scans.data);
            SPDLOG_INFO("scans max before clean {}", ptcdata.scans.data.maxCoeff());
            SPDLOG_INFO("scans min before clean {}", ptcdata.scans.data.minCoeff());
            SPDLOG_INFO("flags before clean {}", ptcdata.flags.data);

            // write rtc timestreams
            if (run_tod_output) {
                if (tod_output_type == "rtc") {
                    SPDLOG_INFO("writing rtcdata");
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], redu_type, telescope.pixel_axes, pointing_offsets_arcsec,
                                             det_indices, calib.apt, tod_output_type, verbose_mode, telescope.d_fsmp);
                }
            }

            // remove flagged dets
            SPDLOG_INFO("removing flagged dets");

            SPDLOG_INFO("array_limits {}", calib.array_limits);

            ptcproc.remove_flagged_dets(ptcdata, calib.apt, det_indices);

            // remove outliers
            SPDLOG_INFO("removing outlier weights");
            //ptcproc.remove_bad_dets(ptcdata, calib.apt, det_indices);
            auto calib_scan = ptcproc.remove_bad_dets_nw(ptcdata, calib, det_indices, nw_indices, array_indices);
            //map_indices = calc_map_indices(det_indices, nw_indices, array_indices, stokes_param);

            // run cleaning
            if (stokes_param == "I") {
                SPDLOG_INFO("ptcproc");
                ptcproc.run(ptcdata, ptcdata, calib);
                //ptcproc.run(ptcdata, ptcdata, calib_scan);
            }

            // calculate weights
            SPDLOG_INFO("calculating weights");
            ptcproc.calc_weights(ptcdata, calib.apt, telescope);
            //ptcproc.calc_weights(ptcdata, calib_scan.apt, telescope);

            // write ptc timestreams
            if (run_tod_output) {
                if (tod_output_type == "ptc") {
                    SPDLOG_INFO("writing ptcdata");
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], redu_type, telescope.pixel_axes, pointing_offsets_arcsec,
                                             det_indices, calib.apt, tod_output_type, verbose_mode, telescope.d_fsmp);
                }
            }
            SPDLOG_INFO("scans after{}", ptcdata.scans.data);
            SPDLOG_INFO("scans max after clean {}", ptcdata.scans.data.maxCoeff());
            SPDLOG_INFO("scans min after clean {}", ptcdata.scans.data.minCoeff());
            SPDLOG_INFO("flags after {}", ptcdata.flags.data);
            SPDLOG_INFO("weight after {}", ptcdata.weights.data);

            // populate maps
            SPDLOG_INFO("populating maps");
            mapmaking::populate_maps_naive(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
                                           redu_type, calib.apt, pointing_offsets_arcsec, telescope.d_fsmp, run_noise);
            //mapmaking::populate_maps_naive(ptcdata, omb, cmb, map_indices, det_indices, telescope.pixel_axes,
            //                               redu_type, calib_scan.apt, pointing_offsets_arcsec, telescope.d_fsmp, run_noise);
            SPDLOG_INFO("signal map {}", omb.signal);
            SPDLOG_INFO("cov map {}", omb.coverage);
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

                SPDLOG_INFO("telescope.scan_indices.col(scan) {}",telescope.scan_indices);

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

    // calculate map psds
    SPDLOG_INFO("calc_map_psd");
    omb.calc_map_psd();
    // calculate map histograms
    SPDLOG_INFO("calc_map_hist");
    omb.calc_map_hist();

    // fit maps
    for (Eigen::Index i=0; i<n_maps; i++) {
        //auto array_index = ptcs[stokes_param][0].array_indices.data(i);
        auto init_fwhm = toltec_io.array_fwhm_arcsec[i]*ASEC_TO_RAD/omb.pixel_size_rad;
        auto [det_params, det_perror, good_fit] =
            map_fitter.fit_to_gaussian<engine_utils::mapFitter::peakValue>(omb.signal[i], omb.weight[i], init_fwhm);
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

template <mapmaking::MapType map_type>
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
            SPDLOG_INFO("i {}, j {}, k {}",i,j,k);
            if constexpr (map_type == mapmaking::Obs) {
                write_maps(fits_io_vec,omb,i,j,j);
            }
            else if constexpr (map_type == mapmaking::Coadd) {
                write_maps(coadd_fits_io_vec,cmb,i,j,j);
            }

            k++;
        }
    }

    // empty fits vector
    fits_io_vec.clear();

    SPDLOG_INFO("done with writing maps");

    write_psd();
    SPDLOG_INFO("done with psd");
    write_hist();
    SPDLOG_INFO("done with hist");
}
