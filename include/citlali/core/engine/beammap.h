#pragma once

#include <map>
#include <vector>
#include <string>

#include <citlali/core/engine/engine.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Beammap: public Engine {
public:
    // vector to store each scan's PTCData
    std::vector<TCData<TCDataKind::PTC,Eigen::MatrixXd>> ptcs0, ptcs;

    // beammap iteration parameters
    Eigen::Index current_iter;

    // vector for convergence check
    Eigen::Matrix<bool, Eigen::Dynamic, 1> converged;

    // vector to record convergence iteration
    Eigen::Vector<int, Eigen::Dynamic> converge_iter;

    // number of parameters for map fitting
    Eigen::Index n_params;

    // previous iteration fit parameters
    Eigen::MatrixXd p0;

    // current iteration fit parameters
    Eigen::MatrixXd params;

    // previous iteration fit errors
    Eigen::MatrixXd perror0;

    // current iteration fit errors
    Eigen::MatrixXd perrors;

    // good fits
    Eigen::Matrix<bool, Eigen::Dynamic, 1> good_fits;

    // placeholder vectors for grppi maps
    std::vector<int> scan_in_vec, scan_out_vec;
    std::vector<int> det_in_vec, det_out_vec;

    void setup();
    auto run_timestream();
    auto run_loop();

    template <class KidsProc, class RawObs>
    auto timestream_pipeline(KidsProc &, RawObs &);

    template<typename array_indices_t, typename nw_indices_t>
    void flag_dets(array_indices_t &, nw_indices_t &);
    void adjust_apt();

    auto loop_pipeline();

    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    template <mapmaking::MapType map_type>
    void output();
};

void Beammap::setup() {

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

    // create kids tone apt row
    calib.apt["kids_tone"].resize(calib.n_dets);

    Eigen::Index j = 0;
    calib.apt["kids_tone"](0) = 0;
    for (Eigen::Index i=1; i<calib.n_dets; i++) {
        if (calib.apt["nw"](i) > calib.apt["nw"](i-1)) {
            j = 0;
        }

        else {
            j++;
        }

        calib.apt["kids_tone"](i) = j;
    }

    calib.apt_header_keys.push_back("kids_tone");

    // set number of parameters for map fitting
    n_params = 6;

    // resize the PTCData vector to number of scans
    ptcs0.resize(telescope.scan_indices.cols());

    // resize the initial fit matrix
    p0.setZero(n_maps, n_params);
    // resize the initial fit error matrix
    perror0.setZero(n_maps, n_params);
    // resize the current fit matrix
    params.setZero(n_maps, n_params);
    perrors.setZero(n_maps, n_params);

    // resize good fits
    good_fits.setZero(n_maps);

    // initially all detectors are unconverged
    converged.setZero(n_maps);
    // convergence iteration
    converge_iter.resize(n_maps);
    converge_iter.setConstant(1);
    // set the initial iteration
    current_iter = 0;

    // setup kernel
    if (rtcproc.run_kernel) {
        rtcproc.kernel.setup(n_maps);
    }

    // set despiker sample rate
    rtcproc.despiker.fsmp = telescope.fsmp;

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
            fs::create_directories(obsnum_dir_name + "raw/" + tod_output_subdir_name);
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

    // tod output mode require sequential policy so set explicitly
    if (run_tod_output || verbose_mode) {
        SPDLOG_WARN("tod output mode require sequential policy");
        parallel_policy = "seq";
    }

    /* update apt table meta data */

    // add obsnum to meta data
    calib.apt_meta["obsnum"] = obsnum;

    // add source name
    calib.apt_meta["Source"] = telescope.source_name;

    // add date
    calib.apt_meta["Date"] = engine_utils::current_date_time();

    // detector id
    calib.apt_meta["uid"].push_back("units: N/A");
    calib.apt_meta["uid"].push_back("unique id");

    // tone freq
    calib.apt_meta["tone_freq"].push_back("units: Hz");
    calib.apt_meta["tone_freq"].push_back("tone frequency");

    // detector array
    calib.apt_meta["array"].push_back("units: N/A");
    calib.apt_meta["array"].push_back("array index");

    // detector nw
    calib.apt_meta["nw"].push_back("units: N/A");
    calib.apt_meta["nw"].push_back("network index");

    // detector freq group
    calib.apt_meta["fg"].push_back("units: N/A");
    calib.apt_meta["fg"].push_back("frequency group");

    // detector polarization
    calib.apt_meta["pg"].push_back("units: N/A");
    calib.apt_meta["pg"].push_back("polarization group");

    // detector orientation
    calib.apt_meta["ori"].push_back("units: N/A");
    calib.apt_meta["ori"].push_back("orientation");

    // detector responsivity
    calib.apt_meta["responsivity"].push_back("units: N/A");
    calib.apt_meta["responsivity"].push_back("responsivity");

    // detector flux scale
    calib.apt_meta["flxscale"].push_back("units: mJy/beam/" + omb.sig_unit);
    calib.apt_meta["flxscale"].push_back("flux conversion scale");

    // detector sensitivity
    calib.apt_meta["sens"].push_back("units: V/Hz^-0.5");
    calib.apt_meta["sens"].push_back("sensitivity");

    // detector derotation elevation
    calib.apt_meta["derot_elev"].push_back("units: radians");
    calib.apt_meta["derot_elev"].push_back("derotation elevation angle");

    // detector amplitude
    calib.apt_meta["amp"].push_back("units: N/A");
    calib.apt_meta["amp"].push_back("fitted amplitude");

    // detector amplitude error
    calib.apt_meta["amp_err"].push_back("units: N/A");
    calib.apt_meta["amp_err"].push_back("fitted amplitude error");

    // detector x position
    calib.apt_meta["x_t"].push_back("units: arcsec");
    calib.apt_meta["x_t"].push_back("fitted azimuthal offset");

    // detector x position error
    calib.apt_meta["x_t_err"].push_back("units: arcsec");
    calib.apt_meta["x_t_err"].push_back("fitted azimuthal offset error");

    // detector y position
    calib.apt_meta["y_t"].push_back("units: arcsec");
    calib.apt_meta["y_t"].push_back("fitted altitude offset");

    // detector y position error
    calib.apt_meta["y_t_err"].push_back("units: arcsec");
    calib.apt_meta["y_t_err"].push_back("fitted altitude offset error");

    // detector x fwhm
    calib.apt_meta["a_fwhm"].push_back("units: arcsec");
    calib.apt_meta["a_fwhm"].push_back("fitted azimuthal FWHM");

    // detector x fwhm error
    calib.apt_meta["a_fwhm_err"].push_back("units: arcsec");
    calib.apt_meta["a_fwhm_err"].push_back("fitted azimuthal FWHM error");

    // detector y fwhm
    calib.apt_meta["b_fwhm"].push_back("units: arcsec");
    calib.apt_meta["b_fwhm"].push_back("fitted altitude FWMH");

    // detector y fwhm error
    calib.apt_meta["b_fwhm_err"].push_back("units: arcsec");
    calib.apt_meta["b_fwhm_err"].push_back("fitted altitude FWMH error");

    // detector rotation angle
    calib.apt_meta["angle"].push_back("units: radians");
    calib.apt_meta["angle"].push_back("fitted rotation angle");

    // detector rotation angle error
    calib.apt_meta["angle_err"].push_back("units: radians");
    calib.apt_meta["angle_err"].push_back("fitted rotation angle error");

    // detector convergence iteration
    calib.apt_meta["converge_iter"].push_back("units: N/A");
    calib.apt_meta["converge_iter"].push_back("beammap convergence iteration");

    // detector flag
    calib.apt_meta["flag"].push_back("units: N/A");
    calib.apt_meta["flag"].push_back("good detector");

    // kids tone
    calib.apt_meta["kids_tone"].push_back("units: N/A");
    calib.apt_meta["kids_tone"].push_back("index of tone in network");

    // detector map signal-to-noise
    calib.apt_meta["sig2noise"].push_back("units: N/A");
    calib.apt_meta["sig2noise"].push_back("signal to noise");

    // is the detector rotated
    calib.apt_meta["is_derotated"] = beammap_derotate;
    // is the detector rotated
    calib.apt_meta["reference_detector_subtracted"] = beammap_subtract_reference;
    // reference detector
    calib.apt_meta["reference_det"] = beammap_reference_det;

    // print basic info for obs reduction
    print_summary();
}

auto Beammap::run_timestream() {
    auto farm = grppi::farm(n_threads,[&](auto &input_tuple) -> TCData<TCDataKind::PTC,Eigen::MatrixXd> {
        // RTCData input
        auto rtcdata = std::get<0>(input_tuple);
        // kidsproc
        auto kidsproc = std::get<1>(input_tuple);
        // start index input
        auto scan_rawobs = std::get<2>(input_tuple);

        // get tone flags
        Eigen::Index j = 0;
        Eigen::VectorXd tone_flags(calib.n_dets);
        for (Eigen::Index i=0; i<scan_rawobs.size(); i++) {
            auto tone_axis = scan_rawobs[i].wcs.tone_axis("flag");
            tone_flags.segment(j,tone_axis.size()) = tone_axis;
            j = j + tone_axis.size();
        }

        rtcdata.flags2.data.setConstant(timestream::TimestreamFlags::Good);

        // starting index for scan (outer scan)
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
            //tula::logging::scoped_loglevel<spdlog::level::off> _0;
            rtcdata.scans.data = kidsproc.populate_rtc(scan_rawobs,rtcdata.scan_indices.data, sl, calib.n_dets, tod_type);
        }

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;

        // loop through polarizations
        for (const auto &[stokes_index,stokes_param]: rtcproc.polarization.stokes_params) {
            SPDLOG_INFO("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

            SPDLOG_INFO("reducing {} timestream",stokes_param);
            // create a new rtcdata for each polarization
            TCData<TCDataKind::RTC,Eigen::MatrixXd> rtcdata_pol;
            // demodulate
            auto [array_indices, nw_indices, det_indices] = rtcproc.polarization.demodulate_timestream(rtcdata, rtcdata_pol,
                                                                                                       stokes_param,
                                                                                                       redu_type, calib);
            // get indices for maps
            SPDLOG_INFO("calculating map indices");
            auto map_indices = calc_map_indices(det_indices, nw_indices, array_indices, stokes_param);

            // run rtcproc
            SPDLOG_INFO("raw time chunk processing");
            rtcproc.run(rtcdata_pol, ptcdata, telescope.pixel_axes, redu_type, calib, telescope, pointing_offsets_arcsec,
                        det_indices, array_indices, map_indices, omb.pixel_size_rad);

            // write rtc timestreams
            if (run_tod_output) {
                if (tod_output_type == "rtc" || tod_output_type=="both") {
                    SPDLOG_INFO("writing raw time chunk");
                    ptcproc.append_to_netcdf(ptcdata, tod_filename["rtc_" + stokes_param], redu_type, telescope.pixel_axes,
                                             pointing_offsets_arcsec, det_indices, calib.apt, tod_output_type, verbose_mode, telescope.d_fsmp);
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

        n_scans_done++;
        SPDLOG_INFO("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

        return ptcdata;
    });

    return farm;
}

auto Beammap::run_loop() {
    // variable to control iteration
    bool keep_going = true;

    // iterative loop
    while (keep_going) {
        // copy ptcs
        ptcs = ptcs0;
        // set maps to zero for each iteration
        for (Eigen::Index i=0; i<n_maps; i++) {
            omb.signal[i].setZero();
            omb.weight[i].setZero();

            if (!omb.coverage.empty()) {
                omb.coverage[i].setZero();
            }

            if (rtcproc.run_kernel) {
                omb.kernel[i].setZero();
            }
        }

        // progress bar
        tula::logging::progressbar pb(
            [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 100, "PTC progress ");

        // cleaning
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {

            if (run_mapmaking) {
                // subtract gaussian
                if (current_iter > 0) {
                    SPDLOG_INFO("subtracting gaussian from tod");
                    ptcproc.add_gaussian(ptcs[i], params, telescope.pixel_axes, map_grouping, calib.apt, pointing_offsets_arcsec,
                                         omb.pixel_size_rad, omb.n_rows, omb.n_cols, ptcs[i].map_indices.data, ptcs[i].det_indices.data,
                                         "subtract");
                }
            }

            // subtract scan means
            SPDLOG_INFO("subtracting detector means");
            ptcproc.subtract_mean(ptcs[i]);

            if (map_grouping!="detector") {
                SPDLOG_INFO("removing flagged dets");
                ptcproc.remove_flagged_dets(ptcs[i], calib.apt, ptcs[i].det_indices.data);
            }

            auto calib_scan = rtcproc.remove_bad_dets(ptcs[i], calib, ptcs[i].det_indices.data, ptcs[i].nw_indices.data,
                                                      ptcs[i].array_indices.data, redu_type, map_grouping);

            // remove duplicate tones
            calib_scan = rtcproc.remove_nearby_tones(ptcs[i], calib_scan, ptcs[i].det_indices.data, ptcs[i].nw_indices.data,
                                                     ptcs[i].array_indices.data, redu_type, map_grouping);

            SPDLOG_INFO("processed time chunk processing");
            ptcproc.run(ptcs[i], ptcs[i], calib_scan);

            // remove outliers after clean
            calib_scan = ptcproc.remove_bad_dets(ptcs[i], calib, ptcs[i].det_indices.data, ptcs[i].nw_indices.data,
                                                 ptcs[i].array_indices.data, redu_type, map_grouping);
            // write chunk summary
            if (verbose_mode && current_iter==0) {
                SPDLOG_DEBUG("writing chunk summary");
                write_chunk_summary(ptcs[i]);
            }

            if (run_mapmaking) {
                // add gaussan back
                if (current_iter > 0) {
                    SPDLOG_INFO("adding gaussian to tod");
                    ptcproc.add_gaussian(ptcs[i], params, telescope.pixel_axes, map_grouping, calib.apt, pointing_offsets_arcsec,
                                         omb.pixel_size_rad, omb.n_rows, omb.n_cols, ptcs[i].map_indices.data, ptcs[i].det_indices.data,
                                         "add");
                }
            }

            // update progress bar
            pb.count(telescope.scan_indices.cols(), 1);

            // set weights to a constant value
            ptcs[i].weights.data.resize(ptcs[i].scans.data.cols());
            ptcs[i].weights.data.setOnes();

            // write ptc timestreams
            if (run_tod_output) {
                if (tod_output_type == "ptc" || tod_output_type=="both") {
                    SPDLOG_INFO("writing processed time chunk");
                    if (current_iter == beammap_tod_output_iter) {
                        // hardcoded to stokes I for now
                        ptcproc.append_to_netcdf(ptcs[i], tod_filename["ptc_I"], redu_type, telescope.pixel_axes,
                                                 pointing_offsets_arcsec, ptcs[i].det_indices.data, calib.apt, "ptc", verbose_mode,
                                                 telescope.d_fsmp);
                    }
                }
            }

            // populate maps
            if (run_mapmaking) {
                if (map_method=="naive") {
                    mapmaking::populate_maps_naive(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                                   ptcs[i].det_indices.data, telescope.pixel_axes,
                                                   redu_type, calib.apt, pointing_offsets_arcsec,
                                                   telescope.d_fsmp, run_noise);
                }

                else if (map_method=="jinc") {
                    mapmaking::populate_maps_jinc(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                                  ptcs[i].det_indices.data, telescope.pixel_axes,
                                                  redu_type, calib.apt, pointing_offsets_arcsec,
                                                  telescope.d_fsmp, run_noise, jinc_r_max, jinc_a,
                                                  jinc_b, jinc_c);
                }
            }
            return 0;
        });

        if (run_mapmaking) {
            // normalize maps
            SPDLOG_INFO("normalizing maps");
            omb.normalize_maps();

            SPDLOG_INFO("fitting maps");
            grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
                // only fit if not converged
                if (!converged(i)) {
                    // get array number
                    auto array_index = ptcs[0].array_indices.data(i);
                    // get initial guess fwhm from theoretical fwhms for the arrays
                    auto init_fwhm = toltec_io.array_fwhm_arcsec[array_index]*ASEC_TO_RAD/omb.pixel_size_rad;
                    auto [det_params, det_perror, good_fit] =
                        map_fitter.fit_to_gaussian<engine_utils::mapFitter::peakValue>(omb.signal[i], omb.weight[i], init_fwhm);

                    params.row(i) = det_params;
                    perrors.row(i) = det_perror;
                    good_fits(i) = good_fit;
                }
                // otherwise keep value from previous iteration
                else {
                    params.row(i) = p0.row(i);
                    perrors.row(i) = perror0.row(i);
                }

                return 0;}
            );

            SPDLOG_INFO("max good fits {} {}", good_fits.maxCoeff(), good_fits.minCoeff());
            SPDLOG_INFO("number of good beammap fits {}/{}", good_fits.cast<double>().sum(), n_maps);
        }

        // increment loop iteration
        current_iter++;

        if (current_iter < beammap_iter_max) {
            // check if all detectors are converged
            if ((converged.array() == true).all()) {
                SPDLOG_INFO("all detectors converged");
                keep_going = false;
            }
            else if (current_iter > 1) {
                // only do convergence test if tolerance is above zero, otherwise run all iterations
                if (beammap_iter_tolerance > 0) {
                    // loop through detectors and check if it is converged
                    SPDLOG_INFO("checking convergence");
                    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
                        if (!converged(i)) {
                            // get relative change from last iteration
                            auto diff = abs((params.row(i).array() - p0.row(i).array())/p0.row(i).array());
                            // if variable is constant, make sure no nans are present
                            auto d = (diff.array()).isNaN().select(0,diff);
                            if ((d.array() <= beammap_iter_tolerance).all()) {
                                // set as converged
                                converged(i) = true;
                                // set convergence iteration
                                converge_iter(i) = current_iter;
                            }
                        }
                        return 0;
                    });
                }

                SPDLOG_INFO("{} detectors converged", (converged.array() == true).count());
            }

            // set previous iteration fits to current iteration fits
            p0 = params;
            perror0 = perrors;
        }
        else {
            SPDLOG_INFO("max iteration reached");
            keep_going = false;
        }
    }
}

template <class KidsProc, class RawObs>
auto Beammap::timestream_pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // initialize number of completed scans
    n_scans_done = 0;

    // progress bar
    tula::logging::progressbar pb(
        [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 100, "RTC progress ");

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
        run_timestream());
}

template<typename array_indices_t, typename nw_indices_t>
void Beammap::flag_dets(array_indices_t &array_indices, nw_indices_t &nw_indices) {
    // mean values of x_t and y_t
    std::map<std::string, double> array_mean_x_t, array_mean_y_t;
    std::map<Eigen::Index, double> nw_mean_sens;

    // calc median x_t and y_t values for arrays
    for (Eigen::Index i=0; i<calib.n_arrays; i++) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // x_t
        array_mean_x_t[array_name] = tula::alg::median(calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                                   std::get<1>(calib.array_limits[array])-1)));
        // y_t
        array_mean_y_t[array_name] = tula::alg::median(calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                 std::get<1>(calib.array_limits[array])-1)));
    }

    // calc median sensitivity for each network
    for (Eigen::Index i=0; i<calib.n_nws; i++) {
        Eigen::Index nw = calib.nws(i);
        nw_mean_sens[nw] = tula::alg::median(calib.apt["sens"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                                         std::get<1>(calib.nw_limits[nw])-1)));
    }

    // track number of flagged detectors
    int n_flagged_dets = 0;

    // mean elevation for tau calc
    double mean_elev = telescope.tel_data["TelElAct"].mean();

    // calculate extinction in each waveband
    Eigen::VectorXd tau_el(1);
    tau_el << mean_elev;
    auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);

    SPDLOG_INFO("flagging detectors");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = array_indices(i);
        std::string array_name = toltec_io.array_name_map[array_index];

        // get nw of current detector
        auto nw_index = nw_indices(i);

        // calculate map standard deviation
        double map_std_dev = engine_utils::calc_std_dev(omb.signal[i]);

        // set apt signal to noise
        calib.apt["sig2noise"](i) = params(i,0)/map_std_dev;

        // calculate distance of detector from mean position of all detectors
        double dist = sqrt(pow(calib.apt["x_t"](i) - array_mean_x_t[array_name],2) +
                           pow(calib.apt["y_t"](i) - array_mean_y_t[array_name],2));

        // flag bad fits
        if (!good_fits(i)) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag detectors with outler a_fwhm values
        else if (calib.apt["a_fwhm"](i) < lower_fwhm_arcsec[array_name] ||
                 (calib.apt["a_fwhm"](i) > upper_fwhm_arcsec[array_name]) && upper_fwhm_arcsec[array_name] > 0) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag detectors with outler b_fwhm values
        else if (calib.apt["b_fwhm"](i) < lower_fwhm_arcsec[array_name] ||
                 (calib.apt["b_fwhm"](i) > upper_fwhm_arcsec[array_name] && upper_fwhm_arcsec[array_name] > 0)) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag detectors with outler S/N values
        else if (params(i,0)/map_std_dev < lower_sig2noise[array_name]) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag detectors that are further than the mean value than the distance limit
        else if (dist > max_dist_arcsec[array_name] && max_dist_arcsec[array_name] > 0) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag outlier sensitivities
        else if (calib.apt["sens"](i) < lower_sens_factor*nw_mean_sens[nw_index] ||
                 (calib.apt["sens"](i) > upper_sens_factor*nw_mean_sens[nw_index] && upper_sens_factor > 0)) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }

        // calc flux scale (always in mJy/beam)
        if (params(i,0) != 0 && calib.apt["flag"](i) != 0) {
            calib.apt["flxscale"](i) = beammap_fluxes_mJy_beam[array_name]/params(i,0);
        }

        // set fluxscale (fcf) to zero if flagged
        else {
            calib.apt["flxscale"](i) = 0;
        }

        // correct fcf for extinction
        calib.apt["flxscale"](i) = calib.apt["flxscale"](i)*exp(-tau_freq[array_index](0));
        return 0;
    });

    // print number of flagged detectors
    SPDLOG_INFO("{} detectors were flagged", n_flagged_dets);

    // get average fwhms and beam areas
    calib.setup();

    // calculate source flux in MJy/Sr from average beamsizes
    for (Eigen::Index i=0; i<calib.n_arrays; i++) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // get source flux in MJy/Sr
        beammap_fluxes_MJy_Sr[array_name] = mJY_ASEC_to_MJY_SR*(beammap_fluxes_mJy_beam[array_name])/calib.array_beam_areas[array];
    }
}

void Beammap::adjust_apt() {
    // std maps to hold median unflagged x and y positions
    std::map<std::string, double> array_median_x_t, array_median_y_t;
    Eigen::VectorXd x_t, y_t;

    // calc mean x_t and y_t values from unflagged detectors for each arrays
    for (Eigen::Index i=0; i<calib.n_arrays; i++) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // x_t
        auto array_x_t = calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                 std::get<1>(calib.array_limits[array])-1));
        // y_t
        auto array_y_t = calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                 std::get<1>(calib.array_limits[array])-1));

        // number of good detectors
        Eigen::Index n_good_det = calib.apt["flag"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                         std::get<1>(calib.array_limits[array])-1)).sum();

        x_t.resize(n_good_det);
        y_t.resize(n_good_det);

        // remove flagged dets
        Eigen::Index j = std::get<0>(calib.array_limits[array]);
        Eigen::Index k = 0;
        for (Eigen::Index i=0; i<array_x_t.size(); i++) {
            if (calib.apt["flag"](j)) {
                x_t(k) = array_x_t(i);
                y_t(k) = array_y_t(i);
                k++;
            }
            j++;
        }
        array_median_x_t[array_name] = tula::alg::median(x_t);
        array_median_y_t[array_name] = tula::alg::median(y_t);
    }

    // reference detector x and y
    double ref_det_x_t = 0;
    double ref_det_y_t = 0;

    // if particular reference detector is requested
    if (beammap_subtract_reference) {
        if (beammap_reference_det >= 0) {
            // set reference x_t and y_t
            ref_det_x_t = calib.apt["x_t"](beammap_reference_det);
            ref_det_y_t = calib.apt["y_t"](beammap_reference_det);
        }
        // else use closest to a1100 median
        else {
            auto array_name = toltec_io.array_name_map[calib.apt["array"](0)];
            Eigen::VectorXd dist = pow(calib.apt["x_t"].array() - array_median_x_t[array_name],2) +
                pow(calib.apt["y_t"].array() - array_median_y_t[array_name],2);

            // index of detector closest to median
            auto min_dist = dist.minCoeff(&beammap_reference_det);

            // set reference x_t and y_t
            ref_det_x_t = calib.apt["x_t"](beammap_reference_det);
            ref_det_y_t = calib.apt["y_t"](beammap_reference_det);
        }
    }

    // add reference detector to APT meta data
    calib.apt_meta["reference_x_t"] = ref_det_x_t;
    calib.apt_meta["reference_y_t"] = ref_det_y_t;

    // align to reference detector if specified and
    // subtract its position from x and y
    calib.apt["x_t"] =  calib.apt["x_t"].array() - ref_det_x_t;
    calib.apt["y_t"] =  calib.apt["y_t"].array() - ref_det_y_t;

    // derotated detector x and y values
    calib.apt["x_t_derot"] = calib.apt["x_t"];
    calib.apt["y_t_derot"] = calib.apt["y_t"];

    // raw (not derotated) detector x and y values
    calib.apt["x_t_raw"] = calib.apt["x_t"];
    calib.apt["y_t_raw"] = calib.apt["y_t"];

    // derotation elevation
    calib.apt["derot_elev"].setConstant(telescope.tel_data["TelElAct"].mean());

    // calculate derotated positions
    Eigen::VectorXd rot_az_off = cos(-calib.apt["derot_elev"].array())*calib.apt["x_t_derot"].array() -
                                 sin(-calib.apt["derot_elev"].array())*calib.apt["y_t_derot"].array();
    Eigen::VectorXd rot_alt_off = sin(-calib.apt["derot_elev"].array())*calib.apt["x_t_derot"].array() +
                                  cos(-calib.apt["derot_elev"].array())*calib.apt["y_t_derot"].array();

    // overwrite x_t and y_t
    calib.apt["x_t_derot"] = -rot_az_off;
    calib.apt["y_t_derot"] = -rot_alt_off;

    if (beammap_derotate) {
        SPDLOG_INFO("derotating detectors");
        // if derotation requested set default positions to derotated positions
        calib.apt["x_t"] = calib.apt["x_t_derot"];
        calib.apt["y_t"] = calib.apt["y_t_derot"];
    }
}

auto Beammap::loop_pipeline() {
    // run iterative stage
    run_loop();

    // write map summary
    if (verbose_mode) {
        write_map_summary(omb);
    }

    // empty initial ptcdata vector to save memory
    ptcs0.clear();

    // set to input parallel policy
    parallel_policy = omb.parallel_policy;

    SPDLOG_INFO("calculating sensitivity");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        Eigen::MatrixXd det_sens, noise_flux;
        // calc sensitivity within psd freq range
        calc_sensitivity(ptcs, det_sens, noise_flux, telescope.d_fsmp, i, {sens_psd_limits(0), sens_psd_limits(1)});
        // copy into apt table
        calib.apt["sens"](i) = tula::alg::median(det_sens);

        return 0;
    });

    // array indices
    auto array_indices = ptcs[0].array_indices.data;
    // nw indices
    auto nw_indices = ptcs[0].nw_indices.data;
    // detector indices
    auto det_indices = ptcs[0].det_indices.data;

    // apt and sensitivity only relevant if beammapping
    if (map_grouping=="detector") {
        // rescale fit params from pixel to on-sky units
        calib.apt["amp"] = params.col(0);
        calib.apt["x_t"] = RAD_TO_ASEC*omb.pixel_size_rad*(params.col(1).array() - (omb.n_cols)/2);
        calib.apt["y_t"] = RAD_TO_ASEC*omb.pixel_size_rad*(params.col(2).array() - (omb.n_rows)/2);
        calib.apt["a_fwhm"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params.col(3));
        calib.apt["b_fwhm"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params.col(4));
        calib.apt["angle"] = params.col(5);

         // rescale fit errors from pixel to on-sky units
        calib.apt["amp_err"] = perrors.col(0);
        calib.apt["x_t_err"] = RAD_TO_ASEC*omb.pixel_size_rad*(perrors.col(1));
        calib.apt["y_t_err"] = RAD_TO_ASEC*omb.pixel_size_rad*(perrors.col(2));
        calib.apt["a_fwhm_err"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(3));
        calib.apt["b_fwhm_err"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(4));
        calib.apt["angle_err"] = perrors.col(5);

        // add convergence iteration to apt table
        calib.apt["converge_iter"] = converge_iter.cast<double> ();

        // flag detectors
        flag_dets(array_indices, nw_indices);

        // subtract ref detector position and derotate
        adjust_apt();

        // add apt table to timestream files
        if (run_tod_output) {
            // vectors to hold tangent plane pointing
            std::vector<Eigen::MatrixXd> det_lat, det_lon;

            // recalculate tangent plane pointing for tod output
            for (Eigen::Index i=0; i<ptcs.size(); i++) {
                // tangent plane pointing for each detector
                Eigen::MatrixXd lats(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
                Eigen::MatrixXd lons(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());

                grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto j) {
                    double az_off = 0;
                    double el_off = 0;

                    auto det_index = det_indices(j);
                    az_off = calib.apt["x_t"](det_index);
                    el_off = calib.apt["y_t"](det_index);

                    // get physical pointing
                    auto [lat, lon] = engine_utils::calc_det_pointing(ptcs[i].tel_data.data, az_off, el_off,
                                                                      telescope.pixel_axes, pointing_offsets_arcsec);
                    lats.col(j) = std::move(lat);
                    lons.col(j) = std::move(lon);

                    return 0;
                });
                det_lat.push_back(std::move(lats));
                det_lon.push_back(std::move(lons));
            }

            // set parallel policy to sequential for tod output
            parallel_policy = "seq";

            SPDLOG_INFO("adding final apt and detector tangent plane pointing to tod files");
            // loop through tod files
            for (const auto & [key, val]: tod_filename) {
                netCDF::NcFile fo(val, netCDF::NcFile::write);
                auto vars = fo.getVars();
                // overwrite apt table
                for (auto const& x: calib.apt) {
                    // start index for apt table
                    std::vector<std::size_t> start_index_apt = {0};
                    // size for apt
                    std::vector<std::size_t> size_apt = {1};
                    netCDF::NcVar apt_v = fo.getVar("apt_" + x.first);
                    for (std::size_t i=0; i< TULA_SIZET(calib.n_dets); ++i) {
                        start_index_apt[0] = i;
                        apt_v.putVar(start_index_apt, size_apt, &calib.apt[x.first](det_indices(i)));
                    }
                }

                // detector tangent plane pointing
                netCDF::NcVar det_lat_v = fo.getVar("det_lat");
                netCDF::NcVar det_lon_v = fo.getVar("det_lon");

                // detector absolute pointing
                netCDF::NcVar det_ra_v = fo.getVar("det_ra");
                netCDF::NcVar det_dec_v = fo.getVar("det_dec");

                // start indices for data
                std::vector<std::size_t> start_index = {0, 0};
                // size for data
                std::vector<std::size_t> size = {1, TULA_SIZET(calib.n_dets)};
                std::size_t k = 0;
                for (Eigen::Index i=0; i<det_lat.size(); i++) {
                    for (std::size_t j=0; j < TULA_SIZET(det_lat[i].rows()); ++j) {
                        start_index[0] = k;
                        k++;
                        // append detector latitudes
                        Eigen::VectorXd lats_row = det_lat[i].row(j);
                        det_lat_v.putVar(start_index, size, lats_row.data());

                        // append detector longitudes
                        Eigen::VectorXd lons_row = det_lon[i].row(j);
                        det_lon_v.putVar(start_index, size, lons_row.data());

                        if (telescope.pixel_axes == "icrs") {
                            // get absolute pointing
                            auto [decs, ras] = engine_utils::phys_to_abs(lats_row, lons_row, telescope.tel_header["Header.Source.Ra"](0),
                                                                         telescope.tel_header["Header.Source.Dec"](0));
                            // append detector ra
                            det_ra_v.putVar(start_index, size, ras.data());

                            // append detector dec
                            det_dec_v.putVar(start_index, size, decs.data());
                        }
                    }
                }
            }

            // empty ptcdata vector to save memory
            ptcs.clear();
        }
    }
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
    det_in_vec.resize(n_maps);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    det_out_vec.resize(n_maps);

    // run iterative pipeline
    loop_pipeline();
}

template <mapmaking::MapType map_type>
void Beammap::output() {
    // pointer to map buffer
    mapmaking::ObsMapBuffer* mb = NULL;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = NULL;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = NULL;

    // directory name
    std::string dir_name;

    // raw obs maps
    if constexpr (map_type == mapmaking::RawObs) {
        mb = &omb;
        f_io = &fits_io_vec;
        n_io = &noise_fits_io_vec;
        dir_name = obsnum_dir_name + "raw/";

        // only write apt table if beammapping
        if (map_grouping=="detector") {
            SPDLOG_INFO("writing apt table");
            auto apt_filename = toltec_io.create_filename<engine_utils::toltecIO::apt, engine_utils::toltecIO::map,
                                                          engine_utils::toltecIO::raw>
                                (obsnum_dir_name + "raw/", redu_type, "", obsnum, telescope.sim_obs);

            //calib.apt_header_keys.push_back("x_t_derot");
            //calib.apt_header_keys.push_back("y_t_derot");
            //calib.apt_header_keys.push_back("x_t_raw");
            //calib.apt_header_keys.push_back("y_t_raw");

            Eigen::MatrixXd apt_table(calib.n_dets, calib.apt_header_keys.size());

            // convert to floats
            Eigen::Index i = 0;
            for (auto const& x: calib.apt_header_keys) {
                apt_table.col(i) = calib.apt[x].cast<double> ();
                i++;
            }

            // write to ecsv
            to_ecsv_from_matrix(apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);

            SPDLOG_INFO("done writing apt table");
        }
    }

    // filtered obs maps
    else if constexpr (map_type == mapmaking::FilteredObs) {
        mb = &omb;
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        dir_name = obsnum_dir_name + "filtered/";
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

    // progress bar
    {
        tula::logging::progressbar pb(
            [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 100, "output progress ");

        for (Eigen::Index i=0; i<f_io->size(); i++) {
            // get the array for the given map
            Eigen::Index map_index = maps_to_arrays(i);
            // add primary hdu
            add_phdu(f_io, mb, map_index);

            if (!mb->noise.empty()) {
                add_phdu(n_io, mb, map_index);
            }
        }

        // write the maps
        Eigen::Index k = 0;
        Eigen::Index step = 2;

        if (!mb->kernel.empty()) {
            step++;
        }
        if (!mb->coverage.empty()) {
            step++;
        }

        // write the maps
        for (Eigen::Index i=0; i<n_maps; i++) {
            // update progress bar
            pb.count(n_maps, 1);
            write_maps(f_io,n_io,mb,i);

            if (map_grouping=="detector") {
                if constexpr (map_type == mapmaking::RawObs) {
                    // get the array for the given map
                    Eigen::Index map_index = maps_to_arrays(i);

                    // check if we move from one file to the next
                    // if so go back to first hdu layer
                    if (i>0) {
                        if (map_index > maps_to_arrays(i-1)) {
                            k = 0;
                        }
                    }

                    // add apt table
                    for (auto const& key: calib.apt_header_keys) {
                        if (calib.apt[key](i) == calib.apt[key](i) && !std::isinf(calib.apt[key](i))) {
                            f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, static_cast<float>(calib.apt[key](i)), key
                                                                  + " (" + calib.apt_header_units[key] + ")");
                        }
                        else {
                            f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, 0.0, key
                                                                   + " (" + calib.apt_header_units[key] + ")");
                        }
                    }
                    // increment hdu layer
                    k = k + step;
                }
            }
        }
    }

    SPDLOG_INFO("files have been written to:");
    for (Eigen::Index i=0; i<f_io->size(); i++) {
        SPDLOG_INFO("{}.fits",f_io->at(i).filepath);
    }

    // clear fits file vectors to ensure its closed.
    f_io->clear();
    n_io->clear();
}

