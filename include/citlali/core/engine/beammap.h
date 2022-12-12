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

    // vector to hold derotation elevation
    Eigen::VectorXd derot_elev;

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

    auto loop_pipeline();

    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    template <mapmaking::MapType map_type>
    void output();
};

void Beammap::setup() {
    // ensure all detectors are initially flagged as good
    calib.apt["flag"].setOnes();
    calib.apt["flxscale"].setOnes();

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

    // resize good fits
    good_fits.setZero(calib.n_dets);

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
        create_tod_files();
    }

    // update apt table meta data
    calib.apt_meta["uid"].push_back("units: N/A");
    calib.apt_meta["uid"].push_back("unique id");

    calib.apt_meta["array"].push_back("units: N/A");
    calib.apt_meta["array"].push_back("array index");

    calib.apt_meta["nw"].push_back("units: N/A");
    calib.apt_meta["nw"].push_back("network index");

    calib.apt_meta["fg"].push_back("units: N/A");
    calib.apt_meta["fg"].push_back("frequency group");

    calib.apt_meta["pg"].push_back("units: N/A");
    calib.apt_meta["pg"].push_back("polarization group");

    calib.apt_meta["ori"].push_back("units: N/A");
    calib.apt_meta["ori"].push_back("orientation");

    calib.apt_meta["responsivity"].push_back("units: N/A");
    calib.apt_meta["responsivity"].push_back("responsivity");

    calib.apt_meta["flxscale"].push_back("units: " + omb.sig_unit);
    calib.apt_meta["flxscale"].push_back("flux conversion scale");

    calib.apt_meta["sens"].push_back("units: V/Hz^-0.5");
    calib.apt_meta["sens"].push_back("sensitivity");

    calib.apt_meta["derot_elev"].push_back("units: radians");
    calib.apt_meta["derot_elev"].push_back("derotation elevation angle");

    calib.apt_meta["amp"].push_back("units: N/A");
    calib.apt_meta["amp"].push_back("fitted amplitude");

    calib.apt_meta["amp_err"].push_back("units: N/A");
    calib.apt_meta["amp_err"].push_back("fitted amplitude error");

    calib.apt_meta["x_t"].push_back("units: arcsec");
    calib.apt_meta["x_t"].push_back("fitted azimuthal offset");

    calib.apt_meta["x_t_err"].push_back("units: arcsec");
    calib.apt_meta["x_t_err"].push_back("fitted azimuthal offset error");

    calib.apt_meta["y_t"].push_back("units: arcsec");
    calib.apt_meta["y_t"].push_back("fitted altitude offset");

    calib.apt_meta["y_t_err"].push_back("units: arcsec");
    calib.apt_meta["y_t_err"].push_back("fitted altitude offset error");

    calib.apt_meta["a_fwhm"].push_back("units: arcsec");
    calib.apt_meta["a_fwhm"].push_back("fitted azimuthal FWHM");

    calib.apt_meta["a_fwhm_err"].push_back("units: arcsec");
    calib.apt_meta["a_fwhm_err"].push_back("fitted azimuthal FWHM error");

    calib.apt_meta["b_fwhm"].push_back("units: arcsec");
    calib.apt_meta["b_fwhm"].push_back("fitted altitude FWMH");

    calib.apt_meta["b_fwhm_err"].push_back("units: arcsec");
    calib.apt_meta["b_fwhm_err"].push_back("fitted altitude FWMH error");

    calib.apt_meta["angle"].push_back("units: radians");
    calib.apt_meta["angle"].push_back("fitted rotation angle");

    calib.apt_meta["angle_err"].push_back("units: radians");
    calib.apt_meta["angle_err"].push_back("fitted rotation angle error");

    calib.apt_meta["converge_iter"].push_back("units: N/A");
    calib.apt_meta["converge_iter"].push_back("beammap convergence iteration");

    calib.apt_meta["flag"].push_back("units: N/A");
    calib.apt_meta["flag"].push_back("good detector");

    calib.apt_meta["sig2noise"].push_back("units: N/A");
    calib.apt_meta["sig2noise"].push_back("signal to noise");

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

        Eigen::Index j = 0;
        Eigen::VectorXd tone_flags(calib.n_dets);
        for (Eigen::Index i=0; i<scan_rawobs.size(); i++) {
            auto tone_axis = scan_rawobs[i].wcs.tone_axis("flag");
            tone_flags.segment(j,tone_axis.size()) = tone_axis;
            j = j + tone_axis.size();
        }

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
        for (const auto &[stokes_index,stokes_param]: rtcproc.polarization.stokes_params) {
            SPDLOG_INFO("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

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
            rtcproc.run(rtcdata_pol, ptcdata, telescope.pixel_axes, redu_type, calib, telescope, pointing_offsets_arcsec,
                        det_indices, array_indices, map_indices, omb.pixel_size_rad);

            // write rtc timestreams
            if (run_tod_output) {
                SPDLOG_INFO("writing rtcdata");
                if (tod_output_type == "rtc") {
                    ptcproc.append_to_netcdf(ptcdata, tod_filename[stokes_param], redu_type, telescope.pixel_axes, pointing_offsets_arcsec,
                                             det_indices, calib.apt, tod_output_type, verbose_mode, telescope.d_fsmp);
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
        ptcs = ptcs0;
        // set maps to zero for each iteration
        for (Eigen::Index i=0; i<n_maps; i++) {
            omb.signal[i].setZero();
            omb.weight[i].setZero();
            //omb.coverage[i].setZero();

            if (rtcproc.run_kernel) {
                omb.kernel[i].setZero();
            }
        }

        // cleaning
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {

            if (run_mapmaking) {
                // subtract gaussian
                if (current_iter > 0) {
                    SPDLOG_INFO("subtract gaussian");
                    // negate the amplitude
                    params.col(0) = -params.col(0);
                    ptcproc.add_gaussian(ptcs[i], params, telescope.pixel_axes,redu_type, calib.apt,pointing_offsets_arcsec,
                                         omb.pixel_size_rad, omb.n_rows, omb.n_cols,ptcs[i].map_indices.data, ptcs[i].det_indices.data);
                }
            }

            SPDLOG_INFO("removing outlier weights");
            auto calib_scan = ptcproc.remove_bad_dets_nw(ptcs[i], calib, ptcs[i].det_indices.data, ptcs[i].nw_indices.data,
                                                         ptcs[i].array_indices.data, redu_type);
            SPDLOG_INFO("ptcproc");
            ptcproc.run(ptcs[i], ptcs[i], calib);

            if (run_mapmaking) {
                // add gaussan back
                if (current_iter > 0) {
                    SPDLOG_INFO("add gaussian");
                    // params is negative due to earlier gaussian subtraction
                    params.col(0) = -params.col(0);
                    ptcproc.add_gaussian(ptcs[i], params, telescope.pixel_axes,redu_type, calib.apt,pointing_offsets_arcsec,
                                         omb.pixel_size_rad, omb.n_rows, omb.n_cols,ptcs[i].map_indices.data, ptcs[i].det_indices.data);
                }
            }

            return 0;
        });

        // sensitivity
        SPDLOG_INFO("calculating sensitivity");
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            Eigen::MatrixXd det_sens, noise_flux;
            calc_sensitivity(ptcs, det_sens, noise_flux, telescope.d_fsmp, i, {sens_psd_limits(0), sens_psd_limits(1)});
            calib.apt["sens"](i) = tula::alg::median(det_sens);
            return 0;
        });

        // calculate weights, populate maps
        SPDLOG_INFO("populating maps");
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
            // calculate weights
            ptcproc.calc_weights(ptcs[i], calib.apt, telescope);

            // write ptc timestreams
            if (run_tod_output) {
                SPDLOG_INFO("writing ptcdata");
                if (tod_output_type == "ptc") {
                    if (current_iter == 0) {
                        ptcproc.append_to_netcdf(ptcs[i], tod_filename["I"], redu_type, telescope.pixel_axes, pointing_offsets_arcsec,
                                                ptcs[i].det_indices.data, calib.apt, tod_output_type, verbose_mode, telescope.d_fsmp);
                    }
                }
            }

            // populate maps
            if (run_mapmaking) {
                mapmaking::populate_maps_naive(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                               ptcs[i].det_indices.data, telescope.pixel_axes,
                                               redu_type, calib.apt, pointing_offsets_arcsec,
                                               telescope.d_fsmp, run_noise);
            }
            return 0;
        });

        if (run_mapmaking) {
            // normalize maps
            SPDLOG_INFO("normalizing maps");
            omb.normalize_maps();

            SPDLOG_INFO("fitting maps");
            grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
                // get array number
                auto array_index = ptcs[0].array_indices.data(i);
                // get initial guess fwhm from theoretical fwhms for the arrays
                auto init_fwhm = toltec_io.array_fwhm_arcsec[array_index]*ASEC_TO_RAD/omb.pixel_size_rad;
                auto [det_params, det_perror, good_fit] =
                    map_fitter.fit_to_gaussian<engine_utils::mapFitter::peakValue>(omb.signal[i], omb.weight[i], init_fwhm);

                params.row(i) = det_params;
                perrors.row(i) = det_perror;
                good_fits(i) = good_fit;

                return 0;}
            );

            SPDLOG_INFO("number of good beammap fits {}/{}", (good_fits.array()==true).count(), calib.n_dets);
        }

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

auto Beammap::loop_pipeline() {
    // run iterative stage
    run_loop();

    // empty initial ptcdata vector to save memory
    ptcs0.clear();

    // rescale fit params from pixel to on-sky units
    calib.apt["amp"] = params.col(0);
    calib.apt["x_t"] = omb.pixel_size_rad*(params.col(1).array() - (omb.n_cols)/2)*RAD_TO_ASEC;
    calib.apt["y_t"] = omb.pixel_size_rad*(params.col(2).array() - (omb.n_rows)/2)*RAD_TO_ASEC;
    calib.apt["a_fwhm"] = STD_TO_FWHM*omb.pixel_size_rad*(params.col(3))*RAD_TO_ASEC;
    calib.apt["b_fwhm"] = STD_TO_FWHM*omb.pixel_size_rad*(params.col(4))*RAD_TO_ASEC;
    calib.apt["angle"] = params.col(5);

     // rescale fit errors from pixel to on-sky units
    calib.apt["amp"] = params.col(0);
    calib.apt["x_t_err"] = omb.pixel_size_rad*(perrors.col(1))*RAD_TO_ASEC;
    calib.apt["y_t_err"] = omb.pixel_size_rad*(perrors.col(2))*RAD_TO_ASEC;
    calib.apt["a_fwhm_err"] = STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(3))*RAD_TO_ASEC;
    calib.apt["b_fwhm_err"] = STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(4))*RAD_TO_ASEC;
    calib.apt["angle_err"] = perrors.col(5);

    calib.apt["converge_iter"] = converge_iter.cast<double> ();

    // array indices for current polarization
    auto array_indices = ptcs[0].array_indices.data;

    std::map<std::string, double> array_mean_x_t, array_mean_y_t;

    // get mean x_t and y_t values for arrays
    for (Eigen::Index i=0; i<calib.n_arrays; i++) {
        Eigen::Index array = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[calib.apt["array"](i)];

        array_mean_x_t[array_name] = calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                               std::get<1>(calib.array_limits[array])-1)).mean();

        array_mean_y_t[array_name] = calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                 std::get<1>(calib.array_limits[array])-1)).mean();
    }

    // mean values of fitted detector positions
    double mean_x_t = calib.apt["x_t"].mean();
    double mean_y_t = calib.apt["y_t"].mean();

    // track number of flagged detectors
    int n_flagged_dets = 0;

    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        auto array_index = array_indices(i);
        std::string array_name = toltec_io.array_name_map[calib.apt["array"](array_index)];

        // calculate map standard deviation
        double map_std_dev = engine_utils::calc_std_dev(omb.signal[i]);

        // calculate distance of detector from mean position of all detectors
        //double dist = sqrt(pow(calib.apt["x_t"](i) - mean_x_t,2) + pow(calib.apt["y_t"](i) - mean_y_t,2));
        double dist = sqrt(pow(calib.apt["x_t"](i) - array_mean_x_t[array_name],2) + pow(calib.apt["y_t"](i) - array_mean_y_t[array_name],2));

        // remove bad fits
        if (!good_fits(i)) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag detectors with outler a_fwhm values
        else if (calib.apt["a_fwhm"](i) < lower_fwhm_arcsec[array_name] || calib.apt["a_fwhm"](i) > upper_fwhm_arcsec[array_name]) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag detectors with outler b_fwhm values
        else if (calib.apt["b_fwhm"](i) < lower_fwhm_arcsec[array_name] || calib.apt["b_fwhm"](i) > upper_fwhm_arcsec[array_name]) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag detectors with outler S/N values
        else if (params(i,0)/map_std_dev < lower_sig2noise[array_name]) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }
        // flag detectors that are further than the mean value than the distance limit
        else if (dist > max_dist_arcsec[array_name] && max_dist_arcsec[array_name] != 0) {
            calib.apt["flag"](i) = 0;
            n_flagged_dets++;
        }

        // calculate detector beamsize
        double det_fwhm = (calib.apt["a_fwhm"](i) + calib.apt["b_fwhm"](i))/2;
        double det_beamsize = 2.*pi*pow(det_fwhm*FWHM_TO_STD,2);

        // set flux scale (always in MJy/sr)
        if (params(i,0) != 0 && det_beamsize !=0) {
            calib.apt["flxscale"](i) = mJY_ASEC_to_MJY_SR*(beammap_fluxes[array_name]/params(i,0))/det_beamsize;
        }

        else {
            calib.apt["flxscale"](i) = 0;
        }

        // get detector pointing
        Eigen::VectorXd lat = -(calib.apt["y_t"](i)*ASEC_TO_RAD) + telescope.tel_data["TelElAct"].array();
        Eigen::VectorXd lon = -(calib.apt["x_t"](i)*ASEC_TO_RAD) + telescope.tel_data["TelAzAct"].array();

        // index of minimum elevation distance
        Eigen::Index min_index;

        // minimum cartesian distance from source
        double min_dist = ((telescope.tel_data["SourceEl"] - lat).array().pow(2) +
                           (telescope.tel_data["SourceAz"] - lon).array().pow(2)).minCoeff(&min_index);

        // elevation at which we are on source
        derot_elev(i) = telescope.tel_data["TelElAct"](min_index);

        // derotate detectors to zero elevation
        if (beammap_derotate) {
            // rotation az/el angles
            double rot_azoff = cos(-derot_elev(i))*calib.apt["x_t"](i) - sin(-derot_elev(i))*calib.apt["y_t"](i);
            double rot_eloff = sin(-derot_elev(i))*calib.apt["x_t"](i) + cos(-derot_elev(i))*calib.apt["y_t"](i);

            // overwrite x_t and y_t
            calib.apt["x_t"](i) = -rot_azoff;
            calib.apt["y_t"](i) = -rot_eloff;
        }

        // correct for extinction
        Eigen::VectorXd tau_el(1);
        tau_el << derot_elev(i);
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);

        calib.apt["flxscale"](i) = calib.apt["flxscale"](i)/exp(-tau_freq[array_index](0));

        return 0;
    });

    // print number of flagged detectors
    SPDLOG_INFO("{} detectors were flagged", n_flagged_dets);

    // rescale sens to MJy/sr units
    //calib.apt["sens"] = calib.apt["sens"].array()*calib.apt["flxscale"].array();
    //SPDLOG_INFO("sens after {}",calib.apt["sens"]);

    // align to reference detector if specified
    if (beammap_reference_det > 0) {
        SPDLOG_INFO("subtracting reference detector {} position", beammap_reference_det);
        auto ref_det_x_t = params(beammap_reference_det,1);
        auto ref_det_y_t = params(beammap_reference_det,2);

        params.col(1) =  params.col(1).array() - ref_det_x_t;
        params.col(2) =  params.col(2).array() - ref_det_y_t;
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
    det_in_vec.resize(calib.n_dets);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    det_out_vec.resize(calib.n_dets);

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
        dir_name = obsnum_dir_name + "/raw/";

        SPDLOG_INFO("writing apt table");
        auto apt_filename = toltec_io.create_filename<engine_utils::toltecIO::apt, engine_utils::toltecIO::map,
                                                      engine_utils::toltecIO::raw>
                            (obsnum_dir_name + "/raw/", redu_type, "", obsnum, telescope.sim_obs);

        Eigen::MatrixXd apt_table(calib.n_dets, calib.apt_header_keys.size());
        SPDLOG_INFO("done writing apt table");

        // convert to floats
        Eigen::Index i = 0;
        for (auto const& x: calib.apt_header_keys) {
            apt_table.col(i) = calib.apt[x].cast<double> ();
            i++;
        }

        // write to ecsv
        to_ecsv_from_matrix(apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);
    }

    // filtered obs maps
    else if constexpr (map_type == mapmaking::FilteredObs) {
        mb = &omb;
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        dir_name = obsnum_dir_name + "/filtered/";
    }

    // raw coadded maps
    else if constexpr (map_type == mapmaking::RawCoadd) {
        mb = &cmb;
        f_io = &coadd_fits_io_vec;
        n_io = &coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "/raw/";
    }

    // filtered coadded maps
    else if constexpr (map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        f_io = &filtered_coadd_fits_io_vec;
        n_io = &filtered_coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "/filtered/";
    }

    // write the maps
    for (Eigen::Index i=0; i<n_maps; i++) {
        write_maps(f_io,n_io,mb,i);
    }

    // clear fits file vectors to ensure its closed.
    f_io->clear();
    n_io->clear();

    // write psd and histogram files
    write_psd<map_type>(mb, dir_name);
    write_hist<map_type>(mb, dir_name);

    /*
    SPDLOG_INFO("writing apt table");
    auto apt_filename = toltec_io.create_filename<engine_utils::toltecIO::apt, engine_utils::toltecIO::map,
                                                  engine_utils::toltecIO::raw>
                        (obsnum_dir_name, redu_type, "", obsnum, telescope.sim_obs);

    Eigen::MatrixXd apt_table(calib.n_dets, calib.apt_header_keys.size());

    // convert to floats
    Eigen::Index i = 0;
    for (auto const& x: calib.apt_header_keys) {
        apt_table.col(i) = calib.apt[x].cast<double> ();
        i++;
    }

    // write to ecsv
    to_ecsv_from_matrix(apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);

    SPDLOG_INFO("writing maps");
    for (Eigen::Index i=0; i<rtcproc.polarization.stokes_params.size(); i++) {
        for (Eigen::Index j=0; j<calib.n_arrays; j++) {
            auto array = calib.arrays[j];
            std::string name = toltec_io.array_name_map[j];
            for (Eigen::Index k=0; k<calib.n_dets; k++) {
                if (calib.apt["array"](k) == array) {
                    // signal
                    fits_io_vec[array].add_hdu("signal_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i], omb.signal[k]);
                    fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);
                    fits_io_vec[array].hdus.back()->addKey("UNIT", omb.sig_unit, "Unit of map");

                    // weight
                    fits_io_vec[array].add_hdu("weight_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i], omb.weight[k]);
                    fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);
                    fits_io_vec[array].hdus.back()->addKey("UNIT", "1/("+omb.sig_unit+")", "Unit of map");

                    if (rtcproc.run_kernel) {
                        // kernel
                        fits_io_vec[array].add_hdu("kernel_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i],
                                                   omb.kernel[k]);
                        fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);
                        fits_io_vec[array].hdus.back()->addKey("UNIT", omb.sig_unit, "Unit of map");
                    }
                }
            }

            fits_io_vec[array].pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
            fits_io_vec[array].pfits->pHDU().addKey("CITLALI_VER", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");

            if (rtcproc.run_calibrate) {
                if (omb.sig_unit == "Mjy/sr") {
                    fits_io_vec[array].pfits->pHDU().addKey("to_mJy/beam", calib.array_beam_areas[calib.arrays(j)]*MJY_SR_TO_mJY_ASEC,
                                                            "Conversion to mJy/beam");
                    fits_io_vec[array].pfits->pHDU().addKey("to_Mjy/sr", 1, "Conversion to MJy/sr");
                }

                else if (omb.sig_unit == "mJy/beam") {
                    fits_io_vec[array].pfits->pHDU().addKey("to_mJy/beam", 1, "Conversion to mJy/beam");
                    fits_io_vec[array].pfits->pHDU().addKey("to_Mjy/sr", 1/calib.mean_flux_conversion_factor[name], "Conversion to MJy/sr");
                }
            }

            else {
                fits_io_vec[array].pfits->pHDU().addKey("to_mJy/beam", "N/A", "Conversion to mJy/beam");
                fits_io_vec[array].pfits->pHDU().addKey("to_Mjy/sr", "N/A", "Conversion to MJy/sr");
            }

            // add obsnum
            fits_io_vec[array].pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
            // add citlali version
            fits_io_vec[array].pfits->pHDU().addKey("VERSION", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");
            // add tod type
            fits_io_vec[array].pfits->pHDU().addKey("TYPE", tod_type, "TOD Type");
            // add exposure time
            fits_io_vec[array].pfits->pHDU().addKey("EXPTIME", omb.exposure_time, "Exposure Time");

            // add source ra
            fits_io_vec[j].pfits->pHDU().addKey("SRC_RA", telescope.tel_header["Header.Source.Ra"][0], "Source RA (radians)");
            // add source dec
            fits_io_vec[j].pfits->pHDU().addKey("SRC_DEC", telescope.tel_header["Header.Source.Dec"][0], "Source Dec (radians)");
            // add map tangent point ra
            fits_io_vec[j].pfits->pHDU().addKey("TAN_RA", telescope.tel_header["Header.Source.Ra"][0], "Map Tangent Point RA (radians)");
            // add map tangent point dec
            fits_io_vec[j].pfits->pHDU().addKey("TAN_DEC", telescope.tel_header["Header.Source.Dec"][0], "Map Tangent Point Dec (radians)");

            // add telescope file header information
            for (auto const& [key, val] : telescope.tel_header) {
                fits_io_vec[j].pfits->pHDU().addKey(key, val(0), key);
            }
        }
    }
    */
}

