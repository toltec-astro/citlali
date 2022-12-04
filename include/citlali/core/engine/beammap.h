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
    // convert beammap flux and error to MJy/sr (default) units
    SPDLOG_INFO("beammap_fluxes {}",beammap_fluxes);
    /*Eigen::Index i = 0;
    for (auto const& [key, val] : beammap_fluxes) {
        std::string name = toltec_io.array_name_map[i];
        beammap_fluxes[name] = beammap_fluxes[name]/calib.array_beam_areas[calib.arrays(i)]/MJY_SR_TO_mJY_ASEC;
        beammap_err[name] = beammap_err[name]/calib.array_beam_areas[calib.arrays(i)]/MJY_SR_TO_mJY_ASEC;
        i++;
    }*/

    SPDLOG_INFO("beammap_fluxes {}",beammap_fluxes);

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

    calib.apt_meta["sens"].push_back("units: N/A");
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
            SPDLOG_INFO("tone_axis(flag) {}", tone_axis);
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
            // remove outliers
            //SPDLOG_INFO("removing outlier weights");
            //ptcproc.remove_bad_dets(ptcdata, calib.apt, det_indices);
            //ptcproc.remove_bad_dets_nw(ptcs[i], calib, ptcs[i].det_indices.data);

            // run cleaning
            SPDLOG_INFO("ptcproc");
            ptcproc.run(ptcs[i], ptcs[i], calib);
            SPDLOG_INFO("i {}",i);
            return 0;
        });

        // sensitivity
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            SPDLOG_INFO("sens");
            Eigen::MatrixXd det_sens, noise_flux;
            calc_sensitivity(ptcs, det_sens, noise_flux, telescope.d_fsmp, i, {sens_psd_limits(0), sens_psd_limits(1)});
            calib.apt["sens"](i) = tula::alg::median(det_sens);
            SPDLOG_INFO("i {}",i);
            return 0;
        });

        SPDLOG_INFO("sens {}",calib.apt["sens"]);

        // calculate weights, populate maps
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
            // calculate weights
            SPDLOG_INFO("calculating weights");
            ptcproc.calc_weights(ptcs[i], calib.apt, telescope);

            // write ptc timestreams
            if (run_tod_output) {
                SPDLOG_INFO("writing ptcdata");
                if (tod_output_type == "ptc") {
                    ptcproc.append_to_netcdf(ptcs[i], tod_filename["I"], redu_type, telescope.pixel_axes, pointing_offsets_arcsec,
                                             ptcs[i].det_indices.data, calib.apt, tod_output_type, verbose_mode, telescope.d_fsmp);
                }
            }

            // populate maps
            SPDLOG_INFO("populating maps");
            mapmaking::populate_maps_naive(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                           ptcs[i].det_indices.data, telescope.pixel_axes,
                                           redu_type, calib.apt, pointing_offsets_arcsec,
                                           telescope.d_fsmp, run_noise);
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
                map_fitter.fit_to_gaussian<engine_utils::mapFitter::peakValue>(omb.signal[i], omb.weight[i], init_fwhm);

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

    SPDLOG_INFO("params {}", params);
    SPDLOG_INFO("1/ASEC_TO_RAD {}",1/ASEC_TO_RAD);
    SPDLOG_INFO("RAD_TO_ASEC {}",RAD_TO_ASEC);

    SPDLOG_INFO("omb.n_cols/2 {}",omb.n_cols/2);
    SPDLOG_INFO("omb.pixel_size_rad {}", omb.pixel_size_rad);
    SPDLOG_INFO("x_t {}", params(0,1));

    // rescale fit params from pixel to on-sky units
    calib.apt["amp"] = params.col(0);
    calib.apt["x_t"] = omb.pixel_size_rad*(params.col(1).array() - (omb.n_cols)/2)*RAD_TO_ASEC;
    calib.apt["y_t"] = omb.pixel_size_rad*(params.col(2).array() - (omb.n_rows)/2)*RAD_TO_ASEC;
    calib.apt["a_fwhm"] = STD_TO_FWHM*omb.pixel_size_rad*(params.col(3))*RAD_TO_ASEC;
    calib.apt["b_fwhm"] = STD_TO_FWHM*omb.pixel_size_rad*(params.col(4))*RAD_TO_ASEC;
    calib.apt["angle"] = params.col(5);

    SPDLOG_INFO("x_t {}", calib.apt["x_t"]);
    SPDLOG_INFO("a_fwhm {}", calib.apt["a_fwhm"]);

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

    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        auto array_index = array_indices(i);
        std::string array_name = toltec_io.array_name_map[calib.apt["array"](array_index)];

        SPDLOG_INFO("ARRAY NAME {}", array_name);

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

        // set flux scale (always in MJy/sr)
        calib.apt["flxscale"](i) = beammap_fluxes["array_name"]/params(i,0);

        SPDLOG_INFO("beammap_fluxes[array_name] {} params(i,0) {}",beammap_fluxes["array_name"], params(i,0));

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

    // rescale sens to MJy/sr units
    calib.apt["sens"] = calib.apt["sens"].array()*calib.apt["flxscale"].array();

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
    SPDLOG_INFO("writing apt table");
    auto apt_filename = toltec_io.create_filename<engine_utils::toltecIO::apt, engine_utils::toltecIO::map>
                        (obsnum_dir_name, redu_type, "", obsnum, telescope.sim_obs);

    Eigen::MatrixXf apt_table(calib.n_dets, calib.apt_header_keys.size());

    // convert to floats
    Eigen::Index i = 0;
    for (auto const& x: calib.apt_header_keys) {
        SPDLOG_INFO("apt header key {}", x);
        SPDLOG_INFO("apt key {}", calib.apt[x]);

        apt_table.col(i) = calib.apt[x].cast<float> ();
        i++;
    }

    // write to ecsv
    to_ecsv_from_matrix(apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);

    SPDLOG_INFO("writing maps");
    for (Eigen::Index i=0; i<rtcproc.polarization.stokes_params.size(); i++) {
        for (Eigen::Index j=0; j<calib.n_arrays; j++) {
            auto array = calib.arrays[j];
            for (Eigen::Index k=0; k<calib.n_dets; k++) {
                if (calib.apt["array"](k) == array) {
                    // signal
                    fits_io_vec[array].add_hdu("signal_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i], omb.signal[k]);
                    fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);

                    // weight
                    fits_io_vec[array].add_hdu("weight_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i], omb.weight[k]);
                    fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);

                    if (rtcproc.run_kernel) {
                        // kernel
                        fits_io_vec[array].add_hdu("kernel_" + std::to_string(k) + "_" + rtcproc.polarization.stokes_params[i],
                                                   omb.kernel[k]);
                        fits_io_vec[array].add_wcs(fits_io_vec[array].hdus.back(),omb.wcs);
                    }
                }
            }

            fits_io_vec[array].pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
            fits_io_vec[array].pfits->pHDU().addKey("CITLALI_VER", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");

            // add telescope file header information
            for (auto const& [key, val] : telescope.tel_header) {
                fits_io_vec[array].pfits->pHDU().addKey(key, val(0), key);
            }
        }
    }
}

