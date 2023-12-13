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
    // parallel policies for each section
    std::string  map_parallel_policy;

    // vector to store each scan's PTCData
    std::vector<TCData<TCDataKind::PTC,Eigen::MatrixXd>> ptcs0, ptcs;

    // copy of obs map buffer for map iteration
    mapmaking::ObsMapBuffer omb_copy{"omb"};

    // vector to store each scan's calib class
    std::vector<engine::Calib> calib_scans0, calib_scans;

    // beammap iteration parameters
    Eigen::Index current_iter;

    // vector for convergence check
    Eigen::Matrix<bool, Eigen::Dynamic, 1> converged;

    // vector to record convergence iteration
    Eigen::Vector<int, Eigen::Dynamic> converge_iter;

    // previous iteration fit parameters
    Eigen::MatrixXd p0, perror0;

    // current iteration fit parameters
    Eigen::MatrixXd params, perrors;

    // reference detector
    Eigen::Index beammap_reference_det_found = -99;

    // bitwise flags
    enum AptFlags {
        Good         = 0,
        BadFit       = 1 << 0,
        AzFWHM       = 1 << 1,
        ElFWHM       = 1 << 2,
        Sig2Noise    = 1 << 3,
        Sens         = 1 << 4,
        Position     = 1 << 5,
        };

    // holds bitwise flags
    Eigen::Matrix<uint16_t,Eigen::Dynamic,1> flag2;

    // good fits
    Eigen::Matrix<bool, Eigen::Dynamic, 1> good_fits;

    // placeholder vectors for grppi maps
    std::vector<int> scan_in_vec, scan_out_vec;
    std::vector<int> det_in_vec, det_out_vec;

    // initial setup for each obs
    void setup();

    // run the raw time chunk processing
    auto run_timestream();

    // run the iterative stage
    auto run_loop();

    // timestream grppi pipeline
    template <class KidsProc, class RawObs>
    auto timestream_pipeline(KidsProc &, RawObs &);

    // flag detectors
    template<typename array_indices_t, typename nw_indices_t>
    void set_apt_flags(array_indices_t &, nw_indices_t &);

    // derotate apt and subtract reference detector
    void process_apt();

    // run the loop pipeline
    auto loop_pipeline();

    // main pipeline process
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // output files
    template <mapmaking::MapType map_type>
    void output();
};

void Beammap::setup() {
    // assign parallel policies
    map_parallel_policy = parallel_policy;

    // run obsnum setup
    obsnum_setup();

    // create kids tone apt row
    calib.apt["kids_tone"].resize(calib.n_dets);

    Eigen::Index j = 0;
    calib.apt["kids_tone"](0) = 0;
    for (Eigen::Index i=1; i<calib.n_dets; ++i) {
        if (calib.apt["nw"](i) > calib.apt["nw"](i-1)) {
            j = 0;
        }
        else {
            j++;
        }

        calib.apt["kids_tone"](i) = j;
    }

    // add kids tone to apt header
    calib.apt_header_keys.push_back("kids_tone");
    calib.apt_header_units["kids_tone"] = "N/A";

    // resize the PTCData vector to number of scans
    ptcs0.resize(telescope.scan_indices.cols());

    // resize the calib vector to number of scans
    calib_scans0.resize(telescope.scan_indices.cols());

    // resize the initial fit matrix
    p0.setZero(n_maps, map_fitter.n_params);
    // resize the initial fit error matrix
    perror0.setZero(n_maps, map_fitter.n_params);
    // resize the current fit matrix
    params.setZero(n_maps, map_fitter.n_params);
    perrors.setZero(n_maps, map_fitter.n_params);

    // resize good fits
    good_fits.setZero(n_maps);

    // initially all detectors are unconverged
    converged.setZero(n_maps);
    // convergence iteration
    converge_iter.resize(n_maps);
    converge_iter.setConstant(1);
    // set the initial iteration
    current_iter = 0;

    // use per detector parallelization for jinc mapmaking
    if (map_method == "jinc") {
        map_parallel_policy = "seq";
    }

    /* update apt table meta data */
    calib.apt_meta.reset();

    // add obsnum to meta data
    calib.apt_meta["obsnum"] = obsnum;

    // add source name
    calib.apt_meta["source"] = telescope.source_name;

    // add project id to meta data
    calib.apt_meta["project_id"] = telescope.project_id;

    // add input source flux
    for (const auto &beammap_flux: beammap_fluxes_mJy_beam) {
        auto key = beammap_flux.first + "_flux";
        calib.apt_meta[key].push_back(beammap_flux.second);
        calib.apt_meta[key].push_back("units: mJy/beam");
        calib.apt_meta[key].push_back(beammap_flux.first + " flux density");
    }

    // add date
    calib.apt_meta["date"] = engine_utils::current_date_time();

    // mean Modified Julian Date
    calib.apt_meta["mjd"] = engine_utils::unix_to_modified_julian_date(telescope.tel_data["TelTime"].mean());

    // reference frame
    calib.apt_meta["Radesys"] = telescope.pixel_axes;

    // add apt header keys
    for (const auto &[param,unit]: calib.apt_header_units) {
        calib.apt_meta[param].push_back("units: " + unit);
    }
    // add apt header descriptions
    for (const auto &[param,description]: calib.apt_header_description) {
        calib.apt_meta[param].push_back(description);
    }

    // kids tone
    calib.apt_meta["kids_tone"].push_back("units: N/A");
    calib.apt_meta["kids_tone"].push_back("index of tone in network");

    // bitwise flag
    calib.apt_meta["flag2"].push_back("units: N/A");
    calib.apt_meta["flag2"].push_back("bitwise flag");
    calib.apt_meta["flag2"].push_back("Good=0");
    calib.apt_meta["flag2"].push_back("BadFit=1");
    calib.apt_meta["flag2"].push_back("AzFWHM=2");
    calib.apt_meta["flag2"].push_back("ElFWHM=3");
    calib.apt_meta["flag2"].push_back("Sig2Noise=4");
    calib.apt_meta["flag2"].push_back("Sens=5");
    calib.apt_meta["flag2"].push_back("Position=6");

    // add array mapping
    for (const auto &[arr_index,arr_name]: toltec_io.array_name_map) {
        calib.apt_meta["array_order"].push_back(std::to_string(arr_index) + ": " + arr_name);
    }

    calib.apt_header_units["flag2"] = "N/A";
    calib.apt_header_keys.push_back("flag2");

    // is the detector rotated?
    calib.apt_meta["is_derotated"] = beammap_derotate;
    // was a reference detector subtracted?
    calib.apt_meta["reference_detector_subtracted"] = beammap_subtract_reference;
    // reference detector
    calib.apt_meta["reference_det"] = beammap_reference_det_found;
}

auto Beammap::run_timestream() {
    auto farm = grppi::farm(n_threads,[&](auto &input_tuple) -> TCData<TCDataKind::PTC,Eigen::MatrixXd> {
        // RTCData input
        auto rtcdata = std::get<0>(input_tuple);
        // kidsproc
        auto kidsproc = std::get<1>(input_tuple);
        // start index input
        auto scan_rawobs = std::get<2>(input_tuple);

        // allocate up bitwise timestream flags
        rtcdata.flags2.data.setConstant(timestream::TimestreamFlags::Good);

        // starting index for scan (outer scan)
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
            rtcdata.hwpr_angle.data = calib.hwpr_angle.segment(si + hwpr_start_indices, sl);
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

        if (map_grouping!="detector") {
            // remove flagged detectors
            rtcproc.remove_flagged_dets(ptcdata, calib.apt, det_indices);
        }

        // remove outliers before cleaning
        auto calib_scan = rtcproc.remove_bad_dets(ptcdata, calib, det_indices, map_grouping);

        // remove duplicate tones
        if (!telescope.sim_obs) {
            calib_scan = rtcproc.remove_nearby_tones(ptcdata, calib_scan, det_indices, map_grouping);
        }

        // write rtc timestreams
        if (run_tod_output && !tod_filename.empty()) {
            if (tod_output_type == "rtc" || tod_output_type == "both") {
                logger->info("writing raw time chunk");
                rtcproc.append_to_netcdf(ptcdata, tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         ptcdata.pointing_offsets_arcsec.data, det_indices, calib_scan);
            }
        }

        // store indices for each ptcdata
        ptcdata.det_indices.data = std::move(det_indices);
        ptcdata.nw_indices.data = std::move(nw_indices);
        ptcdata.array_indices.data = std::move(array_indices);
        ptcdata.map_indices.data = std::move(map_indices);

        // move out ptcdata the PTCData vector at corresponding index
        ptcs0.at(ptcdata.index.data) = std::move(ptcdata);
        calib_scans0.at(ptcdata.index.data) = std::move(calib_scan);

        // increment number of completed scans
        n_scans_done++;
        logger->info("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done, telescope.scan_indices.cols());

        return ptcdata;
    });

    return farm;
}

auto Beammap::run_loop() {
    // variable to control iteration
    bool keep_going = true;

    // iterative loop
    while (keep_going) {
        logger->info("starting iter {}", current_iter);

        // copy ptcs
        ptcs = ptcs0;
        // copy calibs
        calib_scans = calib_scans0;

        // copy signal for convergence test
        if (ptcproc.run_fruit_loops) {
            omb_copy.signal = omb.signal;
            // calc mean rms
            if (current_iter == 1) {
                // use obs map buffer
                if (!omb.noise.empty()) {
                    omb.calc_mean_rms();
                }
            }
        }

        // progress bar
        tula::logging::progressbar pb(
            [&](const auto &msg) { logger->info("{}", msg); }, 100, "PTC progress ");

        // cleaning (separate from mapmaking loop due to jinc mapmaking parallelization)
        grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
            if (run_mapmaking) {
                if (current_iter > 0) {
                    if (!ptcproc.run_fruit_loops) {
                        // if not running fruit loops use source fit
                        logger->info("subtracting gaussian from tod");
                        // subtract gaussian
                        ptcproc.add_gaussian<timestream::TCProc::SourceType::NegativeGaussian>(ptcs[i], params, telescope.pixel_axes, map_grouping,
                                                                                               calib.apt,omb.pixel_size_rad, omb.n_rows, omb.n_cols);
                    }
                    else {
                        logger->info("subtracting map from tod");
                        // subtract map
                        ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(omb, ptcs[i], calib, ptcs[i].det_indices.data,
                                                                                        ptcs[i].map_indices.data, telescope.pixel_axes,
                                                                                        map_grouping);
                    }
                }
            }

            // clean the maps
            logger->info("processed time chunk processing for scan {}", i + 1);
            ptcproc.run(ptcs[i], ptcs[i], calib_scans[i], ptcs[i].det_indices.data, telescope.pixel_axes, map_grouping);

            if (run_mapmaking) {
                if (current_iter > 0) {
                    // if not running fruit loops use source fit
                    if (!ptcproc.run_fruit_loops) {
                        logger->info("adding gaussian to tod");
                        // add gaussian back
                        ptcproc.add_gaussian<timestream::TCProc::SourceType::Gaussian>(ptcs[i], params, telescope.pixel_axes, map_grouping, calib.apt,
                                                                                       omb.pixel_size_rad,omb.n_rows, omb.n_cols);
                    }
                    else {
                        logger->info("adding map to tod");
                        // add map back
                        ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(omb, ptcs[i], calib, ptcs[i].det_indices.data,
                                                                                ptcs[i].map_indices.data, telescope.pixel_axes,
                                                                                map_grouping);
                    }
                }
            }

            // remove outliers after clean
            calib_scans[i] = ptcproc.remove_bad_dets(ptcs[i], calib_scans[i], ptcs[i].det_indices.data, map_grouping);

            if (map_grouping == "detector") {
                // set weights to a constant value
                ptcs[i].weights.data.resize(ptcs[i].scans.data.cols());
                ptcs[i].weights.data.setOnes();
            }
            else {
                // calculate weights
                logger->info("calculating weights");
                ptcproc.calc_weights(ptcs[i], calib.apt, telescope, ptcs[i].det_indices.data);

                // reset weights to median
                ptcproc.reset_weights(ptcs[i], calib, ptcs[i].det_indices.data);
            }

            // write out chunk summary
            if (verbose_mode && current_iter==beammap_tod_output_iter) {
                logger->debug("writing chunk summary");
                write_chunk_summary(ptcs[i]);
            }

            // calc stats
            logger->debug("calculating stats");
            diagnostics.calc_stats(ptcs[i]);

            return 0;
        });

        // write ptc timestreams
        if (run_tod_output && !tod_filename.empty()) {
            if (tod_output_type == "ptc" || tod_output_type == "both") {
                logger->info("writing processed time chunk");
                if (current_iter == beammap_tod_output_iter) {
                    for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                        ptcproc.append_to_netcdf(ptcs[i], tod_filename["ptc"], map_grouping, telescope.pixel_axes,
                                                 ptcs[i].pointing_offsets_arcsec.data, ptcs[i].det_indices.data,
                                                 calib_scans[i]);
                    }
                }
            }
        }

        if (run_mapmaking) {
            // set maps to zero for each iteration
            for (Eigen::Index i=0; i<n_maps; ++i) {
                omb.signal[i].setZero();
                omb.weight[i].setZero();

                // clear coverage
                if (!omb.coverage.empty()) {
                    omb.coverage[i].setZero();
                }
                // clear kernel
                if (rtcproc.run_kernel) {
                    omb.kernel[i].setZero();
                }
                // clear noise
                if (!omb.noise.empty()) {
                    omb.noise[i].setZero();
                }

                if (run_noise) {
                    // declare random number generator
                    thread_local boost::random::mt19937 eng;

                    // boost random number generator (0,1)
                    boost::random::uniform_int_distribution<> rands{0,1};

                    for (auto& ptcdata: ptcs) {
                        if (omb.randomize_dets) {
                            ptcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(omb.n_noise, calib.n_dets)
                                                     .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                        } else {
                            ptcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(omb.n_noise)
                                                     .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
                        }
                    }
                }
            }

            // mapmaking
            grppi::map(tula::grppi_utils::dyn_ex(map_parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
                bool run_omb = true;
                // populate maps
                    if (map_method=="naive") {
                        // naive mapmaker
                        naive_mm.populate_maps_naive(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                                     ptcs[i].det_indices.data, telescope.pixel_axes,
                                                     calib.apt, telescope.d_fsmp, run_omb, run_noise);
                    }
                    else if (map_method=="jinc") {
                        // jinc mapmaker
                        jinc_mm.populate_maps_jinc(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                                   ptcs[i].det_indices.data, telescope.pixel_axes,
                                                   calib.apt,telescope.d_fsmp, run_omb, run_noise);
                    }

                // update progress bar
                pb.count(telescope.scan_indices.cols(), 1);

                return 0;
            });

            // normalize maps
            logger->info("normalizing maps");
            omb.normalize_maps();

            // only run fit if subtracting fitted source
            if (!ptcproc.run_fruit_loops) {
                // initial position for fitting
                double init_row = -99;
                double init_col = -99;

                logger->info("fitting maps");
                grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
                    // only fit if not converged
                    if (!converged(i)) {
                        // get array number
                        auto array = maps_to_arrays(i);
                        // get initial guess fwhm from theoretical fwhms for the arrays
                        double init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/omb.pixel_size_rad;
                        // fit the maps
                        auto [det_params, det_perror, good_fit] =
                            map_fitter.fit_to_gaussian<engine_utils::mapFitter::beammap>(omb.signal[i], omb.weight[i],
                                                                                         init_fwhm, init_row, init_col);

                        params.row(i) = det_params;
                        perrors.row(i) = det_perror;
                        good_fits(i) = good_fit;
                    }
                    // otherwise keep value from previous iteration
                    else {
                        params.row(i) = p0.row(i);
                        perrors.row(i) = perror0.row(i);
                    }

                    return 0;
                });

                logger->info("number of good fits {}/{}", good_fits.cast<double>().sum(), n_maps);
            }
        }

        // increment loop iteration
        current_iter++;

        if (current_iter < beammap_iter_max) {
            // check if all detectors are converged
            if ((converged.array() == true).all()) {
                logger->info("all maps converged");
                keep_going = false;
            }
            else if (current_iter > 1) {
                // only do convergence test if tolerance is above zero, otherwise run all iterations
                if (beammap_iter_tolerance > 0) {
                    // loop through maps and check if it is converged
                    logger->info("checking convergence");
                    grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
                        if (!converged(i)) {
                            // get relative change from last iteration
                            Eigen::ArrayXd diff;
                            if (!ptcproc.run_fruit_loops) {
                                diff = abs((params.row(i).array() - p0.row(i).array())/p0.row(i).array());
                            }
                            else {
                                diff = abs(omb_copy.signal[i].array() - omb.signal[i].array()/omb_copy.signal[i].array());
                            }
                            // if a variable is constant, make sure no nans are present
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

                    logger->info("{} maps converged on iter {}", (converged.array() == true).count(), current_iter);

                    // stop if all maps converged
                    if ((converged.array() == true).all()) {
                        logger->info("all maps converged");
                        keep_going = false;
                    }
                }
                else {
                    logger->info("bypassing convergence check");
                }

            }

            // set previous iteration fits to current iteration fits
            p0 = params;
            perror0 = perrors;
        }
        else {
            logger->info("max iteration reached");
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
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "RTC progress ");

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
        // run the raw time chunk processing
        run_timestream());
}

template<typename array_indices_t, typename nw_indices_t>
void Beammap::set_apt_flags(array_indices_t &array_indices, nw_indices_t &nw_indices) {
    // setup bitwise flags
    flag2.resize(calib.n_dets);
    flag2.setConstant(AptFlags::Good);

    // track number of flagged detectors
    int n_flagged_dets = 0;

    logger->info("flagging detectors");
    // first flag based on fit values and signal-to-noise
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = array_indices(i);
        std::string array_name = toltec_io.array_name_map[array_index];

        // calculate map standard deviation
        double map_std_dev = engine_utils::calc_std_dev(omb.signal[i]);

        // set apt signal to noise
        calib.apt["sig2noise"](i) = params(i,0)/map_std_dev;

        // flag bad fits
        if (!good_fits(i)) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::BadFit;
        }
        // flag detectors with outler a_fwhm values
        if (calib.apt["a_fwhm"](i) < lower_fwhm_arcsec[array_name] ||
            ((calib.apt["a_fwhm"](i) > upper_fwhm_arcsec[array_name]) && upper_fwhm_arcsec[array_name] > 0)) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::AzFWHM;
        }
        // flag detectors with outler b_fwhm values
        if (calib.apt["b_fwhm"](i) < lower_fwhm_arcsec[array_name] ||
            ((calib.apt["b_fwhm"](i) > upper_fwhm_arcsec[array_name] && upper_fwhm_arcsec[array_name] > 0))) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::ElFWHM;
        }
        // flag detectors with outler S/N values
        if ((params(i,0)/map_std_dev < lower_sig2noise[array_name]) ||
            ((params(i,0)/map_std_dev > upper_sig2noise[array_name]) && (upper_sig2noise[array_name] > 0))) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::Sig2Noise;
        }
        return 0;
    });

    // median network sensitivity for flagging
    std::map<Eigen::Index, double> nw_median_sens;

    // calc median sens from unflagged detectors for each nw
    logger->debug("calculating mean sensitivities");
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        Eigen::Index nw = calib.nws(i);

        // nw sensitivity
        auto nw_sens = calib.apt["sens"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                    std::get<1>(calib.nw_limits[nw])-1));
        // number of good detectors
        Eigen::Index n_good_det = (calib.apt["flag"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                               std::get<1>(calib.nw_limits[nw])-1)).array()==0).count();

        if (n_good_det>0) {
            // to hold good detectors
            Eigen::VectorXd sens(n_good_det);

            // remove flagged dets
            Eigen::Index j = std::get<0>(calib.nw_limits[nw]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<sens.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    sens(k) = nw_sens(m);
                    k++;
                }
                j++;
            }
            // calculate median sens
            nw_median_sens[nw] = tula::alg::median(sens);
        }
        else {
            nw_median_sens[nw] = tula::alg::median(nw_sens);
        }
    }

    // flag too low/high sensitivies based on the median unflagged sensitivity of each nw
    logger->debug("flagging sensitivities");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get nw of current detector
        auto nw_index = nw_indices(i);

        // flag outlier sensitivities
        if (calib.apt["sens"](i) < lower_sens_factor*nw_median_sens[nw_index] ||
            (calib.apt["sens"](i) > upper_sens_factor*nw_median_sens[nw_index] && upper_sens_factor > 0)) {
            if (calib.apt["flag"](i)==0) {
                calib.apt["flag"](i) = 1;
                n_flagged_dets++;
            }
            flag2(i) |= AptFlags::Sens;
        }

        return 0;
    });

    // std maps to hold median unflagged x and y positions
    std::map<std::string, double> array_median_x_t, array_median_y_t;

    // calc median x_t and y_t values from unflagged detectors for each arrays
    logger->debug("calculating array median positions");
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // x_t
        auto array_x_t = calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        // y_t
        auto array_y_t = calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        // number of good detectors
        Eigen::Index n_good_det = (calib.apt["flag"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                std::get<1>(calib.array_limits[array])-1)).array()==0).count();

        // to hold good detectors
        Eigen::VectorXd x_t, y_t;

        if (n_good_det>0) {
            x_t.resize(n_good_det);
            y_t.resize(n_good_det);

            // remove flagged dets
            Eigen::Index j = std::get<0>(calib.array_limits[array]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<array_x_t.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    x_t(k) = array_x_t(m);
                    y_t(k) = array_y_t(m);
                    k++;
                }
                j++;
            }
            // calculate medians
            array_median_x_t[array_name] = tula::alg::median(x_t);
            array_median_y_t[array_name] = tula::alg::median(y_t);
        }
        else {
            // if no good dets, use all dets to calculate median
            array_median_x_t[array_name] = tula::alg::median(array_x_t);
            array_median_y_t[array_name] = tula::alg::median(array_y_t);
        }
    }

    // remove detectors above distance limits
    logger->debug("flagging detector positions");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = array_indices(i);
        std::string array_name = toltec_io.array_name_map[array_index];

        // calculate distance of detector from mean position of all detectors
        double dist = sqrt(pow(calib.apt["x_t"](i) - array_median_x_t[array_name],2) +
                           pow(calib.apt["y_t"](i) - array_median_y_t[array_name],2));

        // flag detectors that are further than the mean value than the distance limit
        if (dist > max_dist_arcsec[array_name] && max_dist_arcsec[array_name] > 0) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::Position;
        }

        return 0;
    });

    // print number of flagged detectors
    logger->info("{} detectors were flagged", n_flagged_dets);

    // calculate fcf
    logger->debug("calculating flux conversion factors");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = array_indices(i);
        std::string array_name = toltec_io.array_name_map[array_index];

        // calc flux scale (always in mJy/beam)
        if (params(i,0)!=0) {
            calib.apt["flxscale"](i) = beammap_fluxes_mJy_beam[array_name]/params(i,0);
            calib.apt["sens"](i) = calib.apt["sens"](i)*calib.apt["flxscale"](i);
        }
        // set fluxscale (fcf) to zero if flagged
        else {
            calib.apt["flxscale"](i) = 0;
            calib.apt["sens"](i) = 0;
        }
        return 0;
    });

    // re-run calib setup to get average fwhms and beam areas
    calib.setup();

    // calculate source flux in MJy/sr from average beamsizes
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // get source flux in MJy/Sr
        beammap_fluxes_MJy_Sr[array_name] = mJY_ASEC_to_MJY_SR*(beammap_fluxes_mJy_beam[array_name])/calib.array_beam_areas[array];
    }
}

void Beammap::process_apt() {
    // reference detector x and y
    double ref_det_x_t = 0;
    double ref_det_y_t = 0;

    // initial reference det
    beammap_reference_det_found = -99;

    // if particular reference detector is requested
    if (beammap_subtract_reference) {
        if (beammap_reference_det >= 0) {
            beammap_reference_det_found = beammap_reference_det;
            // set reference x_t and y_t
            ref_det_x_t = calib.apt["x_t"](beammap_reference_det_found);
            ref_det_y_t = calib.apt["y_t"](beammap_reference_det_found);
        }
        // else use closest to (0,0) in map 0 apt
        else {
            logger->info("finding a reference detector");
            // calc x_t and y_t values from unflagged detectors for each arrays
            Eigen::Index array = calib.arrays(0);

             // x_t
            auto array_x_t = calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                         std::get<1>(calib.array_limits[array])-1));
            // y_t
            auto array_y_t = calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                         std::get<1>(calib.array_limits[array])-1));
            // number of good detectors
            Eigen::Index n_good_det = (calib.apt["flag"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                    std::get<1>(calib.array_limits[array])-1)).array()==0).count();

            Eigen::VectorXd x_t, y_t, det_indices, dist;

            if (n_good_det>0) {
                x_t.resize(n_good_det);
                y_t.resize(n_good_det);
                det_indices.resize(n_good_det);

                // get good detector positions
                Eigen::Index j = std::get<0>(calib.array_limits[array]);
                Eigen::Index k = 0;
                for (Eigen::Index i=0; i<array_x_t.size(); ++i) {
                    if (calib.apt["flag"](j)==0) {
                        x_t(k) = array_x_t(i);
                        y_t(k) = array_y_t(i);
                        det_indices(k) = j;
                        k++;
                    }
                    j++;
                }

                // distance from (0,0)
                dist = pow(x_t.array(),2) + pow(y_t.array(),2);
            }
            else {
                dist = pow(array_x_t.array(),2) + pow(array_y_t.array(),2);
                det_indices = Eigen::VectorXd::LinSpaced(array_x_t.size(), std::get<0>(calib.array_limits[array]),
                                                         std::get<1>(calib.array_limits[array]));
            }

            // index of detector closest to zero
            auto min_dist = dist.minCoeff(&beammap_reference_det_found);

            // get row in apt table
            beammap_reference_det_found = det_indices(beammap_reference_det_found);

            // set reference x_t and y_t
            ref_det_x_t = calib.apt["x_t"](beammap_reference_det_found);
            ref_det_y_t = calib.apt["y_t"](beammap_reference_det_found);
        }
        logger->info("using detector {} at ({},{}) arcsec",beammap_reference_det_found,
                    static_cast<float>(ref_det_x_t),static_cast<float>(ref_det_y_t));
    }
    else {
        logger->info("no reference detector selected");
    }

    // add reference detector to APT meta data
    calib.apt_meta["reference_x_t"] = ref_det_x_t;
    calib.apt_meta["reference_y_t"] = ref_det_y_t;

    // raw (not derotated or reference detector subtracted) detector x and y values
    calib.apt["x_t_raw"] = calib.apt["x_t"];
    calib.apt["y_t_raw"] = calib.apt["y_t"];

    // align to reference detector if specified and subtract its position from x and y
    calib.apt["x_t"] =  calib.apt["x_t"].array() - ref_det_x_t;
    calib.apt["y_t"] =  calib.apt["y_t"].array() - ref_det_y_t;

    // derotated detector x and y values
    calib.apt["x_t_derot"] = calib.apt["x_t"];
    calib.apt["y_t_derot"] = calib.apt["y_t"];

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
        logger->info("derotating apt");
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

    if (map_grouping=="detector") {
        logger->info("calculating sensitivity");
        // parallelize on detectors
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            Eigen::MatrixXd det_sens, noise_flux;
            // calc sensitivity within psd freq range
            calc_sensitivity(ptcs, det_sens, noise_flux, telescope.d_fsmp, i, {sens_psd_limits_Hz(0), sens_psd_limits_Hz(1)});
            // copy into apt table
            calib.apt["sens"](i) = tula::alg::median(det_sens);

            return 0;
        });
    }

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

        // flag detectors in apt based on config limits
        set_apt_flags(array_indices, nw_indices);

        // subtract reference detector position and derotate
        process_apt();

        // add final apt table to timestream files
        if (run_tod_output && !tod_filename.empty()) {
            // vectors to hold tangent plane pointing for all ptcs (n_chunks x [n_pts x n_dets])
            std::vector<Eigen::MatrixXd> lat, lon;

            // recalculate tangent plane pointing for tod output
            for (Eigen::Index i=0; i<ptcs.size(); ++i) {
                // tangent plane pointing for each detector
                Eigen::MatrixXd ptc_lat(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
                Eigen::MatrixXd ptc_lon(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
                // loop through detectors
                grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto j) {
                    // det indices
                    auto det_index = det_indices(j);
                    double az_off = calib.apt["x_t"](det_index);
                    double el_off = calib.apt["y_t"](det_index);

                    // get tangent pointing
                    auto [det_lat, det_lon] = engine_utils::calc_det_pointing(ptcs[i].tel_data.data, az_off,
                                                                              el_off, telescope.pixel_axes,
                                                                              ptcs[i].pointing_offsets_arcsec.data,
                                                                              map_grouping);
                    ptc_lat.col(j) = std::move(det_lat);
                    ptc_lon.col(j) = std::move(det_lon);

                    return 0;
                });
                lat.push_back(std::move(ptc_lat));
                lon.push_back(std::move(ptc_lon));
            }

            // set parallel policy to sequential for tod output
            parallel_policy = "seq";

            logger->info("adding final apt and detector pointing to tod files");
            // loop through tod files
            for (const auto & [key, val]: tod_filename) {
                netCDF::NcFile fo(val, netCDF::NcFile::write);
                // overwrite apt table
                for (auto const& x: calib.apt) {
                    if (x.first!="flag2") {
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
                // loop through ptcs
                for (Eigen::Index i=0; i<lat.size(); ++i) {
                    // loop through n_pts
                    for (std::size_t j=0; j < TULA_SIZET(lat[i].rows()); ++j) {
                        start_index[0] = k;
                        k++;
                        // append detector latitudes
                        Eigen::VectorXd lat_row = lat[i].row(j);
                        det_lat_v.putVar(start_index, size, lat_row.data());

                        // append detector longitudes
                        Eigen::VectorXd lon_row = lon[i].row(j);
                        det_lon_v.putVar(start_index, size, lon_row.data());

                        if (telescope.pixel_axes == "radec") {
                            // get absolute pointing
                            auto [dec, ra] = engine_utils::tangent_to_abs(lat_row, lon_row, telescope.tel_header["Header.Source.Ra"](0),
                                                                          telescope.tel_header["Header.Source.Dec"](0));
                            // append detector ra
                            det_ra_v.putVar(start_index, size, ra.data());

                            // append detector dec
                            det_dec_v.putVar(start_index, size, dec.data());
                        }
                    }
                }
            }

            // empty ptcdata vector to save memory
            ptcs.clear();
        }
    }

    else {
        // calculate map psds
        logger->info("calculating map psd");
        omb.calc_map_psd();
        // calculate map histograms
        logger->info("calculating map histogram");
        omb.calc_map_hist();
    }
}

template <class KidsProc, class RawObs>
void Beammap::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // only get kids params if not simulation
    if (!telescope.sim_obs) {
        // add kids models to apt
        auto [kids_models, kids_model_header] = kidsproc.load_fit_report(rawobs);

        Eigen::Index i = 0;
        // loop through kids header
        for (const auto &h: kids_model_header) {
            std::string name = h;
            if (name=="flag") {
                name = "kids_flag";
            }
            calib.apt[name].resize(calib.n_dets);
            Eigen::Index j = 0;
            for (const auto &v: kids_models) {
                calib.apt[name].segment(j,v.rows()) = v.col(i);
                j = j + v.rows();
            }

            // search for key
            bool found = false;
            for (const auto &key: calib.apt_header_keys){
                if (key==name) {
                    found = true;
                }
            }
            // if not found, push back placeholder
            if (!found) {
                calib.apt_header_keys.push_back(name);
                calib.apt_header_units[name] = "N/A";
            }

            // detector orientation
            calib.apt_meta[name].push_back("units: N/A");
            calib.apt_meta[name].push_back(name);
            i++;
        }
    }

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
    mapmaking::ObsMapBuffer* mb = nullptr;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;

    // directory name
    std::string dir_name;

    // raw obs maps
    if constexpr (map_type == mapmaking::RawObs) {
        // create output map files
        if (run_mapmaking) {
            create_obs_map_files();
        }
        mb = &omb;
        f_io = &fits_io_vec;
        n_io = &noise_fits_io_vec;
        dir_name = obsnum_dir_name + "raw/";

        // write stats file
        write_stats();

        // add header informqtion to tod
        if (run_tod_output && !tod_filename.empty()) {
            add_tod_header();
        }

        // only write apt table if beammapping
        if (map_grouping=="detector") {
            logger->info("writing apt table");
            auto apt_filename = toltec_io.create_filename<engine_utils::toltecIO::apt, engine_utils::toltecIO::map,
                                                          engine_utils::toltecIO::raw>
                                (obsnum_dir_name + "raw/", redu_type, "", obsnum, telescope.sim_obs);

            Eigen::MatrixXd apt_table(calib.n_dets, calib.apt_header_keys.size());

            // copy to table
            Eigen::Index i = 0;
            for (auto const& x: calib.apt_header_keys) {
                if (x != "flag2") {
                    apt_table.col(i) = calib.apt[x];
                }
                else {
                    apt_table.col(i) = flag2.cast<double> ();
                }
                i++;
            }

            // write to ecsv
            to_ecsv_from_matrix(apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);

            logger->info("done writing apt table {}",apt_filename);
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
                Eigen::Index k = 0;
                Eigen::Index step = 2;

                if (!mb->kernel.empty()) {
                    step++;
                }
                if (!mb->coverage.empty()) {
                    step++;
                }

                // write the maps
                for (Eigen::Index i=0; i<n_maps; ++i) {
                    // update progress bar
                    pb.count(n_maps, 1);
                    write_maps(f_io,n_io,mb,i);

                    if (map_grouping=="detector") {
                        if constexpr (map_type == mapmaking::RawObs) {
                            // get the array for the given map
                            Eigen::Index map_index = arrays_to_maps(i);

                            // check if we move from one file to the next
                            // if so go back to first hdu layer
                            if (i>0) {
                                if (map_index > arrays_to_maps(i-1)) {
                                    k = 0;
                                }
                            }

                            // add apt table
                            for (auto const& key: calib.apt_header_keys) {
                                if (key!="flag2") {
                                    try {
                                        f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, calib.apt[key](i), key
                                                                              + " (" + calib.apt_header_units[key] + ")");
                                    } catch(...) {
                                        f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, 0.0, key
                                                                               + " (" + calib.apt_header_units[key] + ")");
                                    }
                                }
                                else {
                                    f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, flag2(i), key
                                                                           + " (" + calib.apt_header_units[key] + ")");
                                }
                            }
                            // increment hdu layer
                            k = k + step;
                        }
                    }
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

        if (map_grouping!="detector") {
            // write psd and histogram files
            logger->debug("writing psds");
            write_psd<map_type>(mb, dir_name);
            logger->debug("writing histograms");
            write_hist<map_type>(mb, dir_name);
        }
    }
}

