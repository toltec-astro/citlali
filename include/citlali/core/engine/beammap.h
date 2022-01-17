#pragma once

#include <vector>
#include <Eigen/Core>

#include <tula/grppi.h>

#include <citlali/core/engine/engine.h>
#include <citlali/core/utils/fitting.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Beammap: public EngineBase {
public:
    // vector to store each scan's PTCData
    std::vector<TCData<TCDataKind::PTC,Eigen::MatrixXd>> ptcs0;
    std::vector<TCData<TCDataKind::PTC,Eigen::MatrixXd>> ptcs;

    // beammap iteration parameters
    Eigen::Index iteration;

    // number of fit parameters
    int nparams;

    // vector for convergence check
    Eigen::Matrix<bool, Eigen::Dynamic, 1> converged;

    // vector to record convergence iteration
    Eigen::Vector<int, Eigen::Dynamic> converge_iter;

    // previous iteration fit parameters
    Eigen::MatrixXd p0;

    // previous iteration fit parameter errors
    Eigen::MatrixXd perror0;

    // placeholder vectors for grppi maps
    std::vector<int> scan_in_vec, scan_out_vec;
    std::vector<int> det_in_vec, det_out_vec;

    void setup();
    auto run_timestream();
    auto run_loop();

    template <class KidsProc, class RawObs>
    auto timestream_pipeline(KidsProc &, RawObs &);

    template <class KidsProc, class RawObs>
    auto loop_pipeline(KidsProc &, RawObs &);

    template <class KidsProc, class RawObs>
    auto pipeline(KidsProc &, RawObs &);

    template <MapBase::MapType out_type, class MC, typename fits_out_vec_t>
    void output(MC&, fits_out_vec_t &, fits_out_vec_t &);
};

void Beammap::setup() {
    // set number of fit parameters
    nparams = 6;
    // initially all detectors are unconverged
    converged.setZero(ndet);
    // convergence iteration
    converge_iter.resize(ndet);
    converge_iter.setConstant(1);
    // set the initial iteration
    iteration = 0;

    // resize the PTCData vector to number of scans
    ptcs0.resize(scanindices.cols());

    // resize the initial fit vector
    p0.resize(nparams, ndet);
    // set initial fit to nan to pass first cutoff test
    p0.setConstant(std::nan(""));
    // resize the initial fit error vector
    perror0.setZero(nparams, ndet);
    // resize the current fit vector
    mb.pfit.setZero(nparams, ndet);
    mb.perror.setZero(nparams, ndet);

    // set initial elevations for when det is on source
    mb.min_el.setZero(ndet);

    // set initial elevation dist for when det is on source
    mb.el_dist.resize(ndet);
    mb.el_dist.setConstant(std::numeric_limits<double>::max());

    // make filter if requested
    if (run_filter) {
      filter.make_filter();
    }

    if (run_downsample) {
        // set the downsampled sample rate
        dfsmp = fsmp/downsampler.dsf;
    }

    else {
        dfsmp = fsmp;
    }

    // toltec input/output class
    ToltecIO toltec_io;

    // empty the fits vector for subsequent observations
    fits_ios.clear();

    // create empty FITS files at start
    for (Eigen::Index i=0; i<array_indices.size(); i++) {
        std::string filename;
        filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                ToltecIO::beammap, ToltecIO::no_prod_type, ToltecIO::obsnum_true>(filepath,obsnum,i);

        FitsIO<fileType::write_fits, CCfits::ExtHDU*> fits_io(filename);
        fits_ios.push_back(std::move(fits_io));
    }
}

auto Beammap::run_timestream() {
    auto farm = grppi::farm(nthreads,[&](auto input_tuple) -> TCData<TCDataKind::PTC,Eigen::MatrixXd> {
        // RTCData input
        auto in = std::get<0>(input_tuple);
        // start index
        auto start_index = std::get<1>(input_tuple);

        // current length of outer scans
        Eigen::Index scan_length = in.scans.data.rows();

        // set up flag matrix
        in.flags.data.setOnes(scan_length, ndet);

        // copy tel_meta_data for scan
        in.tel_meta_data.data["TelElDes"] = tel_meta_data["TelElDes"].segment(start_index, scan_length);
        in.tel_meta_data.data["ParAng"] = tel_meta_data["ParAng"].segment(start_index, scan_length);

        in.tel_meta_data.data["TelLatPhys"] = tel_meta_data["TelLatPhys"].segment(start_index, scan_length);
        in.tel_meta_data.data["TelLonPhys"] = tel_meta_data["TelLonPhys"].segment(start_index, scan_length);

        in.tel_meta_data.data["SourceEl"] = tel_meta_data["SourceEl"].segment(start_index, scan_length);

        /*Stage 1: RTCProc*/
        RTCProc rtcproc;
        TCData<TCDataKind::PTC,Eigen::MatrixXd> out;
        rtcproc.run(in, out, this);

        // move out into the PTCData vector
        ptcs0.at(out.index.data - 1) = std::move(out);

        SPDLOG_INFO("done with scan {}", out.index.data);

        return out;
    });

    return farm;
}

auto Beammap::run_loop() {
    auto loop = grppi::repeat_until([&](auto in) {
        // vector to hold the ptcs for each iteration
        // reset it to the original ptc vector each time
        ptcs = ptcs0;

        /*Stage 2: PTCProc*/
        PTCProc ptcproc;

        // set maps to zero on each iteration
        for (Eigen::Index i=0; i<mb.map_count; i++) {
            mb.signal.at(i).setZero();
            mb.weight.at(i).setZero();
            //mb.coverage.at(i).setZero();

            if (run_kernel) {
                mb.kernel.at(i).setZero();
            }
        }

        grppi::map(tula::grppi_utils::dyn_ex(ex_name), scan_in_vec, scan_out_vec, [&](auto s) {
            SPDLOG_INFO("reducing scan {}/{}", s+1, ptcs.size());
            // subtract gaussian if iteration > 0
            if (iteration > 0) {
                {
                    tula::logging::scoped_timeit timer("subtract gaussian");
                    mb.pfit.row(0) = -mb.pfit.row(0);
                    add_gaussian(this,ptcs.at(s).scans.data, ptcs.at(s).tel_meta_data.data);
                }
            }

            // clean scan
            {
                tula::logging::scoped_timeit timer("ptcproc.run()");
                ptcproc.run(ptcs.at(s), ptcs.at(s), this);
            }

            // add gaussian if iteration > 0
            if (iteration > 0) {
                {
                    tula::logging::scoped_timeit timer("add gaussian()");
                    mb.pfit.row(0) = -mb.pfit.row(0);
                    add_gaussian(this,ptcs.at(s).scans.data, ptcs.at(s).tel_meta_data.data);
                }
            }

            /*Stage 3 Populate Map*/
            if (mapping_method == "naive") {
                {
                    tula::logging::scoped_timeit timer("populate_maps_naive()");
                    populate_maps_naive(ptcs.at(s), this);
                }
            }

            return 0;});

        SPDLOG_INFO("normalizing maps");
        mb.normalize_maps(run_kernel);

        SPDLOG_INFO("fitting maps");
        grppi::map(tula::grppi_utils::dyn_ex(ex_name), det_in_vec, det_out_vec, [&](auto d) {
            if (converged(d) == false) {
                SPDLOG_INFO("fitting detector {}/{}",d+1, det_in_vec.size());
                // declare fitter class for detector
                gaussfit::MapFitter fitter;
                // size of region to fit in pixels
                fitter.bounding_box_pix = bounding_box_pix;
                mb.pfit.col(d) = fitter.fit<gaussfit::MapFitter::peakValue>(mb.signal[d], mb.weight[d], calib_data);
                mb.perror.col(d) = fitter.error;
            }
            else {
                mb.pfit.col(d) = p0.col(d);
                mb.perror.col(d) = perror0.col(d);
            }
            return 0;});

        return in;
    },

    [&](auto in) {
        SPDLOG_INFO("iteration {} done", iteration);
        // increment iteration
        iteration++;

        // variable to return if we're complete (true) or not (false)
        bool complete = false;

        if (iteration < max_iterations) {
            // check if all detectors are converged
            if ((converged.array() == 1).all()) {
                complete = true;
            }
            else {
                // loop through and find if any are converged
                grppi::map(tula::grppi_utils::dyn_ex(ex_name), det_in_vec, det_out_vec, [&](auto d) {
                    if (converged(d) == false) {
                        // percent difference between current and previous iteration's fit
                        auto ratio = abs((mb.pfit.col(d).array() - p0.col(d).array())/p0.col(d).array());
                        // if the detector is converged, set it to converged
                        if ((ratio.array() <= cutoff).all()) {
                            converged(d) = true;
                            converge_iter(d) = iteration;
                        }
                    }
                    return 0;});
                SPDLOG_INFO("converged detectors {}",(converged.array() == true).count());
            }
            // set previous iteration fit to current fit
            p0 = mb.pfit;
            perror0 = mb.perror;
        }
        else {
            // we're done!
            complete = true;
        }

        // if this is false, the loop_pipeline will restart
        return complete;
    });

    return loop;
}

template <class KidsProc, class RawObs>
auto Beammap::timestream_pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    grppi::pipeline(tula::grppi_utils::dyn_ex(ex_name),
        [&]() -> std::optional<std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, Eigen::Index>> {
        // variable to hold current scan
        static auto scan = 0;
        // length of current outer scan
        Eigen::Index scan_length;
        // start index of current scan
        Eigen::Index start_index = 0;
        // main grppi loop
        while (scan < scanindices.cols()) {
            SPDLOG_INFO("reducing scan {}", scan + 1);
            // start index of current scan
            start_index = scanindices(2, scan);
            // length of current scan
            scan_length = scanindices(3, scan) - scanindices(2, scan) + 1;

            TCData<TCDataKind::RTC, Eigen::MatrixXd> rtc;
            // current scanindices (inner si, inner ei, outer si, outer ei)
            rtc.scan_indices.data = scanindices.col(scan);
            // current scan number for outputting progress
            rtc.index.data = scan + 1;

            // run kidsproc to get correct units
            rtc.scans.data = kidsproc.populate_rtc(rawobs, rtc.scan_indices.data, scan_length, ndet);

            // increment scan
            scan++;

            return std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, Eigen::Index> (rtc, start_index);
       }
       return {};
    },
    run_timestream());

}

template <class KidsProc, class RawObs>
auto Beammap::loop_pipeline(KidsProc &kidproc, RawObs &rawobs) {
    // note that this pipeline is forced to be sequential
    // this ensures that grppi::map parallelization works!
    grppi::pipeline(tula::grppi_utils::dyn_ex("seq"),
        [&]() -> std::optional<Eigen::Index> {

         // place holder variable to make pipeline work
         static auto place_holder = 0;
         while (place_holder < 1) {
               return place_holder++;
         }
         return {};
        },
        run_loop(),

        [&](auto in) {
            // convert to map units (arcsec and radians)
            mb.pfit.row(1) = pixel_size*(mb.pfit.row(1).array() - (mb.ncols)/2)/RAD_ASEC;
            mb.pfit.row(2) = pixel_size*(mb.pfit.row(2).array() - (mb.nrows)/2)/RAD_ASEC;
            mb.pfit.row(3) = STD_TO_FWHM*pixel_size*(mb.pfit.row(3))/RAD_ASEC;
            mb.pfit.row(4) = STD_TO_FWHM*pixel_size*(mb.pfit.row(4))/RAD_ASEC;

            // rescale errors from pixel to on-sky units
            mb.perror.row(1) = pixel_size*(mb.perror.row(1))/RAD_ASEC;
            mb.perror.row(2) = pixel_size*(mb.perror.row(2))/RAD_ASEC;
            mb.perror.row(3) = STD_TO_FWHM*pixel_size*(mb.perror.row(3))/RAD_ASEC;
            mb.perror.row(4) = STD_TO_FWHM*pixel_size*(mb.perror.row(4))/RAD_ASEC;

            // derotate x_t and y_t
            SPDLOG_INFO("min el {}", mb.min_el);
            SPDLOG_INFO("el dist {}", mb.el_dist);

            // need to copy because of aliasing?
            Eigen::VectorXd rot_azoff = cos(-mb.min_el.array())*mb.pfit.row(1).array() -
                    sin(-mb.min_el.array())*mb.pfit.row(2).array();
            Eigen::VectorXd rot_eloff = cos(-mb.min_el.array())*mb.pfit.row(2).array() +
                    sin(-mb.min_el.array())*mb.pfit.row(1).array();

            mb.pfit.row(1) = -rot_azoff;
            mb.pfit.row(2) = -rot_eloff;

            // calculate sensitivity for detectors
            grppi::map(tula::grppi_utils::dyn_ex(ex_name), det_in_vec, det_out_vec, [&](int d) {
                SPDLOG_INFO("calculating sensitivity for det {}", d);
                Eigen::MatrixXd det_sens;
                Eigen::MatrixXd noise_flux;
                calc_sensitivity(ptcs, det_sens, noise_flux, dfsmp, d);
                sensitivity(d) = det_sens.mean();
            return 0;});

            return in;
        });
}

template <class KidsProc, class RawObs>
auto Beammap::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // start the timestream pipeline (RTCProc only)
    SPDLOG_INFO("starting beammap timestream pipeline");
    timestream_pipeline(kidsproc, rawobs);

    // placeholder vectors of size nscans for grppi maps
    scan_in_vec.resize(ptcs0.size());
    std::iota(scan_in_vec.begin(), scan_in_vec.end(), 0);
    scan_out_vec.resize(ptcs0.size());

    // placeholder vectors of size ndet for grppi maps
    det_in_vec.resize(ndet);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    det_out_vec.resize(ndet);

    // start the iterative pipeline (clean, map, and fit)
    SPDLOG_INFO("starting iterative pipeline");
    loop_pipeline(kidsproc, rawobs);

    SPDLOG_INFO("beammapping finished");
}

template <MapBase::MapType out_type, class MC, typename fits_out_vec_t>
void Beammap::output(MC &mout, fits_out_vec_t &f_ios, fits_out_vec_t & nf_ios) {
    if constexpr (out_type==MapType::obs) {
        // apt table
        SPDLOG_INFO("writing apt table");
        ToltecIO toltec_io;
        // get output path from citlali_config
        auto filename = toltec_io.setup_filepath<ToltecIO::apt,ToltecIO::simu,
                ToltecIO::beammap, ToltecIO::no_prod_type, ToltecIO::obsnum_true>(filepath,obsnum,-1);

        // check in debug mode for row/col error (seems fine)
        Eigen::MatrixXf table(toltec_io.beammap_apt_header.size(), ndet);
        table.row(0) = calib_data["array"].cast <float> ();
        table.row(1) = calib_data["nw"].cast <float> ();
        table.row(2) = calib_data["flxscale"].cast <float> ();
        table.row(3) = sensitivity.cast <float> ();
        //table.block(3,0,nparams,ndet) = mout.pfit.template cast <float> ();
        //table.row(9) = converge_iter.cast <float> ();

        int ci = 0;
        for (int ti=0; ti < toltec_io.apt_header.size()-1; ti=ti+2) {
            table.row(ti+4) = mout.pfit.row(ci).template cast <float> ();
            table.row(ti+4 + 1) = mout.perror.row(ci).template cast <float> ();
            ci++;
        }

        table.row(toltec_io.beammap_apt_header.size()-1) = converge_iter.cast <float> ();

        table.transposeInPlace();

        SPDLOG_INFO("beammap fit table header {}", toltec_io.beammap_apt_header);
        SPDLOG_INFO("beammap fit table {}", table);

        // Yaml node for ecsv table meta data (units and description)
        YAML::Node meta;
        meta["array"].push_back("units: N/A");
        meta["array"].push_back("array index");

        meta["nw"].push_back("units: N/A");
        meta["nw"].push_back("network index");

        meta["flxscale"].push_back("units: Mjy/sr");
        meta["flxscale"].push_back("flux conversion scale");

        meta["amp"].push_back("units: Mjy/sr");
        meta["amp"].push_back("fitted amplitude");

        meta["amp_err"].push_back("units: Mjy/sr");
        meta["amp_err"].push_back("fitted amplitude error");

        meta["x_t"].push_back("units: arcsec");
        meta["x_t"].push_back("fitted azimuthal offset");

        meta["x_t_err"].push_back("units: arcsec");
        meta["x_t_err"].push_back("fitted azimuthal offset error");

        meta["y_t"].push_back("units: arcsec");
        meta["y_t"].push_back("fitted altitude offset");

        meta["y_t_err"].push_back("units: arcsec");
        meta["y_t_err"].push_back("fitted altitude offset error");

        meta["a_fwhm"].push_back("units: arcsec");
        meta["a_fwhm"].push_back("fitted azimuthal FWHM");

        meta["a_fwhm_err"].push_back("units: arcsec");
        meta["a_fwhm_err"].push_back("fitted azimuthal FWHM error");

        meta["b_fwhm"].push_back("units: arcsec");
        meta["b_fwhm"].push_back("fitted altitude FWMH");

        meta["b_fwhm_err"].push_back("units: arcsec");
        meta["b_fwhm_err"].push_back("fitted altitude FWMH error");

        meta["angle"].push_back("units: radians");
        meta["angle"].push_back("fitted rotation angle");

        meta["angle_err"].push_back("units: radians");
        meta["angle_err"].push_back("fitted rotation angle error");

        meta["converge_iter"].push_back("units: N/A");
        meta["converge_iter"].push_back("beammap convergence iteration");

        // write apt table to ecsv file
        to_ecsv_from_matrix(filename, table, toltec_io.beammap_apt_header, meta);
        SPDLOG_INFO("successfully wrote apt table to {}.ecsv", filename);

        SPDLOG_INFO("writing maps");
        // loop through existing files
        for (Eigen::Index i=0; i<array_indices.size(); i++) {
            SPDLOG_INFO("writing {}.fits", f_ios.at(i).filepath);
            // loop through maps and save them as an hdu
            // current detector
            auto start_det = std::get<0>(array_indices.at(i));
            // size of block for each grouping
            auto end_det = std::get<1>(array_indices.at(i)) + 1;

            for (Eigen::Index j=start_det; j<end_det; j++) {
                //SPDLOG_INFO("writing sig{}", j);
                f_ios.at(i).add_hdu("sig" + std::to_string(j), mout.signal.at(j));
                //SPDLOG_INFO("writing wt{}", j);
                f_ios.at(i).add_hdu("wt" + std::to_string(j), mout.weight.at(j));

                // write kernel if requested
                if (run_kernel) {
                    //SPDLOG_INFO("writing ker{}", j);
                    f_ios.at(i).add_hdu("ker" + std::to_string(j), mout.kernel.at(j));
                }
            }

            // loop through hdus and add wcs (hacky method)
            int j = start_det;
            int k = 0;
            int nhdus = 2;

            if (run_kernel) {
                nhdus = 3;
            }

            for (auto hdu: f_ios.at(i).hdus) {
                f_ios.at(i).template add_wcs<UnitsType::arcsec>(hdu,map_type,mout.nrows,mout.ncols,
                                                       pixel_size,source_center);
                // add fit parameters to hdus
                hdu->addKey("amp", (float)mout.pfit(0,i),"amplitude (Mjy/sr)");
                hdu->addKey("amp_err", (float)mout.perror(0,i),"amplitude error (Mjy/sr)");
                hdu->addKey("x_t", (float)mout.pfit(1,i),"az offset (arcsec)");
                hdu->addKey("x_t_err", (float)mout.perror(1,i),"az offset error (arcsec)");
                hdu->addKey("y_t", (float)mout.pfit(2,i),"alt offset (arcsec)");
                hdu->addKey("y_t_err", (float)mout.perror(2,i),"alt offset error (arcsec)");
                hdu->addKey("a_fwhm", (float)mout.pfit(3,i),"az fwhm (arcsec)");
                hdu->addKey("a_fwhm_err", (float)mout.perror(3,i),"az fwhm error (arcsec)");
                hdu->addKey("b_fwhm", (float)mout.pfit(4,i),"alt fwhm (arcsec)");
                hdu->addKey("b_fwhm_err", (float)mout.perror(4,i),"alt fwhm error (arcsec)");
                hdu->addKey("angle", (float)mout.pfit(5,i),"position angle (radians)");
                hdu->addKey("angle_err", (float)mout.perror(5,i),"position angle error (radians)");

                k++;

                // only increment every nhdus
                if (k == nhdus) {
                    k = 0;
                    j++;
                }
            }

            // loop through default TolTEC fits header keys and add to primary header
            for (auto const& pair : toltec_io.fits_header_keys) {
                f_ios.at(i).pfits->pHDU().addKey(pair.first, pair.second, " ");
            }

            // add wcs to pHDU
            f_ios.at(i).template add_wcs<UnitsType::arcsec>(&f_ios.at(i).pfits->pHDU(),map_type,mout.nrows,
                                                      mout.ncols,pixel_size,source_center);

            // add wavelength
            f_ios.at(i).pfits->pHDU().addKey("WAV", toltec_io.name_keys[i], "Array Name");
            // add obsnum
            f_ios.at(i).pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
            // add units
            f_ios.at(i).pfits->pHDU().addKey("UNIT", obsnum, "MJy/Sr");
            // add conversion
            f_ios.at(i).pfits->pHDU().addKey("to_mjy/b", toltec_io.barea_keys[i]*MJY_SR_TO_mJY_ASEC, "Conversion to mJy/beam");
        }
    }
}
