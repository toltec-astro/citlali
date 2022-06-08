#pragma once

#include <vector>
#include <Eigen/Core>

#include <tula/grppi.h>
#include <kids/core/kidsdata.h>

#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/constants.h>
#include <citlali/core/engine/engine.h>
#include <citlali/core/utils/fitting.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

using namespace mapmaking;

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
    void output(MC&, fits_out_vec_t &, fits_out_vec_t &, bool);
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

    // empty the fits vector for subsequent observations
    fits_ios.clear();

    // get obsnum directory name inside redu directory name
    std::stringstream ss_redu;
    ss_redu << std::setfill('0') << std::setw(2) << redu_num;

    std::string hdname;

    if (use_subdir) {
        hdname = "redu" + ss_redu.str() + "/";
    }

    else {
        hdname = "";
    }

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << obsnum;
    std::string dname = hdname + ss.str() + "/";

    // create obsnum directory
    toltec_io.setup_output_directory(filepath, dname);

    // create empty FITS files at start
    for (Eigen::Index i=0; i<arrays.size(); i++) {
        std::string filename;
        filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                ToltecIO::beammap, ToltecIO::no_prod_type, ToltecIO::obsnum_true>(filepath + dname,obsnum,i);

        FitsIO<fileType::write_fits, CCfits::ExtHDU*> fits_io(filename);
        fits_ios.push_back(std::move(fits_io));
    }

    if (run_tod_output) {
        if (ts_format == "netcdf") {
            ts_rows = 0;

            std::string filename;

            if (reduction_type == "science") {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::science, ToltecIO::timestream,
                                                    ToltecIO::obsnum_true>(filepath + dname,obsnum,-1);
            }

            else if (reduction_type == "pointing") {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::pointing, ToltecIO::timestream,
                                                    ToltecIO::obsnum_true>(filepath + dname,obsnum,-1);
            }

            Eigen::Index ndim_pol = (calib_data["fg"].array() == 0).count() + (calib_data["fg"].array() == 1).count();

            SPDLOG_INFO("ndim_pol {}", ndim_pol);

            for (auto const& stokes_params: polarization.stokes_params) {

                ts_filepath.push_back(filename + "_" + stokes_params.first + ".nc");

                netCDF::NcFile fo(ts_filepath.back(), netCDF::NcFile::replace);
                netCDF::NcDim nsmp_dim = fo.addDim("nsamples");

                std::vector<netCDF::NcDim> dims;
                dims.push_back(nsmp_dim);

                if (stokes_params.first == "I") {
                    netCDF::NcDim ndet_dim = fo.addDim("ndetectors",ndet);
                    dims.push_back(ndet_dim);
                }

                else if (stokes_params.first == "Q") {
                    netCDF::NcDim ndet_dim = fo.addDim("ndetectors",ndim_pol);
                    dims.push_back(ndet_dim);
                }

                if (stokes_params.first == "U") {
                    netCDF::NcDim ndet_dim = fo.addDim("ndetectors",ndim_pol);
                    dims.push_back(ndet_dim);
                }

                netCDF::NcVar pixid_v = fo.addVar("PIXID",netCDF::ncInt, dims[1]);
                pixid_v.putAtt("Units","N/A");
                netCDF::NcVar a_v = fo.addVar("ARRAYID",netCDF::ncDouble, dims[1]);
                a_v.putAtt("Units","N/A");

                netCDF::NcVar xt_v = fo.addVar("AZOFF",netCDF::ncDouble, dims[1]);
                xt_v.putAtt("Units","radians");
                netCDF::NcVar yt_v = fo.addVar("ELOFF",netCDF::ncDouble, dims[1]);
                yt_v.putAtt("Units","radians");

                netCDF::NcVar afwhm_v = fo.addVar("AFWHM",netCDF::ncDouble, dims[1]);
                afwhm_v.putAtt("Units","radians");
                netCDF::NcVar bfwhm_v = fo.addVar("BFWHM",netCDF::ncDouble, dims[1]);
                bfwhm_v.putAtt("Units","radians");

                netCDF::NcVar t_v = fo.addVar("TIME",netCDF::ncDouble, nsmp_dim);
                t_v.putAtt("Units","seconds");
                netCDF::NcVar e_v = fo.addVar("ELEV",netCDF::ncDouble, nsmp_dim);
                e_v.putAtt("Units","radians");

                netCDF::NcVar data_v = fo.addVar("DATA",netCDF::ncDouble, dims);
                data_v.putAtt("Units","MJy/sr");
                netCDF::NcVar flag_v = fo.addVar("FLAG",netCDF::ncDouble, dims);
                flag_v.putAtt("Units","N/A");

                netCDF::NcVar lat_v = fo.addVar("DY",netCDF::ncDouble, dims);
                lat_v.putAtt("Units","radians");
                netCDF::NcVar lon_v = fo.addVar("DX",netCDF::ncDouble, dims);
                lon_v.putAtt("Units","radians");

                fo.close();
            }
        }
    }
}

auto Beammap::run_timestream() {
    auto farm = grppi::farm(nthreads,[&](auto input_tuple) -> TCData<TCDataKind::PTC,Eigen::MatrixXd> {
        // RTCData input
        auto in = std::get<0>(input_tuple);
        // kidsproc
        auto kidsproc = std::get<1>(input_tuple);
        // start index input
        auto loaded_rawobs = std::get<2>(input_tuple);

        // starting index for scan
        Eigen::Index start_index = in.scan_indices.data(2);

        // current length of outer scans
        Eigen::Index scan_length = in.scan_indices.data(3) - in.scan_indices.data(2) + 1;

        // set up flag matrix
        in.flags.data.setOnes(scan_length, ndet);

        // copy tel_meta_data for scan
        in.tel_meta_data.data["TelTime"] = tel_meta_data["TelTime"].segment(start_index, scan_length);

        in.tel_meta_data.data["TelElDes"] = tel_meta_data["TelElDes"].segment(start_index, scan_length);
        in.tel_meta_data.data["ParAng"] = tel_meta_data["ParAng"].segment(start_index, scan_length);

        in.tel_meta_data.data["TelLatPhys"] = tel_meta_data["TelLatPhys"].segment(start_index, scan_length);
        in.tel_meta_data.data["TelLonPhys"] = tel_meta_data["TelLonPhys"].segment(start_index, scan_length);

        in.tel_meta_data.data["SourceEl"] = tel_meta_data["SourceEl"].segment(start_index, scan_length);

        /*Stage 0: KidsProc*/
        {
            tula::logging::scoped_timeit timer("kidsproc.populate_rtc_load()");
            tula::logging::scoped_loglevel<spdlog::level::critical> _0;
            in.scans.data = kidsproc.populate_rtc_load(loaded_rawobs,in.scan_indices.data, scan_length, ndet);
        }

        // do polarization to get map and detector index vectors
        auto [map_index_vector, det_index_vector] =  polarization.create_rtc(in, in, "I", this);

        /*Stage 1: RTCProc*/
        RTCProc rtcproc;
        TCData<TCDataKind::PTC,Eigen::MatrixXd> out;
        rtcproc.run(in, out, map_index_vector, det_index_vector, this);

        out.map_index_vector.data = map_index_vector;
        out.det_index_vector.data = det_index_vector;

        // timestream output (seq only)
        if (run_tod_output) {
            // we use out here due to filtering and downsampling
            if (ts_chunk_type == "rtc") {

                Eigen::MatrixXd lat(out.scans.data.rows(),out.scans.data.cols());
                Eigen::MatrixXd lon(out.scans.data.rows(),out.scans.data.cols());

                SPDLOG_INFO("writing scan RTC timestream {} to {}", in.index.data, ts_filepath[0]);
                // loop through detectors and get pointing timestream
                for (Eigen::Index i=0; i<out.scans.data.cols(); i++) {

                    // get offsets
                    auto azoff = calib_data["x_t"](i);
                    auto eloff = calib_data["y_t"](i);

                    // get pointing
                    auto [lat_i, lon_i] = engine_utils::get_det_pointing(out.tel_meta_data.data, azoff, eloff, map_type, pointing_offsets);
                    lat.col(i) = lat_i;
                    lon.col(i) = lon_i;
                }
                // append to netcdf file
                append_to_netcdf(ts_filepath[0], out.scans.data, out.flags.data, lat, lon,
                                 out.tel_meta_data.data["TelElDes"], out.tel_meta_data.data["TelTime"],
                                 det_index_vector, calib_data,out.scans.data.cols());
            }
        }

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

        if (iteration == 0) {
            if (run_tod_output) {
                if (ts_chunk_type == "ptc") {
                    for (Eigen::Index s=0; s<ptcs.size(); s++) {
                        Eigen::MatrixXd lat(ptcs[s].scans.data.rows(), ptcs[s].scans.data.cols());
                        Eigen::MatrixXd lon(ptcs[s].scans.data.rows(), ptcs[s].scans.data.cols());

                        SPDLOG_INFO("writing scan PTC timestream {} to {}", ptcs[s].index.data, ts_filepath[0]);
                        // loop through detectors and get pointing timestream
                        for (Eigen::Index i=0; i<ptcs[s].scans.data.cols(); i++) {

                            // get offsets
                            auto azoff = calib_data["x_t"](i);
                            auto eloff = calib_data["y_t"](i);

                            // get pointing
                            auto [lat_i, lon_i] = engine_utils::get_det_pointing(ptcs[s].tel_meta_data.data, azoff, eloff,
                                                                                 map_type, pointing_offsets);
                            lat.col(i) = lat_i;
                            lon.col(i) = lon_i;
                        }
                        append_to_netcdf(ts_filepath[0], ptcs[s].scans.data, ptcs[s].flags.data, lat, lon,
                                         ptcs[s].tel_meta_data.data["TelElDes"], ptcs[s].tel_meta_data.data["TelTime"],
                                         ptcs[s].det_index_vector.data, calib_data, ptcs[s].scans.data.cols());
                    }
                }
            }
        }

        // set maps to zero on each iteration
        for (Eigen::Index i=0; i<mb.map_count; i++) {
            mb.signal.at(i).setZero();
            mb.weight.at(i).setZero();

            if (run_kernel) {
                mb.kernel.at(i).setZero();
            }
        }

        /*Stage 2: PTCProc*/
        PTCProc ptcproc;

        grppi::map(tula::grppi_utils::dyn_ex(ex_name), scan_in_vec, scan_out_vec, [&](auto s) {
            SPDLOG_INFO("reducing scan {}/{}", s+1, ptcs.size());
            // subtract gaussian if iteration > 0
            if (iteration > 0) {
                {
                    tula::logging::scoped_timeit timer("subtract gaussian");
                    mb.pfit.row(0) = -mb.pfit.row(0);
                    add_gaussian_2(this,ptcs.at(s).scans.data, ptcs.at(s).tel_meta_data.data);
                }
            }

            // clean scan
            {
                tula::logging::scoped_timeit timer("ptcproc.run()");
                ptcproc.run(ptcs.at(s), ptcs.at(s), this, run_clean);
            }

            // add gaussian if iteration > 0
            if (iteration > 0) {
                {
                    tula::logging::scoped_timeit timer("add gaussian()");
                    mb.pfit.row(0) = -mb.pfit.row(0);
                    add_gaussian_2(this,ptcs.at(s).scans.data, ptcs.at(s).tel_meta_data.data);
                }
            }

            /*Stage 3 Populate Map*/
            if (mapping_method == "naive") {
                {
                    tula::logging::scoped_timeit timer("populate_maps_naive()");
                    populate_maps_naive(ptcs.at(s), ptcs.at(s).map_index_vector.data, ptcs.at(s).det_index_vector.data, this);
                }
            }

            else if (mapping_method == "jinc") {
                {
                    tula::logging::scoped_timeit timer("populate_maps_naive()");
                    populate_maps_jinc(ptcs.at(s), ptcs.at(s).map_index_vector.data, ptcs.at(s).det_index_vector.data, this);
                }
            }

            return 0;});

        SPDLOG_INFO("normalizing maps");
        mb.normalize_maps(run_kernel,ex_name);

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

    [&](auto &in) {
        SPDLOG_INFO("iteration {} done", iteration);
        // increment iteration
        iteration++;

        // variable to return if we're complete (true) or not (false)
        bool complete = false;

        if (iteration < max_iterations) {
            // check if all detectors are converged
            if ((converged.array() == true).all()) {
                complete = true;
            }
            else {
                // loop through and find if any are converged
                grppi::map(tula::grppi_utils::dyn_ex(ex_name), det_in_vec, det_out_vec, [&](auto d) {
                    if (converged(d) == false) {
                        // percent difference between current and previous iteration's fit
                        auto ratio = abs((mb.pfit.col(d).array() - p0.col(d).array())/p0.col(d).array());
                        SPDLOG_INFO("mb.pfit.col(d) {}", mb.pfit.col(d));
                        SPDLOG_INFO("p0.col(d) {}", p0.col(d));

			SPDLOG_INFO("ratio {}", ratio);
                        // if the detector is converged, set it to converged
                        if ((ratio.array() <= cutoff).all()) {
                            converged(d) = true;
                            converge_iter(d) = iteration;
                        }
                    }
                    return 0;});
                SPDLOG_INFO("converged detectors {}", (converged.array() == true).count());
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
        [&]() -> std::optional<std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                          std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>>> {
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
            //rtc.scans.data = kidsproc.populate_rtc(rawobs, rtc.scan_indices.data, scan_length, ndet);
            // run kidsproc to get correct units
            std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> loaded_rawobs;
            {
                tula::logging::scoped_timeit timer("kidsproc.load_rawobs()");
                tula::logging::scoped_loglevel<spdlog::level::off> _0;
                auto slice = tula::container_utils::Slice<int>{
                                                               scanindices(2,scan), scanindices(3,scan) + 1, std::nullopt};
                loaded_rawobs = kidsproc.load_rawobs(rawobs, slice);
                //rtc.scans.data = kidsproc.populate_rtc(rawobs, rtc.scan_indices.data, scan_length, ndet);
            }

            // increment scan
            scan++;

            return std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                              std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>> (rtc, kidsproc, loaded_rawobs);
       }
        scan = 0;
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
            mb.pfit.row(1) = pixel_size*(mb.pfit.row(1).array() - (mb.ncols)/2)/ASEC_TO_RAD;
            mb.pfit.row(2) = pixel_size*(mb.pfit.row(2).array() - (mb.nrows)/2)/ASEC_TO_RAD;
            mb.pfit.row(3) = STD_TO_FWHM*pixel_size*(mb.pfit.row(3))/ASEC_TO_RAD;
            mb.pfit.row(4) = STD_TO_FWHM*pixel_size*(mb.pfit.row(4))/ASEC_TO_RAD;

            // rescale errors from pixel to on-sky units
            mb.perror.row(1) = pixel_size*(mb.perror.row(1))/ASEC_TO_RAD;
            mb.perror.row(2) = pixel_size*(mb.perror.row(2))/ASEC_TO_RAD;
            mb.perror.row(3) = STD_TO_FWHM*pixel_size*(mb.perror.row(3))/ASEC_TO_RAD;
            mb.perror.row(4) = STD_TO_FWHM*pixel_size*(mb.perror.row(4))/ASEC_TO_RAD;

	    double mean_el = tel_meta_data["TelElDes"].mean();

            Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> map_index_vector = ptcs.back().map_index_vector.data;

            // derotate x_t and y_t and calculate sensitivity for detectors
            SPDLOG_INFO("derotating offsets and calculating sensitivity");
            grppi::map(tula::grppi_utils::dyn_ex(ex_name), det_in_vec, det_out_vec, [&](int d) {

                Eigen::Index mi = map_index_vector(d);

                mb.pfit(0,d) = beammap_fluxes[toltec_io.name_keys[mi]]/mb.pfit(0,d);

                //SPDLOG_INFO("derotating det {}", d);
                Eigen::Index min_index;

                // don't use pointing here as that will rotate by azoff/eloff
                Eigen::VectorXd lat = -(mb.pfit(2,d)*ASEC_TO_RAD) + tel_meta_data["TelElDes"].array();
                Eigen::VectorXd lon = -(mb.pfit(1,d)*ASEC_TO_RAD) + tel_meta_data["TelAzDes"].array();

                // minimum cartesian distance from source
                double min_dist = ((tel_meta_data["SourceEl"] - lat).array().pow(2) +
                        (tel_meta_data["SourceAz"] - lon).array().pow(2)).minCoeff(&min_index);

                double min_el = tel_meta_data["TelElDes"](min_index);

                double rot_azoff = cos(-min_el)*mb.pfit(1,d) -
                        sin(-min_el)*mb.pfit(2,d);
                double rot_eloff = sin(-min_el)*mb.pfit(1,d) +
                        cos(-min_el)*mb.pfit(2,d);

                mb.pfit(1,d) = -rot_azoff;
                mb.pfit(2,d) = -rot_eloff; 

                //SPDLOG_INFO("calculating sensitivity for det {}", d);
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
    {
        tula::logging::scoped_timeit timer("timestream_pipeline");
        timestream_pipeline(kidsproc, rawobs);
    }

    // placeholder vectors of size nscans for grppi maps
    scan_in_vec.resize(ptcs0.size());
    std::iota(scan_in_vec.begin(), scan_in_vec.end(), 0);
    scan_out_vec.resize(ptcs0.size());

    // placeholder vectors of size ndet for grppi maps
    det_in_vec.resize(ndet);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    det_out_vec.resize(ndet);

    // start the iterative pipeline (clean, map, and fit, check convergence)
    SPDLOG_INFO("starting iterative pipeline");
    {
        tula::logging::scoped_timeit timer("loop_pipeline");
        loop_pipeline(kidsproc, rawobs);
    }

    SPDLOG_INFO("beammapping finished");
}

template <MapBase::MapType out_type, class MC, typename fits_out_vec_t>
void Beammap::output(MC &mout, fits_out_vec_t &f_ios, fits_out_vec_t & nf_ios, bool filtered) {

    std::string hdname;

    if (use_subdir) {
        // get obsnum directory name inside redu directory name
        std::stringstream ss_redu;
        ss_redu << std::setfill('0') << std::setw(2) << redu_num;

        hdname = "redu" + ss_redu.str() + "/";
    }

    else {
        hdname = "";
    }

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << obsnum;
    std::string dname = hdname + ss.str() + "/";

    std::string cname = hdname + "coadded/";

    if (filtered==false) {
        cname = cname + "raw/";
    }

    else if (filtered==true) {
        cname = cname + "filtered/";
    }

    if constexpr (out_type==MapType::obs) {
        // apt table
        SPDLOG_INFO("writing apt table");
        // get output path from citlali_config
        auto filename = toltec_io.setup_filepath<ToltecIO::apt, ToltecIO::simu,
                                                 ToltecIO::beammap, ToltecIO::no_prod_type,
                                                 ToltecIO::obsnum_true>(filepath + dname,obsnum,-1);

        // check in debug mode for row/col error (seems fine)
        Eigen::MatrixXf table(toltec_io.beammap_apt_header.size(), ndet);
        table.row(0) = calib_data["array"].cast <float> ();
        table.row(1) = calib_data["nw"].cast <float> ();
        table.row(2) = calib_data["flxscale"].cast <float> ();
        table.row(3) = sensitivity.cast <float> ();

        table.row(4) = mout.pfit.row(0).template cast <float> ();
        table.row(5) = mout.pfit.row(0).template cast <float> ();
        table.row(6) = mout.pfit.row(1).template cast <float> ();
        table.row(7) = mout.pfit.row(1).template cast <float> ();
        table.row(8) = mout.pfit.row(2).template cast <float> ();
        table.row(9) = mout.pfit.row(2).template cast <float> ();
        table.row(10) = mout.pfit.row(3).template cast <float> ();
        table.row(11) = mout.pfit.row(3).template cast <float> ();
        table.row(12) = mout.pfit.row(4).template cast <float> ();
        table.row(13) = mout.pfit.row(4).template cast <float> ();
        table.row(14) = mout.pfit.row(5).template cast <float> ();
        table.row(15) = mout.pfit.row(5).template cast <float> ();


        /*int ci = 0;
        for (int ti=0; ti < toltec_io.apt_header.size()-2; ti=ti+2) {
            table.row(ti + 4) = mout.pfit.row(ci).template cast <float> ();
            table.row(ti + 4 + 1) = mout.perror.row(ci).template cast <float> ();
            ci++;
        }*/

        table.row(toltec_io.beammap_apt_header.size()-1) = converge_iter.cast <float> ();

        table.transposeInPlace();

        // Yaml node for ecsv table meta data (units and description)
        YAML::Node meta;
        meta["array"].push_back("units: N/A");
        meta["array"].push_back("array index");

        meta["nw"].push_back("units: N/A");
        meta["nw"].push_back("network index");

        meta["flxscale"].push_back("units: MJy/Sr");
        meta["flxscale"].push_back("flux conversion scale");

        meta["amp"].push_back("units: MJy/Sr");
        meta["amp"].push_back("fitted amplitude");

        meta["amp_err"].push_back("units: MJy/Sr");
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
        for (Eigen::Index i=0; i<arrays.size(); i++) {
            SPDLOG_INFO("writing {}.fits", f_ios.at(i).filepath);
            // loop through maps and save them as an hdu

            for (Eigen::Index j=0; j<ndet; j++) {
                if (calib_data["array"](j) == i) {
                    // add signal map to file
                    f_ios.at(i).add_hdu("signal_" + std::to_string(j) + "_I", mout.signal.at(j));
                    f_ios.at(i).hdus.back()->addKey("UNIT", cunit, "Unit of map");

                    // add fit parameters to hdus
                    /*f_ios.at(i).hdus.back()->addKey("amp", (float)mout.pfit(0,j),"amplitude (N/A)");
                    f_ios.at(i).hdus.back()->addKey("amp_err", (float)mout.perror(0,j),"amplitude error (N/A)");
                    f_ios.at(i).hdus.back()->addKey("x_t", (float)mout.pfit(1,j),"az offset (arcsec)");
                    f_ios.at(i).hdus.back()->addKey("x_t_err", (float)mout.perror(1,j),"az offset error (arcsec)");
                    f_ios.at(i).hdus.back()->addKey("y_t", (float)mout.pfit(2,j),"alt offset (arcsec)");
                    f_ios.at(i).hdus.back()->addKey("y_t_err", (float)mout.perror(2,j),"alt offset error (arcsec)");
                    f_ios.at(i).hdus.back()->addKey("a_fwhm", (float)mout.pfit(3,j),"az fwhm (arcsec)");
                    f_ios.at(i).hdus.back()->addKey("a_fwhm_err", (float)mout.perror(3,j),"az fwhm error (arcsec)");
                    f_ios.at(i).hdus.back()->addKey("b_fwhm", (float)mout.pfit(4,j),"alt fwhm (arcsec)");
                    f_ios.at(i).hdus.back()->addKey("b_fwhm_err", (float)mout.perror(4,j),"alt fwhm error (arcsec)");
                    f_ios.at(i).hdus.back()->addKey("angle", (float)mout.pfit(5,j),"position angle (radians)");
                    f_ios.at(i).hdus.back()->addKey("angle_err", (float)mout.perror(5,j),"position angle error (radians)");
                    */

                    // add weight map to file
                    f_ios.at(i).add_hdu("weight_" + std::to_string(j) + "_I", mout.weight.at(j));
                    f_ios.at(i).hdus.back()->addKey("UNIT", "(" + cunit + ")^-2", "Unit of map");

                    // add kernel map to file
                    if (run_kernel) {
                        f_ios.at(i).add_hdu("kernel_" + std::to_string(j) + "_I", mout.kernel.at(j));
                        f_ios.at(i).hdus.back()->addKey("UNIT", cunit, "Unit of map");
                    }
                }
            }

            Eigen:: Index k = 0;
            for (auto hdu: f_ios.at(i).hdus) {
                std::string hdu_name = hdu->name();
                f_ios.at(i).template add_wcs<UnitsType::arcsec>(hdu,map_type,mout.nrows,mout.ncols,pixel_size,
                                                                source_center,toltec_io.array_freqs[i],
                                                                polarization.stokes_params,hdu_name);
            }

            // loop through default TolTEC fits header keys and add to primary header
            for (auto const& pair : toltec_io.fits_header_keys) {
                f_ios.at(i).pfits->pHDU().addKey(pair.first, pair.second, " ");
            }

            // add wcs to pHDU
            f_ios.at(i).template add_wcs<UnitsType::arcsec>(&f_ios.at(i).pfits->pHDU(),map_type,mout.nrows,
                                                            mout.ncols,pixel_size,source_center,
                                                            toltec_io.array_freqs[i],
                                                            polarization.stokes_params);

            // add wavelength
            f_ios.at(i).pfits->pHDU().addKey("WAV", toltec_io.name_keys[i], "Array Name");
            // add obsnum
            f_ios.at(i).pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
            // object
            f_ios.at(i).pfits->pHDU().addKey("OBJECT", (std::string)source_name, "");
            // exp time
            f_ios.at(i).pfits->pHDU().addKey("t_exptime", tel_header_data["t_exp"], "Exposure Time (sec)");
            // add source ra
            f_ios.at(i).pfits->pHDU().addKey("s_ra", source_center["Ra"][0], "Source RA (radians)");
            // add source dec
            f_ios.at(i).pfits->pHDU().addKey("s_dec", source_center["Dec"][0], "Source Dec (radians)");
            // add map tangent point ra
            f_ios.at(i).pfits->pHDU().addKey("tan_ra", source_center["Ra"][0], "Map Tangent Point RA (radians)");
            // add map tangent point dec
            f_ios.at(i).pfits->pHDU().addKey("tan_dec", source_center["Dec"][0], "Map Tangent Point Dec (radians)");

            // add conversion
            if (cunit == "MJy/Sr") {
                f_ios.at(i).pfits->pHDU().addKey("to_mJy/beam", toltec_io.barea_keys[i]*MJY_SR_TO_mJY_ASEC, "Conversion to mJy/beam");
                f_ios.at(i).pfits->pHDU().addKey("to_Mjy/Sr", 1.0, "Conversion to MJy/Sr");
                f_ios.at(i).pfits->pHDU().addKey("to_uK/arcmin^2", engine_utils::MJy_Sr_to_uK(1, toltec_io.array_freqs[i],toltec_io.bfwhm_keys[i]),
                                                 "Conversion to uK/arcmin^2");
            }
            else if (cunit == "mJy/beam") {
                f_ios.at(i).pfits->pHDU().addKey("to_mJy/beam", 1.0, "Conversion to mJy/beam");
                f_ios.at(i).pfits->pHDU().addKey("to_MJy/Sr", 1/(toltec_io.barea_keys[i]*MJY_SR_TO_mJY_ASEC), "Conversion to MJy/Sr");
                f_ios.at(i).pfits->pHDU().addKey("to_uK/arcmin^2", MJY_SR_TO_mJY_ASEC/engine_utils::MJy_Sr_to_uK(1, toltec_io.array_freqs[i],toltec_io.bfwhm_keys[i]),
                                                 "Conversion to uK/arcmin^2");
            }
            else if (cunit == "uK/arcmin^2") {
                f_ios.at(i).pfits->pHDU().addKey("to_mJy/beam", MJY_SR_TO_mJY_ASEC/engine_utils::MJy_Sr_to_uK(1, toltec_io.array_freqs[i],
                                                                                                                toltec_io.bfwhm_keys[i]),
                                                 "Conversion to mJy/beam");
                f_ios.at(i).pfits->pHDU().addKey("to_MJy/Sr", 1/engine_utils::MJy_Sr_to_uK(1, toltec_io.array_freqs[i],toltec_io.bfwhm_keys[i]),
                                                 "Conversion to MJy/Sr");
                f_ios.at(i).pfits->pHDU().addKey("to_uK/arcmin^2", 1.0, "Conversion to uK/arcmin^2");
            }
        }
    }

    f_ios.clear();
    // close file since we're done
    SPDLOG_INFO("closing FITS files");
}
