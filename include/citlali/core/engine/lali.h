#pragma once

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

// selects the type of TCData
using timestream::TCDataKind;

class Lali: public EngineBase {
public:

    void setup();
    auto run();

    template <class KidsProc, class RawObs>
    auto pipeline(KidsProc&, RawObs&);

    template <MapBase::MapType out_type, class MC, typename fits_out_vec_t>
    void output(MC&, fits_out_vec_t &, fits_out_vec_t &, bool);
};

void Lali::setup() {
    // if filter is requested, make it here
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

    // toltec i/o class for filenames
    ToltecIO toltec_io;

    // empty the fits vector for subsequent observations
    fits_ios.clear();

    // get obsnum directory name inside redu directory name
    std::stringstream ss_redu;
    ss_redu << std::setfill('0') << std::setw(2) << redu_num;
    std::string hdname = "redu" + ss_redu.str() + "/";

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << obsnum;
    std::string dname = hdname + ss.str() + "/";

    // create obsnum directory
    toltec_io.setup_output_directory(filepath, dname);

    // create files for each member of the array_indices group
    for (Eigen::Index i=0; i<array_indices.size(); i++) {
        std::string filename;
        // generate filename for science maps
        if (reduction_type == "science") {
            filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                    ToltecIO::science, ToltecIO::no_prod_type, ToltecIO::obsnum_true>(filepath + dname,obsnum,i);
        }

        else if (reduction_type == "pointing") {
            // generate filename for pointing maps
            filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                    ToltecIO::pointing, ToltecIO::no_prod_type, ToltecIO::obsnum_true>(filepath + dname,obsnum,i);
        }

        // push the file classes into a vector for storage
        FitsIO<fileType::write_fits, CCfits::ExtHDU*> fits_io(filename);
        fits_ios.push_back(std::move(fits_io));
    }

    if (ts_out) {
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

            ts_filepath = filename + ".nc";

            netCDF::NcFile fo(ts_filepath,netCDF::NcFile::replace);
            netCDF::NcDim nsmp_dim = fo.addDim("nsamples");
            netCDF::NcDim ndet_dim = fo.addDim("ndetectors",ndet);

            std::vector<netCDF::NcDim> dims;
            dims.push_back(nsmp_dim);
            dims.push_back(ndet_dim);

            netCDF::NcVar pixid_v = fo.addVar("PIXID",netCDF::ncInt, ndet_dim);
            pixid_v.putAtt("Units","N/A");
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

auto Lali::run() {
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

        if (run_polarization) {
            in.hwp.data = hwp.segment(start_index, scan_length);
        }

        /*Stage 0: KidsProc*/
        {
            tula::logging::scoped_timeit timer("kidsproc.populate_rtc_load()");
            tula::logging::scoped_loglevel<spdlog::level::critical> _0;
            in.scans.data = kidsproc.populate_rtc_load(loaded_rawobs,in.scan_indices.data, scan_length, ndet);
        }

        TCData<TCDataKind::PTC,Eigen::MatrixXd> out;

        for (auto const& stokes_params: polarization.stokes_params) {
            TCData<TCDataKind::RTC,Eigen::MatrixXd> in2;
            auto [map_index_vector, det_index_vector] =  polarization.create_rtc(in, in2, stokes_params.first, this);

            /*Stage 1: RTCProc*/
            RTCProc rtcproc;
            {
                tula::logging::scoped_timeit timer("rtcproc.run()");
                rtcproc.run(in2, out, this);
            }

            if (ts_out) {
                Eigen::MatrixXd lat(out.scans.data.rows(),out.scans.data.cols());
                Eigen::MatrixXd lon(out.scans.data.rows(),out.scans.data.cols());

                if (ts_format == "netcdf") {
                    SPDLOG_INFO("writing scan {} to {}", in.index.data, ts_filepath);
                    // loop through detectors and get pointing timestream
                    for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {

                        // get offsets
                        auto azoff = calib_data["x_t"](i);
                        auto eloff = calib_data["y_t"](i);

                        // get pointing
                        auto [lat_i, lon_i] = engine_utils::get_det_pointing(out.tel_meta_data.data, azoff, eloff, map_type);
                        lat.col(i) = std::move(lat_i);
                        lon.col(i) = std::move(lon_i);
                    }

                    append_to_netcdf(ts_filepath, out.scans.data, out.flags.data, lat, lon, out.tel_meta_data.data["TelElDes"],
                                 out.tel_meta_data.data["TelTime"]);
                }
            }

            /*Stage 2: PTCProc*/
            PTCProc ptcproc;
            {
                bool rc = true;
                if (stokes_params.first == "Q" || stokes_params.first == "U") {
                    rc = false;
                }

                tula::logging::scoped_timeit timer("ptcproc.run()");
                ptcproc.run(out, out, this, rc);
            }

            /*Stage 3 Populate Map*/
            if (mapping_method == "naive") {
                {
                    tula::logging::scoped_timeit timer("populate_maps_naive()");
                    populate_maps_naive(out, map_index_vector, det_index_vector, this);
                }
            }

            else if (mapping_method == "jinc") {
                tula::logging::scoped_timeit timer("populate_maps_jinc()");
                populate_maps_jinc(out, this);
            }
        }

        SPDLOG_INFO("done with scan {}", out.index.data);
        return out;
    });

    // return the farm object to the pipeline
    return farm;
}

template <class KidsProc, class RawObs>
auto Lali::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    grppi::pipeline(tula::grppi_utils::dyn_ex(ex_name),
        [&]() -> std::optional<std::tuple<TCData<TCDataKind::RTC, Eigen::MatrixXd>, KidsProc,
                                          std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>>> {
        // variable to hold current scan
        static auto scan = 0;
        // length of current outer scan
        Eigen::Index scan_length;
        // main grppi loop
        while (scan < scanindices.cols()) {
            SPDLOG_INFO("reducing scan {}", scan);
            // length of current scan
            scan_length = scanindices(3, scan) - scanindices(2, scan) + 1;

            // create TCData of kind RTC
            TCData<TCDataKind::RTC, Eigen::MatrixXd> rtc;
            // current scanindices (inner si, inner ei, outer si, outer ei)
            rtc.scan_indices.data = scanindices.col(scan);
            // current scan number for outputting progress
            rtc.index.data = scan;

            // run kidsproc to get correct units
            std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> loaded_rawobs;
            {
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
    run());

    SPDLOG_INFO("normalizing maps");
    mb.normalize_maps(run_kernel);

    mb.psd.resize(mb.map_count);
    try {
        for (Eigen::Index i=0; i < mb.map_count; i++) {
            SPDLOG_INFO("calculating map {} psd", i);
            PSD psd;
            SPDLOG_INFO("cov_cut {}",cmb.cov_cut);
            psd.cov_cut = cmb.cov_cut;
            //psd.calc_map_psd(mb.signal.at(i), mb.weight.at(i), mb.rcphys, mb.ccphys);
            //mb.psd.at(i) = std::move(psd);
        }
    } catch (...) {
        SPDLOG_INFO("failed to compute psds");
    }

    mb.histogram.resize(mb.map_count);
    try {
        for (Eigen::Index i=0; i < mb.map_count; i++) {
            SPDLOG_INFO("calculating map {} histogram", i);
            Histogram histogram;
            int nbins = 200;
            histogram.cov_cut = cmb.cov_cut;
            //histogram.calc_hist(mb.signal.at(i), mb.weight.at(i), nbins);
            //mb.histogram.at(i) = std::move(histogram);
        }
    } catch (...) {
        SPDLOG_INFO("failed to compute psds");
    }

    // do fit if map_grouping is pointing
    if (reduction_type == "pointing") {
        // placeholder vectors for grppi loop
        std::vector<int> array_in_vec, array_out_vec;
        array_in_vec.resize(mb.map_count);

        std::iota(array_in_vec.begin(), array_in_vec.end(), 0);
        array_out_vec.resize(mb.map_count);

        // set nparams for fit
        Eigen::Index nparams = 6;
        mb.pfit.setZero(nparams, mb.map_count);
        mb.perror.setZero(nparams, mb.map_count);

        // loop through the arrays and do the fit
        SPDLOG_INFO("fitting pointing maps");
        grppi::map(tula::grppi_utils::dyn_ex(ex_name), array_in_vec, array_out_vec, [&](auto d) {
            // declare fitter class for detector
            gaussfit::MapFitter fitter;
            // size of region to fit in pixels
            fitter.bounding_box_pix = bounding_box_pix;
            mb.pfit.col(d) = fitter.fit<gaussfit::MapFitter::peakValue>(mb.signal[d], mb.weight[d], calib_data);
            mb.perror.col(d) = fitter.error;
            return 0;});

        // rescale params from pixel to on-sky units
        mb.pfit.row(1) = pixel_size*(mb.pfit.row(1).array() - (mb.ncols)/2)/RAD_ASEC;
        mb.pfit.row(2) = pixel_size*(mb.pfit.row(2).array() - (mb.nrows)/2)/RAD_ASEC;
        mb.pfit.row(3) = STD_TO_FWHM*pixel_size*(mb.pfit.row(3))/RAD_ASEC;
        mb.pfit.row(4) = STD_TO_FWHM*pixel_size*(mb.pfit.row(4))/RAD_ASEC;

        // rescale errors from pixel to on-sky units
        mb.perror.row(1) = pixel_size*(mb.perror.row(1))/RAD_ASEC;
        mb.perror.row(2) = pixel_size*(mb.perror.row(2))/RAD_ASEC;
        mb.perror.row(3) = STD_TO_FWHM*pixel_size*(mb.perror.row(3))/RAD_ASEC;
        mb.perror.row(4) = STD_TO_FWHM*pixel_size*(mb.perror.row(4))/RAD_ASEC;
    }
}

template <MapBase::MapType out_type, class MC, typename fits_out_vec_t>
void Lali::output(MC &mout, fits_out_vec_t &f_ios, fits_out_vec_t &nf_ios, bool filtered) {
    // toltec input/output class
    ToltecIO toltec_io;

    // get obsnum directory name inside redu directory name
    std::stringstream ss_redu;
    ss_redu << std::setfill('0') << std::setw(2) << redu_num;
    std::string hdname = "redu" + ss_redu.str() + "/";

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

    // loop through array indices and add hdu's to existing files
    Eigen::Index pp = 0;
    for (Eigen::Index i=0; i<array_indices.size(); i++) {
        SPDLOG_INFO("writing {}.fits", f_ios.at(i).filepath);

        if (run_polarization) {
            for (auto const& stokes_params: polarization.stokes_params) {
                // add signal map to file
                f_ios.at(i).add_hdu("signal_"+stokes_params.first, mout.signal.at(pp));

                //add weight map to file
                f_ios.at(i).add_hdu("weight_"+stokes_params.first, mout.weight.at(pp));

                //add kernel map to file
                if (run_kernel) {
                    f_ios.at(i).add_hdu("kernel_"+stokes_params.first, mout.kernel.at(pp));
                }

                // add coverage map to file
                f_ios.at(i).add_hdu("coverage_"+stokes_params.first, mout.coverage.at(pp));

                // add signal-to-noise map to file.  We calculate it here to save space
                Eigen::MatrixXd signoise = mout.signal.at(pp).array()*sqrt(mout.weight.at(pp).array());
                f_ios.at(i).add_hdu("sig2noise_"+stokes_params.first, signoise);
                pp++;
            }
        }

        else {
            // add signal map to file
            f_ios.at(i).add_hdu("signal", mout.signal.at(i));

                 //add weight map to file
            f_ios.at(i).add_hdu("weight", mout.weight.at(i));

                 //add kernel map to file
            if (run_kernel) {
                f_ios.at(i).add_hdu("kernel", mout.kernel.at(i));
            }

                 // add coverage map to file
            f_ios.at(i).add_hdu("coverage", mout.coverage.at(i));

                 // add signal-to-noise map to file.  We calculate it here to save space
            Eigen::MatrixXd signoise = mout.signal.at(i).array()*sqrt(mout.weight.at(i).array());
            f_ios.at(i).add_hdu("sig2noise", signoise);
        }

        // now loop through hdus and add wcs
        for (auto hdu: f_ios.at(i).hdus) {
            std::string hdu_name = hdu->name();
            // degrees if science map
            if (reduction_type == "science") {
                f_ios.at(i).template add_wcs<UnitsType::deg>(hdu,map_type,mout.nrows,mout.ncols,
                                                       pixel_size,source_center, hdu_name);
            }

            // arcseconds if pointing map
            else if (reduction_type == "pointing") {
                f_ios.at(i). template add_wcs<UnitsType::arcsec>(hdu,map_type,mout.nrows,mout.ncols,
                                                          pixel_size,source_center, hdu_name);

                if constexpr (out_type==MapType::obs) {
                    // add fit parameters
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
                }
            }
        }

        // loop through default TolTEC fits header keys and add to primary header
        for (auto const& pair : toltec_io.fits_header_keys) {
            f_ios.at(i).pfits->pHDU().addKey(pair.first, pair.second, " ");
        }

        // degrees if science map
        if (reduction_type == "science") {
            // add wcs to pHDU
            f_ios.at(i).template add_wcs<UnitsType::deg>(&f_ios.at(i).pfits->pHDU(),map_type,
                                                   mout.nrows,mout.ncols,pixel_size,source_center);
        }
        // arcseconds if pointing map
        else if (reduction_type == "pointing") {
            // add wcs to pHDU
            f_ios.at(i).template add_wcs<UnitsType::arcsec>(&f_ios.at(i).pfits->pHDU(),map_type,
                                                      mout.nrows,mout.ncols,pixel_size,source_center);
        }

        // add wavelength
        f_ios.at(i).pfits->pHDU().addKey("WAV", toltec_io.name_keys[i], "Array Name");
        // add obsnum
        f_ios.at(i).pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
        // add units
        f_ios.at(i).pfits->pHDU().addKey("UNIT", obsnum, "MJy/Sr");
        // add conversion
        f_ios.at(i).pfits->pHDU().addKey("to_mjy/b", toltec_io.barea_keys[i]*MJY_SR_TO_mJY_ASEC, "Conversion to mJy/beam");
        // add source ra
        f_ios.at(i).pfits->pHDU().addKey("SRC_RA_RAD", source_center["Ra"][0], "Source RA (radians)");
        // add source dec
        f_ios.at(i).pfits->pHDU().addKey("SRC_DEC_RAD", source_center["Dec"][0], "Source Dec (radians)");

    }
    // close file since we're done
    SPDLOG_INFO("closing FITS files");
    f_ios.clear();

    // save psds
    SPDLOG_INFO("saving map psd");

    std::string filename;

    if constexpr (out_type==MapType::obs) {
        if constexpr (out_type==MapType::obs) {
            if (reduction_type == "science") {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::science, ToltecIO::psd,
                                                    ToltecIO::obsnum_true>(filepath + dname,obsnum,-1);
            }

            else if (reduction_type == "pointing") {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::pointing, ToltecIO::psd,
                                                    ToltecIO::obsnum_true>(filepath + dname,obsnum,-1);
            }
        }
    }

    else if constexpr (out_type == MapType::coadd) {
        if (reduction_type == "science") {
            if (filtered == false) {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::science, ToltecIO::raw_psd,
                                                    ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
            }

            else if (filtered == true) {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::science, ToltecIO::filtered_psd,
                                                    ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
            }
        }

        else if (reduction_type == "pointing") {
            if (filtered == false) {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::pointing, ToltecIO::raw_psd,
                                                    ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
            }
            if (filtered == true) {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::pointing, ToltecIO::filtered_psd,
                                                    ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
            }
        }
    }

    netCDF::NcFile fo(filename + ".nc",netCDF::NcFile::replace);

    std::map<int, std::string> name_keys;

    if (run_polarization) {
        name_keys = toltec_io.polarized_name_keys;
    }

    else {
        name_keys = toltec_io.name_keys;
    }

    for (Eigen::Index i=0; i<mb.map_count; i++) {

        netCDF::NcDim psd_dim = fo.addDim(name_keys[i] +"_nfreq",mb.psd.at(i).psd.size());
        netCDF::NcDim pds2d_row_dim = fo.addDim(name_keys[i] +"_rows",mb.psd.at(i).psd2d.rows());
        netCDF::NcDim pds2d_col_dim = fo.addDim(name_keys[i] +"_cols",mb.psd.at(i).psd2d.cols());

        std::vector<netCDF::NcDim> dims;
        dims.push_back(pds2d_row_dim);
        dims.push_back(pds2d_col_dim);

        netCDF::NcVar psd_v = fo.addVar(name_keys[i] + "_psd",netCDF::ncDouble, psd_dim);
        psd_v.putVar(mb.psd.at(i).psd.data());

        netCDF::NcVar psdfreq_v = fo.addVar(name_keys[i] + "_psd_freq",netCDF::ncDouble, psd_dim);
        psdfreq_v.putVar(mb.psd.at(i).psd_freq.data());

        Eigen::MatrixXd psd2d_transposed = mb.psd.at(i).psd2d.transpose();
        Eigen::MatrixXd psd2d_freq_transposed = mb.psd.at(i).psd2d_freq.transpose();

        netCDF::NcVar psd2d_v = fo.addVar(name_keys[i] + "_psd2d",netCDF::ncDouble, dims);
        psd2d_v.putVar(psd2d_transposed.data());

        netCDF::NcVar psd2d_freq_v = fo.addVar(name_keys[i] + "_psd2d_freq",netCDF::ncDouble, dims);
        psd2d_freq_v.putVar(psd2d_freq_transposed.data());
    }

    fo.close();

    // save histogram
    SPDLOG_INFO("saving map histogram");

    if constexpr (out_type==MapType::obs) {
        if (reduction_type == "science") {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::science, ToltecIO::hist,
                                                    ToltecIO::obsnum_true>(filepath + dname,obsnum,-1);
        }

        else if (reduction_type == "pointing") {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::pointing, ToltecIO::hist,
                                                    ToltecIO::obsnum_true>(filepath + dname,obsnum,-1);
        }
    }

    else if constexpr (out_type == MapType::coadd) {
        if (reduction_type == "science") {
            if (filtered == false) {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::science, ToltecIO::raw_hist,
                                                    ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
            }

            else if (filtered == true) {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::science, ToltecIO::filtered_hist,
                                                    ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
            }
        }

        else if (reduction_type == "pointing") {
            if (filtered == false) {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::pointing, ToltecIO::raw_hist,
                                                    ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
            }
            if (filtered == true) {
                filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                    ToltecIO::pointing, ToltecIO::filtered_hist,
                                                    ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
            }
        }
    }

    netCDF::NcFile hist_fo(filename + ".nc",netCDF::NcFile::replace);

    for (Eigen::Index i=0; i<mb.map_count; i++) {
        netCDF::NcDim bins_dim = hist_fo.addDim(name_keys[i] +"nbins",mb.psd.at(i).psd.size());

        netCDF::NcVar hist_v = hist_fo.addVar(name_keys[i] + "_values",netCDF::ncDouble, bins_dim);
        hist_v.putVar(mb.histogram.at(i).hist_vals.data());

        netCDF::NcVar bins_v = hist_fo.addVar(name_keys[i] + "_bins",netCDF::ncDouble, bins_dim);
        bins_v.putVar(mb.histogram.at(i).hist_bins.data());
    }

    hist_fo.close();


    // add fitting parameters to file if pointing mode is selected
    if constexpr (out_type==MapType::obs) {
        if (reduction_type == "pointing") {
            // yaml node for ecsv table meta data (units and description)
            YAML::Node meta;
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

            // ppt table
            SPDLOG_INFO("writing pointing fit table");

            // get output path from citlali_config
            auto filename = toltec_io.setup_filepath<ToltecIO::ppt, ToltecIO::simu,
                    ToltecIO::pointing, ToltecIO::no_prod_type, ToltecIO::obsnum_true>(filepath + dname,obsnum,-1);
            Eigen::MatrixXf table(toltec_io.apt_header.size(), mb.map_count);

            //table = mout.pfit.template cast <float> ();
            int ci = 0;
            for (int ti=0; ti < toltec_io.apt_header.size()-1; ti=ti+2) {
                table.row(ti) = mout.pfit.row(ci).template cast <float> ();
                table.row(ti + 1) = mout.perror.row(ci).template cast <float> ();
                ci++;
            }
            table.transposeInPlace();

            SPDLOG_INFO("pointing fit table header {}", toltec_io.apt_header);
            SPDLOG_INFO("pointing fit table {}", table);

            // write the ecsv file
            to_ecsv_from_matrix(filename, table, toltec_io.apt_header,meta);
            SPDLOG_INFO("successfully wrote ppt table to {}.ecsv", filename);
        }
    }

    else if constexpr (out_type == MapType::coadd) {
        if (run_coadd) {
            if (run_noise) {

                if constexpr (out_type == MapType::coadd) {
                    if (reduction_type == "science") {
                        if (filtered == false) {
                            filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                                ToltecIO::science, ToltecIO::noise_raw_psd,
                                                                ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
                        }

                        else if (filtered == true) {
                            filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                                ToltecIO::science, ToltecIO::noise_filtered_psd,
                                                                ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
                        }
                    }

                    else if (reduction_type == "pointing") {
                        if (filtered == false) {
                            filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                                ToltecIO::pointing, ToltecIO::noise_raw_psd,
                                                                ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
                        }
                        if (filtered == true) {
                            filename = toltec_io.setup_filepath<ToltecIO::toltec, ToltecIO::simu,
                                                                ToltecIO::pointing, ToltecIO::noise_filtered_psd,
                                                                ToltecIO::obsnum_false>(filepath + cname,obsnum,-1);
                        }
                    }
                }

                netCDF::NcFile fo(filename + ".nc",netCDF::NcFile::replace);

                /*for (Eigen::Index i=0; i<mb.map_count; i++) {
                    netCDF::NcDim psd_dim = fo.addDim(toltec_io.name_keys[i] +"_nfreq",mb.noise_psd.at(i).psd.size());
                    netCDF::NcDim pds2d_row_dim = fo.addDim(toltec_io.name_keys[i] +"_rows",mb.noise_psd.at(i).psd2d.rows());
                    netCDF::NcDim pds2d_col_dim = fo.addDim(toltec_io.name_keys[i] +"_cols",mb.noise_psd.at(i).psd2d.cols());

                    std::vector<netCDF::NcDim> dims;
                    dims.push_back(pds2d_row_dim);
                    dims.push_back(pds2d_col_dim);

                    netCDF::NcVar psd_v = fo.addVar(toltec_io.name_keys[i] + "_psd",netCDF::ncDouble, psd_dim);
                    psd_v.putVar(mb.noise_psd.at(i).psd.data());

                    netCDF::NcVar psdfreq_v = fo.addVar(toltec_io.name_keys[i] + "_psd_freq",netCDF::ncDouble, psd_dim);
                    psdfreq_v.putVar(mb.noise_psd.at(i).psd_freq.data());

                    Eigen::MatrixXd psd2d_transposed = mb.noise_psd.at(i).psd2d.transpose();
                    Eigen::MatrixXd psd2d_freq_transposed = mb.noise_psd.at(i).psd2d_freq.transpose();

                    netCDF::NcVar psd2d_v = fo.addVar(toltec_io.name_keys[i] + "_psd2d",netCDF::ncDouble, dims);
                    psd2d_v.putVar(psd2d_transposed.data());

                    netCDF::NcVar psd2d_freq_v = fo.addVar(toltec_io.name_keys[i] + "_psd2d_freq",netCDF::ncDouble, dims);
                    psd2d_freq_v.putVar(psd2d_freq_transposed.data());
                }*/

                fo.close();

                SPDLOG_INFO("writing noise maps");
                // loop through array indices and add hdu's to existing files
                for (Eigen::Index i=0; i<array_indices.size(); i++) {
                    SPDLOG_INFO("writing {}.fits", nf_ios.at(i).filepath);
                    // loop through noise map number
                    for (Eigen::Index j=0; j<mout.nnoise; j++) {

                        // get tensor chip on 3rd dimension (nrows,ncols, nnoise)
                        Eigen::Tensor<double,2> out = mout.noise.at(i).chip(j,2);
                        auto out_matrix = Eigen::Map<Eigen::MatrixXd>(out.data(), out.dimension(0),
                                                    out.dimension(1));
                        // add noise map to file
                        nf_ios.at(i).add_hdu("noise" + std::to_string(j),out_matrix);
                    }

                    // now loop through hdus and add wcs
                    for (auto hdu: nf_ios.at(i).hdus) {
                        // degrees if science map
                        //if (reduction_type == "science") {
                            nf_ios.at(i).template add_wcs<UnitsType::deg>(hdu,map_type,mout.nrows,mout.ncols,
                                                                   pixel_size,source_center);
                        //}
                    }

                    // loop through default TolTEC fits header keys and add to primary header
                    for (auto const& pair : toltec_io.fits_header_keys) {
                        nf_ios.at(i).pfits->pHDU().addKey(pair.first, pair.second, " ");
                    }

                    // add wcs to pHDU
                    nf_ios.at(i).template add_wcs<UnitsType::deg>(&nf_ios.at(i).pfits->pHDU(),map_type,
                                                           mout.nrows,mout.ncols,pixel_size,source_center);

                    // add wavelength
                    nf_ios.at(i).pfits->pHDU().addKey("WAV", toltec_io.name_keys[i], "Array Name");
                    // add obsnum
                    nf_ios.at(i).pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
                    // add units
                    nf_ios.at(i).pfits->pHDU().addKey("UNIT", obsnum, "MJy/Sr");
                    // add conversion
                    nf_ios.at(i).pfits->pHDU().addKey("to_mjy/b", toltec_io.barea_keys[i]*MJY_SR_TO_mJY_ASEC, "Conversion to mJy/beam");
                    // add source ra
                    nf_ios.at(i).pfits->pHDU().addKey("SRC_RA_RAD", source_center["Ra"][0], "Source RA (radians)");
                    // add source dec
                    nf_ios.at(i).pfits->pHDU().addKey("SRC_DEC_RAD", source_center["Dec"][0], "Source Dec (radians)");

                }
                // close the file when we're done
                SPDLOG_INFO("closing noise FITS files");
                nf_ios.clear();
            }
        }
    }
}
