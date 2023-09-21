#pragma once

#include <tula/algorithm/ei_stats.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/timestream/rtc/polarization_3.h>
#include <citlali/core/timestream/rtc/kernel.h>
#include <citlali/core/timestream/rtc/despike.h>
#include <citlali/core/timestream/rtc/filter.h>
#include <citlali/core/timestream/rtc/downsample.h>
#include <citlali/core/timestream/rtc/calibrate.h>

namespace timestream {

using timestream::TCData;

class RTCProc: public TCProc {
public:
    // controls for timestream reduction
    bool run_timestream;
    bool run_polarization;
    bool run_kernel;
    bool run_despike;
    bool run_tod_filter;
    bool run_downsample;
    bool run_calibrate;
    bool run_extinction;

    // rtc tod classes
    timestream::Polarization polarization;
    timestream::Kernel kernel;
    timestream::Despiker despiker;
    timestream::Filter filter;
    timestream::Downsampler downsampler;
    timestream::Calibration calibration;

    // minimum allowed frequency distance between tones
    double delta_f_min_Hz;

    // get config file
    template <typename config_t>
    void get_config(config_t &, std::vector<std::vector<std::string>> &, std::vector<std::vector<std::string>> &);

    // get indices to map from detector to index in map vectors
    template <class calib_t, typename Derived>
    auto calc_map_indices(calib_t &, Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
                          Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string, std::string);

    // run the main processing
    template<typename calib_t, typename telescope_t>
    auto run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &,
             TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string &,
             std::string &, calib_t &, telescope_t &, double, std::string,
             std::string);

    // remove nearby tones
    template <typename calib_t, typename Derived>
    auto remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, Eigen::DenseBase<Derived> &,
                             std::string);

    // remove flagged detectors
    template <typename apt_t, typename Derived>
    void remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &, Eigen::DenseBase<Derived> &);

    // append time chunk to tod netcdf file
    template <typename Derived, typename calib_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, Eigen::DenseBase<Derived> &, calib_t &);
};

// get config file
template <typename config_t>
void RTCProc::get_config(config_t &config, std::vector<std::vector<std::string>> &missing_keys,
                         std::vector<std::vector<std::string>> &invalid_keys) {
    // lower weight factor
    get_config_value(config, lower_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","flagging","lower_weight_factor"});
    // upper weight factor
    get_config_value(config, upper_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream", "raw_time_chunk","flagging","upper_weight_factor"});
    // minimum allowed frequency separation between tones
    get_config_value(config, delta_f_min_Hz, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","flagging","delta_f_min_Hz"});

    // run polarization?
    get_config_value(config, run_polarization, missing_keys, invalid_keys,
                     std::tuple{"timestream","polarimetry", "enabled"});

    // add stokes I, Q, and U if polarization is enabled
    if (run_polarization) {
        polarization.stokes_params = {{0,"I"}, {1,"Q"}, {2,"U"}};
    }
    // otherwise only use stokes I
    else {
        polarization.stokes_params[0] = "I";
    }

    // run kernel?
    get_config_value(config, run_kernel, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","kernel","enabled"});

    if (run_kernel) {
        // filepath to kernel
        get_config_value(config, kernel.filepath, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","kernel","filepath"});
        // type of kernel
        get_config_value(config, kernel.type, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","kernel","type"});
        // kernel fwhm in arcsec
        get_config_value(config, kernel.fwhm_rad, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","kernel","fwhm_arcsec"});

        // convert kernel fwhm to radians
        kernel.fwhm_rad *=ASEC_TO_RAD;
        // get kernel stddev
        kernel.sigma_rad = kernel.fwhm_rad*FWHM_TO_STD;

        // if kernel type is FITS input
        if (kernel.type == "fits") {
            // get extension name vector
            auto img_ext_name_node = config.get_node(std::tuple{"timestream","raw_time_chunk","kernel", "image_ext_names"});
            // get images
            for (Eigen::Index i=0; i<img_ext_name_node.size(); i++) {
                std::string img_ext_name = config.template get_str(std::tuple{"timestream","raw_time_chunk","kernel", "image_ext_names",
                                                                              i, std::to_string(i)});
                kernel.img_ext_names.push_back(img_ext_name);
            }
        }
    }

    // run despike?
    get_config_value(config, run_despike, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","despike","enabled"});

    if (run_despike) {
        // minimum spike sigma
        get_config_value(config, despiker.min_spike_sigma, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","despike","min_spike_sigma"});
        // decay time constant
        get_config_value(config, despiker.time_constant_sec, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","despike","time_constant_sec"});
        // window size for spikes
        get_config_value(config, despiker.window_size, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","despike","window_size"});

        // how to group spike finding and replacement
        despiker.grouping = "nw";
    }

    // run filter?
    get_config_value(config, run_tod_filter, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","filter","enabled"});

    if (run_tod_filter) {
        // tod filter gibbs param
        get_config_value(config, filter.a_gibbs, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","filter","a_gibbs"});
        // lower frequency limit
        get_config_value(config, filter.freq_low_Hz, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","filter","freq_low_Hz"});
        // upper frequency limit
        get_config_value(config, filter.freq_high_Hz, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","filter","freq_high_Hz"});
        // filter size
        get_config_value(config, filter.n_terms, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","filter","n_terms"});

        // replace despiker window size
        despiker.window_size = filter.n_terms;
    }
    else {
        // explicitly set filter size to zero for inner time chunks
        filter.n_terms = 0;
    }

    // run downsampling?
    get_config_value(config, run_downsample, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","downsample","enabled"});
    if (run_downsample) {
        // downsample factor
        get_config_value(config, downsampler.factor, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","downsample","factor"},{},{0});
        // downsample frequency
        get_config_value(config, downsampler.downsampled_freq_Hz, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","downsample","downsampled_freq_Hz"});
    }

    // run flux calibration?
    get_config_value(config, run_calibrate, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","flux_calibration","enabled"});
    // run extinction correction?
    get_config_value(config, run_extinction, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","extinction_correction","enabled"});
    // extinction model
    get_config_value(config, calibration.extinction_model, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","extinction_correction","extinction_model"},
                     {"am_q25","am_q50","am_q75","null"});

    // setup atm model
    calibration.setup();
}

template <class calib_t, typename Derived>
auto RTCProc::calc_map_indices(calib_t &calib, Eigen::DenseBase<Derived> &det_indices, Eigen::DenseBase<Derived> &nw_indices,
                               Eigen::DenseBase<Derived> &array_indices, Eigen::DenseBase<Derived> &fg_indices,
                               std::string stokes_param, std::string map_grouping) {
    // indices for maps
    Eigen::VectorXI indices(array_indices.size()), map_indices(array_indices.size());

    // number of maps from grouping
    int n_maps = 0;

    // overwrite map indices for networks
    if (map_grouping == "nw") {
        indices = nw_indices;
        n_maps = calib.n_nws;
    }

    // overwrite map indices for arrays
    else if (map_grouping == "array") {
        indices = array_indices;
        n_maps = calib.n_arrays;
    }

    // overwrite map indices for detectors
    else if (map_grouping == "detector") {
        indices = det_indices;
        n_maps = calib.n_dets;
    }

    // overwrite map indices for fg
    else if (map_grouping == "fg") {
        indices = fg_indices;
        n_maps = calib.fg.size()*calib.n_arrays;
    }

    // start at 0
    if (map_grouping != "fg") {
        Eigen::Index map_index = 0;
        map_indices(0) = 0;
        // loop through and populate map indices
        for (Eigen::Index i=0; i<indices.size()-1; i++) {
            // if next index is larger than current index, increment map index
            if (indices(i+1) > indices(i)) {
                map_index++;
            }
            map_indices(i+1) = map_index;
        }
    }
    else {
        // convert fg to indices
        std::map<Eigen::Index, Eigen::Index> fg_to_index, array_to_index;

        // get mapping from fg to map index
        for (Eigen::Index i=0; i<calib.fg.size(); i++) {
            fg_to_index[calib.fg(i)] = i;
        }

        // get mapping from fg to map index
        for (Eigen::Index i=0; i<calib.arrays.size(); i++) {
            array_to_index[calib.arrays(i)] = i;
        }

        // allocate map indices from fg
        for (Eigen::Index i=0; i<indices.size(); i++) {
            map_indices(i) = fg_to_index[indices(i)] + calib.fg.size()*array_to_index[array_indices(i)];
        }
    }

    // increment if polarization is enabled
    if (run_polarization) {
        if (stokes_param == "Q") {
            // stokes Q takes the second set of n_maps
            map_indices = map_indices.array() + n_maps;
        }
        else if (stokes_param == "U") {
            // stokes U takes the third set of n_maps
            map_indices = map_indices.array() + 2*n_maps;
        }
    }

    // return the map indices
    return std::move(map_indices);
}

template<class calib_t, typename telescope_t>
auto RTCProc::run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in,
                  TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, std::string &pixel_axes,
                  std::string &redu_type, calib_t &calib, telescope_t &telescope, double pixel_size_rad,
                  std::string stokes_param, std::string map_grouping) {

    // new timechunk for current stokes parameter timestream
    TCData<TCDataKind::RTC,Eigen::MatrixXd> in_pol;

    // calculate the stokes timestream
    auto [array_indices, nw_indices, det_indices, fg_indices] = polarization.demodulate_timestream(in, in_pol, stokes_param,
                                                                                                   redu_type, calib, telescope.sim_obs);

    // resize fcf
    in_pol.fcf.data.setOnes(in_pol.scans.data.cols());

    // get indices for maps
    logger->debug("calculating map indices");
    auto map_indices = calc_map_indices(calib, det_indices, nw_indices, array_indices, fg_indices, stokes_param, map_grouping);

    if (run_calibrate) {
        logger->debug("calibrating timestream");
        // calibrate tod
        calibration.calibrate_tod(in_pol, det_indices, array_indices, calib);

        out.calibrated = true;
    }

    if (run_extinction) {
        logger->debug("correcting extinction");
        // calc tau at toltec frequencies
        auto tau_freq = calibration.calc_tau(in_pol.tel_data.data["TelElAct"], telescope.tau_225_GHz);
        // correct for extinction
        calibration.extinction_correction(in_pol, det_indices, array_indices, calib, tau_freq);
    }

    // number of points in scan
    Eigen::Index n_pts = in_pol.scans.data.rows();

    // start index of inner scans
    auto si = filter.n_terms;
    // end index of inner scans
    auto sl = in_pol.scan_indices.data(1) - in_pol.scan_indices.data(0) + 1;

    // set up flags
    in_pol.flags.data.setZero(in_pol.scans.data.rows(), in_pol.scans.data.cols());

    // create kernel if requested
    if (run_kernel) {
        logger->debug("creating kernel timestream");
        // symmetric gaussian kernel
        if (kernel.type == "gaussian") {
            logger->debug("creating symmetric gaussian kernel");
            kernel.create_symmetric_gaussian_kernel(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                                    det_indices);
        }
        // airy kernel
        else if (kernel.type == "airy") {
            logger->debug("creating airy kernel");
            kernel.create_airy_kernel(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                      det_indices);
        }
        // get kernel from fits
        else if (kernel.type == "fits") {
            logger->debug("getting kernel from fits");
            kernel.create_kernel_from_fits(in_pol, pixel_axes, redu_type, calib.apt, in_pol.pointing_offsets_arcsec.data,
                                           pixel_size_rad, map_indices, det_indices);
        }

        out.kernel_generated = true;
    }

    // run despiking
    if (run_despike) {
        logger->debug("despiking");
        // despike data
        despiker.despike(in_pol.scans.data, in_pol.flags.data, calib.apt);

        // we want to replace spikes on a per array or network basis
        auto grp_limits = get_grouping(despiker.grouping, det_indices, calib, in_pol.scans.data.cols());

        logger->debug("replacing spikes");
        for (auto const& [key, val] : grp_limits) {
            // starting index
            auto start_index = std::get<0>(val);
            // size of block for each grouping
            auto n_dets = std::get<1>(val) - std::get<0>(val);

            // get the reference block of in scans that corresponds to the current array
            Eigen::Ref<Eigen::MatrixXd> in_scans_ref = in_pol.scans.data.block(0, start_index, n_pts, n_dets);

            Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                in_scans(in_scans_ref.data(), in_scans_ref.rows(), in_scans_ref.cols(),
                         Eigen::OuterStride<>(in_scans_ref.outerStride()));

            // get the block of in flags that corresponds to the current array
            Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags_ref =
                in_pol.flags.data.block(0, start_index, n_pts, n_dets);

            Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>, 0, Eigen::OuterStride<> >
                in_flags(in_flags_ref.data(), in_flags_ref.rows(), in_flags_ref.cols(),
                         Eigen::OuterStride<>(in_flags_ref.outerStride()));

            // replace spikes
            despiker.replace_spikes(in_scans, in_flags, calib.apt, start_index);
        }

        out.despiked = true;
    }

    // timestream filtering
    if (run_tod_filter) {
        logger->debug("convolving signal with tod filter");
        filter.convolve(in_pol.scans.data);
        //filter.iir(in_pol.scans.data);

        // filter kernel
        if (run_kernel) {
            logger->debug("convolving kernel with tod filter");
            filter.convolve(in_pol.kernel.data);
        }

        out.tod_filtered = true;
    }

    if (run_downsample) {
        logger->debug("downsampling data");
        // get the block of out scans that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Map<Eigen::MatrixXd>> in_scans =
            in_pol.scans.data.block(si, 0, sl, in_pol.scans.data.cols());

        // get the block of in flags that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags =
            in_pol.flags.data.block(si, 0, sl, in_pol.flags.data.cols());

        downsampler.downsample(in_scans, out.scans.data);
        downsampler.downsample(in_flags, out.flags.data);

        // loop through telescope meta data and downsample
        logger->debug("downsampling telescope");
        for (auto const& x: in_pol.tel_data.data) {
            // get the block of in tel data that corresponds to the inner scan indices
            Eigen::Ref<Eigen::VectorXd> in_tel =
                in_pol.tel_data.data[x.first].segment(si, sl);

            downsampler.downsample(in_tel, out.tel_data.data[x.first]);
        }

        // downsample pointing
        for (auto const& x: in_pol.pointing_offsets_arcsec.data) {
        Eigen::Ref<Eigen::VectorXd> in_pointing =
            in_pol.pointing_offsets_arcsec.data[x.first].segment(si, sl);

            downsampler.downsample(in_pointing, out.pointing_offsets_arcsec.data[x.first]);
        }

        // downsample hwpr
        if (run_polarization && calib.run_hwpr) {
            Eigen::Ref<Eigen::VectorXd> in_hwpr =
                in_pol.hwpr_angle.data.segment(si, sl);
            downsampler.downsample(in_hwpr, in_pol.hwpr_angle.data);
        }
        // downsample kernel if requested
        if (run_kernel) {
            logger->debug("downsampling kernel");
            // get the block of in kernel scans that corresponds to the inner scan indices
            Eigen::Ref<Eigen::MatrixXd> in_kernel =
                in_pol.kernel.data.block(si, 0, sl, in_pol.kernel.data.cols());

            downsampler.downsample(in_kernel, out.kernel.data);
        }

        out.downsampled = true;
    }

    else {
        // copy data
        out.scans.data = in_pol.scans.data.block(si, 0, sl, in_pol.scans.data.cols());
        // copy flags
        out.flags.data = in_pol.flags.data.block(si, 0, sl, in_pol.flags.data.cols());
        // copy kernel
        if (run_kernel) {
            out.kernel.data = in_pol.kernel.data.block(si, 0, sl, in_pol.kernel.data.cols());
        }
        // copy telescope data
        for (auto const& x: in_pol.tel_data.data) {
            out.tel_data.data[x.first] = in_pol.tel_data.data[x.first].segment(si, sl);
        }

        // copy pointing offsets
        for (auto const& x: in_pol.pointing_offsets_arcsec.data) {
            out.pointing_offsets_arcsec.data[x.first] = in_pol.pointing_offsets_arcsec.data[x.first].segment(si, sl);
        }

        // copy hwpr angle
        if (run_polarization) {
            if (calib.run_hwpr) {
                out.hwpr_angle.data = in_pol.hwpr_angle.data.segment(si, sl);
            }
        }
    }

    out.scan_indices.data = in_pol.scan_indices.data;
    out.index.data = in_pol.index.data;
    out.fcf.data = in_pol.fcf.data;

    return std::tuple<Eigen::VectorXI,Eigen::VectorXI,Eigen::VectorXI,Eigen::VectorXI>(map_indices, array_indices, nw_indices, det_indices);
}

template <typename apt_t, typename Derived>
void RTCProc::remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_t &apt, Eigen::DenseBase<Derived> &det_indices) {

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    // number of detectors flagged in apt
    Eigen::Index n_flagged = 0;

    // loop through detectors and set flags to zero
    // for those flagged in apt table
    for (Eigen::Index i=0; i<n_dets; i++) {
        Eigen::Index det_index = det_indices(i);
        if (apt["flag"](det_index)!=0) {
            in.flags.data.col(i).setOnes();
            n_flagged++;
        }
    }

    logger->info("removed {} detectors flagged in APT table ({}%)",n_flagged,
                (static_cast<float>(n_flagged)/static_cast<float>(n_dets))*100);
}

template <typename calib_t, typename Derived>
auto RTCProc::remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, Eigen::DenseBase<Derived> &det_indices,
                                  std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    int n_nearby_tones = 0;

    // loop through flag columns
    for (Eigen::Index i=0; i<in.flags.data.cols(); i++) {
        // map from data column to apt row
        Eigen::Index det_index = det_indices(i);
        // if closer than freq separation limit and unflagged, flag it
        if (calib.apt["duplicate_tone"](det_index) && calib_scan.apt["flag"](det_index)==0) {
            n_nearby_tones++;
            // increment number of nearby tones
            if (map_grouping!="detector") {
                in.flags.data.col(i).setOnes();
            }
            else {
                calib_scan.apt["flag"](det_index) = 1;
            }
        }
    }

    logger->info("removed {}/{} ({}%) unflagged tones closer than {} kHz", n_nearby_tones, n_dets,
                (static_cast<float>(n_nearby_tones)/static_cast<float>(n_dets))*100, delta_f_min_Hz/1000);

    // set up scan calib
    calib_scan.setup();

    return std::move(calib_scan);
}

template <typename Derived, typename calib_t, typename pointing_offset_t>
void RTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string map_grouping,
                               std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, Eigen::DenseBase<Derived> &det_indices,
                               calib_t &calib) {
    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        // open netcdf file
        NcFile fo(filepath, netCDF::NcFile::write);

        // append common time chunk variables
        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, det_indices, calib);

        fo.sync();
        fo.close();

    } catch (NcException &e) {
        logger->error("{}", e.what());
    }
}

} // namespace timestream
