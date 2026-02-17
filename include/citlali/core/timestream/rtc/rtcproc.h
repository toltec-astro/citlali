#pragma once

#include <tula/algorithm/ei_stats.h>
#include <Eigen/QR>
#include <cmath>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/timestream/rtc/polarization.h>
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
    bool run_pointing;
    bool run_polarization;
    bool run_kernel;
    bool run_despike;
    bool run_tod_filter;
    bool run_tod_notch;
    bool run_tod_iir_highpass;
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

    struct AltAzDestripeOptions {
        bool enabled = false;
        std::string grouping = "nw";
        bool fit_time_trend = true;
        bool fit_derivs = true;
        Eigen::Index min_samples = 64;
    };
    AltAzDestripeOptions altaz_destripe;

    // get config file
    template <typename config_t>
    void get_config(config_t &, std::vector<std::vector<std::string>> &, std::vector<std::vector<std::string>> &);

    // get indices to map from detector to index in map vectors
    template <class calib_t>
    auto calc_map_indices(calib_t &, std::string);

    // run the main processing
    template<typename calib_t, typename telescope_t>
    auto run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             calib_t &, telescope_t &, double, std::string);

    // remove nearby tones
    template <typename calib_t>
    auto remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, std::string);

    // remove flagged detectors
    template <typename apt_t>
    void remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &);

    // append time chunk to tod netcdf file
    template <typename calib_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, calib_t &, bool apply_det_offsets = false,
                          Eigen::Index scan_row_index = -1);
};

// get config file
template <typename config_t>
void RTCProc::get_config(config_t &config, std::vector<std::vector<std::string>> &missing_keys,
                         std::vector<std::vector<std::string>> &invalid_keys) {
    // lower inv var factor
    get_config_value(config, lower_inv_var_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","flagging","lower_tod_inv_var_factor"});
    // upper inv var factor
    get_config_value(config, upper_inv_var_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream", "raw_time_chunk","flagging","upper_tod_inv_var_factor"});
    // minimum allowed frequency separation between tones
    get_config_value(config, delta_f_min_Hz, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","flagging","delta_f_min_Hz"});

    // run polarization?
    get_config_value(config, run_polarization, missing_keys, invalid_keys,
                     std::tuple{"timestream","polarimetry","enabled"});
    // add stokes I, Q, and U if polarization is enabled
    if (run_polarization) {
        polarization.stokes_params = {{0,"I"}, {1,"Q"}, {2,"U"}};
        // use loc or fg?
        get_config_value(config, polarization.grouping, missing_keys, invalid_keys,
                         std::tuple{"timestream","polarimetry","grouping"});
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
            for (Eigen::Index i=0; i<img_ext_name_node.size(); ++i) {
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
        const bool has_freq_low = config.template has_typed<double>(
            std::tuple{"timestream","raw_time_chunk","filter","freq_low_Hz"});
        const bool has_freq_high = config.template has_typed<double>(
            std::tuple{"timestream","raw_time_chunk","filter","freq_high_Hz"});
        if (has_freq_low && has_freq_high &&
            filter.freq_high_Hz < filter.freq_low_Hz) {
            logger->error("timestream.raw_time_chunk.filter.freq_high_Hz ({}) must be >= freq_low_Hz ({})",
                          filter.freq_high_Hz, filter.freq_low_Hz);
            std::exit(EXIT_FAILURE);
        }
        // filter size
        get_config_value(config, filter.n_terms, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","filter","n_terms"});

        // replace despiker window size
        despiker.window_size = filter.n_terms;

        // optional notch filtering (applied after FIR)
        run_tod_notch = false;
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","notch"})) {
            get_config_value(config, run_tod_notch, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","notch","enabled"});
            if (run_tod_notch) {
                auto freqs = config.template get_typed<std::vector<double>>(
                    std::tuple{"timestream","raw_time_chunk","filter","notch","freqs_Hz"});
                auto deltas = config.template get_typed<std::vector<double>>(
                    std::tuple{"timestream","raw_time_chunk","filter","notch","delta_f_Hz"});
                if (freqs.empty()) {
                    logger->error("notch enabled but freqs_Hz is empty");
                    std::exit(EXIT_FAILURE);
                }
                if (deltas.size() == 1 && freqs.size() > 1) {
                    deltas.resize(freqs.size(), deltas[0]);
                }
                if (deltas.size() != freqs.size()) {
                    logger->error("notch freqs_Hz and delta_f_Hz must have same length (or delta_f_Hz length 1)");
                    std::exit(EXIT_FAILURE);
                }
                filter.w0s.clear();
                filter.qs.clear();
                for (std::size_t i = 0; i < freqs.size(); ++i) {
                    if (freqs[i] <= 0.0 || deltas[i] <= 0.0) {
                        logger->error("notch freqs_Hz and delta_f_Hz must be > 0");
                        std::exit(EXIT_FAILURE);
                    }
                    filter.w0s.push_back(freqs[i]);
                    filter.qs.push_back(freqs[i] / deltas[i]);
                }
            }
        }
    }
    else {
        // explicitly set filter size to zero for inner time chunks
        filter.n_terms = 0;
        run_tod_notch = false;
    }

    // run optional iir highpass filter?
    run_tod_iir_highpass = false;
    filter.iir_highpass_freq_Hz = 0.0;
    filter.iir_highpass_order = 1;
    filter.iir_highpass_zero_phase = false;
    if (config.has(std::tuple{"timestream","raw_time_chunk","IIR_filter"})) {
        get_config_value(config, run_tod_iir_highpass, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","IIR_filter","enabled"});
        if (run_tod_iir_highpass) {
            get_config_value(config, filter.iir_highpass_freq_Hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","IIR_filter","freq_Hz"});
            get_config_value(config, filter.iir_highpass_order, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","IIR_filter","order"}, {}, {1});
            get_config_value(config, filter.iir_highpass_zero_phase, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","IIR_filter","zero_phase"});
            const bool has_iir_freq = config.template has_typed<double>(
                std::tuple{"timestream","raw_time_chunk","IIR_filter","freq_Hz"});
            if (has_iir_freq && filter.iir_highpass_freq_Hz <= 0.0) {
                logger->error("timestream.raw_time_chunk.IIR_filter.freq_Hz ({}) must be > 0",
                              filter.iir_highpass_freq_Hz);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    // keep despike filter-aware
    if (run_despike) {
        despiker.run_filter = run_tod_filter;
    }

    // run downsampling?
    get_config_value(config, run_downsample, missing_keys, invalid_keys,
                     std::tuple{"timestream","raw_time_chunk","downsample","enabled"});
    if (run_downsample) {
        // check if tod filtering is enabled
        if (!run_tod_filter) {
            logger->error("running downsampling without tod filtering will lose data!");
            std::exit(EXIT_FAILURE);
        }
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

    // optional alt-az template destriping on rtc output (before ptc cleaning)
    altaz_destripe = {};
    if (config.has(std::tuple{"timestream","raw_time_chunk","altaz_destripe"})) {
        get_config_value(config, altaz_destripe.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","altaz_destripe","enabled"});
        if (config.has(std::tuple{"timestream","raw_time_chunk","altaz_destripe","grouping"})) {
            get_config_value(config, altaz_destripe.grouping, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","altaz_destripe","grouping"},
                             {"nw", "network", "array", "all"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","altaz_destripe","fit_time_trend"})) {
            get_config_value(config, altaz_destripe.fit_time_trend, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","altaz_destripe","fit_time_trend"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","altaz_destripe","fit_derivs"})) {
            get_config_value(config, altaz_destripe.fit_derivs, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","altaz_destripe","fit_derivs"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","altaz_destripe","min_samples"})) {
            get_config_value(config, altaz_destripe.min_samples, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","altaz_destripe","min_samples"}, {}, {4});
        }
        if (altaz_destripe.enabled) {
            logger->info("raw_time_chunk.altaz_destripe enabled: grouping={} fit_time_trend={} fit_derivs={} min_samples={}",
                         altaz_destripe.grouping, altaz_destripe.fit_time_trend,
                         altaz_destripe.fit_derivs, altaz_destripe.min_samples);
        }
    }
}

template <class calib_t>
auto RTCProc::calc_map_indices(calib_t &calib, std::string map_grouping) {
    // indices for maps
    Eigen::VectorXI indices(calib.n_dets), map_indices(calib.n_dets);

    // overwrite map indices for networks
    if (map_grouping == "nw") {
        indices = calib.apt["nw"].template cast<Eigen::Index> ();
    }
    // overwrite map indices for arrays
    else if (map_grouping == "array") {
        indices = calib.apt["array"].template cast<Eigen::Index> ();
    }
    // overwrite map indices for detectors
    else if (map_grouping == "detector") {
        indices = Eigen::VectorXI::LinSpaced(calib.n_dets,0,calib.n_dets-1);
    }
    // overwrite map indices for fg
    else if (map_grouping == "fg") {
        indices = calib.apt["fg"].template cast<Eigen::Index> ();
    }
    // start at 0
    if (map_grouping != "fg") {
        std::unordered_map<Eigen::Index, Eigen::Index> group_to_index;
        Eigen::Index next_index = 0;
        for (Eigen::Index i=0; i<indices.size(); ++i) {
            const auto key = indices(i);
            auto it = group_to_index.find(key);
            if (it == group_to_index.end()) {
                group_to_index[key] = next_index;
                map_indices(i) = next_index;
                next_index++;
            }
            else {
                map_indices(i) = it->second;
            }
        }
    }
    else {
        // convert fg to indices
        std::map<Eigen::Index, Eigen::Index> fg_to_index, array_to_index;

        // get mapping from fg to map index
        for (Eigen::Index i=0; i<calib.fg.size(); ++i) {
            fg_to_index[calib.fg(i)] = i;
        }
        // get mapping from fg to map index
        for (Eigen::Index i=0; i<calib.arrays.size(); ++i) {
            array_to_index[calib.arrays(i)] = i;
        }
        // allocate map indices from fg
        for (Eigen::Index i=0; i<indices.size(); ++i) {
            map_indices(i) = fg_to_index[indices(i)] + calib.fg.size()*array_to_index[calib.apt["array"](i)];
        }
    }
    // return the map indices
    return std::move(map_indices);
}

template<class calib_t, typename telescope_t>
auto RTCProc::run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, TCData<TCDataKind::PTC, Eigen::MatrixXd> &out,
                  calib_t &calib, telescope_t &telescope, double pixel_size_rad, std::string map_grouping) {

    // number of points in scan
    Eigen::Index n_pts = in.scans.data.rows();

    // start index of inner scans
    auto si = filter.n_terms;
    // end index of inner scans
    auto sl = in.scan_indices.data(1) - in.scan_indices.data(0) + 1;

    // calculate the polarization angle
    if (run_polarization) {
        polarization.calc_angle(in, calib);
    }

    // resize fcf
    in.fcf.data.setOnes(in.scans.data.cols());

    // get indices for maps
    logger->debug("calculating map indices");
    auto map_indices = calc_map_indices(calib, map_grouping);

    if (run_calibrate) {
        logger->debug("calibrating timestream");
        // calibrate tod
        calibration.calibrate_tod(in, calib);

        in.status.calibrated = true;
    }

    if (run_extinction) {
        logger->debug("correcting extinction");
        // calc tau at toltec frequencies
        auto tau_freq = calibration.calc_tau(in.tel_data.data["TelElAct"], telescope.tau_225_GHz);
        // correct for extinction
        calibration.extinction_correction(in, calib, tau_freq);

        in.status.extinction_corrected = true;
    }

    // create kernel if requested
    if (run_kernel) {
        logger->debug("creating kernel timestream");
        // symmetric gaussian kernel
        if (kernel.type == "gaussian") {
            logger->debug("creating symmetric gaussian kernel");
            kernel.create_symmetric_gaussian_kernel(in, telescope.pixel_axes, calib.apt);
        }
        // airy kernel
        else if (kernel.type == "airy") {
            logger->debug("creating airy kernel");
            kernel.create_airy_kernel(in, telescope.pixel_axes, calib.apt);
        }
        // get kernel from fits
        else if (kernel.type == "fits") {
            logger->debug("getting kernel from fits");
            kernel.create_kernel_from_fits(in, telescope.pixel_axes, calib.apt, pixel_size_rad, map_indices);
        }

        in.status.kernel_generated = true;
    }

    // run despiking
    if (run_despike) {
        logger->debug("despiking");
        // despike data
        despiker.despike(in.scans.data, in.flags.data, calib.apt);

        // we want to replace spikes on a per array or network basis
        auto grp_limits = get_grouping(despiker.grouping, calib, in.scans.data.cols());

        logger->debug("replacing spikes");
        for (auto const& [key, val] : grp_limits) {
            // starting index
            auto start_index = std::get<0>(val);
            // size of block for each grouping
            auto n_dets = std::get<1>(val) - std::get<0>(val);

            // get the reference block of in scans that corresponds to the current array
            Eigen::Ref<Eigen::MatrixXd> in_scans_ref = in.scans.data.block(0, start_index, n_pts, n_dets);
            // eigen map to reference for input scans
            Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>
                in_scans(in_scans_ref.data(), in_scans_ref.rows(), in_scans_ref.cols(),
                         Eigen::OuterStride<>(in_scans_ref.outerStride()));

            // get the block of in flags that corresponds to the current array
            Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags_ref =
                in.flags.data.block(0, start_index, n_pts, n_dets);
            // eigen map to reference for input flags
            Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>, 0, Eigen::OuterStride<> >
                in_flags(in_flags_ref.data(), in_flags_ref.rows(), in_flags_ref.cols(),
                         Eigen::OuterStride<>(in_flags_ref.outerStride()));

            // replace spikes
            despiker.replace_spikes(in_scans, in_flags, calib.apt, start_index);
        }

        in.status.despiked = true;
    }

    bool ran_tod_filter_stage = false;

    // timestream filtering
    if (run_tod_filter) {
        logger->debug("convolving signal with tod filter");
        filter.convolve(in.scans.data);
        if (run_tod_notch) {
            logger->debug("applying notch filter to signal");
            filter.iir(in.scans.data);
        }

        // filter kernel
        if (run_kernel) {
            logger->debug("convolving kernel with tod filter");
            filter.convolve(in.kernel.data);
            if (run_tod_notch) {
                logger->debug("applying notch filter to kernel");
                filter.iir(in.kernel.data);
            }
        }
        ran_tod_filter_stage = true;
    }

    if (run_tod_iir_highpass) {
        logger->debug("applying iir highpass filter to signal");
        filter.iir_highpass(in.scans.data, telescope.fsmp);

        if (run_kernel) {
            logger->debug("applying iir highpass filter to kernel");
            filter.iir_highpass(in.kernel.data, telescope.fsmp);
        }
        ran_tod_filter_stage = true;
    }

    if (ran_tod_filter_stage) {
        in.status.tod_filtered = true;
    }

    if (run_downsample) {
        logger->debug("downsampling data");
        // get the block of out scans that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Map<Eigen::MatrixXd>> in_scans =
            in.scans.data.block(si, 0, sl, in.scans.data.cols());

        // get the block of in flags that corresponds to the inner scan indices
        Eigen::Ref<Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>> in_flags =
            in.flags.data.block(si, 0, sl, in.flags.data.cols());

        // downsample scans
        downsampler.downsample(in_scans, out.scans.data);
        // downsample flags
        downsampler.downsample_flags(in_flags, out.flags.data);

        // loop through telescope meta data and downsample
        logger->debug("downsampling telescope");
        for (auto const& x: in.tel_data.data) {
            // get the block of in tel data that corresponds to the inner scan indices
            Eigen::Ref<Eigen::VectorXd> in_tel =
                in.tel_data.data[x.first].segment(si,sl);

            downsampler.downsample(in_tel, out.tel_data.data[x.first]);
        }

        // downsample pointing
        for (auto const& x: in.pointing_offsets_arcsec.data) {
        Eigen::Ref<Eigen::VectorXd> in_pointing =
            in.pointing_offsets_arcsec.data[x.first].segment(si,sl);

            downsampler.downsample(in_pointing, out.pointing_offsets_arcsec.data[x.first]);
        }

        if (run_polarization) {
            if (calib.run_hwpr) {
                // downsample hwpr
                Eigen::Ref<Eigen::VectorXd> in_hwpr =
                    in.hwpr_angle.data.segment(si,sl);
                downsampler.downsample(in_hwpr, out.hwpr_angle.data);
            }
            // downsample detector angle
            Eigen::Ref<Eigen::VectorXd> in_angle =
                in.angle.data.segment(si, sl);
            downsampler.downsample(in_angle, out.angle.data);
        }
        // downsample kernel if requested
        if (run_kernel) {
            logger->debug("downsampling kernel");
            // get the block of in kernel scans that corresponds to the inner scan indices
            Eigen::Ref<Eigen::MatrixXd> in_kernel =
                in.kernel.data.block(si, 0, sl, in.kernel.data.cols());

            downsampler.downsample(in_kernel, out.kernel.data);
        }

        in.status.downsampled = true;
    }

    else {
        // copy data
        out.scans.data = in.scans.data.block(si, 0, sl, in.scans.data.cols());
        // copy flags
        out.flags.data = in.flags.data.block(si, 0, sl, in.flags.data.cols());
        // copy kernel
        if (run_kernel) {
            out.kernel.data = in.kernel.data.block(si, 0, sl, in.kernel.data.cols());
        }
        // copy telescope data
        for (auto const& x: in.tel_data.data) {
            out.tel_data.data[x.first] = in.tel_data.data[x.first].segment(si,sl);
        }
        // copy pointing offsets
        for (auto const& x: in.pointing_offsets_arcsec.data) {
            out.pointing_offsets_arcsec.data[x.first] = in.pointing_offsets_arcsec.data[x.first].segment(si,sl);
        }

        if (run_polarization) {
            // copy hwpr angle
            if (calib.run_hwpr) {
                out.hwpr_angle.data = in.hwpr_angle.data.segment(si,sl);
            }
            // copy detector angle
            out.angle.data = in.angle.data.segment(si,sl);
        }
    }

    if (altaz_destripe.enabled) {
        const auto az_it = out.tel_data.data.find("TelAzAct");
        const auto el_it = out.tel_data.data.find("TelElAct");
        if (az_it == out.tel_data.data.end() || el_it == out.tel_data.data.end()) {
            logger->warn("altaz_destripe enabled but TelAzAct/TelElAct not found; skipping");
        }
        else {
            const auto n_pts_out = out.scans.data.rows();
            const auto n_dets_out = out.scans.data.cols();
            if (n_pts_out > 0 && n_dets_out > 0) {
                Eigen::VectorXd az = az_it->second;
                Eigen::VectorXd el = el_it->second;
                if (az.size() != n_pts_out || el.size() != n_pts_out) {
                    logger->warn("altaz_destripe skipped: tel vector size mismatch (n_pts={} az={} el={})",
                                 n_pts_out, az.size(), el.size());
                }
                else {
                    // unwrap azimuth to avoid 2pi jumps in derivative templates
                    Eigen::VectorXd az_unwrap(n_pts_out);
                    az_unwrap(0) = az(0);
                    double az_offset = 0.0;
                    for (Eigen::Index i = 1; i < n_pts_out; ++i) {
                        const double prev = az_unwrap(i - 1);
                        const double curr_raw = az(i) + az_offset;
                        const double d = curr_raw - prev;
                        if (d > pi) {
                            az_offset -= 2.0 * pi;
                        }
                        else if (d < -pi) {
                            az_offset += 2.0 * pi;
                        }
                        az_unwrap(i) = az(i) + az_offset;
                    }

                    Eigen::VectorXd daz = Eigen::VectorXd::Zero(n_pts_out);
                    Eigen::VectorXd del = Eigen::VectorXd::Zero(n_pts_out);
                    if (n_pts_out > 1) {
                        daz(0) = az_unwrap(1) - az_unwrap(0);
                        del(0) = el(1) - el(0);
                        for (Eigen::Index i = 1; i < n_pts_out - 1; ++i) {
                            daz(i) = 0.5 * (az_unwrap(i + 1) - az_unwrap(i - 1));
                            del(i) = 0.5 * (el(i + 1) - el(i - 1));
                        }
                        daz(n_pts_out - 1) = az_unwrap(n_pts_out - 1) - az_unwrap(n_pts_out - 2);
                        del(n_pts_out - 1) = el(n_pts_out - 1) - el(n_pts_out - 2);
                    }

                    Eigen::Array<bool, Eigen::Dynamic, 1> tel_good(n_pts_out);
                    for (Eigen::Index i = 0; i < n_pts_out; ++i) {
                        tel_good(i) = std::isfinite(az_unwrap(i)) && std::isfinite(el(i)) &&
                                      std::isfinite(daz(i)) && std::isfinite(del(i));
                    }

                    auto zscore = [&](Eigen::VectorXd &v) {
                        double sum = 0.0;
                        Eigen::Index n = 0;
                        for (Eigen::Index i = 0; i < n_pts_out; ++i) {
                            if (tel_good(i)) {
                                sum += v(i);
                                ++n;
                            }
                        }
                        if (n <= 1) {
                            return false;
                        }
                        const double mean = sum / static_cast<double>(n);
                        double ss = 0.0;
                        for (Eigen::Index i = 0; i < n_pts_out; ++i) {
                            if (tel_good(i)) {
                                const double dv = v(i) - mean;
                                ss += dv * dv;
                            }
                        }
                        const double stddev = std::sqrt(ss / static_cast<double>(n - 1));
                        if (!std::isfinite(stddev) || stddev <= 0.0) {
                            return false;
                        }
                        for (Eigen::Index i = 0; i < n_pts_out; ++i) {
                            v(i) = (v(i) - mean) / stddev;
                        }
                        return true;
                    };

                    std::vector<Eigen::VectorXd> cols;
                    cols.reserve(6);
                    cols.push_back(Eigen::VectorXd::Ones(n_pts_out));

                    if (altaz_destripe.fit_time_trend) {
                        Eigen::VectorXd t(n_pts_out);
                        if (n_pts_out > 1) {
                            t = Eigen::VectorXd::LinSpaced(n_pts_out, -1.0, 1.0);
                        }
                        else {
                            t.setZero();
                        }
                        if (zscore(t)) {
                            cols.push_back(std::move(t));
                        }
                    }

                    if (zscore(az_unwrap)) {
                        cols.push_back(std::move(az_unwrap));
                    }
                    if (zscore(el)) {
                        cols.push_back(std::move(el));
                    }
                    if (altaz_destripe.fit_derivs) {
                        if (zscore(daz)) {
                            cols.push_back(std::move(daz));
                        }
                        if (zscore(del)) {
                            cols.push_back(std::move(del));
                        }
                    }

                    const Eigen::Index n_cols = static_cast<Eigen::Index>(cols.size());
                    if (n_cols < 2) {
                        logger->warn("altaz_destripe skipped: insufficient template columns");
                    }
                    else {
                        Eigen::MatrixXd X(n_pts_out, n_cols);
                        for (Eigen::Index c = 0; c < n_cols; ++c) {
                            X.col(c) = cols[static_cast<std::size_t>(c)];
                        }

                        std::string grp = altaz_destripe.grouping;
                        if (grp == "network") {
                            grp = "nw";
                        }
                        if (grp != "nw" && grp != "array" && grp != "all") {
                            logger->warn("altaz_destripe grouping '{}' unsupported; using 'nw'", grp);
                            grp = "nw";
                        }

                        std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;
                        if (grp == "all") {
                            grp_limits[0] = std::make_tuple(0, n_dets_out);
                        }
                        else {
                            grp_limits = get_grouping(grp, calib, n_dets_out);
                        }

                        Eigen::Index n_fit_total = 0;
                        Eigen::Index n_skip_total = 0;
                        for (const auto &[key, val] : grp_limits) {
                            const auto start = std::get<0>(val);
                            const auto end = std::get<1>(val);
                            for (Eigen::Index j = start; j < end; ++j) {
                                std::vector<Eigen::Index> rows;
                                rows.reserve(static_cast<std::size_t>(n_pts_out));
                                for (Eigen::Index i = 0; i < n_pts_out; ++i) {
                                    if (!out.flags.data(i, j) && tel_good(i)) {
                                        rows.push_back(i);
                                    }
                                }

                                const Eigen::Index n_use = static_cast<Eigen::Index>(rows.size());
                                const Eigen::Index n_min = std::max<Eigen::Index>(altaz_destripe.min_samples, n_cols + 2);
                                if (n_use < n_min) {
                                    ++n_skip_total;
                                    continue;
                                }

                                Eigen::MatrixXd X_use(n_use, n_cols);
                                Eigen::VectorXd y_use(n_use);
                                for (Eigen::Index r = 0; r < n_use; ++r) {
                                    const auto ii = rows[static_cast<std::size_t>(r)];
                                    X_use.row(r) = X.row(ii);
                                    y_use(r) = out.scans.data(ii, j);
                                }
                                const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_use);
                                if (qr.rank() < std::min<Eigen::Index>(n_cols, n_use)) {
                                    ++n_skip_total;
                                    continue;
                                }
                                const Eigen::VectorXd beta = qr.solve(y_use);
                                out.scans.data.col(j).noalias() -= X * beta;
                                ++n_fit_total;
                            }
                            logger->debug("altaz_destripe grouping={} key={} det_range=[{}, {})", grp, key, start, end);
                        }
                        logger->info("altaz_destripe applied: grouping={} templates={} fitted_detectors={} skipped_detectors={}",
                                     grp, n_cols, n_fit_total, n_skip_total);
                    }
                }
            }
        }
    }

    // copy scan indices
    out.scan_indices.data = in.scan_indices.data;
    // copy scan index
    out.index.data = in.index.data;
    // copy fcf
    out.fcf.data = in.fcf.data;
    // copy chunk status
    out.status = in.status;
    // copy noise
    out.noise.data = in.noise.data;

    // empty rtcdata
    in.scans.data.resize(0,0);
    in.flags.data.resize(0,0);
    in.kernel.data.resize(0,0);
    in.tel_data.data.clear();
    in.pointing_offsets_arcsec.data.clear();
    if (run_polarization) {
        if (calib.run_hwpr) {
            in.hwpr_angle.data.resize(0);
        }
        in.angle.data.resize(0);
    }

    in.noise.data.resize(0,0);

    return map_indices;
}

template <typename apt_t>
void RTCProc::remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_t &apt) {

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    // number of detectors flagged in apt
    Eigen::Index n_flagged = 0;

    // loop through detectors and set flags to one
    // for those flagged in apt table
    for (Eigen::Index i=0; i<n_dets; ++i) {
        Eigen::Index det_index = i;
        if (apt["flag"](det_index)!=0) {
            in.flags.data.col(i).setOnes();
            n_flagged++;
        }
    }

    logger->info("removed {} detectors flagged in APT table ({}%)",n_flagged,
                (static_cast<float>(n_flagged)/static_cast<float>(n_dets))*100);
}

template <typename calib_t>
auto RTCProc::remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();

    int n_nearby_tones = 0;

    // loop through flag columns
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // map from data column to apt row
        Eigen::Index det_index = i;
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

template <typename calib_t, typename pointing_offset_t>
void RTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string map_grouping,
                               std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, calib_t &calib,
                               bool apply_det_offsets, Eigen::Index scan_row_index) {
    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        // open netcdf file
        NcFile fo(filepath, netCDF::NcFile::write);

        // append common time chunk variables
        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, calib, apply_det_offsets,
                              scan_row_index);

        // sync file to make sure it gets updated
        fo.sync();
        // close file
        fo.close();

        logger->info("tod chunk written to {}", filepath);

    } catch (NcException &e) {
        logger->error("{}", e.what());
    }
}

} // namespace timestream
