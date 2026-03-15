#pragma once

#include <tula/algorithm/ei_stats.h>
#include <Eigen/QR>
#include <unsupported/Eigen/FFT>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

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

    struct NetworkStepMaskOptions {
        bool enabled = false;
        double step_window_sec = 0.5;
        double step_score_thresh = 2.5;
        double min_good_frac = 0.8;
        Eigen::Index min_det_used = 32;
        double min_step_det_frac = 0.05;
        double min_alignment_frac = 0.5;
        double cluster_tol_sec = 0.25;
        double mask_half_width_sec = 0.5;
        double max_flagged_fraction = 0.30;
    };
    NetworkStepMaskOptions network_step_mask;

    struct ImpulsiveCaptureOptions {
        bool enabled = false;
        double min_good_frac = 0.8;
        double min_event_z = 6.0;
        double near_event_z = 4.0;
        Eigen::Index max_events_per_network = 3;
        double snippet_half_width_sec = 0.25;
    };
    ImpulsiveCaptureOptions impulsive_capture;

    struct RTCDetectorDiagSummary : DespikeDetectorDiagSummary {
        Eigen::Index det = -1;
        double final_flagged_frac = std::numeric_limits<double>::quiet_NaN();
        int final_region_count = 0;
        double final_region_len_median = std::numeric_limits<double>::quiet_NaN();
        int final_region_len_max = 0;
        double step_score = std::numeric_limits<double>::quiet_NaN();
        int step_sample = -2147483647;
        double impulsive_peak_abs_z = std::numeric_limits<double>::quiet_NaN();
        int impulsive_peak_abs_sample = -2147483647;
        double impulsive_peak_delta_abs_z = std::numeric_limits<double>::quiet_NaN();
        int impulsive_peak_delta_abs_sample = -2147483647;
        int impulsive_near_abs_count = 0;
        int impulsive_near_delta_count = 0;
        double impulsive_event_score = std::numeric_limits<double>::quiet_NaN();
        int impulsive_event_sample = -2147483647;
        int impulsive_event_kind = -2147483647;
    };

    struct RTCNetworkDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_used = 0;
        double median_step_score = std::numeric_limits<double>::quiet_NaN();
        double max_step_score = std::numeric_limits<double>::quiet_NaN();
        double step_det_frac = std::numeric_limits<double>::quiet_NaN();
        double step_alignment_frac = std::numeric_limits<double>::quiet_NaN();
        int dominant_step_sample = -2147483647;
        double cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();
        double cm_peak_freq_Hz = std::numeric_limits<double>::quiet_NaN();
        double cm_peak_prominence = std::numeric_limits<double>::quiet_NaN();
        bool step_mask_applied = false;
        int step_mask_start_sample = -2147483647;
        int step_mask_end_sample = -2147483647;
        int step_mask_window_samples = 0;
        int step_mask_n_det_masked = 0;
        int step_mask_n_det_samples_flagged = 0;
        double step_mask_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
    };

    struct RTCImpulsiveSnippetSummary {
        int det = -2147483647;
        int event_sample = -2147483647;
        int event_kind = -2147483647;
        double event_score = std::numeric_limits<double>::quiet_NaN();
        double peak_abs_z = std::numeric_limits<double>::quiet_NaN();
        double peak_delta_abs_z = std::numeric_limits<double>::quiet_NaN();
        double added_flagged_frac = std::numeric_limits<double>::quiet_NaN();
        int raw_exceed_count = -2147483647;
        int local_exceed_count = -2147483647;
        int delta_spike_count = -2147483647;
        int local_delta_exceed_count = -2147483647;
        std::vector<double> snippet_z;
        std::vector<int> snippet_flag;
    };

    std::map<Eigen::Index, std::vector<RTCDetectorDiagSummary>> rtc_detector_summary_by_scan;
    std::map<Eigen::Index, std::vector<RTCNetworkDiagSummary>> rtc_network_summary_by_scan;
    std::map<Eigen::Index, std::map<Eigen::Index, std::vector<RTCImpulsiveSnippetSummary>>> rtc_impulsive_summary_by_scan;

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

    // summarize RTC diagnostics for the written output chunk
    template <typename calib_t>
    void capture_rtc_diagnostics(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, bool recompute_step_metrics = true);

    // optionally flag a network-wide window around aligned step-like events
    template <typename calib_t>
    void apply_network_step_mask(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &);

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
    network_step_mask = {};
    if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask"})) {
        get_config_value(config, network_step_mask.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","enabled"});
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","step_window_sec"})) {
            get_config_value(config, network_step_mask.step_window_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","step_window_sec"},
                             {}, {0.01});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","step_score_thresh"})) {
            get_config_value(config, network_step_mask.step_score_thresh, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","step_score_thresh"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","min_good_frac"})) {
            get_config_value(config, network_step_mask.min_good_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","min_good_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","min_det_used"})) {
            get_config_value(config, network_step_mask.min_det_used, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","min_det_used"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","min_step_det_frac"})) {
            get_config_value(config, network_step_mask.min_step_det_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","min_step_det_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","min_alignment_frac"})) {
            get_config_value(config, network_step_mask.min_alignment_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","min_alignment_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","cluster_tol_sec"})) {
            get_config_value(config, network_step_mask.cluster_tol_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","cluster_tol_sec"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","mask_half_width_sec"})) {
            get_config_value(config, network_step_mask.mask_half_width_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","mask_half_width_sec"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","max_flagged_fraction"})) {
            get_config_value(config, network_step_mask.max_flagged_fraction, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","network_step_mask","max_flagged_fraction"},
                             {}, {0.0}, {1.0});
        }
        if (network_step_mask.enabled) {
            logger->info(
                "raw_time_chunk.flagging.network_step_mask enabled: step_window_sec={} step_score_thresh={} min_good_frac={} min_det_used={} min_step_det_frac={} min_alignment_frac={} cluster_tol_sec={} mask_half_width_sec={} max_flagged_fraction={}",
                network_step_mask.step_window_sec,
                network_step_mask.step_score_thresh,
                network_step_mask.min_good_frac,
                network_step_mask.min_det_used,
                network_step_mask.min_step_det_frac,
                network_step_mask.min_alignment_frac,
                network_step_mask.cluster_tol_sec,
                network_step_mask.mask_half_width_sec,
                network_step_mask.max_flagged_fraction);
        }
    }
    impulsive_capture = {};
    if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture"})) {
        get_config_value(config, impulsive_capture.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","enabled"});
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","min_good_frac"})) {
            get_config_value(config, impulsive_capture.min_good_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","min_good_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","min_event_z"})) {
            get_config_value(config, impulsive_capture.min_event_z, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","min_event_z"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","near_event_z"})) {
            get_config_value(config, impulsive_capture.near_event_z, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","near_event_z"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","max_events_per_network"})) {
            get_config_value(config, impulsive_capture.max_events_per_network, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","max_events_per_network"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","snippet_half_width_sec"})) {
            get_config_value(config, impulsive_capture.snippet_half_width_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","snippet_half_width_sec"},
                             {}, {0.0});
        }
        if (impulsive_capture.enabled) {
            logger->info(
                "raw_time_chunk.flagging.impulsive_capture enabled: min_good_frac={} min_event_z={} near_event_z={} max_events_per_network={} snippet_half_width_sec={}",
                impulsive_capture.min_good_frac,
                impulsive_capture.min_event_z,
                impulsive_capture.near_event_z,
                impulsive_capture.max_events_per_network,
                impulsive_capture.snippet_half_width_sec);
        }
    }

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

        despiker.local_residual = {};
        if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual"})) {
            get_config_value(config, despiker.local_residual.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","despike","local_residual","enabled"});
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","window_sec"})) {
                get_config_value(config, despiker.local_residual.window_sec, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","window_sec"},
                                 {}, {0.0});
            }
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","sigma_scale"})) {
                get_config_value(config, despiker.local_residual.sigma_scale, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","sigma_scale"},
                                 {}, {0.0});
            }
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","delta_sigma_scale"})) {
                get_config_value(config, despiker.local_residual.delta_sigma_scale, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","delta_sigma_scale"},
                                 {}, {0.0});
            }
        }
        if (despiker.local_residual.enabled) {
            logger->info(
                "raw_time_chunk.despike.local_residual enabled: window_sec={} sigma_scale={} delta_sigma_scale={}",
                despiker.local_residual.window_sec,
                despiker.local_residual.sigma_scale,
                despiker.local_residual.delta_sigma_scale);
        }

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
                filter.notch_zero_phase = true;
                if (config.has(std::tuple{"timestream","raw_time_chunk","filter","notch","zero_phase"})) {
                    get_config_value(config, filter.notch_zero_phase, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","filter","notch","zero_phase"});
                }
                if (!filter.notch_zero_phase) {
                    logger->error("timestream.raw_time_chunk.filter.notch.zero_phase must be true to avoid phase shifts");
                    std::exit(EXIT_FAILURE);
                }
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
            if (!filter.iir_highpass_zero_phase) {
                logger->error("timestream.raw_time_chunk.IIR_filter.zero_phase must be true to avoid phase shifts");
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

    // preserve per-detector despike summaries for the final RTC output write.
    std::vector<RTCDetectorDiagSummary> rtc_diag_seed(static_cast<std::size_t>(out.scans.data.cols()));
    for (Eigen::Index det = 0; det < out.scans.data.cols(); ++det) {
        auto &row = rtc_diag_seed[static_cast<std::size_t>(det)];
        row.det = det;
        if (det < static_cast<Eigen::Index>(despiker.last_detector_diag.size())) {
            static_cast<DespikeDetectorDiagSummary &>(row) =
                despiker.last_detector_diag[static_cast<std::size_t>(det)];
        }
    }
    rtc_detector_summary_by_scan[out.index.data] = std::move(rtc_diag_seed);
    rtc_network_summary_by_scan.erase(out.index.data);
    rtc_impulsive_summary_by_scan.erase(out.index.data);

    if (network_step_mask.enabled) {
        capture_rtc_diagnostics(out, calib, true);
        apply_network_step_mask(out, calib);
        capture_rtc_diagnostics(out, calib, false);
    }

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
void RTCProc::capture_rtc_diagnostics(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib,
                                      bool recompute_step_metrics) {
    const Eigen::Index scan_id = in.index.data;
    const Eigen::Index n_pts = in.scans.data.rows();
    const Eigen::Index n_dets = in.scans.data.cols();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int fill_int = -2147483647;

    auto median_of = [&](std::vector<double> values) -> double {
        values.erase(
            std::remove_if(values.begin(), values.end(), [](double v) { return !std::isfinite(v); }),
            values.end());
        if (values.empty()) {
            return nan;
        }
        const auto mid = values.size() / 2;
        std::nth_element(values.begin(),
                         values.begin() + static_cast<std::ptrdiff_t>(mid),
                         values.end());
        double med = values[mid];
        if ((values.size() % 2) == 0) {
            auto lo = std::max_element(values.begin(),
                                       values.begin() + static_cast<std::ptrdiff_t>(mid));
            med = 0.5 * (med + *lo);
        }
        return med;
    };

    auto infer_dt_sec = [&]() -> double {
        for (const auto *name : {"TelTime", "TelUTC", "PpsTime"}) {
            const auto it = in.tel_data.data.find(name);
            if (it == in.tel_data.data.end()) {
                continue;
            }
            const auto &t = it->second;
            std::vector<double> dt;
            dt.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(t.size() - 1, 0)));
            for (Eigen::Index i = 1; i < t.size(); ++i) {
                const double diff = t(i) - t(i - 1);
                if (std::isfinite(diff) && diff > 0.0) {
                    dt.push_back(diff);
                }
            }
            const double med = median_of(std::move(dt));
            if (std::isfinite(med) && med > 0.0) {
                return med;
            }
        }
        return 1.0;
    };

    auto robust_center_scale = [&](const Eigen::VectorXd &x,
                                   const Eigen::Array<bool, Eigen::Dynamic, 1> &valid) {
        std::vector<double> good;
        good.reserve(static_cast<std::size_t>(x.size()));
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            if (valid(i) && std::isfinite(x(i))) {
                good.push_back(x(i));
            }
        }
        if (good.size() < 8) {
            return std::make_pair(nan, nan);
        }
        const double med = median_of(good);
        std::vector<double> abs_dev;
        abs_dev.reserve(good.size());
        for (const double v : good) {
            abs_dev.push_back(std::abs(v - med));
        }
        double sigma = median_of(abs_dev);
        if (std::isfinite(sigma) && sigma > 0.0) {
            sigma *= 1.4826;
        }
        else if (good.size() >= 2) {
            double mean = std::accumulate(good.begin(), good.end(), 0.0) /
                          static_cast<double>(good.size());
            double ss = 0.0;
            for (const double v : good) {
                const double dv = v - mean;
                ss += dv * dv;
            }
            sigma = std::sqrt(ss / static_cast<double>(good.size() - 1));
        }
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            sigma = nan;
        }
        return std::make_pair(med, sigma);
    };

    auto region_stats = [&](const auto &mask_expr) {
        Eigen::Array<bool, Eigen::Dynamic, 1> mask = mask_expr;
        std::vector<double> runs;
        runs.reserve(static_cast<std::size_t>(mask.size()));
        int max_run = 0;
        Eigen::Index i = 0;
        while (i < mask.size()) {
            if (mask(i)) {
                Eigen::Index j = i;
                while (j < mask.size() && mask(j)) {
                    ++j;
                }
                const int run_len = static_cast<int>(j - i);
                runs.push_back(static_cast<double>(run_len));
                max_run = std::max(max_run, run_len);
                i = j;
            }
            else {
                ++i;
            }
        }
        return std::make_tuple(static_cast<int>(runs.size()), median_of(std::move(runs)), max_run);
    };

    auto step_metric = [&](const Eigen::VectorXd &x,
                           const Eigen::Array<bool, Eigen::Dynamic, 1> &valid,
                           Eigen::Index window) {
        const Eigen::Index n = x.size();
        if (n < 16) {
            return std::make_pair(nan, fill_int);
        }
        auto [center, scale] = robust_center_scale(x, valid);
        if (!std::isfinite(center) || !std::isfinite(scale) || scale <= 0.0) {
            return std::make_pair(nan, fill_int);
        }
        Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd good = Eigen::VectorXd::Zero(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            if (valid(i) && std::isfinite(x(i))) {
                z(i) = (x(i) - center) / scale;
                good(i) = 1.0;
            }
        }

        const Eigen::Index max_w = std::max<Eigen::Index>(4, n / 4);
        const Eigen::Index w = std::min(std::max<Eigen::Index>(window, 4), max_w);
        if (n < (2 * w + 2)) {
            return std::make_pair(nan, fill_int);
        }

        Eigen::VectorXd csum(n + 1), gsum(n + 1);
        csum(0) = 0.0;
        gsum(0) = 0.0;
        for (Eigen::Index i = 0; i < n; ++i) {
            csum(i + 1) = csum(i) + z(i);
            gsum(i + 1) = gsum(i) + good(i);
        }

        const double min_count = std::max(4.0, 0.5 * static_cast<double>(w));
        double best = nan;
        int best_idx = fill_int;
        for (Eigen::Index center_idx = w; center_idx < n - w; ++center_idx) {
            const double left_n = gsum(center_idx) - gsum(center_idx - w);
            const double right_n = gsum(center_idx + w) - gsum(center_idx);
            if (left_n < min_count || right_n < min_count) {
                continue;
            }
            const double left_mean = (csum(center_idx) - csum(center_idx - w)) / left_n;
            const double right_mean = (csum(center_idx + w) - csum(center_idx)) / right_n;
            const double delta = std::abs(right_mean - left_mean);
            if (!std::isfinite(best) || delta > best) {
                best = delta;
                best_idx = static_cast<int>(center_idx);
            }
        }
        return std::make_pair(best, best_idx);
    };

    struct ImpulsiveMetrics {
        double peak_abs_z = std::numeric_limits<double>::quiet_NaN();
        int peak_abs_sample = -2147483647;
        double peak_delta_abs_z = std::numeric_limits<double>::quiet_NaN();
        int peak_delta_abs_sample = -2147483647;
        int near_abs_count = 0;
        int near_delta_count = 0;
        double event_score = std::numeric_limits<double>::quiet_NaN();
        int event_sample = -2147483647;
        int event_kind = -2147483647;
    };

    auto impulsive_metric = [&](const Eigen::VectorXd &x,
                                const Eigen::Array<bool, Eigen::Dynamic, 1> &valid) {
        ImpulsiveMetrics out;
        const Eigen::Index n = x.size();
        if (n < 4) {
            return out;
        }

        auto [center, scale] = robust_center_scale(x, valid);
        if (std::isfinite(center) && std::isfinite(scale) && scale > 0.0) {
            for (Eigen::Index i = 0; i < n; ++i) {
                if (!valid(i) || !std::isfinite(x(i))) {
                    continue;
                }
                const double abs_z = std::abs((x(i) - center) / scale);
                if (std::isfinite(abs_z) && abs_z >= impulsive_capture.near_event_z) {
                    ++out.near_abs_count;
                }
                if (!std::isfinite(out.peak_abs_z) || abs_z > out.peak_abs_z) {
                    out.peak_abs_z = abs_z;
                    out.peak_abs_sample = static_cast<int>(i);
                }
            }
        }

        std::vector<double> deltas;
        deltas.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(n - 1, 0)));
        for (Eigen::Index i = 0; i < n - 1; ++i) {
            if (valid(i) && valid(i + 1) && std::isfinite(x(i)) && std::isfinite(x(i + 1))) {
                deltas.push_back(x(i + 1) - x(i));
            }
        }
        if (deltas.size() >= 4) {
            const double delta_med = median_of(deltas);
            std::vector<double> delta_abs_dev;
            delta_abs_dev.reserve(deltas.size());
            for (const double v : deltas) {
                delta_abs_dev.push_back(std::abs(v - delta_med));
            }
            double delta_sigma = median_of(delta_abs_dev);
            if (std::isfinite(delta_sigma) && delta_sigma > 0.0) {
                delta_sigma *= 1.4826;
            }
            else if (deltas.size() >= 2) {
                const double mean =
                    std::accumulate(deltas.begin(), deltas.end(), 0.0) / static_cast<double>(deltas.size());
                double ss = 0.0;
                for (const double v : deltas) {
                    const double dv = v - mean;
                    ss += dv * dv;
                }
                delta_sigma = std::sqrt(ss / static_cast<double>(deltas.size() - 1));
            }
            if (std::isfinite(delta_sigma) && delta_sigma > 0.0) {
                for (Eigen::Index i = 0; i < n - 1; ++i) {
                    if (!(valid(i) && valid(i + 1)) || !std::isfinite(x(i)) || !std::isfinite(x(i + 1))) {
                        continue;
                    }
                    const double delta = x(i + 1) - x(i);
                    const double abs_z = std::abs((delta - delta_med) / delta_sigma);
                    if (std::isfinite(abs_z) && abs_z >= impulsive_capture.near_event_z) {
                        ++out.near_delta_count;
                    }
                    if (!std::isfinite(out.peak_delta_abs_z) || abs_z > out.peak_delta_abs_z) {
                        out.peak_delta_abs_z = abs_z;
                        out.peak_delta_abs_sample = static_cast<int>(i + 1);
                    }
                }
            }
        }

        if (std::isfinite(out.peak_abs_z) || std::isfinite(out.peak_delta_abs_z)) {
            const bool use_delta =
                std::isfinite(out.peak_delta_abs_z) &&
                (!std::isfinite(out.peak_abs_z) || out.peak_delta_abs_z > out.peak_abs_z);
            out.event_score = use_delta ? out.peak_delta_abs_z : out.peak_abs_z;
            out.event_sample = use_delta ? out.peak_delta_abs_sample : out.peak_abs_sample;
            out.event_kind = use_delta ? 1 : 0;
        }
        return out;
    };

    auto dominant_cluster = [&](std::vector<double> values, double tol) {
        values.erase(
            std::remove_if(values.begin(), values.end(), [](double v) { return !std::isfinite(v); }),
            values.end());
        if (values.empty()) {
            return std::make_pair(nan, 0.0);
        }
        std::sort(values.begin(), values.end());
        if (values.size() == 1 || tol <= 0.0) {
            return std::make_pair(values.front(), 1.0);
        }
        std::size_t best_i = 0;
        std::size_t best_j = 0;
        std::size_t j = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (j < i) {
                j = i;
            }
            while (j + 1 < values.size() && (values[j + 1] - values[i]) <= tol) {
                ++j;
            }
            if ((j - i) > (best_j - best_i)) {
                best_i = i;
                best_j = j;
            }
        }
        std::vector<double> cluster(values.begin() + static_cast<std::ptrdiff_t>(best_i),
                                    values.begin() + static_cast<std::ptrdiff_t>(best_j + 1));
        const double center = median_of(std::move(cluster));
        const double frac = static_cast<double>(best_j - best_i + 1) / static_cast<double>(values.size());
        return std::make_pair(center, frac);
    };

    const double dt_sec = infer_dt_sec();
    const double fs_hz = (std::isfinite(dt_sec) && dt_sec > 0.0) ? (1.0 / dt_sec) : nan;
    const double dt_for_step = (std::isfinite(dt_sec) && dt_sec > 0.0) ? dt_sec : 1.0e-6;
    const Eigen::Index step_window = std::max<Eigen::Index>(
        4, static_cast<Eigen::Index>(std::llround(network_step_mask.step_window_sec / dt_for_step)));

    auto det_it = rtc_detector_summary_by_scan.find(scan_id);
    auto nw_it = rtc_network_summary_by_scan.find(scan_id);
    const bool have_detector_summary =
        det_it != rtc_detector_summary_by_scan.end() &&
        det_it->second.size() == static_cast<std::size_t>(n_dets);
    const bool have_network_summary =
        nw_it != rtc_network_summary_by_scan.end();
    const bool need_step_metrics = recompute_step_metrics || !have_detector_summary || !have_network_summary;

    std::vector<RTCDetectorDiagSummary> det_summary;
    if (have_detector_summary) {
        det_summary = det_it->second;
    }
    else {
        det_summary.assign(static_cast<std::size_t>(n_dets), RTCDetectorDiagSummary{});
    }

    for (Eigen::Index det = 0; det < n_dets; ++det) {
        auto &row = det_summary[static_cast<std::size_t>(det)];
        row.det = det;
        Eigen::Array<bool, Eigen::Dynamic, 1> valid(n_pts);
        Eigen::Index n_flagged = 0;
        for (Eigen::Index i = 0; i < n_pts; ++i) {
            valid(i) = std::isfinite(in.scans.data(i, det)) && !in.flags.data(i, det);
            if (in.flags.data(i, det)) {
                ++n_flagged;
            }
        }
        row.final_flagged_frac =
            static_cast<double>(n_flagged) /
            static_cast<double>(std::max<Eigen::Index>(n_pts, 1));
        std::tie(row.final_region_count, row.final_region_len_median, row.final_region_len_max) =
            region_stats(in.flags.data.col(det).array());
        const auto impulsive = impulsive_metric(in.scans.data.col(det), valid);
        row.impulsive_peak_abs_z = impulsive.peak_abs_z;
        row.impulsive_peak_abs_sample = impulsive.peak_abs_sample;
        row.impulsive_peak_delta_abs_z = impulsive.peak_delta_abs_z;
        row.impulsive_peak_delta_abs_sample = impulsive.peak_delta_abs_sample;
        row.impulsive_near_abs_count = impulsive.near_abs_count;
        row.impulsive_near_delta_count = impulsive.near_delta_count;
        row.impulsive_event_score = impulsive.event_score;
        row.impulsive_event_sample = impulsive.event_sample;
        row.impulsive_event_kind = impulsive.event_kind;
        if (need_step_metrics) {
            std::tie(row.step_score, row.step_sample) =
                step_metric(in.scans.data.col(det), valid, step_window);
        }
    }
    rtc_detector_summary_by_scan[scan_id] = det_summary;

    if (impulsive_capture.enabled) {
        const Eigen::Index snippet_half_width = std::max<Eigen::Index>(
            0, static_cast<Eigen::Index>(std::llround(impulsive_capture.snippet_half_width_sec /
                                                      std::max(dt_for_step, 1.0e-6))));
        const Eigen::Index snippet_len = 2 * snippet_half_width + 1;
        std::map<Eigen::Index, std::vector<RTCImpulsiveSnippetSummary>> impulsive_by_network;
        auto grp_limits = get_grouping("nw", calib, n_dets);
        for (const auto &[nw, bounds] : grp_limits) {
            const auto start = std::get<0>(bounds);
            const auto end = std::get<1>(bounds);
            std::vector<RTCImpulsiveSnippetSummary> candidates;
            candidates.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(end - start, 0)));
            for (Eigen::Index det = start; det < end; ++det) {
                const auto &row = det_summary[static_cast<std::size_t>(det)];
                const double good_frac = 1.0 - row.final_flagged_frac;
                if (!std::isfinite(good_frac) || good_frac < impulsive_capture.min_good_frac) {
                    continue;
                }
                if (!std::isfinite(row.impulsive_event_score) ||
                    row.impulsive_event_score < impulsive_capture.min_event_z ||
                    row.impulsive_event_sample == fill_int) {
                    continue;
                }

                RTCImpulsiveSnippetSummary slot;
                slot.det = static_cast<int>(det);
                slot.event_sample = row.impulsive_event_sample;
                slot.event_kind = row.impulsive_event_kind;
                slot.event_score = row.impulsive_event_score;
                slot.peak_abs_z = row.impulsive_peak_abs_z;
                slot.peak_delta_abs_z = row.impulsive_peak_delta_abs_z;
                slot.added_flagged_frac = row.added_flagged_frac;
                slot.raw_exceed_count = row.raw_exceed_count;
                slot.local_exceed_count = row.local_exceed_count;
                slot.delta_spike_count = row.delta_spike_count;
                slot.local_delta_exceed_count = row.local_delta_exceed_count;
                slot.snippet_z.assign(static_cast<std::size_t>(std::max<Eigen::Index>(snippet_len, 0)), nan);
                slot.snippet_flag.assign(static_cast<std::size_t>(std::max<Eigen::Index>(snippet_len, 0)), fill_int);

                Eigen::Array<bool, Eigen::Dynamic, 1> valid(n_pts);
                for (Eigen::Index i = 0; i < n_pts; ++i) {
                    valid(i) = std::isfinite(in.scans.data(i, det)) && !in.flags.data(i, det);
                }
                auto [center, scale] = robust_center_scale(in.scans.data.col(det), valid);
                if (!(std::isfinite(center) && std::isfinite(scale) && scale > 0.0)) {
                    center = 0.0;
                    scale = nan;
                }
                for (Eigen::Index k = 0; k < snippet_len; ++k) {
                    const Eigen::Index sample = static_cast<Eigen::Index>(slot.event_sample) + k - snippet_half_width;
                    if (sample < 0 || sample >= n_pts) {
                        continue;
                    }
                    slot.snippet_flag[static_cast<std::size_t>(k)] = in.flags.data(sample, det) ? 1 : 0;
                    const double v = in.scans.data(sample, det);
                    if (std::isfinite(v) && std::isfinite(scale) && scale > 0.0) {
                        slot.snippet_z[static_cast<std::size_t>(k)] = (v - center) / scale;
                    }
                }

                candidates.push_back(std::move(slot));
            }

            std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
                if (std::isfinite(a.event_score) && std::isfinite(b.event_score) && a.event_score != b.event_score) {
                    return a.event_score > b.event_score;
                }
                if (std::isfinite(a.peak_delta_abs_z) && std::isfinite(b.peak_delta_abs_z) &&
                    a.peak_delta_abs_z != b.peak_delta_abs_z) {
                    return a.peak_delta_abs_z > b.peak_delta_abs_z;
                }
                return a.det < b.det;
            });
            if (static_cast<Eigen::Index>(candidates.size()) > impulsive_capture.max_events_per_network) {
                candidates.resize(static_cast<std::size_t>(impulsive_capture.max_events_per_network));
            }
            impulsive_by_network[nw] = std::move(candidates);
        }
        rtc_impulsive_summary_by_scan[scan_id] = std::move(impulsive_by_network);
    }
    else {
        rtc_impulsive_summary_by_scan.erase(scan_id);
    }

    if (need_step_metrics) {
        std::vector<RTCNetworkDiagSummary> nw_summary;
        const double min_good_frac = network_step_mask.min_good_frac;
        const double step_score_thresh = network_step_mask.step_score_thresh;
        const double cluster_tol_samples = std::max(
            2.0,
            ((network_step_mask.cluster_tol_sec > 0.0)
                 ? (network_step_mask.cluster_tol_sec / dt_for_step)
                 : (0.5 * static_cast<double>(step_window))));
        auto grp_limits = get_grouping("nw", calib, n_dets);
        nw_summary.reserve(grp_limits.size());
        for (const auto &[nw, bounds] : grp_limits) {
            const auto start = std::get<0>(bounds);
            const auto end = std::get<1>(bounds);
            RTCNetworkDiagSummary row;
            row.nw = nw;
            row.n_det_input = end - start;

            Eigen::MatrixXd centered = Eigen::MatrixXd::Zero(n_pts, std::max<Eigen::Index>(end - start, 0));
            Eigen::Index n_used = 0;
            std::vector<double> step_scores;
            std::vector<double> step_samples_active;
            step_scores.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(end - start, 0)));
            step_samples_active.reserve(step_scores.capacity());

            for (Eigen::Index det = start; det < end; ++det) {
                Eigen::Array<bool, Eigen::Dynamic, 1> valid(n_pts);
                Eigen::Index n_valid = 0;
                for (Eigen::Index i = 0; i < n_pts; ++i) {
                    valid(i) = std::isfinite(in.scans.data(i, det)) && !in.flags.data(i, det);
                    if (valid(i)) {
                        ++n_valid;
                    }
                }
                const double good_frac = static_cast<double>(n_valid) /
                                         static_cast<double>(std::max<Eigen::Index>(n_pts, 1));
                if (good_frac < min_good_frac) {
                    continue;
                }
                auto [center, scale] = robust_center_scale(in.scans.data.col(det), valid);
                if (!std::isfinite(center) || !std::isfinite(scale) || scale <= 0.0) {
                    continue;
                }
                for (Eigen::Index i = 0; i < n_pts; ++i) {
                    if (valid(i) && std::isfinite(in.scans.data(i, det))) {
                        centered(i, n_used) = in.scans.data(i, det) - center;
                    }
                }
                const auto &det_row = det_summary[static_cast<std::size_t>(det)];
                if (std::isfinite(det_row.step_score)) {
                    step_scores.push_back(det_row.step_score);
                    if (det_row.step_score >= step_score_thresh && det_row.step_sample != fill_int) {
                        step_samples_active.push_back(static_cast<double>(det_row.step_sample));
                    }
                }
                ++n_used;
            }

            row.n_det_used = n_used;
            if (!step_scores.empty()) {
                row.median_step_score = median_of(step_scores);
                row.max_step_score = *std::max_element(step_scores.begin(), step_scores.end());
                const auto n_active = static_cast<double>(step_samples_active.size());
                row.step_det_frac = n_active / static_cast<double>(step_scores.size());
                auto [step_center, step_align] = dominant_cluster(step_samples_active, cluster_tol_samples);
                row.step_alignment_frac = step_align;
                if (std::isfinite(step_center)) {
                    row.dominant_step_sample = static_cast<int>(std::llround(step_center));
                }
            }

            if (n_used >= 1 && n_pts >= 16 && std::isfinite(fs_hz) && fs_hz > 0.0) {
                centered.conservativeResize(Eigen::NoChange, n_used);
                Eigen::VectorXd cm(n_pts);
                std::vector<double> scratch;
                scratch.reserve(static_cast<std::size_t>(n_used));
                for (Eigen::Index i = 0; i < n_pts; ++i) {
                    scratch.clear();
                    for (Eigen::Index j = 0; j < n_used; ++j) {
                        scratch.push_back(centered(i, j));
                    }
                    cm(i) = median_of(scratch);
                }
                const double cm_mean = cm.mean();
                cm.array() -= cm_mean;
                if (n_pts > 1) {
                    constexpr double two_pi = 6.283185307179586476925286766559;
                    for (Eigen::Index i = 0; i < n_pts; ++i) {
                        const double w = 0.5 * (1.0 - std::cos(
                            two_pi * static_cast<double>(i) / static_cast<double>(n_pts - 1)));
                        cm(i) *= w;
                    }
                }

                Eigen::FFT<double> fft;
                fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
                fft.SetFlag(Eigen::FFT<double>::Unscaled);
                Eigen::VectorXcd spec;
                fft.fwd(spec, cm);
                if (spec.size() > 1) {
                    std::vector<double> power_low;
                    std::vector<double> power_mid;
                    std::vector<double> power_local;
                    power_low.reserve(static_cast<std::size_t>(spec.size()));
                    power_mid.reserve(static_cast<std::size_t>(spec.size()));
                    power_local.reserve(static_cast<std::size_t>(spec.size()));
                    double peak_power = -1.0;
                    double peak_freq = nan;
                    for (Eigen::Index k = 1; k < spec.size(); ++k) {
                        const double freq = static_cast<double>(k) * fs_hz / static_cast<double>(n_pts);
                        const double power = std::norm(spec(k));
                        if (!std::isfinite(power) || !std::isfinite(freq)) {
                            continue;
                        }
                        if (freq >= 0.05 && freq < 0.5) {
                            power_low.push_back(power);
                        }
                        if (freq >= 0.5 && freq < 2.0) {
                            power_mid.push_back(power);
                        }
                        if (freq >= 0.05 && freq <= std::min(16.0, 0.5 * fs_hz)) {
                            power_local.push_back(power);
                            if (power > peak_power) {
                                peak_power = power;
                                peak_freq = freq;
                            }
                        }
                    }
                    const double bp_low = median_of(power_low);
                    const double bp_mid = median_of(power_mid);
                    if (std::isfinite(bp_low) && std::isfinite(bp_mid) && bp_mid > 0.0) {
                        row.cm_low_mid_ratio = bp_low / bp_mid;
                    }
                    row.cm_peak_freq_Hz = peak_freq;
                    const double local_med = median_of(power_local);
                    if (std::isfinite(local_med) && local_med > 0.0 && peak_power > 0.0) {
                        row.cm_peak_prominence = peak_power / local_med;
                    }
                }
            }

            nw_summary.push_back(row);
        }
        rtc_network_summary_by_scan[scan_id] = std::move(nw_summary);
    }
}

template <typename calib_t>
void RTCProc::apply_network_step_mask(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib) {
    if (!network_step_mask.enabled) {
        return;
    }
    const auto scan_id = in.index.data;
    const auto nw_it = rtc_network_summary_by_scan.find(scan_id);
    if (nw_it == rtc_network_summary_by_scan.end()) {
        return;
    }

    auto infer_dt_sec = [&]() -> double {
        for (const auto *name : {"TelTime", "TelUTC", "PpsTime"}) {
            const auto it = in.tel_data.data.find(name);
            if (it == in.tel_data.data.end()) {
                continue;
            }
            const auto &t = it->second;
            std::vector<double> dt;
            dt.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(t.size() - 1, 0)));
            for (Eigen::Index i = 1; i < t.size(); ++i) {
                const double diff = t(i) - t(i - 1);
                if (std::isfinite(diff) && diff > 0.0) {
                    dt.push_back(diff);
                }
            }
            if (!dt.empty()) {
                const auto mid = dt.size() / 2;
                std::nth_element(dt.begin(),
                                 dt.begin() + static_cast<std::ptrdiff_t>(mid),
                                 dt.end());
                return dt[mid];
            }
        }
        return 1.0;
    };

    const double dt_sec =
        (network_step_mask.mask_half_width_sec > 0.0 || network_step_mask.cluster_tol_sec > 0.0)
            ? infer_dt_sec()
            : 1.0;
    const Eigen::Index n_pts = in.scans.data.rows();
    const auto grp_limits = get_grouping("nw", calib, in.scans.data.cols());

    for (auto &row : nw_it->second) {
        row.step_mask_applied = false;
        row.step_mask_start_sample = -2147483647;
        row.step_mask_end_sample = -2147483647;
        row.step_mask_window_samples = 0;
        row.step_mask_n_det_masked = 0;
        row.step_mask_n_det_samples_flagged = 0;
        row.step_mask_flagged_fraction = std::numeric_limits<double>::quiet_NaN();

        const auto grp_it = grp_limits.find(row.nw);
        if (grp_it == grp_limits.end()) {
            continue;
        }
        if (!std::isfinite(row.step_det_frac) || !std::isfinite(row.step_alignment_frac) ||
            row.dominant_step_sample <= -2147483647) {
            continue;
        }
        if (row.n_det_used < network_step_mask.min_det_used ||
            row.step_det_frac < network_step_mask.min_step_det_frac ||
            row.step_alignment_frac < network_step_mask.min_alignment_frac) {
            continue;
        }

        const auto start_det = std::get<0>(grp_it->second);
        const auto end_det = std::get<1>(grp_it->second);
        const Eigen::Index half_width = std::max<Eigen::Index>(
            0, static_cast<Eigen::Index>(std::llround(network_step_mask.mask_half_width_sec /
                                                      std::max(dt_sec, 1.0e-6))));
        const Eigen::Index center = static_cast<Eigen::Index>(row.dominant_step_sample);
        const Eigen::Index start_sample = std::max<Eigen::Index>(0, center - half_width);
        const Eigen::Index end_sample = std::min<Eigen::Index>(n_pts - 1, center + half_width);
        const Eigen::Index window_samples = std::max<Eigen::Index>(0, end_sample - start_sample + 1);
        if (window_samples <= 0 || end_det <= start_det) {
            continue;
        }

        Eigen::Index good_detector_samples = 0;
        Eigen::Index newly_flagged = 0;
        for (Eigen::Index det = start_det; det < end_det; ++det) {
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (!in.flags.data(i, det) && std::isfinite(in.scans.data(i, det))) {
                    ++good_detector_samples;
                }
            }
            for (Eigen::Index i = start_sample; i <= end_sample; ++i) {
                if (!in.flags.data(i, det) && std::isfinite(in.scans.data(i, det))) {
                    ++newly_flagged;
                }
            }
        }

        const double flagged_fraction =
            static_cast<double>(newly_flagged) /
            static_cast<double>(std::max<Eigen::Index>(1, good_detector_samples));
        if (network_step_mask.max_flagged_fraction > 0.0 &&
            flagged_fraction > network_step_mask.max_flagged_fraction) {
            logger->info(
                "network_step_mask rejected for scan {} nw {}: dominant_sample={} window_samples={} proposed_fraction={} exceeds max_flagged_fraction={}",
                scan_id + 1,
                row.nw,
                row.dominant_step_sample,
                window_samples,
                flagged_fraction,
                network_step_mask.max_flagged_fraction);
            continue;
        }

        in.flags.data.block(start_sample, start_det, window_samples, end_det - start_det).setOnes();
        row.step_mask_applied = true;
        row.step_mask_start_sample = static_cast<int>(start_sample);
        row.step_mask_end_sample = static_cast<int>(end_sample);
        row.step_mask_window_samples = static_cast<int>(window_samples);
        row.step_mask_n_det_masked = static_cast<int>(end_det - start_det);
        row.step_mask_n_det_samples_flagged = static_cast<int>(newly_flagged);
        row.step_mask_flagged_fraction = flagged_fraction;

        logger->info(
            "network_step_mask applied for scan {} nw {}: dominant_sample={} window=[{}, {}] n_det_masked={} newly_flagged={} flagged_fraction={}",
            scan_id + 1,
            row.nw,
            row.dominant_step_sample,
            start_sample,
            end_sample,
            end_det - start_det,
            newly_flagged,
            flagged_fraction);
    }
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
        const bool have_step_diag =
            rtc_detector_summary_by_scan.find(in.index.data) != rtc_detector_summary_by_scan.end() &&
            rtc_network_summary_by_scan.find(in.index.data) != rtc_network_summary_by_scan.end();
        capture_rtc_diagnostics(in, calib, !have_step_diag);

        // open netcdf file
        NcFile fo(filepath, netCDF::NcFile::write);

        // append common time chunk variables
        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, calib, apply_det_offsets,
                              scan_row_index);

        const int fill_int = -2147483647;
        const double fill_double = std::numeric_limits<double>::quiet_NaN();
        const auto scan_row = static_cast<unsigned long>((scan_row_index >= 0) ? scan_row_index : in.index.data);

        const auto det_diag_it = rtc_detector_summary_by_scan.find(in.index.data);
        const auto nw_diag_it = rtc_network_summary_by_scan.find(in.index.data);
        const auto impulsive_it = rtc_impulsive_summary_by_scan.find(in.index.data);

        NcDim n_dets_dim = fo.getDim("n_dets");
        if (!n_dets_dim.isNull()) {
            const auto n_dets = n_dets_dim.getSize();
            std::vector<std::size_t> start_scan_det = {scan_row, 0};
            std::vector<std::size_t> size_scan_det = {1, n_dets};

            auto det_double_values = [&](auto getter) {
                std::vector<double> values(n_dets, fill_double);
                if (det_diag_it != rtc_detector_summary_by_scan.end()) {
                    const auto n_copy = std::min<std::size_t>(n_dets, det_diag_it->second.size());
                    for (std::size_t i = 0; i < n_copy; ++i) {
                        values[i] = getter(det_diag_it->second[i]);
                    }
                }
                return values;
            };
            auto det_int_values = [&](auto getter) {
                std::vector<int> values(n_dets, fill_int);
                if (det_diag_it != rtc_detector_summary_by_scan.end()) {
                    const auto n_copy = std::min<std::size_t>(n_dets, det_diag_it->second.size());
                    for (std::size_t i = 0; i < n_copy; ++i) {
                        values[i] = getter(det_diag_it->second[i]);
                    }
                }
                return values;
            };

            auto write_det_double = [&](const std::string &name, auto getter) {
                NcVar v = fo.getVar(name);
                if (!v.isNull()) {
                    auto values = det_double_values(getter);
                    v.putVar(start_scan_det, size_scan_det, values.data());
                }
            };
            auto write_det_int = [&](const std::string &name, auto getter) {
                NcVar v = fo.getVar(name);
                if (!v.isNull()) {
                    auto values = det_int_values(getter);
                    v.putVar(start_scan_det, size_scan_det, values.data());
                }
            };

            write_det_int("rtc_despike_raw_exceed_count",
                          [](const auto &row) { return row.raw_exceed_count; });
            write_det_int("rtc_despike_local_exceed_count",
                          [](const auto &row) { return row.local_exceed_count; });
            write_det_int("rtc_despike_delta_spike_count",
                          [](const auto &row) { return row.delta_spike_count; });
            write_det_int("rtc_despike_local_delta_exceed_count",
                          [](const auto &row) { return row.local_delta_exceed_count; });
            write_det_double("rtc_despike_added_flagged_frac",
                             [](const auto &row) { return row.added_flagged_frac; });
            write_det_int("rtc_despike_added_region_count",
                          [](const auto &row) { return row.added_region_count; });
            write_det_double("rtc_despike_added_region_len_median",
                             [](const auto &row) { return row.added_region_len_median; });
            write_det_int("rtc_despike_added_region_len_max",
                          [](const auto &row) { return row.added_region_len_max; });
            write_det_double("rtc_despike_max_raw_abs_z",
                             [](const auto &row) { return row.max_raw_abs_z; });
            write_det_double("rtc_despike_max_local_abs_z",
                             [](const auto &row) { return row.max_local_abs_z; });
            write_det_double("rtc_despike_max_delta_abs_z",
                             [](const auto &row) { return row.max_delta_abs_z; });
            write_det_double("rtc_despike_max_local_delta_abs_z",
                             [](const auto &row) { return row.max_local_delta_abs_z; });
            write_det_double("rtc_final_flagged_frac",
                             [](const auto &row) { return row.final_flagged_frac; });
            write_det_int("rtc_final_region_count",
                          [](const auto &row) { return row.final_region_count; });
            write_det_double("rtc_final_region_len_median",
                             [](const auto &row) { return row.final_region_len_median; });
            write_det_int("rtc_final_region_len_max",
                          [](const auto &row) { return row.final_region_len_max; });
            write_det_double("rtc_step_score",
                             [](const auto &row) { return row.step_score; });
            write_det_int("rtc_step_sample",
                          [](const auto &row) { return row.step_sample; });
            write_det_double("rtc_impulsive_peak_abs_z",
                             [](const auto &row) { return row.impulsive_peak_abs_z; });
            write_det_int("rtc_impulsive_peak_abs_sample",
                          [](const auto &row) { return row.impulsive_peak_abs_sample; });
            write_det_double("rtc_impulsive_peak_delta_abs_z",
                             [](const auto &row) { return row.impulsive_peak_delta_abs_z; });
            write_det_int("rtc_impulsive_peak_delta_abs_sample",
                          [](const auto &row) { return row.impulsive_peak_delta_abs_sample; });
            write_det_int("rtc_impulsive_near_abs_count",
                          [](const auto &row) { return row.impulsive_near_abs_count; });
            write_det_int("rtc_impulsive_near_delta_count",
                          [](const auto &row) { return row.impulsive_near_delta_count; });
            write_det_double("rtc_impulsive_event_score",
                             [](const auto &row) { return row.impulsive_event_score; });
            write_det_int("rtc_impulsive_event_sample",
                          [](const auto &row) { return row.impulsive_event_sample; });
            write_det_int("rtc_impulsive_event_kind",
                          [](const auto &row) { return row.impulsive_event_kind; });
        }

        NcVar nw_ids_v = fo.getVar("rtc_diag_network_ids");
        if (!nw_ids_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_rtcdiag");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }
                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};
                auto nw_double_values = [&](auto getter) {
                    std::vector<double> values(n_nws, fill_double);
                    if (nw_diag_it != rtc_network_summary_by_scan.end()) {
                        for (const auto &row : nw_diag_it->second) {
                            const auto it = nw_to_index.find(row.nw);
                            if (it == nw_to_index.end() || it->second >= n_nws) {
                                continue;
                            }
                            values[it->second] = getter(row);
                        }
                    }
                    return values;
                };
                auto nw_int_values = [&](auto getter) {
                    std::vector<int> values(n_nws, fill_int);
                    if (nw_diag_it != rtc_network_summary_by_scan.end()) {
                        for (const auto &row : nw_diag_it->second) {
                            const auto it = nw_to_index.find(row.nw);
                            if (it == nw_to_index.end() || it->second >= n_nws) {
                                continue;
                            }
                            values[it->second] = getter(row);
                        }
                    }
                    return values;
                };
                auto write_nw_double = [&](const std::string &name, auto getter) {
                    NcVar v = fo.getVar(name);
                    if (!v.isNull()) {
                        auto values = nw_double_values(getter);
                        v.putVar(start_scan_nw, size_scan_nw, values.data());
                    }
                };
                auto write_nw_int = [&](const std::string &name, auto getter) {
                    NcVar v = fo.getVar(name);
                    if (!v.isNull()) {
                        auto values = nw_int_values(getter);
                        v.putVar(start_scan_nw, size_scan_nw, values.data());
                    }
                };

                write_nw_int("rtc_network_n_det_input",
                             [](const auto &row) { return static_cast<int>(row.n_det_input); });
                write_nw_int("rtc_network_n_det_used",
                             [](const auto &row) { return static_cast<int>(row.n_det_used); });
                write_nw_double("rtc_network_step_score_median",
                                [](const auto &row) { return row.median_step_score; });
                write_nw_double("rtc_network_step_score_max",
                                [](const auto &row) { return row.max_step_score; });
                write_nw_double("rtc_network_step_det_frac",
                                [](const auto &row) { return row.step_det_frac; });
                write_nw_double("rtc_network_step_alignment_frac",
                                [](const auto &row) { return row.step_alignment_frac; });
                write_nw_int("rtc_network_step_dominant_sample",
                             [](const auto &row) { return row.dominant_step_sample; });
                write_nw_double("rtc_network_cm_low_mid_ratio",
                                [](const auto &row) { return row.cm_low_mid_ratio; });
                write_nw_double("rtc_network_cm_peak_freq_hz",
                                [](const auto &row) { return row.cm_peak_freq_Hz; });
                write_nw_double("rtc_network_cm_peak_prominence",
                                [](const auto &row) { return row.cm_peak_prominence; });
                write_nw_int("rtc_network_step_mask_applied",
                             [](const auto &row) { return row.step_mask_applied ? 1 : 0; });
                write_nw_int("rtc_network_step_mask_start_sample",
                             [](const auto &row) { return row.step_mask_start_sample; });
                write_nw_int("rtc_network_step_mask_end_sample",
                             [](const auto &row) { return row.step_mask_end_sample; });
                write_nw_int("rtc_network_step_mask_window_samples",
                             [](const auto &row) { return row.step_mask_window_samples; });
                write_nw_int("rtc_network_step_mask_n_det_masked",
                             [](const auto &row) { return row.step_mask_n_det_masked; });
                write_nw_int("rtc_network_step_mask_n_det_samples_flagged",
                             [](const auto &row) { return row.step_mask_n_det_samples_flagged; });
                write_nw_double("rtc_network_step_mask_flagged_fraction",
                                [](const auto &row) { return row.step_mask_flagged_fraction; });

                NcDim n_slots_dim = fo.getDim("n_rtc_impulsive_slots");
                NcDim n_snip_dim = fo.getDim("n_rtc_impulsive_samples");
                if (!n_slots_dim.isNull() && !n_snip_dim.isNull()) {
                    const auto n_slots = n_slots_dim.getSize();
                    const auto n_snip = n_snip_dim.getSize();
                    std::vector<std::size_t> start_scan_nw_slot = {scan_row, 0, 0};
                    std::vector<std::size_t> size_scan_nw_slot = {1, n_nws, n_slots};
                    std::vector<std::size_t> start_scan_nw_slot_snip = {scan_row, 0, 0, 0};
                    std::vector<std::size_t> size_scan_nw_slot_snip = {1, n_nws, n_slots, n_snip};
                    const auto total_slots = n_nws * n_slots;
                    const auto total_snip = total_slots * n_snip;

                    auto imp_slot_int_values = [&](auto getter) {
                        std::vector<int> values(total_slots, fill_int);
                        if (impulsive_it != rtc_impulsive_summary_by_scan.end()) {
                            for (const auto &[nw, slots] : impulsive_it->second) {
                                const auto it = nw_to_index.find(nw);
                                if (it == nw_to_index.end() || it->second >= n_nws) {
                                    continue;
                                }
                                const auto nw_index = it->second;
                                const auto n_copy = std::min<std::size_t>(n_slots, slots.size());
                                for (std::size_t slot = 0; slot < n_copy; ++slot) {
                                    values[nw_index * n_slots + slot] = getter(slots[slot]);
                                }
                            }
                        }
                        return values;
                    };
                    auto imp_slot_double_values = [&](auto getter) {
                        std::vector<double> values(total_slots, fill_double);
                        if (impulsive_it != rtc_impulsive_summary_by_scan.end()) {
                            for (const auto &[nw, slots] : impulsive_it->second) {
                                const auto it = nw_to_index.find(nw);
                                if (it == nw_to_index.end() || it->second >= n_nws) {
                                    continue;
                                }
                                const auto nw_index = it->second;
                                const auto n_copy = std::min<std::size_t>(n_slots, slots.size());
                                for (std::size_t slot = 0; slot < n_copy; ++slot) {
                                    values[nw_index * n_slots + slot] = getter(slots[slot]);
                                }
                            }
                        }
                        return values;
                    };
                    auto imp_snip_double_values = [&](auto getter) {
                        std::vector<double> values(total_snip, fill_double);
                        if (impulsive_it != rtc_impulsive_summary_by_scan.end()) {
                            for (const auto &[nw, slots] : impulsive_it->second) {
                                const auto it = nw_to_index.find(nw);
                                if (it == nw_to_index.end() || it->second >= n_nws) {
                                    continue;
                                }
                                const auto nw_index = it->second;
                                const auto n_copy = std::min<std::size_t>(n_slots, slots.size());
                                for (std::size_t slot = 0; slot < n_copy; ++slot) {
                                    const auto &snippet = getter(slots[slot]);
                                    const auto n_copy_snip = std::min<std::size_t>(n_snip, snippet.size());
                                    for (std::size_t k = 0; k < n_copy_snip; ++k) {
                                        values[(nw_index * n_slots + slot) * n_snip + k] = snippet[k];
                                    }
                                }
                            }
                        }
                        return values;
                    };
                    auto imp_snip_int_values = [&](auto getter) {
                        std::vector<int> values(total_snip, fill_int);
                        if (impulsive_it != rtc_impulsive_summary_by_scan.end()) {
                            for (const auto &[nw, slots] : impulsive_it->second) {
                                const auto it = nw_to_index.find(nw);
                                if (it == nw_to_index.end() || it->second >= n_nws) {
                                    continue;
                                }
                                const auto nw_index = it->second;
                                const auto n_copy = std::min<std::size_t>(n_slots, slots.size());
                                for (std::size_t slot = 0; slot < n_copy; ++slot) {
                                    const auto &snippet = getter(slots[slot]);
                                    const auto n_copy_snip = std::min<std::size_t>(n_snip, snippet.size());
                                    for (std::size_t k = 0; k < n_copy_snip; ++k) {
                                        values[(nw_index * n_slots + slot) * n_snip + k] = snippet[k];
                                    }
                                }
                            }
                        }
                        return values;
                    };
                    auto write_imp_slot_int = [&](const std::string &name, auto getter) {
                        NcVar v = fo.getVar(name);
                        if (!v.isNull()) {
                            auto values = imp_slot_int_values(getter);
                            v.putVar(start_scan_nw_slot, size_scan_nw_slot, values.data());
                        }
                    };
                    auto write_imp_slot_double = [&](const std::string &name, auto getter) {
                        NcVar v = fo.getVar(name);
                        if (!v.isNull()) {
                            auto values = imp_slot_double_values(getter);
                            v.putVar(start_scan_nw_slot, size_scan_nw_slot, values.data());
                        }
                    };
                    auto write_imp_snip_double = [&](const std::string &name, auto getter) {
                        NcVar v = fo.getVar(name);
                        if (!v.isNull()) {
                            auto values = imp_snip_double_values(getter);
                            v.putVar(start_scan_nw_slot_snip, size_scan_nw_slot_snip, values.data());
                        }
                    };
                    auto write_imp_snip_int = [&](const std::string &name, auto getter) {
                        NcVar v = fo.getVar(name);
                        if (!v.isNull()) {
                            auto values = imp_snip_int_values(getter);
                            v.putVar(start_scan_nw_slot_snip, size_scan_nw_slot_snip, values.data());
                        }
                    };

                    write_imp_slot_int("rtc_impulsive_slot_det_index",
                                       [](const auto &slot) { return slot.det; });
                    write_imp_slot_int("rtc_impulsive_slot_event_sample",
                                       [](const auto &slot) { return slot.event_sample; });
                    write_imp_slot_int("rtc_impulsive_slot_event_kind",
                                       [](const auto &slot) { return slot.event_kind; });
                    write_imp_slot_double("rtc_impulsive_slot_event_score",
                                          [](const auto &slot) { return slot.event_score; });
                    write_imp_slot_double("rtc_impulsive_slot_peak_abs_z",
                                          [](const auto &slot) { return slot.peak_abs_z; });
                    write_imp_slot_double("rtc_impulsive_slot_peak_delta_abs_z",
                                          [](const auto &slot) { return slot.peak_delta_abs_z; });
                    write_imp_slot_double("rtc_impulsive_slot_added_flagged_frac",
                                          [](const auto &slot) { return slot.added_flagged_frac; });
                    write_imp_slot_int("rtc_impulsive_slot_raw_exceed_count",
                                       [](const auto &slot) { return slot.raw_exceed_count; });
                    write_imp_slot_int("rtc_impulsive_slot_local_exceed_count",
                                       [](const auto &slot) { return slot.local_exceed_count; });
                    write_imp_slot_int("rtc_impulsive_slot_delta_spike_count",
                                       [](const auto &slot) { return slot.delta_spike_count; });
                    write_imp_slot_int("rtc_impulsive_slot_local_delta_exceed_count",
                                       [](const auto &slot) { return slot.local_delta_exceed_count; });
                    write_imp_snip_double("rtc_impulsive_slot_snippet_z",
                                          [](const auto &slot) -> const auto & { return slot.snippet_z; });
                    write_imp_snip_int("rtc_impulsive_slot_snippet_flag",
                                       [](const auto &slot) -> const auto & { return slot.snippet_flag; });
                }
            }
        }

        if (det_diag_it != rtc_detector_summary_by_scan.end()) {
            rtc_detector_summary_by_scan.erase(det_diag_it);
        }
        if (nw_diag_it != rtc_network_summary_by_scan.end()) {
            rtc_network_summary_by_scan.erase(nw_diag_it);
        }
        if (impulsive_it != rtc_impulsive_summary_by_scan.end()) {
            rtc_impulsive_summary_by_scan.erase(impulsive_it);
        }

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
