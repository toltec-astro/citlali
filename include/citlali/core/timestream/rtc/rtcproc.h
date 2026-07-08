#pragma once

#include <tula/algorithm/ei_stats.h>
#include <Eigen/QR>
#include <unsupported/Eigen/FFT>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/timestream/timestream.h>
#include <citlali/core/engine/io.h>
#include <citlali/core/utils/pointing.h>

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

    bool despike_source_protection_config_enabled = true;

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
        double snippet_pre_window_sec = 0.25;
        double snippet_post_window_sec = 0.25;
    };
    ImpulsiveCaptureOptions impulsive_capture;

    struct ImpulsiveCoincidenceOptions {
        bool enabled = false;
        double min_good_frac = 0.8;
        double event_score_thresh = 6.0;
        Eigen::Index min_det_used = 32;
        double min_impulsive_det_frac = 0.05;
        double min_alignment_frac = 0.5;
        Eigen::Index min_networks_aligned = 3;
        double high_score_override_thresh = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index high_score_min_networks_aligned = 0;
        double cluster_tol_sec = 0.03;
        double mask_pre_window_sec = 0.03;
        double mask_post_window_sec = 0.03;
        double max_flagged_fraction = 0.10;
    };
    ImpulsiveCoincidenceOptions impulsive_coincidence;

    struct RTCLineAuditOptions {
        bool enabled = false;
        double line_min_hz = 1.0;
        double line_max_hz = 60.0;
        double segment_sec = 4.0;
        double min_segment_sec = 2.0;
        double overlap_frac = 0.5;
        Eigen::Index continuum_radius_bins = 8;
        double prominence_thresh = 8.0;
        double cm_prominence_thresh = 6.0;
        double min_good_frac = 0.8;
        Eigen::Index min_windows = 2;
        Eigen::Index max_peaks_per_detector = 3;
        Eigen::Index max_det = 128;
        Eigen::Index min_det_for_network = 16;
        double cluster_tol_hz = 0.15;
        double notch_min_detector_frac = 0.10;
        Eigen::Index notch_min_detectors = 8;
        double notch_min_cm_prominence = 10.0;
        double detector_min_prominence = 12.0;
        double detector_min_line_power_frac = 0.10;
        double bad_detector_max_cluster_frac = 0.10;
        bool pre_filter_enabled = true;
        bool post_filter_enabled = false;
        bool post_filter_apply_shared_notches = false;
        bool post_filter_apply_detector_notches = false;
        Eigen::Index post_filter_apply_iterations = 1;
        double post_filter_line_min_hz = std::numeric_limits<double>::quiet_NaN();
        double post_filter_line_max_hz = std::numeric_limits<double>::quiet_NaN();
        bool ptc_model_protected_enabled = false;
        bool ptc_require_model_subtracted = true;
        bool ptc_apply_fixed_notches = false;
        bool ptc_apply_shared_notches = false;
        bool ptc_apply_detector_notches = false;
        Eigen::Index ptc_apply_iterations = 1;
        double ptc_line_min_hz = std::numeric_limits<double>::quiet_NaN();
        double ptc_line_max_hz = std::numeric_limits<double>::quiet_NaN();
        bool fixed_notch_enabled = false;
        std::vector<double> fixed_notch_freqs_hz;
        std::vector<double> fixed_notch_widths_hz{0.25};
        double fixed_notch_exclusion_half_width_hz = 0.25;
        bool apply_shared_notches = false;
        Eigen::Index apply_min_support_networks = 2;
        double apply_min_detector_frac = 0.90;
        double apply_min_common_mode_prominence = 150.0;
        double apply_width_scale = 1.5;
        double apply_min_width_hz = 0.25;
        double apply_max_width_hz = 1.50;
        Eigen::Index apply_max_notches = 3;
        double apply_cluster_tol_hz = 0.25;
        double detector_notch_min_prominence = 8.0;
        double detector_notch_min_line_power_frac = 0.0;
        Eigen::Index detector_notch_max_notches = 3;
        double detector_notch_width_scale = 1.0;
        double detector_notch_min_width_hz = 0.25;
        double detector_notch_max_width_hz = 1.50;
        Eigen::Index detector_notch_context_samples = 0;
    };
    RTCLineAuditOptions line_audit;

    struct RTCLineAuditSharedCandidate {
        double freq_hz = std::numeric_limits<double>::quiet_NaN();
        int detector_count = 0;
        double detector_frac = std::numeric_limits<double>::quiet_NaN();
        double median_prominence = std::numeric_limits<double>::quiet_NaN();
        double max_prominence = std::numeric_limits<double>::quiet_NaN();
        double width_hz = std::numeric_limits<double>::quiet_NaN();
        double freq_min_hz = std::numeric_limits<double>::quiet_NaN();
        double freq_max_hz = std::numeric_limits<double>::quiet_NaN();
        double line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double common_mode_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double common_mode_prominence = std::numeric_limits<double>::quiet_NaN();
        double notch_score = std::numeric_limits<double>::quiet_NaN();
        bool recommend_notch = false;
        bool applied_notch = false;
        double applied_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double applied_width_hz = std::numeric_limits<double>::quiet_NaN();
        int applied_support_network_count = 0;
    };

    struct RTCLineAuditDiagSummary {
        int n_det_used = 0;
        double shared_freq_hz = std::numeric_limits<double>::quiet_NaN();
        int shared_detector_count = 0;
        double shared_detector_frac = std::numeric_limits<double>::quiet_NaN();
        double shared_median_prominence = std::numeric_limits<double>::quiet_NaN();
        double shared_max_prominence = std::numeric_limits<double>::quiet_NaN();
        double shared_width_hz = std::numeric_limits<double>::quiet_NaN();
        double shared_line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double shared_common_mode_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double shared_common_mode_prominence = std::numeric_limits<double>::quiet_NaN();
        double shared_notch_score = std::numeric_limits<double>::quiet_NaN();
        bool shared_recommend_notch = false;
        int n_applied_notches = 0;
        bool shared_applied_notch = false;
        double shared_applied_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double shared_applied_width_hz = std::numeric_limits<double>::quiet_NaN();
        int shared_applied_support_network_count = 0;
        int detector_candidate_uid = kTransientFillInt;
        double detector_candidate_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double detector_candidate_prominence = std::numeric_limits<double>::quiet_NaN();
        double detector_candidate_line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double detector_candidate_cluster_detector_frac = std::numeric_limits<double>::quiet_NaN();
        bool detector_candidate_recommend_flag = false;
        std::vector<RTCLineAuditSharedCandidate> shared_candidates;
    };

    struct FilterEdgeGuardOptions {
        bool enabled = false;
        std::string mode = "flag";
        std::string combine = "sum";
        Eigen::Index min_samples = 0;
        Eigen::Index extra_samples = 0;
        Eigen::Index max_samples = 128;
        double iir_settle_attenuation = 0.01;
        bool apply_fir = true;
        bool apply_notch = true;
        bool apply_dynamic_notch = true;
        bool apply_iir_highpass = true;
        bool apply_downsample = true;
        Eigen::Index context_samples = 0;
        Eigen::Index guard_samples = 0;
    };
    FilterEdgeGuardOptions filter_edge_guard;

    struct RTCDetectorDiagSummary : DespikeDetectorDiagSummary {
        Eigen::Index det = -1;
        double final_flagged_frac = std::numeric_limits<double>::quiet_NaN();
        int final_region_count = 0;
        double final_region_len_median = std::numeric_limits<double>::quiet_NaN();
        int final_region_len_max = 0;
        TransientEvent step_event;
        double step_score = std::numeric_limits<double>::quiet_NaN();
        int step_sample = kTransientFillInt;
        TransientEvent impulsive_event;
        double impulsive_peak_abs_z = std::numeric_limits<double>::quiet_NaN();
        int impulsive_peak_abs_sample = kTransientFillInt;
        double impulsive_peak_delta_abs_z = std::numeric_limits<double>::quiet_NaN();
        int impulsive_peak_delta_abs_sample = kTransientFillInt;
        int impulsive_near_abs_count = 0;
        int impulsive_near_delta_count = 0;
        double impulsive_event_score = std::numeric_limits<double>::quiet_NaN();
        int impulsive_event_sample = kTransientFillInt;
        int impulsive_event_kind = kTransientFillInt;
        int detector_notch_n_applied = 0;
        double detector_notch_primary_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double detector_notch_primary_width_hz = std::numeric_limits<double>::quiet_NaN();
        double detector_notch_primary_prominence = std::numeric_limits<double>::quiet_NaN();
        double detector_notch_primary_line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double detector_notch_rms_before = std::numeric_limits<double>::quiet_NaN();
        double detector_notch_rms_after = std::numeric_limits<double>::quiet_NaN();
    };

    struct RTCNetworkDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index impulsive_n_det_used = 0;
        int line_audit_n_det_used = 0;
        double line_audit_shared_freq_hz = std::numeric_limits<double>::quiet_NaN();
        int line_audit_shared_detector_count = 0;
        double line_audit_shared_detector_frac = std::numeric_limits<double>::quiet_NaN();
        double line_audit_shared_median_prominence = std::numeric_limits<double>::quiet_NaN();
        double line_audit_shared_max_prominence = std::numeric_limits<double>::quiet_NaN();
        double line_audit_shared_width_hz = std::numeric_limits<double>::quiet_NaN();
        double line_audit_shared_line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double line_audit_shared_common_mode_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double line_audit_shared_common_mode_prominence = std::numeric_limits<double>::quiet_NaN();
        double line_audit_shared_notch_score = std::numeric_limits<double>::quiet_NaN();
        bool line_audit_shared_recommend_notch = false;
        int line_audit_n_applied_notches = 0;
        bool line_audit_shared_applied_notch = false;
        double line_audit_shared_applied_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double line_audit_shared_applied_width_hz = std::numeric_limits<double>::quiet_NaN();
        int line_audit_shared_applied_support_network_count = 0;
        int line_audit_detector_candidate_uid = kTransientFillInt;
        double line_audit_detector_candidate_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double line_audit_detector_candidate_prominence = std::numeric_limits<double>::quiet_NaN();
        double line_audit_detector_candidate_line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double line_audit_detector_candidate_cluster_detector_frac = std::numeric_limits<double>::quiet_NaN();
        bool line_audit_detector_candidate_recommend_flag = false;
        std::vector<RTCLineAuditSharedCandidate> line_audit_shared_candidates;
        RTCLineAuditDiagSummary post_line_audit;
        TransientEvent step_event;
        double median_step_score = std::numeric_limits<double>::quiet_NaN();
        double max_step_score = std::numeric_limits<double>::quiet_NaN();
        double step_det_frac = std::numeric_limits<double>::quiet_NaN();
        double step_alignment_frac = std::numeric_limits<double>::quiet_NaN();
        int dominant_step_sample = kTransientFillInt;
        double median_impulsive_score = std::numeric_limits<double>::quiet_NaN();
        double max_impulsive_score = std::numeric_limits<double>::quiet_NaN();
        double impulsive_det_frac = std::numeric_limits<double>::quiet_NaN();
        double impulsive_alignment_frac = std::numeric_limits<double>::quiet_NaN();
        int dominant_impulsive_sample = kTransientFillInt;
        double cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();
        double cm_peak_freq_Hz = std::numeric_limits<double>::quiet_NaN();
        double cm_peak_prominence = std::numeric_limits<double>::quiet_NaN();
        bool step_mask_applied = false;
        int step_mask_start_sample = kTransientFillInt;
        int step_mask_end_sample = kTransientFillInt;
        int step_mask_window_samples = 0;
        int step_mask_n_det_masked = 0;
        int step_mask_n_det_samples_flagged = 0;
        double step_mask_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        bool impulsive_mask_applied = false;
        int impulsive_mask_start_sample = kTransientFillInt;
        int impulsive_mask_end_sample = kTransientFillInt;
        int impulsive_mask_window_samples = 0;
        int impulsive_mask_n_det_masked = 0;
        int impulsive_mask_n_det_samples_flagged = 0;
        double impulsive_mask_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        bool impulsive_mask_candidate_available = false;
        bool impulsive_mask_local_trigger = false;
        bool impulsive_mask_cross_network_trigger = false;
        bool impulsive_mask_high_score_override_trigger = false;
        bool impulsive_mask_rejected_max_fraction = false;
        int impulsive_mask_candidate_center_sample = kTransientFillInt;
        int impulsive_mask_cluster_center_sample = kTransientFillInt;
        int impulsive_mask_cluster_network_count = 0;
        int impulsive_mask_cluster_active_count = 0;
        int impulsive_mask_total_active_count = 0;
        double impulsive_mask_cluster_peak_score = std::numeric_limits<double>::quiet_NaN();
        double impulsive_mask_override_score = std::numeric_limits<double>::quiet_NaN();
        bool impulsive_mask_override_uses_network_peak = false;
        double impulsive_mask_proposed_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
    };

    struct RTCImpulsiveSnippetSummary {
        TransientEvent event;
        int det = -2147483647;
        int event_sample = kTransientFillInt;
        int event_kind = kTransientFillInt;
        double event_score = std::numeric_limits<double>::quiet_NaN();
        double peak_abs_z = std::numeric_limits<double>::quiet_NaN();
        double peak_delta_abs_z = std::numeric_limits<double>::quiet_NaN();
        double added_flagged_frac = std::numeric_limits<double>::quiet_NaN();
        int raw_exceed_count = -2147483647;
        int local_raw_candidate_count = -2147483647;
        int local_raw_accepted_event_count = -2147483647;
        int local_flagged_sample_count = -2147483647;
        int local_raw_reject_count = -2147483647;
        int delta_spike_count = -2147483647;
        int local_delta_candidate_count = -2147483647;
        int local_delta_accepted_event_count = -2147483647;
        int local_delta_reject_count = -2147483647;
        std::vector<double> snippet_z;
        std::vector<int> snippet_flag;
    };

    struct RTCSourceProtectionDiagSummary {
        bool enabled = false;
        int protected_samples = 0;
        int total_samples = 0;
        double radius_arcsec = std::numeric_limits<double>::quiet_NaN();
    };

    std::map<Eigen::Index, std::vector<RTCDetectorDiagSummary>> rtc_detector_summary_by_scan;
    std::map<Eigen::Index, std::vector<RTCNetworkDiagSummary>> rtc_network_summary_by_scan;
    std::map<Eigen::Index, std::map<Eigen::Index, std::vector<RTCImpulsiveSnippetSummary>>> rtc_impulsive_summary_by_scan;
    std::map<Eigen::Index, RTCSourceProtectionDiagSummary> rtc_source_protection_summary_by_scan;
    std::shared_ptr<std::mutex> diag_summary_mutex = std::make_shared<std::mutex>();

    std::vector<RTCDetectorDiagSummary> snapshot_detector_diag_summary(Eigen::Index scan_id) {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto it = rtc_detector_summary_by_scan.find(scan_id);
        if (it == rtc_detector_summary_by_scan.end()) {
            return {};
        }
        return it->second;
    }

    RTCSourceProtectionDiagSummary snapshot_source_protection_diag_summary(Eigen::Index scan_id) {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto it = rtc_source_protection_summary_by_scan.find(scan_id);
        if (it == rtc_source_protection_summary_by_scan.end()) {
            return {};
        }
        return it->second;
    }

    // get config file
    template <typename config_t>
    void get_config(config_t &, std::vector<std::vector<std::string>> &, std::vector<std::vector<std::string>> &);

    // get indices to map from detector to index in map vectors
    template <class calib_t>
    auto calc_map_indices(calib_t &, std::string);

    // run the main processing
    template<typename calib_t, typename telescope_t>
    auto run(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             calib_t &, telescope_t &, double, std::string,
             TCData<TCDataKind::RTC, Eigen::MatrixXd> *tod_outer_output = nullptr);

    // remove nearby tones
    template <typename calib_t>
    auto remove_nearby_tones(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, std::string);

    // remove flagged detectors
    template <typename apt_t>
    void remove_flagged_dets(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_t &);

    // summarize RTC diagnostics for the written output chunk
    template <typename calib_t>
    void capture_rtc_diagnostics(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &,
                                 bool recompute_step_metrics = true,
                                 bool recompute_impulsive_metrics = true);

    // analyze narrowband periodic line structure on the pre-filter RTC chunk
    template <typename tc_t, typename calib_t>
    void capture_rtc_line_audit(tc_t &, calib_t &, Eigen::Index start_sample, Eigen::Index n_samples,
                                const RTCLineAuditOptions &, bool post_filter_stage = false);

    double rtc_line_audit_fixed_notch_width_hz(const RTCLineAuditOptions &, std::size_t) const;

    bool rtc_line_audit_frequency_excluded_by_fixed_notch(double, const RTCLineAuditOptions &) const;

    Eigen::Index count_rtc_line_audit_fixed_notches(double, const RTCLineAuditOptions &,
                                                    double *min_width_hz = nullptr) const;

    // optionally apply a fixed census-derived RTC notch set before the residual dynamic audit
    template <typename tc_t>
    Eigen::Index apply_rtc_line_audit_fixed_notches(tc_t &, double fs_hz,
                                                    const RTCLineAuditOptions &);

    // optionally apply chunk-level shared-line notches from the RTC line audit
    template <typename tc_t>
    Eigen::Index apply_rtc_line_audit_shared_notches(tc_t &, double fs_hz,
                                                     const RTCLineAuditOptions &,
                                                     bool post_filter_stage = false);

    // optionally apply detector-local zero-phase notches from the available scan context
    template <typename tc_t>
    Eigen::Index apply_rtc_line_audit_detector_notches(tc_t &, double fs_hz,
                                                       const RTCLineAuditOptions &,
                                                       Eigen::Index diag_start_sample = 0,
                                                       Eigen::Index diag_n_samples = -1);

    // configure and apply a standard flag guard around filtered scan edges
    void configure_filter_edge_guard(double fs_hz);

    template <typename tc_t>
    void apply_filter_edge_guard(tc_t &, Eigen::Index start_sample, Eigen::Index n_samples,
                                 Eigen::Index guard_samples_override = -1);

    // optional az/el template subtraction on the RTC output chunk
    template <typename calib_t>
    void apply_altaz_destripe(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &);

    // optionally flag a network-wide window around aligned step-like events
    template <typename calib_t>
    void apply_network_step_mask(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &);

    // optionally flag a network-wide window around aligned impulsive coincidences
    template <typename calib_t>
    void apply_impulsive_coincidence_mask(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &);

    // append cached RTC diagnostics to a compact sidecar netcdf file
    template <typename calib_t>
    void append_diag_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, calib_t &,
                               Eigen::Index scan_row_index = -1);

    // write cached RTC diagnostic summaries into an existing netcdf file
    template <typename calib_t>
    void write_cached_diagnostics_to_netcdf(netCDF::NcFile &, TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
                                            calib_t &, Eigen::Index scan_row_index = -1);

    // clear cached RTC summaries for one scan after all output products are written
    void clear_cached_diagnostics(Eigen::Index scan_id);

    // append time chunk to tod netcdf file
    template <typename calib_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, calib_t &, bool apply_det_offsets = false,
                          Eigen::Index scan_row_index = -1);

    // append loaded outer RTC time chunk to tod netcdf file
    template <typename calib_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::RTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
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
                "raw_time_chunk.flagging.network_step_mask enabled: step_window_sec={:.4g} step_score_thresh={:.4g} min_good_frac={:.4f} min_det_used={} min_step_det_frac={:.4f} min_alignment_frac={:.4f} cluster_tol_sec={:.4g} mask_half_width_sec={:.4g} max_flagged_fraction={:.4f}",
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
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","snippet_pre_window_sec"})) {
            get_config_value(config, impulsive_capture.snippet_pre_window_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","snippet_pre_window_sec"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","snippet_post_window_sec"})) {
            get_config_value(config, impulsive_capture.snippet_post_window_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_capture","snippet_post_window_sec"},
                             {}, {0.0});
        }
        if (impulsive_capture.enabled) {
            logger->info(
                "raw_time_chunk.flagging.impulsive_capture enabled: min_good_frac={} min_event_z={} near_event_z={} max_events_per_network={} snippet_pre_window_sec={} snippet_post_window_sec={}",
                impulsive_capture.min_good_frac,
                impulsive_capture.min_event_z,
                impulsive_capture.near_event_z,
                impulsive_capture.max_events_per_network,
                impulsive_capture.snippet_pre_window_sec,
                impulsive_capture.snippet_post_window_sec);
        }
    }
    impulsive_coincidence = {};
    if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence"})) {
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","enabled"})) {
            get_config_value(config, impulsive_coincidence.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","enabled"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_good_frac"})) {
            get_config_value(config, impulsive_coincidence.min_good_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_good_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","event_score_thresh"})) {
            get_config_value(config, impulsive_coincidence.event_score_thresh, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","event_score_thresh"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_det_used"})) {
            get_config_value(config, impulsive_coincidence.min_det_used, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_det_used"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_impulsive_det_frac"})) {
            get_config_value(config, impulsive_coincidence.min_impulsive_det_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_impulsive_det_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_alignment_frac"})) {
            get_config_value(config, impulsive_coincidence.min_alignment_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_alignment_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_networks_aligned"})) {
            get_config_value(config, impulsive_coincidence.min_networks_aligned, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","min_networks_aligned"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","high_score_override_thresh"})) {
            get_config_value(config, impulsive_coincidence.high_score_override_thresh, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","high_score_override_thresh"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","high_score_min_networks_aligned"})) {
            get_config_value(config, impulsive_coincidence.high_score_min_networks_aligned, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","high_score_min_networks_aligned"},
                             {}, {0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","cluster_tol_sec"})) {
            get_config_value(config, impulsive_coincidence.cluster_tol_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","cluster_tol_sec"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","mask_pre_window_sec"})) {
            get_config_value(config, impulsive_coincidence.mask_pre_window_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","mask_pre_window_sec"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","mask_post_window_sec"})) {
            get_config_value(config, impulsive_coincidence.mask_post_window_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","mask_post_window_sec"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","max_flagged_fraction"})) {
            get_config_value(config, impulsive_coincidence.max_flagged_fraction, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","flagging","impulsive_coincidence","max_flagged_fraction"},
                             {}, {0.0}, {1.0});
        }
        logger->info(
            "raw_time_chunk.flagging.impulsive_coincidence configured: enabled={} min_good_frac={:.4f} event_score_thresh={:.4g} min_det_used={} min_impulsive_det_frac={:.4f} min_alignment_frac={:.4f} min_networks_aligned={} high_score_override_thresh={:.4g} high_score_min_networks_aligned={} cluster_tol_sec={:.4g} mask_pre_window_sec={:.4g} mask_post_window_sec={:.4g} max_flagged_fraction={:.4f}",
            impulsive_coincidence.enabled,
            impulsive_coincidence.min_good_frac,
            impulsive_coincidence.event_score_thresh,
            impulsive_coincidence.min_det_used,
            impulsive_coincidence.min_impulsive_det_frac,
            impulsive_coincidence.min_alignment_frac,
            impulsive_coincidence.min_networks_aligned,
            impulsive_coincidence.high_score_override_thresh,
            impulsive_coincidence.high_score_min_networks_aligned,
            impulsive_coincidence.cluster_tol_sec,
            impulsive_coincidence.mask_pre_window_sec,
            impulsive_coincidence.mask_post_window_sec,
            impulsive_coincidence.max_flagged_fraction);
    }

    line_audit = {};
    if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit"})) {
        get_config_value(config, line_audit.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","line_audit","enabled"});
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","line_min_hz"})) {
            get_config_value(config, line_audit.line_min_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","line_min_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","line_max_hz"})) {
            get_config_value(config, line_audit.line_max_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","line_max_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","segment_sec"})) {
            get_config_value(config, line_audit.segment_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","segment_sec"},
                             {}, {0.1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","min_segment_sec"})) {
            get_config_value(config, line_audit.min_segment_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","min_segment_sec"},
                             {}, {0.1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","overlap_frac"})) {
            get_config_value(config, line_audit.overlap_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","overlap_frac"},
                             {}, {0.0}, {0.95});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","continuum_radius_bins"})) {
            get_config_value(config, line_audit.continuum_radius_bins, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","continuum_radius_bins"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","prominence_thresh"})) {
            get_config_value(config, line_audit.prominence_thresh, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","prominence_thresh"},
                             {}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","cm_prominence_thresh"})) {
            get_config_value(config, line_audit.cm_prominence_thresh, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","cm_prominence_thresh"},
                             {}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","min_good_frac"})) {
            get_config_value(config, line_audit.min_good_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","min_good_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","min_windows"})) {
            get_config_value(config, line_audit.min_windows, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","min_windows"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","max_peaks_per_detector"})) {
            get_config_value(config, line_audit.max_peaks_per_detector, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","max_peaks_per_detector"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","max_det"})) {
            get_config_value(config, line_audit.max_det, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","max_det"},
                             {}, {0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","min_det_for_network"})) {
            get_config_value(config, line_audit.min_det_for_network, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","min_det_for_network"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","cluster_tol_hz"})) {
            get_config_value(config, line_audit.cluster_tol_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","cluster_tol_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","notch_min_detector_frac"})) {
            get_config_value(config, line_audit.notch_min_detector_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","notch_min_detector_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","notch_min_detectors"})) {
            get_config_value(config, line_audit.notch_min_detectors, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","notch_min_detectors"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","notch_min_cm_prominence"})) {
            get_config_value(config, line_audit.notch_min_cm_prominence, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","notch_min_cm_prominence"},
                             {}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_min_prominence"})) {
            get_config_value(config, line_audit.detector_min_prominence, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_min_prominence"},
                             {}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_min_line_power_frac"})) {
            get_config_value(config, line_audit.detector_min_line_power_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_min_line_power_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","bad_detector_max_cluster_frac"})) {
            get_config_value(config, line_audit.bad_detector_max_cluster_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","bad_detector_max_cluster_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","pre_filter_enabled"})) {
            get_config_value(config, line_audit.pre_filter_enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","pre_filter_enabled"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_enabled"})) {
            get_config_value(config, line_audit.post_filter_enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_enabled"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_apply_shared_notches"})) {
            get_config_value(config, line_audit.post_filter_apply_shared_notches, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_apply_shared_notches"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_apply_detector_notches"})) {
            get_config_value(config, line_audit.post_filter_apply_detector_notches, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_apply_detector_notches"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_apply_iterations"})) {
            get_config_value(config, line_audit.post_filter_apply_iterations, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_apply_iterations"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_line_min_hz"})) {
            get_config_value(config, line_audit.post_filter_line_min_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_line_min_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_line_max_hz"})) {
            get_config_value(config, line_audit.post_filter_line_max_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","post_filter_line_max_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","ptc_model_protected_enabled"})) {
            get_config_value(config, line_audit.ptc_model_protected_enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","ptc_model_protected_enabled"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","ptc_require_model_subtracted"})) {
            get_config_value(config, line_audit.ptc_require_model_subtracted, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","ptc_require_model_subtracted"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","ptc_apply_fixed_notches"})) {
            get_config_value(config, line_audit.ptc_apply_fixed_notches, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","ptc_apply_fixed_notches"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","ptc_apply_shared_notches"})) {
            get_config_value(config, line_audit.ptc_apply_shared_notches, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","ptc_apply_shared_notches"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","ptc_apply_detector_notches"})) {
            get_config_value(config, line_audit.ptc_apply_detector_notches, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","ptc_apply_detector_notches"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","ptc_apply_iterations"})) {
            get_config_value(config, line_audit.ptc_apply_iterations, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","ptc_apply_iterations"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","ptc_line_min_hz"})) {
            get_config_value(config, line_audit.ptc_line_min_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","ptc_line_min_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","ptc_line_max_hz"})) {
            get_config_value(config, line_audit.ptc_line_max_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","ptc_line_max_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","fixed_notch_enabled"})) {
            get_config_value(config, line_audit.fixed_notch_enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","fixed_notch_enabled"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","fixed_notch_freqs_hz"})) {
            line_audit.fixed_notch_freqs_hz = config.template get_typed<std::vector<double>>(
                std::tuple{"timestream","raw_time_chunk","line_audit","fixed_notch_freqs_hz"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","fixed_notch_widths_hz"})) {
            line_audit.fixed_notch_widths_hz = config.template get_typed<std::vector<double>>(
                std::tuple{"timestream","raw_time_chunk","line_audit","fixed_notch_widths_hz"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","fixed_notch_exclusion_half_width_hz"})) {
            get_config_value(config, line_audit.fixed_notch_exclusion_half_width_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","fixed_notch_exclusion_half_width_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_shared_notches"})) {
            get_config_value(config, line_audit.apply_shared_notches, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_shared_notches"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_min_support_networks"})) {
            get_config_value(config, line_audit.apply_min_support_networks, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_min_support_networks"},
                             {}, {1});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_min_detector_frac"})) {
            get_config_value(config, line_audit.apply_min_detector_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_min_detector_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_min_common_mode_prominence"})) {
            get_config_value(config, line_audit.apply_min_common_mode_prominence, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_min_common_mode_prominence"},
                             {}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_width_scale"})) {
            get_config_value(config, line_audit.apply_width_scale, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_width_scale"},
                             {}, {0.01});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_min_width_hz"})) {
            get_config_value(config, line_audit.apply_min_width_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_min_width_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_max_width_hz"})) {
            get_config_value(config, line_audit.apply_max_width_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_max_width_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_max_notches"})) {
            get_config_value(config, line_audit.apply_max_notches, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_max_notches"},
                             {}, {0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","apply_cluster_tol_hz"})) {
            get_config_value(config, line_audit.apply_cluster_tol_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","apply_cluster_tol_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_min_prominence"})) {
            get_config_value(config, line_audit.detector_notch_min_prominence, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_min_prominence"},
                             {}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_min_line_power_frac"})) {
            get_config_value(config, line_audit.detector_notch_min_line_power_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_min_line_power_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_max_notches"})) {
            get_config_value(config, line_audit.detector_notch_max_notches, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_max_notches"},
                             {}, {0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_width_scale"})) {
            get_config_value(config, line_audit.detector_notch_width_scale, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_width_scale"},
                             {}, {0.01});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_min_width_hz"})) {
            get_config_value(config, line_audit.detector_notch_min_width_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_min_width_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_max_width_hz"})) {
            get_config_value(config, line_audit.detector_notch_max_width_hz, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_max_width_hz"},
                             {}, {0.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_context_samples"})) {
            get_config_value(config, line_audit.detector_notch_context_samples, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","line_audit","detector_notch_context_samples"},
                             {}, {0});
        }
        if (line_audit.apply_max_width_hz < line_audit.apply_min_width_hz) {
            logger->error(
                "timestream.raw_time_chunk.line_audit.apply_max_width_hz ({}) must be >= apply_min_width_hz ({})",
                line_audit.apply_max_width_hz,
                line_audit.apply_min_width_hz);
            std::exit(EXIT_FAILURE);
        }
        if (std::isfinite(line_audit.ptc_line_min_hz) &&
            std::isfinite(line_audit.ptc_line_max_hz) &&
            line_audit.ptc_line_max_hz < line_audit.ptc_line_min_hz) {
            logger->error(
                "timestream.raw_time_chunk.line_audit.ptc_line_max_hz ({}) must be >= ptc_line_min_hz ({})",
                line_audit.ptc_line_max_hz,
                line_audit.ptc_line_min_hz);
            std::exit(EXIT_FAILURE);
        }
        if (line_audit.detector_notch_max_width_hz < line_audit.detector_notch_min_width_hz) {
            logger->error(
                "timestream.raw_time_chunk.line_audit.detector_notch_max_width_hz ({}) must be >= detector_notch_min_width_hz ({})",
                line_audit.detector_notch_max_width_hz,
                line_audit.detector_notch_min_width_hz);
            std::exit(EXIT_FAILURE);
        }
        if (line_audit.fixed_notch_widths_hz.empty()) {
            line_audit.fixed_notch_widths_hz.push_back(0.25);
        }
        if (line_audit.fixed_notch_enabled) {
            if (line_audit.fixed_notch_freqs_hz.empty()) {
                logger->error(
                    "timestream.raw_time_chunk.line_audit.fixed_notch_enabled is true but fixed_notch_freqs_hz is empty");
                std::exit(EXIT_FAILURE);
            }
            if (line_audit.fixed_notch_widths_hz.size() == 1 &&
                line_audit.fixed_notch_freqs_hz.size() > 1) {
                line_audit.fixed_notch_widths_hz.resize(
                    line_audit.fixed_notch_freqs_hz.size(),
                    line_audit.fixed_notch_widths_hz.front());
            }
            if (line_audit.fixed_notch_widths_hz.size() != line_audit.fixed_notch_freqs_hz.size()) {
                logger->error(
                    "timestream.raw_time_chunk.line_audit.fixed_notch_widths_hz must have length 1 or match fixed_notch_freqs_hz");
                std::exit(EXIT_FAILURE);
            }
            for (std::size_t i = 0; i < line_audit.fixed_notch_freqs_hz.size(); ++i) {
                if (!std::isfinite(line_audit.fixed_notch_freqs_hz[i]) ||
                    line_audit.fixed_notch_freqs_hz[i] <= 0.0 ||
                    !std::isfinite(line_audit.fixed_notch_widths_hz[i]) ||
                    line_audit.fixed_notch_widths_hz[i] <= 0.0) {
                    logger->error(
                        "timestream.raw_time_chunk.line_audit fixed notch frequencies and widths must be finite and > 0");
                    std::exit(EXIT_FAILURE);
                }
            }
        }
        logger->info(
            "raw_time_chunk.line_audit configured: enabled={} line_min_hz={} line_max_hz={} segment_sec={} min_segment_sec={} overlap_frac={} continuum_radius_bins={} prominence_thresh={} cm_prominence_thresh={} min_good_frac={} min_windows={} max_peaks_per_detector={} max_det={} min_det_for_network={} cluster_tol_hz={} notch_min_detector_frac={} notch_min_detectors={} notch_min_cm_prominence={} detector_min_prominence={} detector_min_line_power_frac={} bad_detector_max_cluster_frac={} pre_filter_enabled={} post_filter_enabled={} post_filter_apply_shared_notches={} post_filter_apply_detector_notches={} post_filter_apply_iterations={} post_filter_line_min_hz={} post_filter_line_max_hz={} ptc_model_protected_enabled={} ptc_require_model_subtracted={} ptc_apply_fixed_notches={} ptc_apply_shared_notches={} ptc_apply_detector_notches={} ptc_apply_iterations={} ptc_line_min_hz={} ptc_line_max_hz={} fixed_notch_enabled={} fixed_notch_count={} fixed_notch_exclusion_half_width_hz={} apply_shared_notches={} apply_min_support_networks={} apply_min_detector_frac={} apply_min_common_mode_prominence={} apply_width_scale={} apply_min_width_hz={} apply_max_width_hz={} apply_max_notches={} apply_cluster_tol_hz={} detector_notch_min_prominence={} detector_notch_min_line_power_frac={} detector_notch_max_notches={} detector_notch_width_scale={} detector_notch_min_width_hz={} detector_notch_max_width_hz={} detector_notch_context_samples={}",
            line_audit.enabled,
            line_audit.line_min_hz,
            line_audit.line_max_hz,
            line_audit.segment_sec,
            line_audit.min_segment_sec,
            line_audit.overlap_frac,
            line_audit.continuum_radius_bins,
            line_audit.prominence_thresh,
            line_audit.cm_prominence_thresh,
            line_audit.min_good_frac,
            line_audit.min_windows,
            line_audit.max_peaks_per_detector,
            line_audit.max_det,
            line_audit.min_det_for_network,
            line_audit.cluster_tol_hz,
            line_audit.notch_min_detector_frac,
            line_audit.notch_min_detectors,
            line_audit.notch_min_cm_prominence,
            line_audit.detector_min_prominence,
            line_audit.detector_min_line_power_frac,
            line_audit.bad_detector_max_cluster_frac,
            line_audit.pre_filter_enabled,
            line_audit.post_filter_enabled,
            line_audit.post_filter_apply_shared_notches,
            line_audit.post_filter_apply_detector_notches,
            line_audit.post_filter_apply_iterations,
            line_audit.post_filter_line_min_hz,
            line_audit.post_filter_line_max_hz,
            line_audit.ptc_model_protected_enabled,
            line_audit.ptc_require_model_subtracted,
            line_audit.ptc_apply_fixed_notches,
            line_audit.ptc_apply_shared_notches,
            line_audit.ptc_apply_detector_notches,
            line_audit.ptc_apply_iterations,
            line_audit.ptc_line_min_hz,
            line_audit.ptc_line_max_hz,
            line_audit.fixed_notch_enabled,
            line_audit.fixed_notch_freqs_hz.size(),
            line_audit.fixed_notch_exclusion_half_width_hz,
            line_audit.apply_shared_notches,
            line_audit.apply_min_support_networks,
            line_audit.apply_min_detector_frac,
            line_audit.apply_min_common_mode_prominence,
            line_audit.apply_width_scale,
            line_audit.apply_min_width_hz,
            line_audit.apply_max_width_hz,
            line_audit.apply_max_notches,
            line_audit.apply_cluster_tol_hz,
            line_audit.detector_notch_min_prominence,
            line_audit.detector_notch_min_line_power_frac,
            line_audit.detector_notch_max_notches,
            line_audit.detector_notch_width_scale,
            line_audit.detector_notch_min_width_hz,
            line_audit.detector_notch_max_width_hz,
            line_audit.detector_notch_context_samples);
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
                std::string img_ext_name = config.get_str(std::tuple{"timestream","raw_time_chunk","kernel", "image_ext_names",
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
        despiker.run_legacy = true;
        if (config.has(std::tuple{"timestream","raw_time_chunk","despike","legacy"})) {
            get_config_value(config, despiker.run_legacy, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","despike","legacy","enabled"});
        }

        despike_source_protection_config_enabled = true;
        despiker.source_protection_enabled = false;
        despiker.source_protection_radius_arcsec = 20.0;
        if (config.has(std::tuple{"timestream","raw_time_chunk","despike","source_protection"})) {
            get_config_value(config, despike_source_protection_config_enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","despike","source_protection","enabled"});
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","source_protection","radius_arcsec"})) {
                get_config_value(config, despiker.source_protection_radius_arcsec, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","source_protection","radius_arcsec"},
                                 {}, {0.0});
            }
        }

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
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","expand_with_filter"})) {
                get_config_value(config, despiker.local_residual.expand_with_filter, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","expand_with_filter"});
            }
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","event_padding_sec"})) {
                get_config_value(config, despiker.local_residual.event_padding_sec, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","event_padding_sec"},
                                 {}, {0.0});
            }
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","high_score_event_override"})) {
                get_config_value(config, despiker.local_residual.high_score_event_override, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","high_score_event_override"},
                                 {}, {0.0});
            }
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","max_added_flagged_fraction"})) {
                get_config_value(config, despiker.local_residual.max_added_flagged_fraction, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","max_added_flagged_fraction"},
                                 {}, {0.0}, {1.0});
            }
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate"})) {
                get_config_value(config, despiker.local_residual.compact_raw_gate.enabled, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","enabled"});
                const bool has_candidate_rel_sigma_scale =
                    config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","candidate_rel_sigma_scale"});
                const bool has_legacy_candidate_sigma_scale =
                    config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","candidate_sigma_scale"});
                if (has_candidate_rel_sigma_scale) {
                    get_config_value(config, despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale,
                                     missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","candidate_rel_sigma_scale"},
                                     {}, {0.0});
                    if (has_legacy_candidate_sigma_scale) {
                        logger->warn(
                            "raw_time_chunk.despike.local_residual.compact_raw_gate.candidate_sigma_scale is deprecated; using candidate_rel_sigma_scale={}",
                            despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale);
                    }
                }
                else if (has_legacy_candidate_sigma_scale) {
                    double legacy_candidate_sigma_scale = despiker.local_residual.sigma_scale;
                    get_config_value(config, legacy_candidate_sigma_scale, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","candidate_sigma_scale"},
                                     {}, {0.0});
                    despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale =
                        legacy_candidate_sigma_scale / despiker.local_residual.sigma_scale;
                    logger->warn(
                        "raw_time_chunk.despike.local_residual.compact_raw_gate.candidate_sigma_scale is deprecated; interpreting legacy value {:.4g} as candidate_rel_sigma_scale={:.4g} using sigma_scale={:.4g}",
                        legacy_candidate_sigma_scale,
                        despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale,
                        despiker.local_residual.sigma_scale);
                }
                if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","window_sec"})) {
                    get_config_value(config, despiker.local_residual.compact_raw_gate.window_sec, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","window_sec"},
                                     {}, {0.0});
                }
                if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","half_peak_frac"})) {
                    get_config_value(config, despiker.local_residual.compact_raw_gate.half_peak_frac, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","half_peak_frac"},
                                     {}, {0.0}, {1.0});
                }
                if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","max_width_sec"})) {
                    get_config_value(config, despiker.local_residual.compact_raw_gate.max_width_sec, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","max_width_sec"},
                                     {}, {0.0});
                }
                if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","max_step_shift_z"})) {
                    get_config_value(config, despiker.local_residual.compact_raw_gate.max_step_shift_z, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_raw_gate","max_step_shift_z"},
                                     {}, {0.0});
                }
            }
            if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate"})) {
                get_config_value(config, despiker.local_residual.compact_delta_gate.enabled, missing_keys, invalid_keys,
                                 std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","enabled"});
                if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","window_sec"})) {
                    get_config_value(config, despiker.local_residual.compact_delta_gate.window_sec, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","window_sec"},
                                     {}, {0.0});
                }
                if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","half_peak_frac"})) {
                    get_config_value(config, despiker.local_residual.compact_delta_gate.half_peak_frac, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","half_peak_frac"},
                                     {}, {0.0}, {1.0});
                }
                if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","max_width_sec"})) {
                    get_config_value(config, despiker.local_residual.compact_delta_gate.max_width_sec, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","max_width_sec"},
                                     {}, {0.0});
                }
                if (config.has(std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","max_step_shift_z"})) {
                    get_config_value(config, despiker.local_residual.compact_delta_gate.max_step_shift_z, missing_keys, invalid_keys,
                                     std::tuple{"timestream","raw_time_chunk","despike","local_residual","compact_delta_gate","max_step_shift_z"},
                                     {}, {0.0});
                }
            }
        }
        if (despiker.local_residual.enabled) {
            logger->info(
                "raw_time_chunk.despike.local_residual enabled: legacy_enabled={} window_sec={:.4g} sigma_scale={:.4g} delta_sigma_scale={:.4g} expand_with_filter={} event_padding_sec={:.4g} high_score_event_override={:.4g} max_added_flagged_fraction={:.4f} compact_raw_gate(enabled={} candidate_rel_sigma_scale={:.4g} candidate_sigma_scale_eff={:.4g} window_sec={:.4g} half_peak_frac={:.4f} max_width_sec={:.4g} max_step_shift_z={:.4g}) compact_delta_gate(enabled={} window_sec={:.4g} half_peak_frac={:.4f} max_width_sec={:.4g} max_step_shift_z={:.4g})",
                despiker.run_legacy,
                despiker.local_residual.window_sec,
                despiker.local_residual.sigma_scale,
                despiker.local_residual.delta_sigma_scale,
                despiker.local_residual.expand_with_filter,
                despiker.local_residual.event_padding_sec,
                despiker.local_residual.high_score_event_override,
                despiker.local_residual.max_added_flagged_fraction,
                despiker.local_residual.compact_raw_gate.enabled,
                despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale,
                despiker.local_residual.compact_raw_gate.candidate_rel_sigma_scale *
                    despiker.local_residual.sigma_scale,
                despiker.local_residual.compact_raw_gate.window_sec,
                despiker.local_residual.compact_raw_gate.half_peak_frac,
                despiker.local_residual.compact_raw_gate.max_width_sec,
                despiker.local_residual.compact_raw_gate.max_step_shift_z,
                despiker.local_residual.compact_delta_gate.enabled,
                despiker.local_residual.compact_delta_gate.window_sec,
                despiker.local_residual.compact_delta_gate.half_peak_frac,
                despiker.local_residual.compact_delta_gate.max_width_sec,
                despiker.local_residual.compact_delta_gate.max_step_shift_z);
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

    filter_edge_guard = {};
    if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard"})) {
        get_config_value(config, filter_edge_guard.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","raw_time_chunk","filter","edge_guard","enabled"});
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","mode"})) {
            get_config_value(config, filter_edge_guard.mode, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","mode"},
                             {"flag","none"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","combine"})) {
            get_config_value(config, filter_edge_guard.combine, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","combine"},
                             {"sum","max"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","min_samples"})) {
            get_config_value(config, filter_edge_guard.min_samples, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","min_samples"},
                             {}, {0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","extra_samples"})) {
            get_config_value(config, filter_edge_guard.extra_samples, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","extra_samples"},
                             {}, {0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","max_samples"})) {
            get_config_value(config, filter_edge_guard.max_samples, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","max_samples"},
                             {}, {0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","iir_settle_attenuation"})) {
            get_config_value(config, filter_edge_guard.iir_settle_attenuation, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","iir_settle_attenuation"},
                             {}, {0.0}, {1.0});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_fir"})) {
            get_config_value(config, filter_edge_guard.apply_fir, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_fir"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_notch"})) {
            get_config_value(config, filter_edge_guard.apply_notch, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_notch"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_dynamic_notch"})) {
            get_config_value(config, filter_edge_guard.apply_dynamic_notch, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_dynamic_notch"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_iir_highpass"})) {
            get_config_value(config, filter_edge_guard.apply_iir_highpass, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_iir_highpass"});
        }
        if (config.has(std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_downsample"})) {
            get_config_value(config, filter_edge_guard.apply_downsample, missing_keys, invalid_keys,
                             std::tuple{"timestream","raw_time_chunk","filter","edge_guard","apply_downsample"});
        }
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

inline void RTCProc::configure_filter_edge_guard(double fs_hz) {
    auto combine_samples = [&](Eigen::Index current, Eigen::Index next) {
        if (next <= 0) {
            return current;
        }
        if (citlali::config::is_max_raw_filter_edge_guard_combine(
                filter_edge_guard.combine)) {
            return std::max(current, next);
        }
        return current + next;
    };

    Eigen::Index base_context = 0;
    if (run_tod_filter) {
        base_context += std::max<Eigen::Index>(0, filter.n_terms);
    }
    if (run_tod_iir_highpass) {
        base_context += filter.iir_highpass_settle_samples(fs_hz);
    }

    filter_edge_guard.context_samples = base_context;
    filter_edge_guard.guard_samples = 0;
    if (!filter_edge_guard.enabled ||
        citlali::config::is_none_raw_filter_edge_guard_mode(
            filter_edge_guard.mode)) {
        return;
    }

    Eigen::Index guard = 0;
    if (run_tod_filter && filter_edge_guard.apply_fir) {
        guard = combine_samples(guard, std::max<Eigen::Index>(0, filter.n_terms));
    }
    if (run_tod_filter && run_tod_notch && filter_edge_guard.apply_notch) {
        guard = combine_samples(guard, filter.notch_settle_samples(fs_hz, filter_edge_guard.iir_settle_attenuation));
    }
    if (line_audit.enabled && line_audit.pre_filter_enabled &&
        line_audit.fixed_notch_enabled && filter_edge_guard.apply_dynamic_notch) {
        double min_fixed_width_hz = std::numeric_limits<double>::quiet_NaN();
        const Eigen::Index n_fixed_sections =
            count_rtc_line_audit_fixed_notches(fs_hz, line_audit, &min_fixed_width_hz);
        if (n_fixed_sections > 0 &&
            std::isfinite(min_fixed_width_hz) &&
            min_fixed_width_hz > 0.0) {
            guard = combine_samples(
                guard,
                n_fixed_sections *
                    timestream::Filter::notch_settle_samples_for_width(
                        fs_hz, min_fixed_width_hz, filter_edge_guard.iir_settle_attenuation));
        }
    }
    if (line_audit.enabled && line_audit.apply_shared_notches && filter_edge_guard.apply_dynamic_notch) {
        const Eigen::Index n_dynamic_sections =
            line_audit.apply_max_notches > 0 ? line_audit.apply_max_notches : 1;
        guard = combine_samples(
            guard,
            n_dynamic_sections *
                timestream::Filter::notch_settle_samples_for_width(
                    fs_hz, line_audit.apply_min_width_hz, filter_edge_guard.iir_settle_attenuation));
    }
    if (run_tod_iir_highpass && filter_edge_guard.apply_iir_highpass) {
        guard = combine_samples(guard, filter.iir_highpass_settle_samples(fs_hz));
    }
    if (run_downsample && filter_edge_guard.apply_downsample && downsampler.factor > 1) {
        guard = combine_samples(guard, static_cast<Eigen::Index>(downsampler.factor - 1));
    }

    guard = std::max(guard, filter_edge_guard.min_samples);
    guard += filter_edge_guard.extra_samples;
    if (filter_edge_guard.max_samples > 0) {
        guard = std::min(guard, filter_edge_guard.max_samples);
    }
    guard = std::max<Eigen::Index>(0, guard);

    filter_edge_guard.guard_samples = guard;
    filter_edge_guard.context_samples = std::max(base_context, guard);
}

template <typename tc_t>
void RTCProc::apply_filter_edge_guard(tc_t &in,
                                      Eigen::Index start_sample,
                                      Eigen::Index n_samples,
                                      Eigen::Index guard_samples_override) {
    const bool prev_guarded = in.status.filter_edge_guarded;
    const int prev_pre_samples = in.status.filter_edge_guard_pre_samples;
    const int prev_post_samples = in.status.filter_edge_guard_post_samples;
    const int prev_flagged_samples = in.status.filter_edge_guard_flagged_samples;
    const double prev_flagged_frac = in.status.filter_edge_guard_flagged_frac;

    const Eigen::Index guard_samples =
        guard_samples_override >= 0 ? guard_samples_override : filter_edge_guard.guard_samples;

    if (!filter_edge_guard.enabled ||
        !citlali::config::is_flag_raw_filter_edge_guard_mode(
            filter_edge_guard.mode) ||
        guard_samples <= 0 || n_samples <= 0 ||
        in.flags.data.rows() <= 0 || in.flags.data.cols() <= 0) {
        return;
    }

    start_sample = std::max<Eigen::Index>(0, start_sample);
    if (start_sample >= in.flags.data.rows()) {
        return;
    }
    n_samples = std::min<Eigen::Index>(n_samples, in.flags.data.rows() - start_sample);
    if (n_samples <= 0) {
        return;
    }

    const Eigen::Index pre = std::min(guard_samples, n_samples);
    const Eigen::Index post = std::min(guard_samples, n_samples - pre);
    if (pre > 0) {
        in.flags.data.block(start_sample, 0, pre, in.flags.data.cols()).setConstant(true);
    }
    if (post > 0) {
        in.flags.data.block(start_sample + n_samples - post, 0, post, in.flags.data.cols()).setConstant(true);
    }

    const Eigen::Index guarded_rows = pre + post;
    in.status.filter_edge_guarded = prev_guarded || guarded_rows > 0;
    in.status.filter_edge_guard_pre_samples =
        std::max(prev_pre_samples, static_cast<int>(pre));
    in.status.filter_edge_guard_post_samples =
        std::max(prev_post_samples, static_cast<int>(post));
    in.status.filter_edge_guard_flagged_samples =
        std::max(prev_flagged_samples, static_cast<int>(guarded_rows * in.flags.data.cols()));
    const double flagged_frac =
        static_cast<double>(guarded_rows) / static_cast<double>(n_samples);
    in.status.filter_edge_guard_flagged_frac =
        std::isfinite(prev_flagged_frac) ? std::max(prev_flagged_frac, flagged_frac) : flagged_frac;
}

inline double RTCProc::rtc_line_audit_fixed_notch_width_hz(const RTCLineAuditOptions &audit,
                                                           std::size_t i) const {
    if (audit.fixed_notch_widths_hz.empty()) {
        return 0.25;
    }
    if (i < audit.fixed_notch_widths_hz.size()) {
        return audit.fixed_notch_widths_hz[i];
    }
    return audit.fixed_notch_widths_hz.back();
}

inline bool RTCProc::rtc_line_audit_frequency_excluded_by_fixed_notch(
    double freq_hz,
    const RTCLineAuditOptions &audit) const {
    if (!audit.fixed_notch_enabled || !std::isfinite(freq_hz) || freq_hz <= 0.0) {
        return false;
    }
    for (std::size_t i = 0; i < audit.fixed_notch_freqs_hz.size(); ++i) {
        const double center_hz = audit.fixed_notch_freqs_hz[i];
        const double width_hz = rtc_line_audit_fixed_notch_width_hz(audit, i);
        if (!std::isfinite(center_hz) || center_hz <= 0.0 ||
            !std::isfinite(width_hz) || width_hz <= 0.0) {
            continue;
        }
        double half_width_hz = 0.5 * width_hz;
        if (std::isfinite(audit.fixed_notch_exclusion_half_width_hz) &&
            audit.fixed_notch_exclusion_half_width_hz > 0.0) {
            half_width_hz =
                std::max(half_width_hz, audit.fixed_notch_exclusion_half_width_hz);
        }
        if (std::abs(freq_hz - center_hz) <= half_width_hz) {
            return true;
        }
    }
    return false;
}

inline Eigen::Index RTCProc::count_rtc_line_audit_fixed_notches(
    double fs_hz,
    const RTCLineAuditOptions &audit,
    double *min_width_hz) const {
    if (min_width_hz != nullptr) {
        *min_width_hz = std::numeric_limits<double>::quiet_NaN();
    }
    if (!audit.fixed_notch_enabled || !std::isfinite(fs_hz) || fs_hz <= 0.0) {
        return 0;
    }
    const double nyquist_hz = 0.5 * fs_hz;
    Eigen::Index count = 0;
    double min_width = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < audit.fixed_notch_freqs_hz.size(); ++i) {
        const double freq_hz = audit.fixed_notch_freqs_hz[i];
        const double width_hz = rtc_line_audit_fixed_notch_width_hz(audit, i);
        if (!std::isfinite(freq_hz) || freq_hz <= 0.0 || freq_hz >= nyquist_hz ||
            !std::isfinite(width_hz) || width_hz <= 0.0) {
            continue;
        }
        ++count;
        min_width = std::min(min_width, width_hz);
    }
    if (min_width_hz != nullptr && count > 0) {
        *min_width_hz = min_width;
    }
    return count;
}

template <typename tc_t>
Eigen::Index RTCProc::apply_rtc_line_audit_fixed_notches(
    tc_t &in,
    double fs_hz,
    const RTCLineAuditOptions &audit) {
    if (!audit.enabled || !audit.fixed_notch_enabled ||
        !std::isfinite(fs_hz) || fs_hz <= 0.0) {
        return 0;
    }

    const double nyquist_hz = 0.5 * fs_hz;
    Filter fixed_notch_filter;
    fixed_notch_filter.notch_zero_phase = true;

    struct FixedAppliedNotch {
        double freq_hz = std::numeric_limits<double>::quiet_NaN();
        double width_hz = std::numeric_limits<double>::quiet_NaN();
    };
    std::vector<FixedAppliedNotch> applied_notches;
    applied_notches.reserve(audit.fixed_notch_freqs_hz.size());

    for (std::size_t i = 0; i < audit.fixed_notch_freqs_hz.size(); ++i) {
        const double freq_hz = audit.fixed_notch_freqs_hz[i];
        const double width_hz = rtc_line_audit_fixed_notch_width_hz(audit, i);
        if (!std::isfinite(freq_hz) || freq_hz <= 0.0 ||
            !std::isfinite(width_hz) || width_hz <= 0.0) {
            continue;
        }
        if (freq_hz >= nyquist_hz) {
            logger->warn(
                "rtc_line_audit fixed_notch scan {} skipped freq_hz={:.4f} at/above Nyquist {:.4f}",
                in.index.data + 1,
                freq_hz,
                nyquist_hz);
            continue;
        }
        fixed_notch_filter.w0s.push_back(freq_hz);
        fixed_notch_filter.qs.push_back(freq_hz / width_hz);
        applied_notches.push_back({freq_hz, width_hz});
    }

    if (applied_notches.empty()) {
        return 0;
    }

    fixed_notch_filter.make_notch_filter(fs_hz);
    fixed_notch_filter.iir(in.scans.data);
    if (run_kernel) {
        fixed_notch_filter.iir(in.kernel.data);
    }

    for (const auto &notch : applied_notches) {
        logger->info(
            "rtc_line_audit apply_fixed_notch scan {}: center_hz={:.4f} width_hz={:.4f} zero_phase=true",
            in.index.data + 1,
            notch.freq_hz,
            notch.width_hz);
    }

    return static_cast<Eigen::Index>(applied_notches.size());
}

template <class calib_t>
auto RTCProc::calc_map_indices(calib_t &calib, std::string map_grouping) {
    // indices for maps
    Eigen::VectorXI indices(calib.n_dets), map_indices(calib.n_dets);

    // overwrite map indices for networks
    if (citlali::config::is_network_map_grouping(map_grouping)) {
        indices = calib.apt["nw"].template cast<Eigen::Index> ();
    }
    // overwrite map indices for arrays
    else if (citlali::config::is_array_map_grouping(map_grouping)) {
        indices = calib.apt["array"].template cast<Eigen::Index> ();
    }
    // overwrite map indices for detectors
    else if (citlali::config::is_detector_map_grouping(map_grouping)) {
        indices = Eigen::VectorXI::LinSpaced(calib.n_dets,0,calib.n_dets-1);
    }
    // overwrite map indices for fg
    else if (citlali::config::is_frequency_group_map_grouping(map_grouping)) {
        indices = calib.apt["fg"].template cast<Eigen::Index> ();
    }
    // start at 0
    if (!citlali::config::is_frequency_group_map_grouping(map_grouping)) {
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
                  calib_t &calib, telescope_t &telescope, double pixel_size_rad, std::string map_grouping,
                  TCData<TCDataKind::RTC, Eigen::MatrixXd> *tod_outer_output) {

    // number of points in scan
    Eigen::Index n_pts = in.scans.data.rows();

    // start index of the science scan inside the loaded outer filter context
    auto si = in.scan_indices.data(0) - in.scan_indices.data(2);
    si = std::max<Eigen::Index>(0, si);
    // end index of inner scans
    auto sl = in.scan_indices.data(1) - in.scan_indices.data(0) + 1;
    sl = std::max<Eigen::Index>(0, std::min<Eigen::Index>(sl, in.scans.data.rows() - si));

    // calculate the polarization angle
    if (run_polarization) {
        polarization.calc_angle(in, calib);
    }

    // resize fcf
    in.fcf.data.setOnes(in.scans.data.cols());

    // get indices for maps
    logger->debug("calculating map indices");
    auto map_indices = calc_map_indices(calib, map_grouping);
    auto despiker_local = despiker;
    RTCSourceProtectionDiagSummary despike_source_summary;

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
        log_kernel_matrix_diag(logger, "rtc after kernel create", in.kernel.data, in.index.data);
    }

    // run despiking
    if (run_despike) {
        logger->debug("despiking");
        despike_source_summary.enabled = despiker_local.source_protection_enabled;
        despike_source_summary.radius_arcsec =
            despiker_local.source_protection_radius_arcsec;
        if (despiker_local.source_protection_enabled) {
            auto [source_mask, source_info] = engine_utils::calc_source_protection_mask(
                in, calib.apt, telescope.pixel_axes, map_grouping,
                "map_center_radius", despiker_local.source_protection_radius_arcsec);
            despike_source_summary.protected_samples =
                static_cast<int>(source_info.protected_samples);
            despike_source_summary.total_samples =
                static_cast<int>(source_mask.size());
            despiker_local.source_protection_mask = std::move(source_mask);
            despiker_local.last_source_protection_sample_count =
                source_info.protected_samples;
            logger->debug(
                "despike source protection scan={} mode={} radius_arcsec={:.4g} protected_samples={} detectors_with_source={}",
                in.index.data, source_info.mode, source_info.radius_arcsec,
                despiker_local.last_source_protection_sample_count,
                source_info.detectors_with_source);
        }
        else {
            despiker_local.clear_source_protection_mask();
        }
        // despike data
        despiker_local.despike(in.scans.data, in.flags.data, calib.apt);

        // we want to replace spikes on a per array or network basis
        auto grp_limits = get_grouping(despiker_local.grouping, calib, in.scans.data.cols());

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
            despiker_local.replace_spikes(in_scans, in_flags, calib.apt, start_index);
        }

        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            if (despike_source_summary.enabled) {
                rtc_source_protection_summary_by_scan[in.index.data] =
                    despike_source_summary;
            }
            else {
                rtc_source_protection_summary_by_scan.erase(in.index.data);
            }
        }

        in.status.despiked = true;
    }

    Eigen::Index n_applied_line_audit_notches = 0;
    if (line_audit.enabled && line_audit.pre_filter_enabled) {
        n_applied_line_audit_notches +=
            apply_rtc_line_audit_fixed_notches(in, telescope.fsmp, line_audit);
        capture_rtc_line_audit(in, calib, si, sl, line_audit, false);
        if (line_audit.apply_shared_notches) {
            n_applied_line_audit_notches +=
                apply_rtc_line_audit_shared_notches(in, telescope.fsmp, line_audit, false);
        }
        if (run_kernel && n_applied_line_audit_notches > 0) {
            log_kernel_matrix_diag(logger, "rtc after pre-filter line audit notches",
                                   in.kernel.data, in.index.data);
        }
    }

    bool ran_tod_filter_stage = false;

    if (n_applied_line_audit_notches > 0) {
        ran_tod_filter_stage = true;
    }

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
            log_kernel_matrix_diag(logger, "rtc after tod filter", in.kernel.data, in.index.data);
        }
        ran_tod_filter_stage = true;
    }

    if (run_tod_iir_highpass) {
        logger->debug("applying iir highpass filter to signal");
        filter.iir_highpass(in.scans.data, telescope.fsmp);

        if (run_kernel) {
            logger->debug("applying iir highpass filter to kernel");
            filter.iir_highpass(in.kernel.data, telescope.fsmp);
            log_kernel_matrix_diag(logger, "rtc after highpass filter", in.kernel.data, in.index.data);
        }
        ran_tod_filter_stage = true;
    }

    if (ran_tod_filter_stage) {
        in.status.tod_filtered = true;
    }

    apply_filter_edge_guard(in, si, sl);
    if (run_kernel) {
        log_kernel_matrix_diag(logger, "rtc after primary edge guard", in.kernel.data, in.index.data);
    }

    RTCLineAuditOptions post_line_audit = line_audit;
    post_line_audit.enabled = line_audit.enabled && line_audit.post_filter_enabled;
    post_line_audit.apply_shared_notches = line_audit.post_filter_apply_shared_notches;
    if (std::isfinite(line_audit.post_filter_line_min_hz)) {
        post_line_audit.line_min_hz = line_audit.post_filter_line_min_hz;
    }
    if (std::isfinite(line_audit.post_filter_line_max_hz)) {
        post_line_audit.line_max_hz = line_audit.post_filter_line_max_hz;
    }

    auto seed_rtc_detector_diag = [&](Eigen::Index scan_id, Eigen::Index n_dets) {
        std::vector<RTCDetectorDiagSummary> existing;
        std::vector<RTCDetectorDiagSummary> summary(
            static_cast<std::size_t>(n_dets), RTCDetectorDiagSummary{});
        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            const auto it = rtc_detector_summary_by_scan.find(scan_id);
            if (it != rtc_detector_summary_by_scan.end() &&
                it->second.size() == static_cast<std::size_t>(n_dets)) {
                existing = it->second;
            }
        }
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            auto &row = summary[static_cast<std::size_t>(det)];
            row.det = det;
            if (det < static_cast<Eigen::Index>(despiker_local.last_detector_diag.size())) {
                static_cast<DespikeDetectorDiagSummary &>(row) =
                    despiker_local.last_detector_diag[static_cast<std::size_t>(det)];
            }
            if (existing.size() == static_cast<std::size_t>(n_dets)) {
                const auto &old = existing[static_cast<std::size_t>(det)];
                row.detector_notch_n_applied = old.detector_notch_n_applied;
                row.detector_notch_primary_freq_hz = old.detector_notch_primary_freq_hz;
                row.detector_notch_primary_width_hz = old.detector_notch_primary_width_hz;
                row.detector_notch_primary_prominence = old.detector_notch_primary_prominence;
                row.detector_notch_primary_line_power_frac = old.detector_notch_primary_line_power_frac;
                row.detector_notch_rms_before = old.detector_notch_rms_before;
                row.detector_notch_rms_after = old.detector_notch_rms_after;
            }
        }
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_detector_summary_by_scan[scan_id] = std::move(summary);
    };

    if (post_line_audit.enabled && post_line_audit.post_filter_apply_detector_notches) {
        seed_rtc_detector_diag(in.index.data, in.scans.data.cols());
        const auto n_detector_notches =
            apply_rtc_line_audit_detector_notches(in, telescope.fsmp, post_line_audit, si, sl);
        if (n_detector_notches > 0) {
            in.status.tod_filtered = true;
            if (run_kernel) {
                log_kernel_matrix_diag(logger, "rtc after detector line audit notches",
                                       in.kernel.data, in.index.data);
            }
            Eigen::Index detector_guard_samples = 0;
            if (filter_edge_guard.apply_dynamic_notch) {
                const Eigen::Index guard_notch_count =
                    (post_line_audit.detector_notch_max_notches > 0)
                        ? std::min<Eigen::Index>(post_line_audit.detector_notch_max_notches,
                                                 n_detector_notches)
                        : n_detector_notches;
                detector_guard_samples =
                    guard_notch_count *
                    timestream::Filter::notch_settle_samples_for_width(
                        telescope.fsmp,
                        post_line_audit.detector_notch_min_width_hz,
                        filter_edge_guard.iir_settle_attenuation);
                detector_guard_samples =
                    std::max(detector_guard_samples, filter_edge_guard.min_samples);
                detector_guard_samples += filter_edge_guard.extra_samples;
                if (filter_edge_guard.max_samples > 0) {
                    detector_guard_samples =
                        std::min(detector_guard_samples, filter_edge_guard.max_samples);
                }
                detector_guard_samples = std::max<Eigen::Index>(0, detector_guard_samples);
            }
            if (detector_guard_samples > 0) {
                const Eigen::Index pre_context = std::max<Eigen::Index>(0, si);
                const Eigen::Index post_context =
                    std::max<Eigen::Index>(0, in.scans.data.rows() - (si + sl));
                const Eigen::Index missing_guard = std::max<Eigen::Index>(
                    std::max<Eigen::Index>(0, detector_guard_samples - pre_context),
                    std::max<Eigen::Index>(0, detector_guard_samples - post_context));
                if (missing_guard > 0) {
                    apply_filter_edge_guard(in, si, sl, missing_guard);
                }
            }
        }
    }

    if (tod_outer_output != nullptr) {
        *tod_outer_output = in;
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
            log_kernel_matrix_diag(logger, "rtc output inner after downsample", out.kernel.data, in.index.data);
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
            log_kernel_matrix_diag(logger, "rtc output inner copy", out.kernel.data, in.index.data);
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

    double post_filter_fs_hz = telescope.fsmp;
    if (run_downsample && downsampler.factor > 1) {
        post_filter_fs_hz /= static_cast<double>(downsampler.factor);
    }
    if (post_line_audit.enabled) {
        Eigen::Index n_post_notches = 0;
        const Eigen::Index post_apply_iterations =
            post_line_audit.apply_shared_notches
                ? std::max<Eigen::Index>(1, line_audit.post_filter_apply_iterations)
                : 1;
        for (Eigen::Index iter = 0; iter < post_apply_iterations; ++iter) {
            capture_rtc_line_audit(out, calib, 0, out.scans.data.rows(), post_line_audit, true);
            if (!post_line_audit.apply_shared_notches) {
                break;
            }
            const auto n_iter_notches =
                apply_rtc_line_audit_shared_notches(out, post_filter_fs_hz, post_line_audit, true);
            n_post_notches += n_iter_notches;
            if (n_iter_notches <= 0) {
                break;
            }
        }
        if (n_post_notches > 0) {
            out.status.tod_filtered = true;
            Eigen::Index post_guard_samples = 0;
            if (filter_edge_guard.apply_dynamic_notch) {
                post_guard_samples =
                    n_post_notches *
                    timestream::Filter::notch_settle_samples_for_width(
                        post_filter_fs_hz,
                        post_line_audit.apply_min_width_hz,
                        filter_edge_guard.iir_settle_attenuation);
                post_guard_samples = std::max(post_guard_samples, filter_edge_guard.min_samples);
                post_guard_samples += filter_edge_guard.extra_samples;
                if (filter_edge_guard.max_samples > 0) {
                    post_guard_samples = std::min(post_guard_samples, filter_edge_guard.max_samples);
                }
                post_guard_samples = std::max<Eigen::Index>(0, post_guard_samples);
            }
            if (post_guard_samples > 0) {
                apply_filter_edge_guard(out, 0, out.scans.data.rows(), post_guard_samples);
            }
        }
    }

    // Preserve per-detector despike summaries for the final RTC output write while
    // retaining detector-local notch diagnostics selected from the outer scan.
    seed_rtc_detector_diag(out.index.data, out.scans.data.cols());
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        if (!line_audit.enabled) {
            rtc_network_summary_by_scan.erase(out.index.data);
        }
        rtc_impulsive_summary_by_scan.erase(out.index.data);
    }

    if (network_step_mask.enabled || impulsive_coincidence.enabled) {
        capture_rtc_diagnostics(out, calib, true, true);
    }

    if (network_step_mask.enabled) {
        apply_network_step_mask(out, calib);
    }
    if (impulsive_coincidence.enabled) {
        apply_impulsive_coincidence_mask(out, calib);
    }

    apply_altaz_destripe(out, calib);

    if (network_step_mask.enabled || impulsive_coincidence.enabled) {
        capture_rtc_diagnostics(out, calib, false, false);
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

    logger->info("removed {} detectors flagged in APT table ({:.2f}%)",n_flagged,
                (static_cast<double>(n_flagged)/static_cast<double>(n_dets))*100.0);
}

template <typename calib_t>
void RTCProc::apply_altaz_destripe(TCData<TCDataKind::PTC, Eigen::MatrixXd> &out, calib_t &calib) {
    if (!altaz_destripe.enabled) {
        return;
    }

    const auto az_it = out.tel_data.data.find("TelAzAct");
    const auto el_it = out.tel_data.data.find("TelElAct");
    if (az_it == out.tel_data.data.end() || el_it == out.tel_data.data.end()) {
        logger->warn("altaz_destripe enabled but TelAzAct/TelElAct not found; skipping");
        return;
    }

    const auto n_pts_out = out.scans.data.rows();
    const auto n_dets_out = out.scans.data.cols();
    if (n_pts_out <= 0 || n_dets_out <= 0) {
        return;
    }

    Eigen::VectorXd az = az_it->second;
    Eigen::VectorXd el = el_it->second;
    if (az.size() != n_pts_out || el.size() != n_pts_out) {
        logger->warn("altaz_destripe skipped: tel vector size mismatch (n_pts={} az={} el={})",
                     n_pts_out, az.size(), el.size());
        return;
    }

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
        return;
    }

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

template <typename tc_t, typename calib_t>
void RTCProc::capture_rtc_line_audit(tc_t &in,
                                     calib_t &calib,
                                     Eigen::Index start_sample,
                                     Eigen::Index n_samples,
                                     const RTCLineAuditOptions &audit,
                                     bool post_filter_stage) {
    if (!audit.enabled) {
        return;
    }

    const Eigen::Index scan_id = in.index.data;
    const Eigen::Index n_total_pts = in.scans.data.rows();
    const Eigen::Index n_total_dets = in.scans.data.cols();
    if (n_total_pts <= 0 || n_total_dets <= 0) {
        return;
    }

    start_sample = std::max<Eigen::Index>(0, std::min(start_sample, n_total_pts - 1));
    if (n_samples <= 0) {
        n_samples = n_total_pts - start_sample;
    }
    n_samples = std::max<Eigen::Index>(0, std::min(n_samples, n_total_pts - start_sample));
    if (n_samples < 16) {
        return;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int fill_int = kTransientFillInt;
    constexpr double two_pi = 6.283185307179586476925286766559;

    auto assign_legacy_line_audit = [](RTCNetworkDiagSummary &row,
                                       const RTCLineAuditDiagSummary &diag) {
        row.line_audit_n_det_used = diag.n_det_used;
        row.line_audit_shared_freq_hz = diag.shared_freq_hz;
        row.line_audit_shared_detector_count = diag.shared_detector_count;
        row.line_audit_shared_detector_frac = diag.shared_detector_frac;
        row.line_audit_shared_median_prominence = diag.shared_median_prominence;
        row.line_audit_shared_max_prominence = diag.shared_max_prominence;
        row.line_audit_shared_width_hz = diag.shared_width_hz;
        row.line_audit_shared_line_power_frac = diag.shared_line_power_frac;
        row.line_audit_shared_common_mode_freq_hz = diag.shared_common_mode_freq_hz;
        row.line_audit_shared_common_mode_prominence = diag.shared_common_mode_prominence;
        row.line_audit_shared_notch_score = diag.shared_notch_score;
        row.line_audit_shared_recommend_notch = diag.shared_recommend_notch;
        row.line_audit_n_applied_notches = diag.n_applied_notches;
        row.line_audit_shared_applied_notch = diag.shared_applied_notch;
        row.line_audit_shared_applied_freq_hz = diag.shared_applied_freq_hz;
        row.line_audit_shared_applied_width_hz = diag.shared_applied_width_hz;
        row.line_audit_shared_applied_support_network_count =
            diag.shared_applied_support_network_count;
        row.line_audit_detector_candidate_uid = diag.detector_candidate_uid;
        row.line_audit_detector_candidate_freq_hz = diag.detector_candidate_freq_hz;
        row.line_audit_detector_candidate_prominence = diag.detector_candidate_prominence;
        row.line_audit_detector_candidate_line_power_frac = diag.detector_candidate_line_power_frac;
        row.line_audit_detector_candidate_cluster_detector_frac =
            diag.detector_candidate_cluster_detector_frac;
        row.line_audit_detector_candidate_recommend_flag = diag.detector_candidate_recommend_flag;
        row.line_audit_shared_candidates = diag.shared_candidates;
    };

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
            dt.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(n_samples - 1, 0)));
            const Eigen::Index end_idx = std::min<Eigen::Index>(t.size(), start_sample + n_samples);
            for (Eigen::Index i = start_sample + 1; i < end_idx; ++i) {
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
            const double mean =
                std::accumulate(good.begin(), good.end(), 0.0) / static_cast<double>(good.size());
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

    auto contiguous_runs = [&](const Eigen::Array<bool, Eigen::Dynamic, 1> &valid_mask) {
        std::vector<std::pair<Eigen::Index, Eigen::Index>> runs;
        Eigen::Index i = 0;
        while (i < valid_mask.size()) {
            if (valid_mask(i)) {
                Eigen::Index j = i + 1;
                while (j < valid_mask.size() && valid_mask(j)) {
                    ++j;
                }
                runs.emplace_back(i, j);
                i = j;
            }
            else {
                ++i;
            }
        }
        return runs;
    };

    auto rolling_median = [&](const std::vector<double> &values, Eigen::Index radius) {
        std::vector<double> out(values.size(), nan);
        radius = std::max<Eigen::Index>(1, radius);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(values.size()); ++i) {
            const Eigen::Index j0 = std::max<Eigen::Index>(0, i - radius);
            const Eigen::Index j1 =
                std::min<Eigen::Index>(static_cast<Eigen::Index>(values.size()), i + radius + 1);
            std::vector<double> window;
            window.reserve(static_cast<std::size_t>(j1 - j0));
            for (Eigen::Index j = j0; j < j1; ++j) {
                if (std::isfinite(values[static_cast<std::size_t>(j)])) {
                    window.push_back(values[static_cast<std::size_t>(j)]);
                }
            }
            out[static_cast<std::size_t>(i)] = median_of(std::move(window));
        }
        return out;
    };

    struct LinePeak {
        int uid = kTransientFillInt;
        double freq_hz = std::numeric_limits<double>::quiet_NaN();
        double prominence = std::numeric_limits<double>::quiet_NaN();
        double width_hz = std::numeric_limits<double>::quiet_NaN();
        double line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double cluster_detector_frac = std::numeric_limits<double>::quiet_NaN();
        bool cluster_recommend_notch = false;
    };

    auto masked_welch_psd = [&](const Eigen::VectorXd &x,
                                const Eigen::Array<bool, Eigen::Dynamic, 1> &valid_mask) {
        struct Result {
            std::vector<double> freq_hz;
            std::vector<double> psd;
            int n_windows = 0;
        };
        Result result;
        const double dt_sec = infer_dt_sec();
        if (!std::isfinite(dt_sec) || dt_sec <= 0.0 || x.size() != valid_mask.size() || x.size() < 16) {
            return result;
        }
        const double fs_hz = 1.0 / dt_sec;
        const auto valid_runs = contiguous_runs(valid_mask);
        Eigen::Index longest_run = 0;
        for (const auto &[i0, i1] : valid_runs) {
            longest_run = std::max<Eigen::Index>(longest_run, i1 - i0);
        }

        Eigen::Index nperseg =
            std::max<Eigen::Index>(16, static_cast<Eigen::Index>(std::llround(audit.segment_sec * fs_hz)));
        const Eigen::Index min_seg_n =
            std::max<Eigen::Index>(16, static_cast<Eigen::Index>(std::llround(audit.min_segment_sec * fs_hz)));
        if (nperseg < min_seg_n) {
            nperseg = min_seg_n;
        }
        if (longest_run < min_seg_n) {
            return result;
        }

        const double hop_frac = std::max(0.05, 1.0 - audit.overlap_frac);
        if (audit.min_windows > 1) {
            const double denom =
                1.0 + hop_frac * static_cast<double>(std::max<Eigen::Index>(0, audit.min_windows - 1));
            if (denom > 0.0) {
                const Eigen::Index max_nperseg_for_windows =
                    static_cast<Eigen::Index>(std::floor(static_cast<double>(longest_run) / denom));
                if (max_nperseg_for_windows >= min_seg_n && nperseg > max_nperseg_for_windows) {
                    nperseg = max_nperseg_for_windows;
                }
            }
        }
        nperseg = std::min(nperseg, longest_run);

        const Eigen::Index hop = std::max<Eigen::Index>(
            1, static_cast<Eigen::Index>(std::llround(nperseg * hop_frac)));

        Eigen::VectorXd window = Eigen::VectorXd::Zero(nperseg);
        if (nperseg > 1) {
            for (Eigen::Index i = 0; i < nperseg; ++i) {
                window(i) = 0.5 * (1.0 - std::cos(two_pi * static_cast<double>(i) /
                                                  static_cast<double>(nperseg - 1)));
            }
        }
        else {
            window(0) = 1.0;
        }
        const double win_norm = fs_hz * window.array().square().sum();
        if (!std::isfinite(win_norm) || win_norm <= 0.0) {
            return result;
        }

        Eigen::VectorXd accum;
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);
        for (const auto &[i0, i1] : valid_runs) {
            const Eigen::Index seg_len = i1 - i0;
            if (seg_len < min_seg_n) {
                continue;
            }
            std::vector<Eigen::Index> starts;
            if (seg_len < nperseg) {
                starts.push_back(i0);
            }
            else {
                for (Eigen::Index s = i0; s <= i1 - nperseg; s += hop) {
                    starts.push_back(s);
                }
                if (!starts.empty() && starts.back() != (i1 - nperseg)) {
                    starts.push_back(i1 - nperseg);
                }
            }
            for (const auto s : starts) {
                const Eigen::Index e = std::min<Eigen::Index>(i1, s + nperseg);
                Eigen::VectorXd chunk = x.segment(s, e - s);
                if (chunk.size() < min_seg_n) {
                    continue;
                }
                const double med = median_of(std::vector<double>(chunk.data(), chunk.data() + chunk.size()));
                if (chunk.size() < nperseg) {
                    Eigen::VectorXd padded = Eigen::VectorXd::Zero(nperseg);
                    if (std::isfinite(med)) {
                        padded.head(chunk.size()) = chunk.array() - med;
                    }
                    else {
                        padded.head(chunk.size()) = chunk;
                    }
                    chunk = std::move(padded);
                }
                else {
                    chunk = chunk.head(nperseg);
                    if (std::isfinite(med)) {
                        chunk.array() -= med;
                    }
                }

                Eigen::VectorXd chunk_windowed = chunk.cwiseProduct(window);
                Eigen::VectorXcd spec;
                fft.fwd(spec, chunk_windowed);
                Eigen::VectorXd psd = spec.array().abs2() / win_norm;
                if (psd.size() > 2) {
                    psd.segment(1, psd.size() - 2) *= 2.0;
                }
                if (accum.size() == 0) {
                    accum = Eigen::VectorXd::Zero(psd.size());
                }
                accum += psd;
                ++result.n_windows;
            }
        }

        if (result.n_windows <= 0 || accum.size() == 0) {
            return result;
        }
        const Eigen::Index n_freq = accum.size();
        result.freq_hz.resize(static_cast<std::size_t>(n_freq));
        result.psd.resize(static_cast<std::size_t>(n_freq));
        for (Eigen::Index k = 0; k < n_freq; ++k) {
            result.freq_hz[static_cast<std::size_t>(k)] =
                static_cast<double>(k) * fs_hz / static_cast<double>(nperseg);
            result.psd[static_cast<std::size_t>(k)] =
                accum(k) / static_cast<double>(result.n_windows);
        }
        return result;
    };

    auto find_line_peaks = [&](const std::vector<double> &freq_hz,
                               const std::vector<double> &psd,
                               double prominence_thresh) {
        std::vector<LinePeak> peaks;
        if (freq_hz.size() != psd.size() || freq_hz.size() < 8) {
            return peaks;
        }

        std::vector<double> good_freq;
        std::vector<double> good_psd;
        good_freq.reserve(freq_hz.size());
        good_psd.reserve(psd.size());
        for (std::size_t i = 0; i < freq_hz.size(); ++i) {
            const double f = freq_hz[i];
            const double p = psd[i];
            if (!std::isfinite(f) || !std::isfinite(p) || p <= 0.0) {
                continue;
            }
            if (audit.line_min_hz > 0.0 && f < audit.line_min_hz) {
                continue;
            }
            if (audit.line_max_hz > 0.0 && f > audit.line_max_hz) {
                continue;
            }
            if (rtc_line_audit_frequency_excluded_by_fixed_notch(f, audit)) {
                continue;
            }
            good_freq.push_back(f);
            good_psd.push_back(p);
        }
        if (good_freq.size() < 8) {
            return peaks;
        }

        auto continuum = rolling_median(good_psd, audit.continuum_radius_bins);
        double continuum_fallback = median_of(good_psd);
        if (!std::isfinite(continuum_fallback) || continuum_fallback <= 0.0) {
            continuum_fallback = 1.0;
        }
        std::vector<double> prominence(good_psd.size(), nan);
        for (std::size_t i = 0; i < good_psd.size(); ++i) {
            double base = continuum[i];
            if (!std::isfinite(base) || base <= 0.0) {
                base = continuum_fallback;
            }
            prominence[i] = good_psd[i] / base;
        }

        for (std::size_t i = 1; i + 1 < good_freq.size(); ++i) {
            if (!std::isfinite(prominence[i]) || prominence[i] < prominence_thresh) {
                continue;
            }
            if (prominence[i] < prominence[i - 1] || prominence[i] < prominence[i + 1]) {
                continue;
            }
            const double target = 1.0 + 0.5 * std::max(prominence[i] - 1.0, 0.0);
            std::size_t j0 = i;
            while (j0 > 0 && prominence[j0 - 1] >= target) {
                --j0;
            }
            std::size_t j1 = i;
            while (j1 + 1 < good_freq.size() && prominence[j1 + 1] >= target) {
                ++j1;
            }
            const double min_bin_width =
                (good_freq.size() > 1) ? std::max(good_freq[1] - good_freq[0], 1.0e-6) : 1.0e-6;
            const double width_hz = std::max(good_freq[j1] - good_freq[j0], min_bin_width);
            double total_power = 0.0;
            for (std::size_t k = 1; k < good_freq.size(); ++k) {
                const double df = good_freq[k] - good_freq[k - 1];
                total_power += 0.5 * (good_psd[k] + good_psd[k - 1]) * df;
            }
            double line_power = 0.0;
            auto continuum_at = [&](std::size_t k) {
                double base = continuum[std::min<std::size_t>(k, continuum.size() - 1)];
                if (!std::isfinite(base) || base <= 0.0) {
                    base = continuum_fallback;
                }
                return base;
            };
            if (j0 == j1) {
                const double df_left = (i > 0) ? (good_freq[i] - good_freq[i - 1]) : min_bin_width;
                const double df_right =
                    (i + 1 < good_freq.size()) ? (good_freq[i + 1] - good_freq[i]) : min_bin_width;
                const double df = std::max(0.5 * (df_left + df_right), min_bin_width);
                line_power = std::max(good_psd[i] - continuum_at(i), 0.0) * df;
            }
            else {
                for (std::size_t k = j0 + 1; k <= j1; ++k) {
                    const double df = good_freq[k] - good_freq[k - 1];
                    const double local0 = std::max(good_psd[k - 1] - continuum_at(k - 1), 0.0);
                    const double local1 = std::max(good_psd[k] - continuum_at(k), 0.0);
                    line_power += 0.5 * (local0 + local1) * df;
                }
            }

            LinePeak peak;
            peak.freq_hz = good_freq[i];
            peak.prominence = prominence[i];
            peak.width_hz = width_hz;
            peak.line_power_frac =
                (total_power > 0.0) ? (line_power / total_power) : nan;
            peaks.push_back(peak);
        }
        std::sort(peaks.begin(), peaks.end(), [](const auto &a, const auto &b) {
            if (a.prominence != b.prominence) {
                return a.prominence > b.prominence;
            }
            return a.freq_hz < b.freq_hz;
        });
        return peaks;
    };

    struct SelectedDet {
        Eigen::Index det = -1;
        int uid = kTransientFillInt;
        Eigen::VectorXd centered;
        Eigen::Array<bool, Eigen::Dynamic, 1> valid;
    };
    struct SharedCluster {
        double center_hz = std::numeric_limits<double>::quiet_NaN();
        int detector_count = 0;
        double detector_frac = std::numeric_limits<double>::quiet_NaN();
        double median_prominence = std::numeric_limits<double>::quiet_NaN();
        double max_prominence = std::numeric_limits<double>::quiet_NaN();
        double median_width_hz = std::numeric_limits<double>::quiet_NaN();
        double notch_width_hz = std::numeric_limits<double>::quiet_NaN();
        double freq_min_hz = std::numeric_limits<double>::quiet_NaN();
        double freq_max_hz = std::numeric_limits<double>::quiet_NaN();
        double median_line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double common_mode_freq_hz = std::numeric_limits<double>::quiet_NaN();
        double common_mode_prominence = std::numeric_limits<double>::quiet_NaN();
        double notch_score = std::numeric_limits<double>::quiet_NaN();
        bool recommend_notch = false;
    };

    const double dt_sec = infer_dt_sec();
    const double fs_hz = (std::isfinite(dt_sec) && dt_sec > 0.0) ? (1.0 / dt_sec) : nan;

    std::map<Eigen::Index, RTCNetworkDiagSummary> prev_nw_summary;
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto prev_it = rtc_network_summary_by_scan.find(scan_id);
        if (prev_it != rtc_network_summary_by_scan.end()) {
            for (const auto &row : prev_it->second) {
                prev_nw_summary[row.nw] = row;
            }
        }
    }

    std::vector<RTCNetworkDiagSummary> nw_summary;
    auto grp_limits = get_grouping("nw", calib, n_total_dets);
    nw_summary.reserve(grp_limits.size());

    for (const auto &[nw, bounds] : grp_limits) {
        const auto start_det = std::get<0>(bounds);
        const auto end_det = std::get<1>(bounds);
        RTCNetworkDiagSummary row;
        const auto old_row_it = prev_nw_summary.find(nw);
        if (old_row_it != prev_nw_summary.end()) {
            row = old_row_it->second;
        }
        row.nw = nw;
        RTCLineAuditDiagSummary line_row;
        auto push_line_audit_row = [&](RTCNetworkDiagSummary row_to_push,
                                       const RTCLineAuditDiagSummary &diag) {
            if (post_filter_stage) {
                row_to_push.post_line_audit = diag;
            }
            else {
                assign_legacy_line_audit(row_to_push, diag);
            }
            nw_summary.push_back(std::move(row_to_push));
        };

        std::vector<SelectedDet> eligible;
        eligible.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(end_det - start_det, 0)));
        for (Eigen::Index det = start_det; det < end_det; ++det) {
            Eigen::Array<bool, Eigen::Dynamic, 1> valid(n_samples);
            Eigen::Index n_valid = 0;
            for (Eigen::Index i = 0; i < n_samples; ++i) {
                const Eigen::Index src_i = start_sample + i;
                valid(i) = std::isfinite(in.scans.data(src_i, det)) && !in.flags.data(src_i, det);
                if (valid(i)) {
                    ++n_valid;
                }
            }
            const double good_frac =
                static_cast<double>(n_valid) / static_cast<double>(std::max<Eigen::Index>(n_samples, 1));
            if (!std::isfinite(good_frac) || good_frac < audit.min_good_frac) {
                continue;
            }

            Eigen::VectorXd signal = in.scans.data.block(start_sample, det, n_samples, 1);
            auto [center, scale] = robust_center_scale(signal, valid);
            if (!std::isfinite(center) || !std::isfinite(scale) || scale <= 0.0) {
                continue;
            }

            SelectedDet selected;
            selected.det = det;
            selected.uid = static_cast<int>(std::llround(calib.apt["uid"](det)));
            selected.centered = Eigen::VectorXd::Zero(n_samples);
            selected.valid = valid;
            for (Eigen::Index i = 0; i < n_samples; ++i) {
                if (valid(i) && std::isfinite(signal(i))) {
                    selected.centered(i) = signal(i) - center;
                }
            }
            eligible.push_back(std::move(selected));
        }

        if (eligible.empty()) {
            push_line_audit_row(std::move(row), line_row);
            continue;
        }

        std::vector<Eigen::Index> selected_idx;
        if (audit.max_det <= 0 || static_cast<Eigen::Index>(eligible.size()) <= audit.max_det) {
            selected_idx.resize(eligible.size());
            std::iota(selected_idx.begin(), selected_idx.end(), 0);
        }
        else {
            selected_idx.reserve(static_cast<std::size_t>(audit.max_det));
            for (Eigen::Index k = 0; k < audit.max_det; ++k) {
                const double alpha = (audit.max_det == 1)
                    ? 0.0
                    : static_cast<double>(k) / static_cast<double>(audit.max_det - 1);
                const Eigen::Index idx = static_cast<Eigen::Index>(std::llround(
                    alpha * static_cast<double>(eligible.size() - 1)));
                if (selected_idx.empty() || idx != selected_idx.back()) {
                    selected_idx.push_back(idx);
                }
            }
        }

        std::vector<const SelectedDet *> selected;
        selected.reserve(selected_idx.size());
        for (const auto idx : selected_idx) {
            if (idx >= 0 && idx < static_cast<Eigen::Index>(eligible.size())) {
                selected.push_back(&eligible[static_cast<std::size_t>(idx)]);
            }
        }
        line_row.n_det_used = static_cast<int>(selected.size());
        if (static_cast<Eigen::Index>(selected.size()) < audit.min_det_for_network) {
            push_line_audit_row(std::move(row), line_row);
            continue;
        }

        std::vector<LinePeak> detector_peaks;
        for (const auto *det_row : selected) {
            auto psd = masked_welch_psd(det_row->centered, det_row->valid);
            if (psd.n_windows < audit.min_windows) {
                continue;
            }
            auto peaks = find_line_peaks(psd.freq_hz, psd.psd, audit.prominence_thresh);
            const auto n_keep = std::min<std::size_t>(
                static_cast<std::size_t>(std::max<Eigen::Index>(audit.max_peaks_per_detector, 0)),
                peaks.size());
            for (std::size_t k = 0; k < n_keep; ++k) {
                peaks[k].uid = det_row->uid;
                detector_peaks.push_back(peaks[k]);
            }
        }

        Eigen::VectorXd cm = Eigen::VectorXd::Zero(n_samples);
        Eigen::Array<bool, Eigen::Dynamic, 1> cm_valid = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(n_samples, false);
        const Eigen::Index min_cm_count = std::max<Eigen::Index>(4, static_cast<Eigen::Index>(0.25 * selected.size()));
        for (Eigen::Index i = 0; i < n_samples; ++i) {
            double sum = 0.0;
            Eigen::Index count = 0;
            for (const auto *det_row : selected) {
                if (det_row->valid(i)) {
                    sum += det_row->centered(i);
                    ++count;
                }
            }
            if (count >= min_cm_count) {
                cm_valid(i) = true;
                cm(i) = sum / static_cast<double>(count);
            }
        }
        std::vector<LinePeak> cm_peaks;
        auto cm_psd = masked_welch_psd(cm, cm_valid);
        if (cm_psd.n_windows >= audit.min_windows) {
            cm_peaks = find_line_peaks(cm_psd.freq_hz, cm_psd.psd, audit.cm_prominence_thresh);
        }

        if (detector_peaks.empty()) {
            push_line_audit_row(std::move(row), line_row);
            continue;
        }

        const double tol_hz = std::max(
            audit.cluster_tol_hz,
            (std::isfinite(fs_hz) && n_samples > 0) ? (2.0 * fs_hz / static_cast<double>(n_samples)) : audit.cluster_tol_hz);

        std::sort(detector_peaks.begin(), detector_peaks.end(), [](const auto &a, const auto &b) {
            if (a.freq_hz != b.freq_hz) {
                return a.freq_hz < b.freq_hz;
            }
            return a.prominence > b.prominence;
        });

        std::vector<SharedCluster> shared_clusters;
        std::size_t i = 0;
        while (i < detector_peaks.size()) {
            std::size_t j = i + 1;
            while (j < detector_peaks.size() &&
                   std::abs(detector_peaks[j].freq_hz - detector_peaks[i].freq_hz) <= tol_hz) {
                ++j;
            }

            std::vector<double> freqs;
            std::vector<double> proms;
            std::vector<double> widths;
            std::vector<double> pfracs;
            std::vector<int> uids;
            freqs.reserve(j - i);
            proms.reserve(j - i);
            widths.reserve(j - i);
            pfracs.reserve(j - i);
            uids.reserve(j - i);
            for (std::size_t k = i; k < j; ++k) {
                freqs.push_back(detector_peaks[k].freq_hz);
                proms.push_back(detector_peaks[k].prominence);
                widths.push_back(detector_peaks[k].width_hz);
                pfracs.push_back(detector_peaks[k].line_power_frac);
                uids.push_back(detector_peaks[k].uid);
            }
            std::sort(uids.begin(), uids.end());
            uids.erase(std::unique(uids.begin(), uids.end()), uids.end());

            SharedCluster cluster;
            cluster.center_hz = median_of(freqs);
            cluster.freq_min_hz = *std::min_element(freqs.begin(), freqs.end());
            cluster.freq_max_hz = *std::max_element(freqs.begin(), freqs.end());
            cluster.detector_count = static_cast<int>(uids.size());
            cluster.detector_frac = static_cast<double>(uids.size()) /
                                    static_cast<double>(std::max<std::size_t>(1, selected.size()));
            cluster.median_prominence = median_of(proms);
            cluster.max_prominence = *std::max_element(proms.begin(), proms.end());
            cluster.median_width_hz = median_of(widths);
            cluster.notch_width_hz = cluster.median_width_hz;
            if (std::isfinite(cluster.center_hz) &&
                std::isfinite(cluster.freq_min_hz) &&
                std::isfinite(cluster.freq_max_hz)) {
                const double half_span_hz =
                    std::max(std::abs(cluster.center_hz - cluster.freq_min_hz),
                             std::abs(cluster.freq_max_hz - cluster.center_hz));
                if (std::isfinite(half_span_hz) && half_span_hz > 0.0) {
                    const double span_width_hz =
                        (std::isfinite(cluster.median_width_hz) ? cluster.median_width_hz : 0.0) +
                        2.0 * half_span_hz;
                    cluster.notch_width_hz = std::isfinite(cluster.notch_width_hz)
                        ? std::max(cluster.notch_width_hz, span_width_hz)
                        : span_width_hz;
                }
            }
            cluster.median_line_power_frac = median_of(pfracs);
            cluster.notch_score = cluster.detector_frac * cluster.median_prominence;
            for (const auto &cm_peak : cm_peaks) {
                if (std::abs(cm_peak.freq_hz - cluster.center_hz) <= tol_hz) {
                    cluster.common_mode_freq_hz = cm_peak.freq_hz;
                    cluster.common_mode_prominence = cm_peak.prominence;
                    break;
                }
            }
            cluster.recommend_notch =
                cluster.detector_frac >= audit.notch_min_detector_frac ||
                (std::isfinite(cluster.common_mode_prominence) &&
                 cluster.common_mode_prominence >= audit.notch_min_cm_prominence &&
                 cluster.detector_count >= audit.notch_min_detectors);
            shared_clusters.push_back(cluster);
            i = j;
        }

        auto better_shared = [](const SharedCluster &a, const SharedCluster &b) {
            if (a.recommend_notch != b.recommend_notch) {
                return a.recommend_notch && !b.recommend_notch;
            }
            if (a.notch_score != b.notch_score) {
                return a.notch_score > b.notch_score;
            }
            return a.median_prominence > b.median_prominence;
        };
        std::sort(shared_clusters.begin(), shared_clusters.end(), better_shared);

        auto shared_candidate_from_cluster = [](const SharedCluster &cluster) {
            RTCLineAuditSharedCandidate candidate;
            candidate.freq_hz = cluster.center_hz;
            candidate.detector_count = cluster.detector_count;
            candidate.detector_frac = cluster.detector_frac;
            candidate.median_prominence = cluster.median_prominence;
            candidate.max_prominence = cluster.max_prominence;
            candidate.width_hz = cluster.notch_width_hz;
            candidate.freq_min_hz = cluster.freq_min_hz;
            candidate.freq_max_hz = cluster.freq_max_hz;
            candidate.line_power_frac = cluster.median_line_power_frac;
            candidate.common_mode_freq_hz = cluster.common_mode_freq_hz;
            candidate.common_mode_prominence = cluster.common_mode_prominence;
            candidate.notch_score = cluster.notch_score;
            candidate.recommend_notch = cluster.recommend_notch;
            return candidate;
        };

        line_row.shared_candidates.clear();
        line_row.shared_candidates.reserve(shared_clusters.size());
        for (const auto &cluster : shared_clusters) {
            if (cluster.recommend_notch) {
                line_row.shared_candidates.push_back(shared_candidate_from_cluster(cluster));
            }
        }
        const SharedCluster *best_shared =
            shared_clusters.empty() ? nullptr : &shared_clusters.front();
        if (best_shared != nullptr) {
            line_row.shared_freq_hz = best_shared->center_hz;
            line_row.shared_detector_count = best_shared->detector_count;
            line_row.shared_detector_frac = best_shared->detector_frac;
            line_row.shared_median_prominence = best_shared->median_prominence;
            line_row.shared_max_prominence = best_shared->max_prominence;
            line_row.shared_width_hz = best_shared->median_width_hz;
            line_row.shared_line_power_frac = best_shared->median_line_power_frac;
            line_row.shared_common_mode_freq_hz = best_shared->common_mode_freq_hz;
            line_row.shared_common_mode_prominence = best_shared->common_mode_prominence;
            line_row.shared_notch_score = best_shared->notch_score;
            line_row.shared_recommend_notch = best_shared->recommend_notch;
        }

        for (auto &peak : detector_peaks) {
            double best_delta = std::numeric_limits<double>::infinity();
            const SharedCluster *best_cluster = nullptr;
            for (const auto &cluster : shared_clusters) {
                const double delta = std::abs(cluster.center_hz - peak.freq_hz);
                if (delta <= tol_hz && delta < best_delta) {
                    best_delta = delta;
                    best_cluster = &cluster;
                }
            }
            if (best_cluster != nullptr) {
                peak.cluster_detector_frac = best_cluster->detector_frac;
                peak.cluster_recommend_notch = best_cluster->recommend_notch;
            }
        }

        auto better_detector_peak = [](const LinePeak &a, const LinePeak &b) {
            if (a.prominence != b.prominence) {
                return a.prominence > b.prominence;
            }
            if (a.line_power_frac != b.line_power_frac) {
                return a.line_power_frac > b.line_power_frac;
            }
            return a.freq_hz < b.freq_hz;
        };
        const LinePeak *best_detector = nullptr;
        for (const auto &peak : detector_peaks) {
            const bool is_detector_local =
                peak.prominence >= audit.detector_min_prominence &&
                std::isfinite(peak.line_power_frac) &&
                peak.line_power_frac >= audit.detector_min_line_power_frac &&
                (!std::isfinite(peak.cluster_detector_frac) ||
                 peak.cluster_detector_frac <= audit.bad_detector_max_cluster_frac) &&
                !peak.cluster_recommend_notch;
            if (!is_detector_local) {
                continue;
            }
            if (best_detector == nullptr || better_detector_peak(peak, *best_detector)) {
                best_detector = &peak;
            }
        }
        if (best_detector != nullptr) {
            line_row.detector_candidate_uid = best_detector->uid;
            line_row.detector_candidate_freq_hz = best_detector->freq_hz;
            line_row.detector_candidate_prominence = best_detector->prominence;
            line_row.detector_candidate_line_power_frac = best_detector->line_power_frac;
            line_row.detector_candidate_cluster_detector_frac = best_detector->cluster_detector_frac;
            line_row.detector_candidate_recommend_flag = true;
        }

        if (line_row.shared_recommend_notch || line_row.detector_candidate_recommend_flag) {
            logger->info(
                "rtc_line_audit scan {} nw {}: n_det_used={} shared_freq_hz={:.4f} det_count={} det_frac={:.4f} shared_prom={:.4g} cm_prom={:.4g} recommend_notch={} recommended_shared_clusters={} detector_uid={} detector_freq_hz={:.4f} detector_prom={:.4g} recommend_bad_detector={}",
                scan_id + 1,
                nw,
                line_row.n_det_used,
                line_row.shared_freq_hz,
                line_row.shared_detector_count,
                line_row.shared_detector_frac,
                line_row.shared_median_prominence,
                line_row.shared_common_mode_prominence,
                line_row.shared_recommend_notch,
                line_row.shared_candidates.size(),
                line_row.detector_candidate_uid,
                line_row.detector_candidate_freq_hz,
                line_row.detector_candidate_prominence,
                line_row.detector_candidate_recommend_flag);
        }

        push_line_audit_row(std::move(row), line_row);
    }

    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_network_summary_by_scan[scan_id] = std::move(nw_summary);
    }
}
template <typename tc_t>
Eigen::Index RTCProc::apply_rtc_line_audit_shared_notches(tc_t &in,
                                                          double fs_hz,
                                                          const RTCLineAuditOptions &audit,
                                                          bool post_filter_stage) {
    if (!audit.enabled || !audit.apply_shared_notches ||
        !std::isfinite(fs_hz) || fs_hz <= 0.0) {
        return 0;
    }

    const auto scan_id = in.index.data;
    std::vector<RTCNetworkDiagSummary> nw_summary;
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto nw_it = rtc_network_summary_by_scan.find(scan_id);
        if (nw_it == rtc_network_summary_by_scan.end()) {
            return 0;
        }
        nw_summary = nw_it->second;
    }
    auto publish_nw_summary = [&]() {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_network_summary_by_scan[scan_id] = nw_summary;
    };
    if (nw_summary.empty()) {
        return 0;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();

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

    struct NetworkCandidate {
        Eigen::Index nw = -1;
        double freq_hz = std::numeric_limits<double>::quiet_NaN();
        double width_hz = std::numeric_limits<double>::quiet_NaN();
        double freq_min_hz = std::numeric_limits<double>::quiet_NaN();
        double freq_max_hz = std::numeric_limits<double>::quiet_NaN();
        double detector_frac = std::numeric_limits<double>::quiet_NaN();
        double common_mode_prominence = std::numeric_limits<double>::quiet_NaN();
        double notch_score = std::numeric_limits<double>::quiet_NaN();
    };
    struct AppliedCluster {
        double center_hz = std::numeric_limits<double>::quiet_NaN();
        double width_hz = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index support_network_count = 0;
        double freq_min_hz = std::numeric_limits<double>::quiet_NaN();
        double freq_max_hz = std::numeric_limits<double>::quiet_NaN();
        double max_detector_frac = std::numeric_limits<double>::quiet_NaN();
        double max_common_mode_prominence = std::numeric_limits<double>::quiet_NaN();
        double max_notch_score = std::numeric_limits<double>::quiet_NaN();
    };
    auto legacy_line_audit_diag = [](const RTCNetworkDiagSummary &row) {
        RTCLineAuditDiagSummary diag;
        diag.n_det_used = row.line_audit_n_det_used;
        diag.shared_freq_hz = row.line_audit_shared_freq_hz;
        diag.shared_detector_count = row.line_audit_shared_detector_count;
        diag.shared_detector_frac = row.line_audit_shared_detector_frac;
        diag.shared_median_prominence = row.line_audit_shared_median_prominence;
        diag.shared_max_prominence = row.line_audit_shared_max_prominence;
        diag.shared_width_hz = row.line_audit_shared_width_hz;
        diag.shared_line_power_frac = row.line_audit_shared_line_power_frac;
        diag.shared_common_mode_freq_hz = row.line_audit_shared_common_mode_freq_hz;
        diag.shared_common_mode_prominence = row.line_audit_shared_common_mode_prominence;
        diag.shared_notch_score = row.line_audit_shared_notch_score;
        diag.shared_recommend_notch = row.line_audit_shared_recommend_notch;
        diag.n_applied_notches = row.line_audit_n_applied_notches;
        diag.shared_applied_notch = row.line_audit_shared_applied_notch;
        diag.shared_applied_freq_hz = row.line_audit_shared_applied_freq_hz;
        diag.shared_applied_width_hz = row.line_audit_shared_applied_width_hz;
        diag.shared_applied_support_network_count =
            row.line_audit_shared_applied_support_network_count;
        diag.detector_candidate_uid = row.line_audit_detector_candidate_uid;
        diag.detector_candidate_freq_hz = row.line_audit_detector_candidate_freq_hz;
        diag.detector_candidate_prominence = row.line_audit_detector_candidate_prominence;
        diag.detector_candidate_line_power_frac = row.line_audit_detector_candidate_line_power_frac;
        diag.detector_candidate_cluster_detector_frac =
            row.line_audit_detector_candidate_cluster_detector_frac;
        diag.detector_candidate_recommend_flag = row.line_audit_detector_candidate_recommend_flag;
        diag.shared_candidates = row.line_audit_shared_candidates;
        return diag;
    };
    auto assign_legacy_line_audit = [](RTCNetworkDiagSummary &row,
                                       const RTCLineAuditDiagSummary &diag) {
        row.line_audit_n_det_used = diag.n_det_used;
        row.line_audit_shared_freq_hz = diag.shared_freq_hz;
        row.line_audit_shared_detector_count = diag.shared_detector_count;
        row.line_audit_shared_detector_frac = diag.shared_detector_frac;
        row.line_audit_shared_median_prominence = diag.shared_median_prominence;
        row.line_audit_shared_max_prominence = diag.shared_max_prominence;
        row.line_audit_shared_width_hz = diag.shared_width_hz;
        row.line_audit_shared_line_power_frac = diag.shared_line_power_frac;
        row.line_audit_shared_common_mode_freq_hz = diag.shared_common_mode_freq_hz;
        row.line_audit_shared_common_mode_prominence = diag.shared_common_mode_prominence;
        row.line_audit_shared_notch_score = diag.shared_notch_score;
        row.line_audit_shared_recommend_notch = diag.shared_recommend_notch;
        row.line_audit_n_applied_notches = diag.n_applied_notches;
        row.line_audit_shared_applied_notch = diag.shared_applied_notch;
        row.line_audit_shared_applied_freq_hz = diag.shared_applied_freq_hz;
        row.line_audit_shared_applied_width_hz = diag.shared_applied_width_hz;
        row.line_audit_shared_applied_support_network_count =
            diag.shared_applied_support_network_count;
        row.line_audit_detector_candidate_uid = diag.detector_candidate_uid;
        row.line_audit_detector_candidate_freq_hz = diag.detector_candidate_freq_hz;
        row.line_audit_detector_candidate_prominence = diag.detector_candidate_prominence;
        row.line_audit_detector_candidate_line_power_frac = diag.detector_candidate_line_power_frac;
        row.line_audit_detector_candidate_cluster_detector_frac =
            diag.detector_candidate_cluster_detector_frac;
        row.line_audit_detector_candidate_recommend_flag = diag.detector_candidate_recommend_flag;
        row.line_audit_shared_candidates = diag.shared_candidates;
    };
    auto get_line_audit_diag = [&](const RTCNetworkDiagSummary &row) {
        return post_filter_stage ? row.post_line_audit : legacy_line_audit_diag(row);
    };
    auto set_line_audit_diag = [&](RTCNetworkDiagSummary &row,
                                   const RTCLineAuditDiagSummary &diag) {
        if (post_filter_stage) {
            row.post_line_audit = diag;
        }
        else {
            assign_legacy_line_audit(row, diag);
        }
    };

    auto push_network_candidate = [](std::vector<NetworkCandidate> &candidates,
                                     Eigen::Index nw,
                                     const RTCLineAuditSharedCandidate &shared) {
        if (!shared.recommend_notch ||
            !std::isfinite(shared.freq_hz) ||
            shared.freq_hz <= 0.0) {
            return false;
        }
        NetworkCandidate candidate;
        candidate.nw = nw;
        candidate.freq_hz = shared.freq_hz;
        candidate.width_hz = shared.width_hz;
        candidate.freq_min_hz = shared.freq_min_hz;
        candidate.freq_max_hz = shared.freq_max_hz;
        candidate.detector_frac = shared.detector_frac;
        candidate.common_mode_prominence = shared.common_mode_prominence;
        candidate.notch_score = shared.notch_score;
        candidates.push_back(candidate);
        return true;
    };

    std::vector<NetworkCandidate> candidates;
    candidates.reserve(nw_summary.size() * 2);
    for (const auto &row : nw_summary) {
        const auto diag = get_line_audit_diag(row);
        bool added_multi_candidate = false;
        for (const auto &shared : diag.shared_candidates) {
            added_multi_candidate =
                push_network_candidate(candidates, row.nw, shared) || added_multi_candidate;
        }
        if (added_multi_candidate) {
            continue;
        }
        RTCLineAuditSharedCandidate legacy_candidate;
        legacy_candidate.freq_hz = diag.shared_freq_hz;
        legacy_candidate.detector_count = diag.shared_detector_count;
        legacy_candidate.detector_frac = diag.shared_detector_frac;
        legacy_candidate.median_prominence = diag.shared_median_prominence;
        legacy_candidate.max_prominence = diag.shared_max_prominence;
        legacy_candidate.width_hz = diag.shared_width_hz;
        if (std::isfinite(diag.shared_freq_hz) &&
            std::isfinite(diag.shared_width_hz) &&
            diag.shared_width_hz > 0.0) {
            legacy_candidate.freq_min_hz = diag.shared_freq_hz - 0.5 * diag.shared_width_hz;
            legacy_candidate.freq_max_hz = diag.shared_freq_hz + 0.5 * diag.shared_width_hz;
        }
        legacy_candidate.line_power_frac = diag.shared_line_power_frac;
        legacy_candidate.common_mode_freq_hz = diag.shared_common_mode_freq_hz;
        legacy_candidate.common_mode_prominence = diag.shared_common_mode_prominence;
        legacy_candidate.notch_score = diag.shared_notch_score;
        legacy_candidate.recommend_notch = diag.shared_recommend_notch;
        push_network_candidate(candidates, row.nw, legacy_candidate);
    }

    for (auto &row : nw_summary) {
        auto diag = get_line_audit_diag(row);
        diag.n_applied_notches = 0;
        diag.shared_applied_notch = false;
        diag.shared_applied_freq_hz = nan;
        diag.shared_applied_width_hz = nan;
        diag.shared_applied_support_network_count = 0;
        for (auto &shared : diag.shared_candidates) {
            shared.applied_notch = false;
            shared.applied_freq_hz = nan;
            shared.applied_width_hz = nan;
            shared.applied_support_network_count = 0;
        }
        set_line_audit_diag(row, diag);
    }

    if (candidates.empty()) {
        publish_nw_summary();
        return 0;
    }

    const double cluster_tol_hz = std::max(audit.cluster_tol_hz, audit.apply_cluster_tol_hz);
    if (!(cluster_tol_hz > 0.0)) {
        publish_nw_summary();
        return 0;
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
        if (a.freq_hz != b.freq_hz) {
            return a.freq_hz < b.freq_hz;
        }
        return a.notch_score > b.notch_score;
    });

    std::vector<AppliedCluster> clusters;
    std::size_t i = 0;
    while (i < candidates.size()) {
        std::size_t j = i + 1;
        while (j < candidates.size() &&
               std::abs(candidates[j].freq_hz - candidates[i].freq_hz) <= cluster_tol_hz) {
            ++j;
        }

        std::vector<double> freqs;
        std::vector<double> widths;
        std::vector<double> scores;
        std::vector<Eigen::Index> nws;
        double freq_min_hz = nan;
        double freq_max_hz = nan;
        double max_detector_frac = nan;
        double max_cm_prom = nan;
        for (std::size_t k = i; k < j; ++k) {
            freqs.push_back(candidates[k].freq_hz);
            widths.push_back(candidates[k].width_hz);
            scores.push_back(candidates[k].notch_score);
            nws.push_back(candidates[k].nw);
            double cand_min_hz = candidates[k].freq_min_hz;
            double cand_max_hz = candidates[k].freq_max_hz;
            if ((!std::isfinite(cand_min_hz) || !std::isfinite(cand_max_hz)) &&
                std::isfinite(candidates[k].freq_hz)) {
                const double half_width_hz =
                    (std::isfinite(candidates[k].width_hz) && candidates[k].width_hz > 0.0)
                        ? 0.5 * candidates[k].width_hz
                        : 0.0;
                cand_min_hz = candidates[k].freq_hz - half_width_hz;
                cand_max_hz = candidates[k].freq_hz + half_width_hz;
            }
            if (std::isfinite(cand_min_hz) &&
                (!std::isfinite(freq_min_hz) || cand_min_hz < freq_min_hz)) {
                freq_min_hz = cand_min_hz;
            }
            if (std::isfinite(cand_max_hz) &&
                (!std::isfinite(freq_max_hz) || cand_max_hz > freq_max_hz)) {
                freq_max_hz = cand_max_hz;
            }
            if (!std::isfinite(max_detector_frac) || candidates[k].detector_frac > max_detector_frac) {
                max_detector_frac = candidates[k].detector_frac;
            }
            if (!std::isfinite(max_cm_prom) || candidates[k].common_mode_prominence > max_cm_prom) {
                max_cm_prom = candidates[k].common_mode_prominence;
            }
        }
        std::sort(nws.begin(), nws.end());
        nws.erase(std::unique(nws.begin(), nws.end()), nws.end());

        const bool enough_networks =
            static_cast<Eigen::Index>(nws.size()) >= audit.apply_min_support_networks;
        const bool strong_cm =
            std::isfinite(max_cm_prom) &&
            max_cm_prom >= audit.apply_min_common_mode_prominence &&
            std::isfinite(max_detector_frac) &&
            max_detector_frac >= audit.apply_min_detector_frac;
        if (enough_networks || strong_cm) {
            AppliedCluster cluster;
            cluster.center_hz = median_of(std::move(freqs));
            cluster.width_hz = median_of(std::move(widths));
            cluster.freq_min_hz = freq_min_hz;
            cluster.freq_max_hz = freq_max_hz;
            if (std::isfinite(cluster.freq_min_hz) &&
                std::isfinite(cluster.freq_max_hz) &&
                cluster.freq_max_hz > cluster.freq_min_hz) {
                const double span_hz = cluster.freq_max_hz - cluster.freq_min_hz;
                cluster.width_hz = std::isfinite(cluster.width_hz)
                    ? std::max(cluster.width_hz, span_hz)
                    : span_hz;
            }
            cluster.support_network_count = static_cast<Eigen::Index>(nws.size());
            cluster.max_detector_frac = max_detector_frac;
            cluster.max_common_mode_prominence = max_cm_prom;
            cluster.max_notch_score = median_of(std::move(scores));
            clusters.push_back(cluster);
        }
        i = j;
    }

    if (clusters.empty()) {
        publish_nw_summary();
        return 0;
    }

    auto better_cluster = [](const AppliedCluster &a, const AppliedCluster &b) {
        if (a.support_network_count != b.support_network_count) {
            return a.support_network_count > b.support_network_count;
        }
        if (a.max_detector_frac != b.max_detector_frac) {
            return a.max_detector_frac > b.max_detector_frac;
        }
        if (a.max_common_mode_prominence != b.max_common_mode_prominence) {
            return a.max_common_mode_prominence > b.max_common_mode_prominence;
        }
        if (a.max_notch_score != b.max_notch_score) {
            return a.max_notch_score > b.max_notch_score;
        }
        return a.center_hz < b.center_hz;
    };
    std::sort(clusters.begin(), clusters.end(), better_cluster);
    if (audit.apply_max_notches > 0 &&
        static_cast<Eigen::Index>(clusters.size()) > audit.apply_max_notches) {
        clusters.resize(static_cast<std::size_t>(audit.apply_max_notches));
    }

    Filter dynamic_notch_filter;
    dynamic_notch_filter.notch_zero_phase = true;
    std::vector<AppliedCluster> applied_clusters;
    applied_clusters.reserve(clusters.size());
    const double nyquist_hz = 0.5 * fs_hz;
    for (auto cluster : clusters) {
        if (!std::isfinite(cluster.center_hz) || cluster.center_hz <= 0.0 || cluster.center_hz >= nyquist_hz) {
            continue;
        }
        double width_hz = cluster.width_hz;
        if (!std::isfinite(width_hz) || width_hz <= 0.0) {
            width_hz = audit.apply_min_width_hz;
        }
        width_hz *= audit.apply_width_scale;
        width_hz = std::max(width_hz, audit.apply_min_width_hz);
        width_hz = std::min(width_hz, audit.apply_max_width_hz);
        width_hz = std::min(width_hz, std::max(0.05, 0.5 * cluster.center_hz));
        if (!std::isfinite(width_hz) || width_hz <= 0.0) {
            continue;
        }
        cluster.width_hz = width_hz;
        dynamic_notch_filter.w0s.push_back(cluster.center_hz);
        dynamic_notch_filter.qs.push_back(cluster.center_hz / width_hz);
        applied_clusters.push_back(cluster);
    }

    if (applied_clusters.empty()) {
        publish_nw_summary();
        return 0;
    }

    dynamic_notch_filter.make_notch_filter(fs_hz);
    logger->debug("applying {} dynamic shared-line RTC notch(es) {}",
                  applied_clusters.size(),
                  post_filter_stage ? "after filtering/downsampling" : "before FIR");
    dynamic_notch_filter.iir(in.scans.data);
    if (run_kernel) {
        dynamic_notch_filter.iir(in.kernel.data);
    }

    for (auto &row : nw_summary) {
        auto diag = get_line_audit_diag(row);
        diag.n_applied_notches = static_cast<int>(applied_clusters.size());
        auto find_applied_cluster = [&](double freq_hz) -> const AppliedCluster * {
            if (!std::isfinite(freq_hz) || freq_hz <= 0.0) {
                return nullptr;
            }
            const AppliedCluster *best_match = nullptr;
            double best_delta = std::numeric_limits<double>::infinity();
            for (const auto &cluster : applied_clusters) {
                const double delta = std::abs(cluster.center_hz - freq_hz);
                const double match_tol_hz =
                    std::max(cluster_tol_hz,
                             (std::isfinite(cluster.width_hz) && cluster.width_hz > 0.0)
                                 ? 0.5 * cluster.width_hz
                                 : 0.0);
                if (delta <= match_tol_hz && delta < best_delta) {
                    best_delta = delta;
                    best_match = &cluster;
                }
            }
            return best_match;
        };

        for (auto &shared : diag.shared_candidates) {
            const auto *match = find_applied_cluster(shared.freq_hz);
            if (match == nullptr) {
                continue;
            }
            shared.applied_notch = true;
            shared.applied_freq_hz = match->center_hz;
            shared.applied_width_hz = match->width_hz;
            shared.applied_support_network_count =
                static_cast<int>(match->support_network_count);
        }

        const AppliedCluster *best_match = find_applied_cluster(diag.shared_freq_hz);
        if (best_match != nullptr) {
            diag.shared_applied_notch = true;
            diag.shared_applied_freq_hz = best_match->center_hz;
            diag.shared_applied_width_hz = best_match->width_hz;
            diag.shared_applied_support_network_count =
                static_cast<int>(best_match->support_network_count);
        }
        set_line_audit_diag(row, diag);
    }
    publish_nw_summary();

    for (const auto &cluster : applied_clusters) {
        logger->info(
            "rtc_line_audit apply_shared_notch scan {}: center_hz={:.4f} width_hz={:.4f} support_networks={} max_detector_frac={:.4f} max_cm_prominence={:.4g}",
            scan_id + 1,
            cluster.center_hz,
            cluster.width_hz,
            cluster.support_network_count,
            cluster.max_detector_frac,
            cluster.max_common_mode_prominence);
    }

    return static_cast<Eigen::Index>(applied_clusters.size());
}

template <typename tc_t>
Eigen::Index RTCProc::apply_rtc_line_audit_detector_notches(tc_t &in,
                                                            double fs_hz,
                                                            const RTCLineAuditOptions &audit,
                                                            Eigen::Index diag_start_sample,
                                                            Eigen::Index diag_n_samples) {
    if (!audit.enabled || !audit.post_filter_apply_detector_notches ||
        !std::isfinite(fs_hz) || fs_hz <= 0.0) {
        return 0;
    }

    const Eigen::Index scan_id = in.index.data;
    const Eigen::Index n_pts = in.scans.data.rows();
    const Eigen::Index n_dets = in.scans.data.cols();
    if (n_pts < 16 || n_dets <= 0) {
        return 0;
    }

    diag_start_sample = std::max<Eigen::Index>(0, diag_start_sample);
    if (diag_start_sample >= n_pts) {
        diag_start_sample = 0;
    }
    if (diag_n_samples < 0) {
        diag_n_samples = n_pts - diag_start_sample;
    }
    diag_n_samples = std::min<Eigen::Index>(diag_n_samples, n_pts - diag_start_sample);
    if (diag_n_samples <= 0) {
        diag_start_sample = 0;
        diag_n_samples = n_pts;
    }

    const bool has_kernel =
        run_kernel &&
        in.kernel.data.rows() == n_pts &&
        in.kernel.data.cols() == n_dets;
    if (run_kernel && !has_kernel) {
        logger->warn(
            "rtc_line_audit detector notch skipped for scan {} because kernel dimensions do not match RTC data",
            scan_id + 1);
        return 0;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double two_pi = 6.283185307179586476925286766559;
    const double nyquist_hz = 0.5 * fs_hz;

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

    auto contiguous_runs = [&](const Eigen::Array<bool, Eigen::Dynamic, 1> &valid_mask) {
        std::vector<std::pair<Eigen::Index, Eigen::Index>> runs;
        Eigen::Index i = 0;
        while (i < valid_mask.size()) {
            if (valid_mask(i)) {
                Eigen::Index j = i + 1;
                while (j < valid_mask.size() && valid_mask(j)) {
                    ++j;
                }
                runs.emplace_back(i, j);
                i = j;
            }
            else {
                ++i;
            }
        }
        return runs;
    };

    auto rolling_median = [&](const std::vector<double> &values, Eigen::Index radius) {
        std::vector<double> out(values.size(), nan);
        radius = std::max<Eigen::Index>(1, radius);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(values.size()); ++i) {
            const Eigen::Index j0 = std::max<Eigen::Index>(0, i - radius);
            const Eigen::Index j1 =
                std::min<Eigen::Index>(static_cast<Eigen::Index>(values.size()), i + radius + 1);
            std::vector<double> window;
            window.reserve(static_cast<std::size_t>(j1 - j0));
            for (Eigen::Index j = j0; j < j1; ++j) {
                const double v = values[static_cast<std::size_t>(j)];
                if (std::isfinite(v)) {
                    window.push_back(v);
                }
            }
            out[static_cast<std::size_t>(i)] = median_of(std::move(window));
        }
        return out;
    };

    auto robust_center = [&](const Eigen::VectorXd &x,
                             const Eigen::Array<bool, Eigen::Dynamic, 1> &valid) {
        std::vector<double> good;
        good.reserve(static_cast<std::size_t>(x.size()));
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            if (valid(i) && std::isfinite(x(i))) {
                good.push_back(x(i));
            }
        }
        return median_of(std::move(good));
    };

    auto robust_rms = [&](const Eigen::VectorXd &x,
                          const Eigen::Array<bool, Eigen::Dynamic, 1> &valid,
                          Eigen::Index start_sample,
                          Eigen::Index n_samples) {
        std::vector<double> good;
        start_sample = std::max<Eigen::Index>(0, start_sample);
        if (start_sample >= x.size()) {
            return nan;
        }
        n_samples = std::min<Eigen::Index>(n_samples, x.size() - start_sample);
        if (n_samples <= 0) {
            return nan;
        }
        good.reserve(static_cast<std::size_t>(n_samples));
        for (Eigen::Index i = start_sample; i < start_sample + n_samples; ++i) {
            if (valid(i) && std::isfinite(x(i))) {
                good.push_back(x(i));
            }
        }
        const double center = median_of(good);
        if (!std::isfinite(center) || good.size() < 2) {
            return nan;
        }
        double ss = 0.0;
        Eigen::Index count = 0;
        for (const double v : good) {
            const double dv = v - center;
            ss += dv * dv;
            ++count;
        }
        return (count > 1) ? std::sqrt(ss / static_cast<double>(count - 1)) : nan;
    };

    struct PsdResult {
        std::vector<double> freq_hz;
        std::vector<double> psd;
        int n_windows = 0;
    };

    auto masked_welch_psd = [&](const Eigen::VectorXd &x,
                                const Eigen::Array<bool, Eigen::Dynamic, 1> &valid_mask) {
        PsdResult result;
        if (x.size() != valid_mask.size() || x.size() < 16) {
            return result;
        }

        const auto valid_runs = contiguous_runs(valid_mask);
        Eigen::Index longest_run = 0;
        for (const auto &[i0, i1] : valid_runs) {
            longest_run = std::max<Eigen::Index>(longest_run, i1 - i0);
        }

        Eigen::Index nperseg =
            std::max<Eigen::Index>(16, static_cast<Eigen::Index>(std::llround(audit.segment_sec * fs_hz)));
        const Eigen::Index min_seg_n =
            std::max<Eigen::Index>(16, static_cast<Eigen::Index>(std::llround(audit.min_segment_sec * fs_hz)));
        if (nperseg < min_seg_n) {
            nperseg = min_seg_n;
        }
        if (longest_run < min_seg_n) {
            return result;
        }

        const double hop_frac = std::max(0.05, 1.0 - audit.overlap_frac);
        if (audit.min_windows > 1) {
            const double denom =
                1.0 + hop_frac * static_cast<double>(std::max<Eigen::Index>(0, audit.min_windows - 1));
            if (denom > 0.0) {
                const Eigen::Index max_nperseg_for_windows =
                    static_cast<Eigen::Index>(std::floor(static_cast<double>(longest_run) / denom));
                if (max_nperseg_for_windows >= min_seg_n && nperseg > max_nperseg_for_windows) {
                    nperseg = max_nperseg_for_windows;
                }
            }
        }
        nperseg = std::min(nperseg, longest_run);
        if (nperseg < min_seg_n) {
            return result;
        }

        const Eigen::Index hop = std::max<Eigen::Index>(
            1, static_cast<Eigen::Index>(std::llround(nperseg * hop_frac)));
        Eigen::VectorXd window = Eigen::VectorXd::Zero(nperseg);
        for (Eigen::Index i = 0; i < nperseg; ++i) {
            window(i) = (nperseg > 1)
                ? 0.5 * (1.0 - std::cos(two_pi * static_cast<double>(i) /
                                          static_cast<double>(nperseg - 1)))
                : 1.0;
        }
        const double win_norm = fs_hz * window.array().square().sum();
        if (!std::isfinite(win_norm) || win_norm <= 0.0) {
            return result;
        }

        Eigen::VectorXd accum;
        Eigen::FFT<double> fft;
        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        fft.SetFlag(Eigen::FFT<double>::Unscaled);
        for (const auto &[i0, i1] : valid_runs) {
            const Eigen::Index seg_len = i1 - i0;
            if (seg_len < min_seg_n) {
                continue;
            }
            std::vector<Eigen::Index> starts;
            if (seg_len < nperseg) {
                starts.push_back(i0);
            }
            else {
                for (Eigen::Index s = i0; s <= i1 - nperseg; s += hop) {
                    starts.push_back(s);
                }
                if (!starts.empty() && starts.back() != (i1 - nperseg)) {
                    starts.push_back(i1 - nperseg);
                }
            }
            for (const auto s : starts) {
                const Eigen::Index e = std::min<Eigen::Index>(i1, s + nperseg);
                Eigen::VectorXd chunk = x.segment(s, e - s);
                if (chunk.size() < min_seg_n) {
                    continue;
                }
                const double med = median_of(std::vector<double>(chunk.data(), chunk.data() + chunk.size()));
                if (chunk.size() < nperseg) {
                    Eigen::VectorXd padded = Eigen::VectorXd::Zero(nperseg);
                    if (std::isfinite(med)) {
                        padded.head(chunk.size()) = chunk.array() - med;
                    }
                    else {
                        padded.head(chunk.size()) = chunk;
                    }
                    chunk = std::move(padded);
                }
                else {
                    chunk = chunk.head(nperseg);
                    if (std::isfinite(med)) {
                        chunk.array() -= med;
                    }
                }
                Eigen::VectorXd chunk_windowed = chunk.cwiseProduct(window);
                Eigen::VectorXcd spec;
                fft.fwd(spec, chunk_windowed);
                Eigen::VectorXd psd = spec.array().abs2() / win_norm;
                if (psd.size() > 2) {
                    psd.segment(1, psd.size() - 2) *= 2.0;
                }
                if (accum.size() == 0) {
                    accum = Eigen::VectorXd::Zero(psd.size());
                }
                accum += psd;
                ++result.n_windows;
            }
        }
        if (result.n_windows <= 0 || accum.size() == 0) {
            return result;
        }
        result.freq_hz.resize(static_cast<std::size_t>(accum.size()));
        result.psd.resize(static_cast<std::size_t>(accum.size()));
        for (Eigen::Index k = 0; k < accum.size(); ++k) {
            result.freq_hz[static_cast<std::size_t>(k)] =
                static_cast<double>(k) * fs_hz / static_cast<double>(nperseg);
            result.psd[static_cast<std::size_t>(k)] =
                accum(k) / static_cast<double>(result.n_windows);
        }
        return result;
    };

    struct DetectorPeak {
        double freq_hz = std::numeric_limits<double>::quiet_NaN();
        double prominence = std::numeric_limits<double>::quiet_NaN();
        double width_hz = std::numeric_limits<double>::quiet_NaN();
        double line_power_frac = std::numeric_limits<double>::quiet_NaN();
        double applied_width_hz = std::numeric_limits<double>::quiet_NaN();
    };

    auto find_line_peaks = [&](const std::vector<double> &freq_hz,
                               const std::vector<double> &psd) {
        std::vector<DetectorPeak> peaks;
        if (freq_hz.size() != psd.size() || freq_hz.size() < 8) {
            return peaks;
        }

        std::vector<double> good_freq;
        std::vector<double> good_psd;
        good_freq.reserve(freq_hz.size());
        good_psd.reserve(psd.size());
        for (std::size_t i = 0; i < freq_hz.size(); ++i) {
            const double f = freq_hz[i];
            const double p = psd[i];
            if (!std::isfinite(f) || !std::isfinite(p) || p <= 0.0) {
                continue;
            }
            if (audit.line_min_hz > 0.0 && f < audit.line_min_hz) {
                continue;
            }
            if (audit.line_max_hz > 0.0 && f > audit.line_max_hz) {
                continue;
            }
            if (f <= 0.0 || f >= nyquist_hz) {
                continue;
            }
            if (rtc_line_audit_frequency_excluded_by_fixed_notch(f, audit)) {
                continue;
            }
            good_freq.push_back(f);
            good_psd.push_back(p);
        }
        if (good_freq.size() < 8) {
            return peaks;
        }

        auto continuum = rolling_median(good_psd, audit.continuum_radius_bins);
        double continuum_fallback = median_of(good_psd);
        if (!std::isfinite(continuum_fallback) || continuum_fallback <= 0.0) {
            continuum_fallback = 1.0;
        }
        std::vector<double> prominence(good_psd.size(), nan);
        for (std::size_t i = 0; i < good_psd.size(); ++i) {
            double base = continuum[i];
            if (!std::isfinite(base) || base <= 0.0) {
                base = continuum_fallback;
            }
            prominence[i] = good_psd[i] / base;
        }

        double total_power = 0.0;
        for (std::size_t k = 1; k < good_freq.size(); ++k) {
            const double df = good_freq[k] - good_freq[k - 1];
            total_power += 0.5 * (good_psd[k] + good_psd[k - 1]) * df;
        }

        for (std::size_t i = 1; i + 1 < good_freq.size(); ++i) {
            if (!std::isfinite(prominence[i]) ||
                prominence[i] < audit.detector_notch_min_prominence) {
                continue;
            }
            if (prominence[i] < prominence[i - 1] || prominence[i] < prominence[i + 1]) {
                continue;
            }
            const double target = 1.0 + 0.5 * std::max(prominence[i] - 1.0, 0.0);
            std::size_t j0 = i;
            while (j0 > 0 && prominence[j0 - 1] >= target) {
                --j0;
            }
            std::size_t j1 = i;
            while (j1 + 1 < good_freq.size() && prominence[j1 + 1] >= target) {
                ++j1;
            }
            const double min_bin_width =
                (good_freq.size() > 1) ? std::max(good_freq[1] - good_freq[0], 1.0e-6) : 1.0e-6;
            const double width_hz = std::max(good_freq[j1] - good_freq[j0], min_bin_width);
            double line_power = 0.0;
            auto continuum_at = [&](std::size_t k) {
                double base = continuum[std::min<std::size_t>(k, continuum.size() - 1)];
                if (!std::isfinite(base) || base <= 0.0) {
                    base = continuum_fallback;
                }
                return base;
            };
            if (j0 == j1) {
                const double df_left = (i > 0) ? (good_freq[i] - good_freq[i - 1]) : min_bin_width;
                const double df_right =
                    (i + 1 < good_freq.size()) ? (good_freq[i + 1] - good_freq[i]) : min_bin_width;
                const double df = std::max(0.5 * (df_left + df_right), min_bin_width);
                line_power = std::max(good_psd[i] - continuum_at(i), 0.0) * df;
            }
            else {
                for (std::size_t k = j0 + 1; k <= j1; ++k) {
                    const double df = good_freq[k] - good_freq[k - 1];
                    const double local0 = std::max(good_psd[k - 1] - continuum_at(k - 1), 0.0);
                    const double local1 = std::max(good_psd[k] - continuum_at(k), 0.0);
                    line_power += 0.5 * (local0 + local1) * df;
                }
            }

            DetectorPeak peak;
            peak.freq_hz = good_freq[i];
            peak.prominence = prominence[i];
            peak.width_hz = width_hz;
            peak.line_power_frac = (total_power > 0.0) ? (line_power / total_power) : nan;
            if (audit.detector_notch_min_line_power_frac > 0.0 &&
                (!std::isfinite(peak.line_power_frac) ||
                 peak.line_power_frac < audit.detector_notch_min_line_power_frac)) {
                continue;
            }
            peaks.push_back(peak);
        }
        std::sort(peaks.begin(), peaks.end(), [](const auto &a, const auto &b) {
            const double a_power = std::isfinite(a.line_power_frac) ? a.line_power_frac : -1.0;
            const double b_power = std::isfinite(b.line_power_frac) ? b.line_power_frac : -1.0;
            if (a_power != b_power) {
                return a_power > b_power;
            }
            if (a.prominence != b.prominence) {
                return a.prominence > b.prominence;
            }
            return a.freq_hz < b.freq_hz;
        });
        return peaks;
    };

    auto filter_column = [&](const Eigen::MatrixXd &data,
                             Eigen::Index det,
                             Filter &notch_filter,
                             Eigen::MatrixXd &filtered_column) {
        if (det < 0 || det >= data.cols() || data.rows() != n_pts) {
            return false;
        }
        if (!data.col(det).allFinite()) {
            return false;
        }
        filtered_column = data.col(det);
        notch_filter.iir(filtered_column);
        return filtered_column.allFinite();
    };

    std::vector<RTCDetectorDiagSummary> det_summary;
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto it = rtc_detector_summary_by_scan.find(scan_id);
        if (it != rtc_detector_summary_by_scan.end() &&
            it->second.size() == static_cast<std::size_t>(n_dets)) {
            det_summary = it->second;
        }
    }
    if (det_summary.size() != static_cast<std::size_t>(n_dets)) {
        det_summary.assign(static_cast<std::size_t>(n_dets), RTCDetectorDiagSummary{});
    }
    for (Eigen::Index det = 0; det < n_dets; ++det) {
        det_summary[static_cast<std::size_t>(det)].det = det;
    }

    Eigen::Index total_notches = 0;
    Eigen::Index touched_dets = 0;
    Eigen::Index max_notches_per_det = 0;
    std::vector<double> primary_freqs;

    for (Eigen::Index det = 0; det < n_dets; ++det) {
        auto &diag = det_summary[static_cast<std::size_t>(det)];
        diag.detector_notch_n_applied = 0;
        diag.detector_notch_primary_freq_hz = nan;
        diag.detector_notch_primary_width_hz = nan;
        diag.detector_notch_primary_prominence = nan;
        diag.detector_notch_primary_line_power_frac = nan;
        diag.detector_notch_rms_before = nan;
        diag.detector_notch_rms_after = nan;

        if (!in.scans.data.col(det).allFinite()) {
            continue;
        }
        if (has_kernel && !in.kernel.data.col(det).allFinite()) {
            continue;
        }

        Eigen::Array<bool, Eigen::Dynamic, 1> valid(n_pts);
        Eigen::Index n_valid = 0;
        for (Eigen::Index i = 0; i < n_pts; ++i) {
            valid(i) = std::isfinite(in.scans.data(i, det)) && !in.flags.data(i, det);
            if (valid(i)) {
                ++n_valid;
            }
        }
        const double good_frac =
            static_cast<double>(n_valid) / static_cast<double>(std::max<Eigen::Index>(n_pts, 1));
        if (!std::isfinite(good_frac) || good_frac < audit.min_good_frac) {
            continue;
        }

        Eigen::VectorXd signal = in.scans.data.col(det);
        const double center = robust_center(signal, valid);
        if (!std::isfinite(center)) {
            continue;
        }
        Eigen::VectorXd centered = Eigen::VectorXd::Zero(n_pts);
        for (Eigen::Index i = 0; i < n_pts; ++i) {
            if (valid(i) && std::isfinite(signal(i))) {
                centered(i) = signal(i) - center;
            }
        }

        auto psd = masked_welch_psd(centered, valid);
        if (psd.n_windows < audit.min_windows) {
            continue;
        }
        auto peaks = find_line_peaks(psd.freq_hz, psd.psd);
        if (peaks.empty()) {
            continue;
        }
        if (audit.detector_notch_max_notches > 0 &&
            static_cast<Eigen::Index>(peaks.size()) > audit.detector_notch_max_notches) {
            peaks.resize(static_cast<std::size_t>(audit.detector_notch_max_notches));
        }

        Filter detector_notch_filter;
        detector_notch_filter.notch_zero_phase = true;
        std::vector<DetectorPeak> applied_peaks;
        applied_peaks.reserve(peaks.size());
        for (auto peak : peaks) {
            if (!std::isfinite(peak.freq_hz) || peak.freq_hz <= 0.0 || peak.freq_hz >= nyquist_hz) {
                continue;
            }
            double width_hz = peak.width_hz;
            if (!std::isfinite(width_hz) || width_hz <= 0.0) {
                width_hz = audit.detector_notch_min_width_hz;
            }
            width_hz *= audit.detector_notch_width_scale;
            width_hz = std::max(width_hz, audit.detector_notch_min_width_hz);
            width_hz = std::min(width_hz, audit.detector_notch_max_width_hz);
            width_hz = std::min(width_hz, std::max(0.05, 0.5 * peak.freq_hz));
            if (!std::isfinite(width_hz) || width_hz <= 0.0) {
                continue;
            }
            peak.applied_width_hz = width_hz;
            detector_notch_filter.w0s.push_back(peak.freq_hz);
            detector_notch_filter.qs.push_back(peak.freq_hz / width_hz);
            applied_peaks.push_back(peak);
        }
        if (applied_peaks.empty()) {
            continue;
        }

        detector_notch_filter.make_notch_filter(fs_hz);
        Eigen::MatrixXd filtered_scan_col;
        Eigen::MatrixXd filtered_kernel_col;
        if (!filter_column(in.scans.data, det, detector_notch_filter, filtered_scan_col)) {
            continue;
        }
        if (has_kernel &&
            !filter_column(in.kernel.data, det, detector_notch_filter, filtered_kernel_col)) {
            logger->warn(
                "rtc_line_audit detector notch skipped for scan {} det {} because kernel filtering failed",
                scan_id + 1,
                det);
            continue;
        }

        diag.detector_notch_rms_before =
            robust_rms(in.scans.data.col(det), valid, diag_start_sample, diag_n_samples);
        in.scans.data.col(det) = filtered_scan_col.col(0);
        if (has_kernel) {
            in.kernel.data.col(det) = filtered_kernel_col.col(0);
        }
        diag.detector_notch_rms_after =
            robust_rms(in.scans.data.col(det), valid, diag_start_sample, diag_n_samples);

        diag.detector_notch_n_applied = static_cast<int>(applied_peaks.size());
        diag.detector_notch_primary_freq_hz = applied_peaks.front().freq_hz;
        diag.detector_notch_primary_width_hz = applied_peaks.front().applied_width_hz;
        diag.detector_notch_primary_prominence = applied_peaks.front().prominence;
        diag.detector_notch_primary_line_power_frac = applied_peaks.front().line_power_frac;
        total_notches += static_cast<Eigen::Index>(applied_peaks.size());
        max_notches_per_det =
            std::max<Eigen::Index>(max_notches_per_det, static_cast<Eigen::Index>(applied_peaks.size()));
        ++touched_dets;
        primary_freqs.push_back(applied_peaks.front().freq_hz);
    }

    if (total_notches > 0) {
        logger->info(
            "rtc_line_audit apply_detector_notches scan {}: dets={} total_notches={} max_notches_per_det={} primary_freq_median_hz={:.4f} zero_phase=true kernel_filtered={}",
            scan_id + 1,
            touched_dets,
            total_notches,
            max_notches_per_det,
            median_of(std::move(primary_freqs)),
            has_kernel);
    }

    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_detector_summary_by_scan[scan_id] = std::move(det_summary);
    }
    return total_notches;
}

template <typename calib_t>
void RTCProc::capture_rtc_diagnostics(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib,
                                      bool recompute_step_metrics,
                                      bool recompute_impulsive_metrics) {
    const Eigen::Index scan_id = in.index.data;
    const Eigen::Index n_pts = in.scans.data.rows();
    const Eigen::Index n_dets = in.scans.data.cols();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int fill_int = kTransientFillInt;

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
        TransientEvent event;
        event.kind = TransientEventKind::step_like;
        const Eigen::Index n = x.size();
        if (n < 16) {
            return event;
        }
        auto [center, scale] = robust_center_scale(x, valid);
        if (!std::isfinite(center) || !std::isfinite(scale) || scale <= 0.0) {
            return event;
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
            return event;
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
        if (best_idx != fill_int && std::isfinite(best)) {
            event.sample = best_idx;
            event.start_sample = static_cast<int>(std::max<Eigen::Index>(0, best_idx - w));
            event.end_sample = static_cast<int>(std::min<Eigen::Index>(n - 1, best_idx + w - 1));
            event.width_samples = static_cast<double>(event.end_sample - event.start_sample + 1);
            event.score = best;
            event.baseline_shift_z = best;
            event.accepted = true;
        }
        return event;
    };

    struct ImpulsiveMetrics {
        TransientEvent event;
        double peak_abs_z = std::numeric_limits<double>::quiet_NaN();
        int peak_abs_sample = kTransientFillInt;
        double peak_delta_abs_z = std::numeric_limits<double>::quiet_NaN();
        int peak_delta_abs_sample = kTransientFillInt;
        int near_abs_count = 0;
        int near_delta_count = 0;
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
            out.event.kind = use_delta ? TransientEventKind::delta_like : TransientEventKind::raw_like;
            out.event.sample = use_delta ? out.peak_delta_abs_sample : out.peak_abs_sample;
            out.event.start_sample = out.event.sample;
            out.event.end_sample = out.event.sample;
            out.event.width_samples = 1.0;
            out.event.score = use_delta ? out.peak_delta_abs_z : out.peak_abs_z;
            out.event.baseline_shift_z = 0.0;
            out.event.peak_abs_z = out.peak_abs_z;
            out.event.peak_delta_abs_z = out.peak_delta_abs_z;
            out.event.accepted = true;
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
    const double impulsive_cluster_tol_samples = std::max(
        1.0,
        ((impulsive_coincidence.cluster_tol_sec > 0.0)
             ? (impulsive_coincidence.cluster_tol_sec / dt_for_step)
             : 1.0));

    std::vector<RTCDetectorDiagSummary> det_summary;
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto det_it = rtc_detector_summary_by_scan.find(scan_id);
        if (det_it != rtc_detector_summary_by_scan.end()) {
            det_summary = det_it->second;
        }
    }
    std::vector<RTCNetworkDiagSummary> existing_nw_summary;
    std::map<Eigen::Index, std::vector<RTCImpulsiveSnippetSummary>> existing_impulsive_summary;
    bool have_network_summary = false;
    bool have_impulsive_summary = false;
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto nw_it = rtc_network_summary_by_scan.find(scan_id);
        if (nw_it != rtc_network_summary_by_scan.end()) {
            existing_nw_summary = nw_it->second;
            have_network_summary = true;
        }
        const auto imp_it = rtc_impulsive_summary_by_scan.find(scan_id);
        if (imp_it != rtc_impulsive_summary_by_scan.end()) {
            existing_impulsive_summary = imp_it->second;
            have_impulsive_summary = true;
        }
    }
    const bool have_detector_summary =
        det_summary.size() == static_cast<std::size_t>(n_dets);
    const bool recompute_detector_step_metrics = recompute_step_metrics || !have_detector_summary;
    const bool recompute_detector_impulsive_metrics =
        recompute_impulsive_metrics || !have_detector_summary;

    if (!have_detector_summary) {
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
        if (recompute_detector_impulsive_metrics) {
            const auto impulsive = impulsive_metric(in.scans.data.col(det), valid);
            row.impulsive_event = impulsive.event;
            row.impulsive_peak_abs_z = impulsive.peak_abs_z;
            row.impulsive_peak_abs_sample = impulsive.peak_abs_sample;
            row.impulsive_peak_delta_abs_z = impulsive.peak_delta_abs_z;
            row.impulsive_peak_delta_abs_sample = impulsive.peak_delta_abs_sample;
            row.impulsive_near_abs_count = impulsive.near_abs_count;
            row.impulsive_near_delta_count = impulsive.near_delta_count;
            row.impulsive_event_score = row.impulsive_event.score;
            row.impulsive_event_sample = row.impulsive_event.sample;
            row.impulsive_event_kind = row.impulsive_event.kind_code();
        }
        if (recompute_detector_step_metrics) {
            row.step_event = step_metric(in.scans.data.col(det), valid, step_window);
            row.step_score = row.step_event.score;
            row.step_sample = row.step_event.sample;
        }
    }
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_detector_summary_by_scan[scan_id] = det_summary;
    }

    if (impulsive_capture.enabled && !(have_impulsive_summary && !recompute_impulsive_metrics)) {
        const Eigen::Index snippet_pre = std::max<Eigen::Index>(
            0, static_cast<Eigen::Index>(std::llround(impulsive_capture.snippet_pre_window_sec /
                                                      std::max(dt_for_step, 1.0e-6))));
        const Eigen::Index snippet_post = std::max<Eigen::Index>(
            0, static_cast<Eigen::Index>(std::llround(impulsive_capture.snippet_post_window_sec /
                                                      std::max(dt_for_step, 1.0e-6))));
        const Eigen::Index snippet_len = snippet_pre + snippet_post + 1;
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
                if (!row.impulsive_event.valid() ||
                    row.impulsive_event.score < impulsive_capture.min_event_z) {
                    continue;
                }

                RTCImpulsiveSnippetSummary slot;
                slot.event = row.impulsive_event;
                slot.det = static_cast<int>(det);
                slot.event_sample = row.impulsive_event.sample;
                slot.event_kind = row.impulsive_event.kind_code();
                slot.event_score = row.impulsive_event.score;
                slot.peak_abs_z = row.impulsive_event.peak_abs_z;
                slot.peak_delta_abs_z = row.impulsive_event.peak_delta_abs_z;
                slot.added_flagged_frac = row.added_flagged_frac;
                slot.raw_exceed_count = row.raw_exceed_count;
                slot.local_raw_candidate_count = row.local_raw_candidate_count;
                slot.local_raw_accepted_event_count = row.local_raw_accepted_event_count;
                slot.local_flagged_sample_count = row.local_flagged_sample_count;
                slot.local_raw_reject_count = row.local_raw_reject_count;
                slot.delta_spike_count = row.delta_spike_count;
                slot.local_delta_candidate_count = row.local_delta_candidate_count;
                slot.local_delta_accepted_event_count = row.local_delta_accepted_event_count;
                slot.local_delta_reject_count = row.local_delta_reject_count;
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
                    const Eigen::Index sample = static_cast<Eigen::Index>(slot.event_sample) - snippet_pre + k;
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
        {
            std::lock_guard<std::mutex> lock(*diag_summary_mutex);
            rtc_impulsive_summary_by_scan[scan_id] = std::move(impulsive_by_network);
        }
    }
    else if (!impulsive_capture.enabled) {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_impulsive_summary_by_scan.erase(scan_id);
    }

    std::map<Eigen::Index, RTCNetworkDiagSummary> prev_nw_summary;
    if (have_network_summary) {
        for (const auto &row : existing_nw_summary) {
            prev_nw_summary[row.nw] = row;
        }
    }

    std::vector<RTCNetworkDiagSummary> nw_summary;
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
        const auto prev_it = prev_nw_summary.find(nw);
        if (prev_it != prev_nw_summary.end()) {
            row = prev_it->second;
        }
        row.nw = nw;
        row.n_det_input = end - start;
        row.n_det_used = 0;
        row.impulsive_n_det_used = 0;
        row.cm_low_mid_ratio = nan;
        row.cm_peak_freq_Hz = nan;
        row.cm_peak_prominence = nan;

        Eigen::MatrixXd centered = Eigen::MatrixXd::Zero(n_pts, std::max<Eigen::Index>(end - start, 0));
        Eigen::Index n_step_used = 0;
        Eigen::Index n_impulsive_used = 0;
        Eigen::Index n_centered_used = 0;
        std::vector<double> step_scores;
        std::vector<double> step_samples_active;
        std::vector<double> impulsive_scores;
        std::vector<double> impulsive_samples_active;
        step_scores.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(end - start, 0)));
        step_samples_active.reserve(step_scores.capacity());
        impulsive_scores.reserve(step_scores.capacity());
        impulsive_samples_active.reserve(step_scores.capacity());

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
            const bool use_for_step = good_frac >= network_step_mask.min_good_frac;
            const bool use_for_impulsive = good_frac >= impulsive_coincidence.min_good_frac;
            if (!use_for_step && !use_for_impulsive) {
                continue;
            }
            auto [center, scale] = robust_center_scale(in.scans.data.col(det), valid);
            if (!std::isfinite(center) || !std::isfinite(scale) || scale <= 0.0) {
                continue;
            }
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (valid(i) && std::isfinite(in.scans.data(i, det))) {
                    centered(i, n_centered_used) = in.scans.data(i, det) - center;
                }
            }
            ++n_centered_used;
            const auto &det_row = det_summary[static_cast<std::size_t>(det)];
            if (use_for_step) {
                ++n_step_used;
            }
            if (use_for_step && det_row.step_event.valid()) {
                step_scores.push_back(det_row.step_event.score);
                if (det_row.step_event.score >= step_score_thresh &&
                    det_row.step_event.sample != fill_int) {
                    step_samples_active.push_back(static_cast<double>(det_row.step_event.sample));
                }
            }
            if (use_for_impulsive) {
                ++n_impulsive_used;
            }
            if (use_for_impulsive && det_row.impulsive_event.valid()) {
                impulsive_scores.push_back(det_row.impulsive_event.score);
                if (det_row.impulsive_event.score >= impulsive_coincidence.event_score_thresh &&
                    det_row.impulsive_event.sample != fill_int) {
                    impulsive_samples_active.push_back(
                        static_cast<double>(det_row.impulsive_event.sample));
                }
            }
        }

        row.n_det_used = n_step_used;
        row.impulsive_n_det_used = n_impulsive_used;
        if (recompute_step_metrics || prev_it == prev_nw_summary.end()) {
            row.median_step_score = nan;
            row.max_step_score = nan;
            row.step_det_frac = nan;
            row.step_alignment_frac = nan;
            row.dominant_step_sample = fill_int;
            row.step_event = {};
            row.step_event.kind = TransientEventKind::step_like;
            if (!step_scores.empty()) {
                row.median_step_score = median_of(step_scores);
                row.max_step_score = *std::max_element(step_scores.begin(), step_scores.end());
                const auto n_active = static_cast<double>(step_samples_active.size());
                row.step_det_frac = n_active / static_cast<double>(step_scores.size());
                auto [step_center, step_align] = dominant_cluster(step_samples_active, cluster_tol_samples);
                row.step_alignment_frac = step_align;
                if (std::isfinite(step_center)) {
                    row.dominant_step_sample = static_cast<int>(std::llround(step_center));
                    row.step_event.sample = row.dominant_step_sample;
                    row.step_event.start_sample = static_cast<int>(
                        std::max<Eigen::Index>(0,
                                               static_cast<Eigen::Index>(row.dominant_step_sample) -
                                                   static_cast<Eigen::Index>(std::llround(cluster_tol_samples))));
                    row.step_event.end_sample = static_cast<int>(
                        std::min<Eigen::Index>(n_pts - 1,
                                               static_cast<Eigen::Index>(row.dominant_step_sample) +
                                                   static_cast<Eigen::Index>(std::llround(cluster_tol_samples))));
                    row.step_event.width_samples =
                        static_cast<double>(row.step_event.end_sample - row.step_event.start_sample + 1);
                    row.step_event.score = row.max_step_score;
                    row.step_event.baseline_shift_z = row.max_step_score;
                    row.step_event.accepted = true;
                }
            }
        }
        if (recompute_impulsive_metrics || prev_it == prev_nw_summary.end()) {
            row.median_impulsive_score = nan;
            row.max_impulsive_score = nan;
            row.impulsive_det_frac = nan;
            row.impulsive_alignment_frac = nan;
            row.dominant_impulsive_sample = fill_int;
            if (n_impulsive_used >= impulsive_coincidence.min_det_used && !impulsive_scores.empty()) {
                row.median_impulsive_score = median_of(impulsive_scores);
                row.max_impulsive_score = *std::max_element(impulsive_scores.begin(), impulsive_scores.end());
                const auto n_active = static_cast<double>(impulsive_samples_active.size());
                row.impulsive_det_frac = n_active / static_cast<double>(impulsive_scores.size());
                auto [imp_center, imp_align] = dominant_cluster(impulsive_samples_active, impulsive_cluster_tol_samples);
                row.impulsive_alignment_frac = imp_align;
                if (std::isfinite(imp_center)) {
                    row.dominant_impulsive_sample = static_cast<int>(std::llround(imp_center));
                }
            }
        }

        if (n_centered_used >= 1 && n_pts >= 16 && std::isfinite(fs_hz) && fs_hz > 0.0) {
            centered.conservativeResize(Eigen::NoChange, n_centered_used);
            Eigen::VectorXd cm(n_pts);
            std::vector<double> scratch;
            scratch.reserve(static_cast<std::size_t>(n_centered_used));
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                scratch.clear();
                for (Eigen::Index j = 0; j < n_centered_used; ++j) {
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
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_network_summary_by_scan[scan_id] = std::move(nw_summary);
    }
}

template <typename calib_t>
void RTCProc::apply_network_step_mask(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib) {
    if (!network_step_mask.enabled) {
        return;
    }
    const auto scan_id = in.index.data;
    std::vector<RTCNetworkDiagSummary> nw_summary;
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto nw_it = rtc_network_summary_by_scan.find(scan_id);
        if (nw_it == rtc_network_summary_by_scan.end()) {
            return;
        }
        nw_summary = nw_it->second;
    }
    if (nw_summary.empty()) {
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

    for (auto &row : nw_summary) {
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
            !row.step_event.valid()) {
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
        const Eigen::Index center = static_cast<Eigen::Index>(row.step_event.sample);
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
                "network_step_mask rejected for scan {} nw {}: dominant_sample={} window_samples={} proposed_fraction={:.4f} exceeds max_flagged_fraction={:.4f}",
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
            "network_step_mask applied for scan {} nw {}: dominant_sample={} window=[{}, {}] n_det_masked={} newly_flagged={} flagged_fraction={:.4f}",
            scan_id + 1,
            row.nw,
            row.dominant_step_sample,
            start_sample,
            end_sample,
            end_det - start_det,
            newly_flagged,
            flagged_fraction);
    }
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_network_summary_by_scan[scan_id] = std::move(nw_summary);
    }
}

template <typename calib_t>
void RTCProc::apply_impulsive_coincidence_mask(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                                               calib_t &calib) {
    if (!impulsive_coincidence.enabled) {
        return;
    }
    const auto scan_id = in.index.data;
    std::vector<RTCNetworkDiagSummary> nw_summary;
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto nw_it = rtc_network_summary_by_scan.find(scan_id);
        if (nw_it == rtc_network_summary_by_scan.end()) {
            return;
        }
        nw_summary = nw_it->second;
    }
    if (nw_summary.empty()) {
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
        (impulsive_coincidence.mask_pre_window_sec > 0.0 ||
         impulsive_coincidence.mask_post_window_sec > 0.0 ||
         impulsive_coincidence.cluster_tol_sec > 0.0)
            ? infer_dt_sec()
            : 1.0;
    const double cluster_tol_samples = std::max(
        1.0,
        ((impulsive_coincidence.cluster_tol_sec > 0.0)
             ? (impulsive_coincidence.cluster_tol_sec / std::max(dt_sec, 1.0e-6))
             : 1.0));
    const Eigen::Index n_pts = in.scans.data.rows();
    const auto grp_limits = get_grouping("nw", calib, in.scans.data.cols());
    const auto det_summary = snapshot_detector_diag_summary(scan_id);
    const bool have_detector_summary =
        det_summary.size() == static_cast<std::size_t>(in.scans.data.cols());

    struct CoincidenceCandidate {
        RTCNetworkDiagSummary *row = nullptr;
        Eigen::Index start_det = 0;
        Eigen::Index end_det = 0;
        Eigen::Index center_sample = 0;
        double max_score = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index total_active_count = 0;
        Eigen::Index cluster_active_count = 0;
        bool local_trigger = false;
        bool cross_network_trigger = false;
        bool high_score_override_trigger = false;
        Eigen::Index cluster_center_sample = 0;
        Eigen::Index cluster_network_count = 0;
        double cluster_peak_score = std::numeric_limits<double>::quiet_NaN();
        double network_max_score = std::numeric_limits<double>::quiet_NaN();
        double override_score = std::numeric_limits<double>::quiet_NaN();
        bool override_uses_network_peak = false;
    };

    std::vector<CoincidenceCandidate> candidates;
    candidates.reserve(nw_summary.size());

    for (auto &row : nw_summary) {
        row.impulsive_mask_applied = false;
        row.impulsive_mask_start_sample = kTransientFillInt;
        row.impulsive_mask_end_sample = kTransientFillInt;
        row.impulsive_mask_window_samples = 0;
        row.impulsive_mask_n_det_masked = 0;
        row.impulsive_mask_n_det_samples_flagged = 0;
        row.impulsive_mask_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        row.impulsive_mask_candidate_available = false;
        row.impulsive_mask_local_trigger = false;
        row.impulsive_mask_cross_network_trigger = false;
        row.impulsive_mask_high_score_override_trigger = false;
        row.impulsive_mask_rejected_max_fraction = false;
        row.impulsive_mask_candidate_center_sample = kTransientFillInt;
        row.impulsive_mask_cluster_center_sample = kTransientFillInt;
        row.impulsive_mask_cluster_network_count = 0;
        row.impulsive_mask_cluster_active_count = 0;
        row.impulsive_mask_total_active_count = 0;
        row.impulsive_mask_cluster_peak_score = std::numeric_limits<double>::quiet_NaN();
        row.impulsive_mask_override_score = std::numeric_limits<double>::quiet_NaN();
        row.impulsive_mask_override_uses_network_peak = false;
        row.impulsive_mask_proposed_flagged_fraction = std::numeric_limits<double>::quiet_NaN();

        const auto grp_it = grp_limits.find(row.nw);
        if (grp_it == grp_limits.end()) {
            continue;
        }

        const auto start_det = std::get<0>(grp_it->second);
        const auto end_det = std::get<1>(grp_it->second);
        const Eigen::Index impulsive_det_used =
            row.impulsive_n_det_used > 0 ? row.impulsive_n_det_used : row.n_det_used;

        if (have_detector_summary) {
            std::vector<std::pair<Eigen::Index, double>> active_events;
            active_events.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(end_det - start_det, 0)));
            for (Eigen::Index det = start_det; det < end_det; ++det) {
                const auto &det_row = det_summary[static_cast<std::size_t>(det)];
                const double good_frac = 1.0 - det_row.final_flagged_frac;
                if (!std::isfinite(good_frac) || good_frac < impulsive_coincidence.min_good_frac) {
                    continue;
                }
                if (det_row.impulsive_event_sample == kTransientFillInt ||
                    !std::isfinite(det_row.impulsive_event_score) ||
                    det_row.impulsive_event_score < impulsive_coincidence.event_score_thresh) {
                    continue;
                }
                active_events.emplace_back(
                    static_cast<Eigen::Index>(det_row.impulsive_event_sample),
                    det_row.impulsive_event_score);
            }
            if (impulsive_det_used < impulsive_coincidence.min_det_used || active_events.empty()) {
                continue;
            }

            std::sort(active_events.begin(), active_events.end(),
                      [](const auto &a, const auto &b) {
                          if (a.first != b.first) {
                              return a.first < b.first;
                          }
                          return a.second > b.second;
                      });

            const Eigen::Index total_active_count =
                static_cast<Eigen::Index>(active_events.size());
            const double det_frac =
                static_cast<double>(total_active_count) /
                static_cast<double>(std::max<Eigen::Index>(impulsive_det_used, 1));
            double network_max_score = std::numeric_limits<double>::quiet_NaN();
            for (const auto &event : active_events) {
                network_max_score = std::isfinite(network_max_score)
                    ? std::max(network_max_score, event.second)
                    : event.second;
            }

            for (std::size_t i = 0; i < active_events.size();) {
                std::size_t j = i;
                double cluster_max_score = active_events[i].second;
                while (j + 1 < active_events.size() &&
                       static_cast<double>(active_events[j + 1].first - active_events[i].first) <= cluster_tol_samples) {
                    ++j;
                    cluster_max_score = std::max(cluster_max_score, active_events[j].second);
                }

                std::vector<Eigen::Index> cluster_samples;
                cluster_samples.reserve(j - i + 1);
                for (std::size_t k = i; k <= j; ++k) {
                    cluster_samples.push_back(active_events[k].first);
                }
                const auto mid = cluster_samples.begin() +
                                 static_cast<std::ptrdiff_t>(cluster_samples.size() / 2);
                std::nth_element(cluster_samples.begin(), mid, cluster_samples.end());
                const Eigen::Index center_sample = *mid;
                const Eigen::Index cluster_active_count =
                    static_cast<Eigen::Index>(j - i + 1);
                const double align_frac =
                    static_cast<double>(cluster_active_count) /
                    static_cast<double>(std::max<Eigen::Index>(total_active_count, 1));

                CoincidenceCandidate cand;
                cand.row = &row;
                cand.start_det = start_det;
                cand.end_det = end_det;
                cand.center_sample = center_sample;
                cand.max_score = cluster_max_score;
                cand.total_active_count = total_active_count;
                cand.cluster_active_count = cluster_active_count;
                cand.network_max_score = network_max_score;
                cand.local_trigger =
                    det_frac >= impulsive_coincidence.min_impulsive_det_frac &&
                    align_frac >= impulsive_coincidence.min_alignment_frac;
                candidates.push_back(cand);

                i = j + 1;
            }
            continue;
        }

        if (impulsive_det_used < impulsive_coincidence.min_det_used ||
            row.dominant_impulsive_sample == kTransientFillInt ||
            !std::isfinite(row.max_impulsive_score) ||
            row.max_impulsive_score < impulsive_coincidence.event_score_thresh) {
            continue;
        }

        CoincidenceCandidate cand;
        cand.row = &row;
        cand.start_det = start_det;
        cand.end_det = end_det;
        cand.center_sample = static_cast<Eigen::Index>(row.dominant_impulsive_sample);
        cand.max_score = row.max_impulsive_score;
        cand.total_active_count = 0;
        cand.cluster_active_count = 0;
        cand.network_max_score = row.max_impulsive_score;
        cand.local_trigger =
            std::isfinite(row.impulsive_det_frac) &&
            std::isfinite(row.impulsive_alignment_frac) &&
            row.impulsive_det_frac >= impulsive_coincidence.min_impulsive_det_frac &&
            row.impulsive_alignment_frac >= impulsive_coincidence.min_alignment_frac;
        candidates.push_back(cand);
    }

    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (candidates[a].center_sample != candidates[b].center_sample) {
            return candidates[a].center_sample < candidates[b].center_sample;
        }
        return candidates[a].row->nw < candidates[b].row->nw;
    });

    for (std::size_t i = 0; i < order.size();) {
        std::size_t j = i;
        while (j + 1 < order.size() &&
               static_cast<double>(candidates[order[j + 1]].center_sample -
                                   candidates[order[i]].center_sample) <= cluster_tol_samples) {
            ++j;
        }
        std::vector<Eigen::Index> cluster_networks;
        cluster_networks.reserve(static_cast<std::size_t>(j - i + 1));
        std::vector<Eigen::Index> cluster_samples;
        cluster_samples.reserve(static_cast<std::size_t>(j - i + 1));
        for (std::size_t k = i; k <= j; ++k) {
            cluster_networks.push_back(candidates[order[k]].row->nw);
            cluster_samples.push_back(candidates[order[k]].center_sample);
        }
        std::sort(cluster_networks.begin(), cluster_networks.end());
        cluster_networks.erase(std::unique(cluster_networks.begin(), cluster_networks.end()),
                               cluster_networks.end());
        const Eigen::Index cluster_network_count =
            static_cast<Eigen::Index>(cluster_networks.size());
        const auto mid = cluster_samples.begin() +
                         static_cast<std::ptrdiff_t>(cluster_samples.size() / 2);
        std::nth_element(cluster_samples.begin(), mid, cluster_samples.end());
        const Eigen::Index cluster_center_sample = *mid;
        double cluster_peak_score = std::numeric_limits<double>::quiet_NaN();
        double cluster_network_max_score = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t k = i; k <= j; ++k) {
            cluster_peak_score = std::isfinite(cluster_peak_score)
                ? std::max(cluster_peak_score, candidates[order[k]].max_score)
                : candidates[order[k]].max_score;
            cluster_network_max_score = std::isfinite(cluster_network_max_score)
                ? std::max(cluster_network_max_score, candidates[order[k]].network_max_score)
                : candidates[order[k]].network_max_score;
        }
        double override_score = cluster_peak_score;
        bool override_uses_network_peak = false;
        if (std::isfinite(cluster_network_max_score) &&
            (!std::isfinite(override_score) || cluster_network_max_score > override_score)) {
            override_score = cluster_network_max_score;
            override_uses_network_peak = true;
        }
        for (std::size_t k = i; k <= j; ++k) {
            auto &cand = candidates[order[k]];
            cand.cluster_center_sample = cluster_center_sample;
            cand.cluster_network_count = cluster_network_count;
            cand.cluster_peak_score = cluster_peak_score;
            cand.override_score = override_score;
            cand.override_uses_network_peak = override_uses_network_peak;
        }
        if (cluster_network_count >= impulsive_coincidence.min_networks_aligned) {
            for (std::size_t k = i; k <= j; ++k) {
                auto &cand = candidates[order[k]];
                cand.cross_network_trigger = true;
            }
        }
        else if (impulsive_coincidence.high_score_min_networks_aligned > 0 &&
                 impulsive_coincidence.high_score_override_thresh > 0.0 &&
                 cluster_network_count >= impulsive_coincidence.high_score_min_networks_aligned) {
            if (std::isfinite(override_score) &&
                override_score >= impulsive_coincidence.high_score_override_thresh) {
                for (std::size_t k = i; k <= j; ++k) {
                    auto &cand = candidates[order[k]];
                    cand.cross_network_trigger = true;
                    cand.high_score_override_trigger = true;
                }
            }
        }
        i = j + 1;
    }

    auto better_candidate = [](const CoincidenceCandidate &a, const CoincidenceCandidate &b) {
        if (a.cross_network_trigger != b.cross_network_trigger) {
            return a.cross_network_trigger && !b.cross_network_trigger;
        }
        if (a.high_score_override_trigger != b.high_score_override_trigger) {
            return a.high_score_override_trigger && !b.high_score_override_trigger;
        }
        if (a.cluster_network_count != b.cluster_network_count) {
            return a.cluster_network_count > b.cluster_network_count;
        }
        if (a.local_trigger != b.local_trigger) {
            return a.local_trigger && !b.local_trigger;
        }
        if (a.cluster_active_count != b.cluster_active_count) {
            return a.cluster_active_count > b.cluster_active_count;
        }
        if (std::isfinite(a.max_score) != std::isfinite(b.max_score)) {
            return std::isfinite(a.max_score);
        }
        if (std::isfinite(a.max_score) && std::isfinite(b.max_score) && a.max_score != b.max_score) {
            return a.max_score > b.max_score;
        }
        if (a.total_active_count != b.total_active_count) {
            return a.total_active_count > b.total_active_count;
        }
        return a.center_sample < b.center_sample;
    };

    std::map<Eigen::Index, std::size_t> best_candidate_by_network;
    for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
        const auto &cand = candidates[idx];
        const auto nw = cand.row->nw;
        const auto it = best_candidate_by_network.find(nw);
        if (it == best_candidate_by_network.end() ||
            better_candidate(cand, candidates[it->second])) {
            best_candidate_by_network[nw] = idx;
        }
    }

    for (const auto &[nw, idx] : best_candidate_by_network) {
        auto &cand = candidates[idx];
        auto &row = *cand.row;

        row.impulsive_mask_candidate_available = true;
        row.impulsive_mask_local_trigger = cand.local_trigger;
        row.impulsive_mask_cross_network_trigger = cand.cross_network_trigger;
        row.impulsive_mask_high_score_override_trigger = cand.high_score_override_trigger;
        row.impulsive_mask_rejected_max_fraction = false;
        row.impulsive_mask_candidate_center_sample = static_cast<int>(cand.center_sample);
        row.impulsive_mask_cluster_center_sample =
            (cand.cluster_network_count > 0)
                ? static_cast<int>(cand.cluster_center_sample)
                : kTransientFillInt;
        row.impulsive_mask_cluster_network_count = static_cast<int>(cand.cluster_network_count);
        row.impulsive_mask_cluster_active_count = static_cast<int>(cand.cluster_active_count);
        row.impulsive_mask_total_active_count = static_cast<int>(cand.total_active_count);
        row.impulsive_mask_cluster_peak_score = cand.cluster_peak_score;
        row.impulsive_mask_override_score = cand.override_score;
        row.impulsive_mask_override_uses_network_peak = cand.override_uses_network_peak;

        const Eigen::Index pre_width = std::max<Eigen::Index>(
            0, static_cast<Eigen::Index>(std::llround(impulsive_coincidence.mask_pre_window_sec /
                                                      std::max(dt_sec, 1.0e-6))));
        const Eigen::Index post_width = std::max<Eigen::Index>(
            0, static_cast<Eigen::Index>(std::llround(impulsive_coincidence.mask_post_window_sec /
                                                      std::max(dt_sec, 1.0e-6))));
        const Eigen::Index center =
            cand.cross_network_trigger ? cand.cluster_center_sample : cand.center_sample;
        const Eigen::Index start_sample = std::max<Eigen::Index>(0, center - pre_width);
        const Eigen::Index end_sample = std::min<Eigen::Index>(n_pts - 1, center + post_width);
        const Eigen::Index window_samples = std::max<Eigen::Index>(0, end_sample - start_sample + 1);
        if (window_samples <= 0 || cand.end_det <= cand.start_det) {
            continue;
        }

        if (!(cand.local_trigger || cand.cross_network_trigger)) {
            continue;
        }

        Eigen::Index good_detector_samples = 0;
        Eigen::Index newly_flagged = 0;
        for (Eigen::Index det = cand.start_det; det < cand.end_det; ++det) {
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
        row.impulsive_mask_proposed_flagged_fraction = flagged_fraction;
        if (impulsive_coincidence.max_flagged_fraction > 0.0 &&
            flagged_fraction > impulsive_coincidence.max_flagged_fraction) {
            row.impulsive_mask_rejected_max_fraction = true;
            logger->info(
                "impulsive_coincidence_mask rejected for scan {} nw {}: dominant_sample={} center_sample={} local_trigger={} cross_network_trigger={} high_score_override_trigger={} cluster_networks={} cluster_peak_score={:.4g} override_score={:.4g} override_network_peak={} cluster_active={} total_active={} window_samples={} proposed_fraction={:.4f} exceeds max_flagged_fraction={:.4f}",
                scan_id + 1,
                row.nw,
                row.dominant_impulsive_sample,
                center,
                cand.local_trigger,
                cand.cross_network_trigger,
                cand.high_score_override_trigger,
                cand.cluster_network_count,
                cand.cluster_peak_score,
                cand.override_score,
                cand.override_uses_network_peak,
                cand.cluster_active_count,
                cand.total_active_count,
                window_samples,
                flagged_fraction,
                impulsive_coincidence.max_flagged_fraction);
            continue;
        }

        in.flags.data.block(start_sample, cand.start_det, window_samples,
                            cand.end_det - cand.start_det).setOnes();
        row.impulsive_mask_applied = true;
        row.impulsive_mask_start_sample = static_cast<int>(start_sample);
        row.impulsive_mask_end_sample = static_cast<int>(end_sample);
        row.impulsive_mask_window_samples = static_cast<int>(window_samples);
        row.impulsive_mask_n_det_masked = static_cast<int>(cand.end_det - cand.start_det);
        row.impulsive_mask_n_det_samples_flagged = static_cast<int>(newly_flagged);
        row.impulsive_mask_flagged_fraction = flagged_fraction;

        logger->info(
            "impulsive_coincidence_mask applied for scan {} nw {}: dominant_sample={} center_sample={} local_trigger={} cross_network_trigger={} high_score_override_trigger={} cluster_networks={} cluster_peak_score={:.4g} override_score={:.4g} override_network_peak={} cluster_active={} total_active={} window=[{}, {}] n_det_masked={} newly_flagged={} flagged_fraction={:.4f}",
            scan_id + 1,
            nw,
            row.dominant_impulsive_sample,
            center,
            cand.local_trigger,
            cand.cross_network_trigger,
            cand.high_score_override_trigger,
            cand.cluster_network_count,
            cand.cluster_peak_score,
            cand.override_score,
            cand.override_uses_network_peak,
            cand.cluster_active_count,
            cand.total_active_count,
            start_sample,
            end_sample,
            cand.end_det - cand.start_det,
            newly_flagged,
            flagged_fraction);
    }
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_network_summary_by_scan[scan_id] = std::move(nw_summary);
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
            in.flags.data.col(i).setOnes();
            if (citlali::config::is_detector_map_grouping(map_grouping)) {
                calib_scan.apt["flag"](det_index) = 1;
            }
        }
    }

    logger->info("removed {}/{} ({:.2f}%) unflagged tones closer than {:.4g} kHz", n_nearby_tones, n_dets,
                (static_cast<double>(n_nearby_tones)/static_cast<double>(n_dets))*100.0, delta_f_min_Hz/1000.0);

    // set up scan calib
    calib_scan.setup();

    return std::move(calib_scan);
}

template <typename calib_t>
void RTCProc::write_cached_diagnostics_to_netcdf(netCDF::NcFile &fo,
                                                 TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                                                 calib_t &calib,
                                                 Eigen::Index scan_row_index) {
    using netCDF::NcDim;
    using netCDF::NcVar;

    const int fill_int = -2147483647;
    const double fill_double = std::numeric_limits<double>::quiet_NaN();
    const auto scan_row = static_cast<unsigned long>((scan_row_index >= 0) ? scan_row_index : in.index.data);

    const auto det_diag = snapshot_detector_diag_summary(in.index.data);
    std::vector<RTCNetworkDiagSummary> nw_diag;
    std::map<Eigen::Index, std::vector<RTCImpulsiveSnippetSummary>> impulsive_diag;
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        const auto nw_diag_it = rtc_network_summary_by_scan.find(in.index.data);
        if (nw_diag_it != rtc_network_summary_by_scan.end()) {
            nw_diag = nw_diag_it->second;
        }
        const auto impulsive_it = rtc_impulsive_summary_by_scan.find(in.index.data);
        if (impulsive_it != rtc_impulsive_summary_by_scan.end()) {
            impulsive_diag = impulsive_it->second;
        }
    }
    std::vector<RemoveBadDetsWindowDiagSummary> window_diag;
    {
        std::lock_guard<std::mutex> lock(*diag_cache_mutex);
        const auto window_diag_it = remove_bad_dets_window_summary_by_scan.find(in.index.data);
        if (window_diag_it != remove_bad_dets_window_summary_by_scan.end()) {
            window_diag = window_diag_it->second;
        }
    }

    NcDim n_dets_dim = fo.getDim("n_dets");
    if (!n_dets_dim.isNull()) {
        const auto n_dets = n_dets_dim.getSize();
        std::vector<std::size_t> start_scan_det = {scan_row, 0};
        std::vector<std::size_t> size_scan_det = {1, n_dets};

        auto det_double_values = [&](auto getter) {
            std::vector<double> values(n_dets, fill_double);
            if (!det_diag.empty()) {
                const auto n_copy = std::min<std::size_t>(n_dets, det_diag.size());
                for (std::size_t i = 0; i < n_copy; ++i) {
                    values[i] = getter(det_diag[i]);
                }
            }
            return values;
        };
        auto det_int_values = [&](auto getter) {
            std::vector<int> values(n_dets, fill_int);
            if (!det_diag.empty()) {
                const auto n_copy = std::min<std::size_t>(n_dets, det_diag.size());
                for (std::size_t i = 0; i < n_copy; ++i) {
                    values[i] = getter(det_diag[i]);
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
        auto window_double_values = [&](auto getter) {
            std::vector<double> values(n_dets, fill_double);
            if (!window_diag.empty()) {
                const auto n_copy = std::min<std::size_t>(n_dets, window_diag.size());
                for (std::size_t i = 0; i < n_copy; ++i) {
                    values[i] = getter(window_diag[i]);
                }
            }
            return values;
        };
        auto window_int_values = [&](auto getter) {
            std::vector<int> values(n_dets, fill_int);
            if (!window_diag.empty()) {
                const auto n_copy = std::min<std::size_t>(n_dets, window_diag.size());
                for (std::size_t i = 0; i < n_copy; ++i) {
                    values[i] = getter(window_diag[i]);
                }
            }
            return values;
        };
        auto write_window_double = [&](const std::string &name, auto getter) {
            NcVar v = fo.getVar(name);
            if (!v.isNull()) {
                auto values = window_double_values(getter);
                v.putVar(start_scan_det, size_scan_det, values.data());
            }
        };
        auto write_window_int = [&](const std::string &name, auto getter) {
            NcVar v = fo.getVar(name);
            if (!v.isNull()) {
                auto values = window_int_values(getter);
                v.putVar(start_scan_det, size_scan_det, values.data());
            }
        };

        write_det_int("rtc_despike_raw_exceed_count",
                      [](const auto &row) { return row.raw_exceed_count; });
        write_det_int("rtc_despike_local_raw_candidate_count",
                      [](const auto &row) { return row.local_raw_candidate_count; });
        write_det_int("rtc_despike_local_raw_accepted_event_count",
                      [](const auto &row) { return row.local_raw_accepted_event_count; });
        write_det_int("rtc_despike_local_flagged_sample_count",
                      [](const auto &row) { return row.local_flagged_sample_count; });
        write_det_int("rtc_despike_local_exceed_count",
                      [](const auto &row) { return row.local_flagged_sample_count; });
        write_det_int("rtc_despike_local_raw_reject_count",
                      [](const auto &row) { return row.local_raw_reject_count; });
        write_det_int("rtc_despike_delta_spike_count",
                      [](const auto &row) { return row.delta_spike_count; });
        write_det_int("rtc_despike_local_delta_candidate_count",
                      [](const auto &row) { return row.local_delta_candidate_count; });
        write_det_int("rtc_despike_local_delta_accepted_event_count",
                      [](const auto &row) { return row.local_delta_accepted_event_count; });
        write_det_int("rtc_despike_local_delta_exceed_count",
                      [](const auto &row) { return row.local_delta_accepted_event_count; });
        write_det_int("rtc_despike_local_delta_reject_count",
                      [](const auto &row) { return row.local_delta_reject_count; });
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
        write_det_int("rtc_detector_notch_n_applied",
                      [](const auto &row) { return row.detector_notch_n_applied; });
        write_det_double("rtc_detector_notch_primary_freq_hz",
                         [](const auto &row) { return row.detector_notch_primary_freq_hz; });
        write_det_double("rtc_detector_notch_primary_width_hz",
                         [](const auto &row) { return row.detector_notch_primary_width_hz; });
        write_det_double("rtc_detector_notch_primary_prominence",
                         [](const auto &row) { return row.detector_notch_primary_prominence; });
        write_det_double("rtc_detector_notch_primary_line_power_frac",
                         [](const auto &row) { return row.detector_notch_primary_line_power_frac; });
        write_det_double("rtc_detector_notch_rms_before",
                         [](const auto &row) { return row.detector_notch_rms_before; });
        write_det_double("rtc_detector_notch_rms_after",
                         [](const auto &row) { return row.detector_notch_rms_after; });
        write_window_int("rtc_invvar_window_n_total",
                         [](const auto &row) { return row.n_total_windows; });
        write_window_int("rtc_invvar_window_n_valid",
                         [](const auto &row) { return row.n_valid_windows; });
        write_window_double("rtc_invvar_window_valid_fraction",
                            [](const auto &row) { return row.valid_window_fraction; });
        write_window_double("rtc_invvar_window_median",
                            [](const auto &row) { return row.inv_var_median; });
        write_window_double("rtc_invvar_window_q10",
                            [](const auto &row) { return row.inv_var_q10; });
        write_window_double("rtc_invvar_window_q90",
                            [](const auto &row) { return row.inv_var_q90; });
        write_window_double("rtc_invvar_window_flagged_frac_median",
                            [](const auto &row) { return row.flagged_frac_median; });
        write_window_double("rtc_invvar_window_flagged_frac_max",
                            [](const auto &row) { return row.flagged_frac_max; });
        write_window_double("rtc_invvar_window_heavy_flagged_fraction",
                            [](const auto &row) { return row.heavily_flagged_window_fraction; });
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
                if (!nw_diag.empty()) {
                    for (const auto &row : nw_diag) {
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
                if (!nw_diag.empty()) {
                    for (const auto &row : nw_diag) {
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
            write_nw_int("rtc_network_impulsive_n_det_used",
                         [](const auto &row) { return static_cast<int>(row.impulsive_n_det_used); });
            auto legacy_line_audit_diag = [](const auto &row) {
                RTCLineAuditDiagSummary diag;
                diag.n_det_used = row.line_audit_n_det_used;
                diag.shared_freq_hz = row.line_audit_shared_freq_hz;
                diag.shared_detector_count = row.line_audit_shared_detector_count;
                diag.shared_detector_frac = row.line_audit_shared_detector_frac;
                diag.shared_median_prominence = row.line_audit_shared_median_prominence;
                diag.shared_max_prominence = row.line_audit_shared_max_prominence;
                diag.shared_width_hz = row.line_audit_shared_width_hz;
                diag.shared_line_power_frac = row.line_audit_shared_line_power_frac;
                diag.shared_common_mode_freq_hz = row.line_audit_shared_common_mode_freq_hz;
                diag.shared_common_mode_prominence = row.line_audit_shared_common_mode_prominence;
                diag.shared_notch_score = row.line_audit_shared_notch_score;
                diag.shared_recommend_notch = row.line_audit_shared_recommend_notch;
                diag.n_applied_notches = row.line_audit_n_applied_notches;
                diag.shared_applied_notch = row.line_audit_shared_applied_notch;
                diag.shared_applied_freq_hz = row.line_audit_shared_applied_freq_hz;
                diag.shared_applied_width_hz = row.line_audit_shared_applied_width_hz;
                diag.shared_applied_support_network_count =
                    row.line_audit_shared_applied_support_network_count;
                diag.detector_candidate_uid = row.line_audit_detector_candidate_uid;
                diag.detector_candidate_freq_hz = row.line_audit_detector_candidate_freq_hz;
                diag.detector_candidate_prominence = row.line_audit_detector_candidate_prominence;
                diag.detector_candidate_line_power_frac =
                    row.line_audit_detector_candidate_line_power_frac;
                diag.detector_candidate_cluster_detector_frac =
                    row.line_audit_detector_candidate_cluster_detector_frac;
                diag.detector_candidate_recommend_flag =
                    row.line_audit_detector_candidate_recommend_flag;
                return diag;
            };
            auto write_line_audit_diag = [&](const std::string &prefix, auto getter) {
                write_nw_int(prefix + "_n_det_used",
                             [&](const auto &row) { return getter(row).n_det_used; });
                write_nw_double(prefix + "_shared_freq_hz",
                                [&](const auto &row) { return getter(row).shared_freq_hz; });
                write_nw_int(prefix + "_shared_detector_count",
                             [&](const auto &row) { return getter(row).shared_detector_count; });
                write_nw_double(prefix + "_shared_detector_frac",
                                [&](const auto &row) { return getter(row).shared_detector_frac; });
                write_nw_double(prefix + "_shared_median_prominence",
                                [&](const auto &row) { return getter(row).shared_median_prominence; });
                write_nw_double(prefix + "_shared_max_prominence",
                                [&](const auto &row) { return getter(row).shared_max_prominence; });
                write_nw_double(prefix + "_shared_width_hz",
                                [&](const auto &row) { return getter(row).shared_width_hz; });
                write_nw_double(prefix + "_shared_line_power_frac",
                                [&](const auto &row) { return getter(row).shared_line_power_frac; });
                write_nw_double(prefix + "_shared_common_mode_freq_hz",
                                [&](const auto &row) { return getter(row).shared_common_mode_freq_hz; });
                write_nw_double(prefix + "_shared_common_mode_prominence",
                                [&](const auto &row) { return getter(row).shared_common_mode_prominence; });
                write_nw_double(prefix + "_shared_notch_score",
                                [&](const auto &row) { return getter(row).shared_notch_score; });
                write_nw_int(prefix + "_shared_recommend_notch",
                             [&](const auto &row) { return getter(row).shared_recommend_notch ? 1 : 0; });
                write_nw_int(prefix + "_n_applied_notches",
                             [&](const auto &row) { return getter(row).n_applied_notches; });
                write_nw_int(prefix + "_shared_applied_notch",
                             [&](const auto &row) { return getter(row).shared_applied_notch ? 1 : 0; });
                write_nw_double(prefix + "_shared_applied_freq_hz",
                                [&](const auto &row) { return getter(row).shared_applied_freq_hz; });
                write_nw_double(prefix + "_shared_applied_width_hz",
                                [&](const auto &row) { return getter(row).shared_applied_width_hz; });
                write_nw_int(prefix + "_shared_applied_support_network_count",
                             [&](const auto &row) { return getter(row).shared_applied_support_network_count; });
                write_nw_int(prefix + "_detector_candidate_uid",
                             [&](const auto &row) { return getter(row).detector_candidate_uid; });
                write_nw_double(prefix + "_detector_candidate_freq_hz",
                                [&](const auto &row) { return getter(row).detector_candidate_freq_hz; });
                write_nw_double(prefix + "_detector_candidate_prominence",
                                [&](const auto &row) { return getter(row).detector_candidate_prominence; });
                write_nw_double(prefix + "_detector_candidate_line_power_frac",
                                [&](const auto &row) { return getter(row).detector_candidate_line_power_frac; });
                write_nw_double(prefix + "_detector_candidate_cluster_detector_frac",
                                [&](const auto &row) { return getter(row).detector_candidate_cluster_detector_frac; });
                write_nw_int(prefix + "_detector_candidate_recommend_flag",
                             [&](const auto &row) { return getter(row).detector_candidate_recommend_flag ? 1 : 0; });
            };
            write_line_audit_diag("rtc_network_line_audit", legacy_line_audit_diag);
            write_line_audit_diag("rtc_network_post_line_audit",
                                  [](const auto &row) { return row.post_line_audit; });
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
            write_nw_double("rtc_network_impulsive_score_median",
                            [](const auto &row) { return row.median_impulsive_score; });
            write_nw_double("rtc_network_impulsive_score_max",
                            [](const auto &row) { return row.max_impulsive_score; });
            write_nw_double("rtc_network_impulsive_det_frac",
                            [](const auto &row) { return row.impulsive_det_frac; });
            write_nw_double("rtc_network_impulsive_alignment_frac",
                            [](const auto &row) { return row.impulsive_alignment_frac; });
            write_nw_int("rtc_network_impulsive_dominant_sample",
                         [](const auto &row) { return row.dominant_impulsive_sample; });
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
            write_nw_int("rtc_network_impulsive_mask_applied",
                         [](const auto &row) { return row.impulsive_mask_applied ? 1 : 0; });
            write_nw_int("rtc_network_impulsive_mask_start_sample",
                         [](const auto &row) { return row.impulsive_mask_start_sample; });
            write_nw_int("rtc_network_impulsive_mask_end_sample",
                         [](const auto &row) { return row.impulsive_mask_end_sample; });
            write_nw_int("rtc_network_impulsive_mask_window_samples",
                         [](const auto &row) { return row.impulsive_mask_window_samples; });
            write_nw_int("rtc_network_impulsive_mask_n_det_masked",
                         [](const auto &row) { return row.impulsive_mask_n_det_masked; });
            write_nw_int("rtc_network_impulsive_mask_n_det_samples_flagged",
                         [](const auto &row) { return row.impulsive_mask_n_det_samples_flagged; });
            write_nw_double("rtc_network_impulsive_mask_flagged_fraction",
                            [](const auto &row) { return row.impulsive_mask_flagged_fraction; });
            write_nw_int("rtc_network_impulsive_mask_candidate_available",
                         [](const auto &row) { return row.impulsive_mask_candidate_available ? 1 : 0; });
            write_nw_int("rtc_network_impulsive_mask_local_trigger",
                         [](const auto &row) { return row.impulsive_mask_local_trigger ? 1 : 0; });
            write_nw_int("rtc_network_impulsive_mask_cross_network_trigger",
                         [](const auto &row) { return row.impulsive_mask_cross_network_trigger ? 1 : 0; });
            write_nw_int("rtc_network_impulsive_mask_high_score_override_trigger",
                         [](const auto &row) { return row.impulsive_mask_high_score_override_trigger ? 1 : 0; });
            write_nw_int("rtc_network_impulsive_mask_rejected_max_fraction",
                         [](const auto &row) { return row.impulsive_mask_rejected_max_fraction ? 1 : 0; });
            write_nw_int("rtc_network_impulsive_mask_candidate_center_sample",
                         [](const auto &row) { return row.impulsive_mask_candidate_center_sample; });
            write_nw_int("rtc_network_impulsive_mask_cluster_center_sample",
                         [](const auto &row) { return row.impulsive_mask_cluster_center_sample; });
            write_nw_int("rtc_network_impulsive_mask_cluster_network_count",
                         [](const auto &row) { return row.impulsive_mask_cluster_network_count; });
            write_nw_int("rtc_network_impulsive_mask_cluster_active_count",
                         [](const auto &row) { return row.impulsive_mask_cluster_active_count; });
            write_nw_int("rtc_network_impulsive_mask_total_active_count",
                         [](const auto &row) { return row.impulsive_mask_total_active_count; });
            write_nw_double("rtc_network_impulsive_mask_cluster_peak_score",
                            [](const auto &row) { return row.impulsive_mask_cluster_peak_score; });
            write_nw_double("rtc_network_impulsive_mask_override_score",
                            [](const auto &row) { return row.impulsive_mask_override_score; });
            write_nw_int("rtc_network_impulsive_mask_override_uses_network_peak",
                         [](const auto &row) { return row.impulsive_mask_override_uses_network_peak ? 1 : 0; });
            write_nw_double("rtc_network_impulsive_mask_proposed_flagged_fraction",
                            [](const auto &row) { return row.impulsive_mask_proposed_flagged_fraction; });

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
                    if (!impulsive_diag.empty()) {
                        for (const auto &[nw, slots] : impulsive_diag) {
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
                    if (!impulsive_diag.empty()) {
                        for (const auto &[nw, slots] : impulsive_diag) {
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
                    if (!impulsive_diag.empty()) {
                        for (const auto &[nw, slots] : impulsive_diag) {
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
                    if (!impulsive_diag.empty()) {
                        for (const auto &[nw, slots] : impulsive_diag) {
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
                write_imp_slot_int("rtc_impulsive_slot_local_raw_candidate_count",
                                   [](const auto &slot) { return slot.local_raw_candidate_count; });
                write_imp_slot_int("rtc_impulsive_slot_local_raw_accepted_event_count",
                                   [](const auto &slot) { return slot.local_raw_accepted_event_count; });
                write_imp_slot_int("rtc_impulsive_slot_local_flagged_sample_count",
                                   [](const auto &slot) { return slot.local_flagged_sample_count; });
                write_imp_slot_int("rtc_impulsive_slot_local_exceed_count",
                                   [](const auto &slot) { return slot.local_flagged_sample_count; });
                write_imp_slot_int("rtc_impulsive_slot_local_raw_reject_count",
                                   [](const auto &slot) { return slot.local_raw_reject_count; });
                write_imp_slot_int("rtc_impulsive_slot_delta_spike_count",
                                   [](const auto &slot) { return slot.delta_spike_count; });
                write_imp_slot_int("rtc_impulsive_slot_local_delta_candidate_count",
                                   [](const auto &slot) { return slot.local_delta_candidate_count; });
                write_imp_slot_int("rtc_impulsive_slot_local_delta_accepted_event_count",
                                   [](const auto &slot) { return slot.local_delta_accepted_event_count; });
                write_imp_slot_int("rtc_impulsive_slot_local_delta_exceed_count",
                                   [](const auto &slot) { return slot.local_delta_accepted_event_count; });
                write_imp_slot_int("rtc_impulsive_slot_local_delta_reject_count",
                                   [](const auto &slot) { return slot.local_delta_reject_count; });
                write_imp_snip_double("rtc_impulsive_slot_snippet_z",
                                      [](const auto &slot) -> const auto & { return slot.snippet_z; });
                write_imp_snip_int("rtc_impulsive_slot_snippet_flag",
                                   [](const auto &slot) -> const auto & { return slot.snippet_flag; });
            }
        }
    }
}

inline void RTCProc::clear_cached_diagnostics(Eigen::Index scan_id) {
    {
        std::lock_guard<std::mutex> lock(*diag_cache_mutex);
        remove_bad_dets_window_summary_by_scan.erase(scan_id);
    }
    {
        std::lock_guard<std::mutex> lock(*diag_summary_mutex);
        rtc_detector_summary_by_scan.erase(scan_id);
        rtc_network_summary_by_scan.erase(scan_id);
        rtc_impulsive_summary_by_scan.erase(scan_id);
        rtc_source_protection_summary_by_scan.erase(scan_id);
    }
}

template <typename calib_t>
void RTCProc::append_diag_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                                    std::string filepath,
                                    calib_t &calib,
                                    Eigen::Index scan_row_index) {
    using netCDF::NcFile;
    using namespace netCDF::exceptions;

    try {
        capture_rtc_diagnostics(in, calib, true, true);

        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        NcFile fo(filepath, netCDF::NcFile::write);
        write_cached_diagnostics_to_netcdf(fo, in, calib, scan_row_index);
        fo.sync();
        fo.close();

        logger->info("rtc diagnostics sidecar chunk written to {}", filepath);
    } catch (NcException &e) {
        logger->error("{}", e.what());
    }
}

template <typename calib_t, typename pointing_offset_t>
void RTCProc::append_to_netcdf(TCData<TCDataKind::RTC, Eigen::MatrixXd> &in, std::string filepath, std::string map_grouping,
                               std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, calib_t &calib,
                               bool apply_det_offsets, Eigen::Index scan_row_index) {
    using netCDF::NcFile;
    using namespace netCDF::exceptions;

    try {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        NcFile fo(filepath, netCDF::NcFile::write);

        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, calib, apply_det_offsets,
                              scan_row_index, true);

        fo.sync();
        fo.close();

        logger->info("outer tod chunk written to {}", filepath);

    } catch (NcException &e) {
        logger->error("{}", e.what());
    }
}

template <typename calib_t, typename pointing_offset_t>
void RTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string map_grouping,
                               std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, calib_t &calib,
                               bool apply_det_offsets, Eigen::Index scan_row_index) {
    using netCDF::NcFile;
    using namespace netCDF::exceptions;

    try {
        capture_rtc_diagnostics(in, calib, true, true);

        // open netcdf file
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        NcFile fo(filepath, netCDF::NcFile::write);

        // append common time chunk variables
        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, calib, apply_det_offsets,
                              scan_row_index);
        write_cached_diagnostics_to_netcdf(fo, in, calib, scan_row_index);

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
