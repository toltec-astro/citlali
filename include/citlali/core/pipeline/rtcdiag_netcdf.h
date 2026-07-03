#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/utils/netcdf_io.h>

namespace citlali::pipeline {

template <class Calib>
std::vector<int> diagnostic_array_ids(const Calib &calib, int fill_value) {
    std::vector<int> ids(static_cast<std::size_t>(calib.n_arrays),
                         fill_value);
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.arrays(i));
    }
    return ids;
}

template <class Calib>
void add_rtcdiag_array_ids(netCDF::NcFile &fo, const Calib &calib,
                           netCDF::NcDim n_arrays_dim, int fill_value) {
    netCDF::NcVar array_ids_v =
        fo.addVar("rtc_diag_array_ids", netCDF::ncInt, n_arrays_dim);
    array_ids_v.putAtt("units", "N/A");
    array_ids_v.putAtt("comment",
                       "array IDs corresponding to n_arrays axis");
    const auto array_ids = diagnostic_array_ids(calib, fill_value);
    array_ids_v.putVar(array_ids.data());
}

template <class Calib>
void add_rtcdiag_network_ids(netCDF::NcFile &fo, const Calib &calib,
                             netCDF::NcDim n_nws_rtcdiag_dim,
                             int fill_value) {
    netCDF::NcVar nw_ids_v =
        fo.addVar("rtc_diag_network_ids", netCDF::ncInt,
                  n_nws_rtcdiag_dim);
    nw_ids_v.putAtt("units", "N/A");
    nw_ids_v.putAtt("comment",
                    "network IDs corresponding to n_nws_rtcdiag axis");
    std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws),
                            fill_value);
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
    }
    nw_ids_v.putVar(nw_ids.data());
}

template <class Calib>
void add_rtcdiag_apt_double_vars(netCDF::NcFile &fo, Calib &calib,
                                 netCDF::NcDim n_dets_dim) {
    for (auto const &x : calib.apt) {
        netCDF::NcVar apt_v =
            fo.addVar("apt_" + x.first, netCDF::ncDouble, n_dets_dim);
        apt_v.putAtt("units", calib.apt_header_units[x.first]);
        apt_v.putVar(x.second.data());
    }
}

template <class LineAudit>
void add_rtc_line_audit_config_vars(netCDF::NcFile &fo,
                                    const LineAudit &line_audit) {
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.ENABLED",
                   line_audit.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MIN_HZ",
                   line_audit.line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.LINE_MAX_HZ",
                   line_audit.line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.SEGMENT_SEC",
                   line_audit.segment_sec);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_SEGMENT_SEC",
                   line_audit.min_segment_sec);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.OVERLAP_FRAC",
                   line_audit.overlap_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CONTINUUM_RADIUS_BINS",
                   line_audit.continuum_radius_bins);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PROMINENCE_THRESH",
                   line_audit.prominence_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CM_PROMINENCE_THRESH",
                   line_audit.cm_prominence_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_GOOD_FRAC",
                   line_audit.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_WINDOWS",
                   line_audit.min_windows);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_PEAKS_PER_DETECTOR",
                   line_audit.max_peaks_per_detector);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MAX_DET",
                   line_audit.max_det);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.MIN_DET_FOR_NETWORK",
                   line_audit.min_det_for_network);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.CLUSTER_TOL_HZ",
                   line_audit.cluster_tol_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTOR_FRAC",
                   line_audit.notch_min_detector_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_DETECTORS",
                   line_audit.notch_min_detectors);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.NOTCH_MIN_CM_PROMINENCE",
                   line_audit.notch_min_cm_prominence);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_PROMINENCE",
                   line_audit.detector_min_prominence);
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_MIN_LINE_POWER_FRAC",
        line_audit.detector_min_line_power_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.BAD_DETECTOR_MAX_CLUSTER_FRAC",
                   line_audit.bad_detector_max_cluster_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PRE_FILTER_ENABLED",
                   line_audit.pre_filter_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_ENABLED",
                   line_audit.post_filter_enabled);
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_SHARED_NOTCHES",
        line_audit.post_filter_apply_shared_notches);
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_DETECTOR_NOTCHES",
        line_audit.post_filter_apply_detector_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_APPLY_ITERATIONS",
                   line_audit.post_filter_apply_iterations);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MIN_HZ",
                   line_audit.post_filter_line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.POST_FILTER_LINE_MAX_HZ",
                   line_audit.post_filter_line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_MODEL_PROTECTED_ENABLED",
                   line_audit.ptc_model_protected_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_REQUIRE_MODEL_SUBTRACTED",
                   line_audit.ptc_require_model_subtracted);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_FIXED_NOTCHES",
                   line_audit.ptc_apply_fixed_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_SHARED_NOTCHES",
                   line_audit.ptc_apply_shared_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_DETECTOR_NOTCHES",
                   line_audit.ptc_apply_detector_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_APPLY_ITERATIONS",
                   line_audit.ptc_apply_iterations);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MIN_HZ",
                   line_audit.ptc_line_min_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.PTC_LINE_MAX_HZ",
                   line_audit.ptc_line_max_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_ENABLED",
                   line_audit.fixed_notch_enabled);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_COUNT",
                   static_cast<int>(line_audit.fixed_notch_freqs_hz.size()));
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_WIDTH_COUNT",
                   static_cast<int>(line_audit.fixed_notch_widths_hz.size()));
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.FIXED_NOTCH_EXCLUSION_HALF_WIDTH_HZ",
        line_audit.fixed_notch_exclusion_half_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_SHARED_NOTCHES",
                   line_audit.apply_shared_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_SUPPORT_NETWORKS",
                   line_audit.apply_min_support_networks);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_DETECTOR_FRAC",
                   line_audit.apply_min_detector_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_CM_PROMINENCE",
                   line_audit.apply_min_common_mode_prominence);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_WIDTH_SCALE",
                   line_audit.apply_width_scale);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MIN_WIDTH_HZ",
                   line_audit.apply_min_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_WIDTH_HZ",
                   line_audit.apply_max_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_MAX_NOTCHES",
                   line_audit.apply_max_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.APPLY_CLUSTER_TOL_HZ",
                   line_audit.apply_cluster_tol_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_PROMINENCE",
                   line_audit.detector_notch_min_prominence);
    add_netcdf_var(
        fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_LINE_POWER_FRAC",
        line_audit.detector_notch_min_line_power_frac);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_NOTCHES",
                   line_audit.detector_notch_max_notches);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_WIDTH_SCALE",
                   line_audit.detector_notch_width_scale);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MIN_WIDTH_HZ",
                   line_audit.detector_notch_min_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_MAX_WIDTH_HZ",
                   line_audit.detector_notch_max_width_hz);
    add_netcdf_var(fo, "CONFIG.RTC.LINE_AUDIT.DETECTOR_NOTCH_CONTEXT_SAMPLES",
                   line_audit.detector_notch_context_samples);
}

inline double rtcdiag_percentile_sorted(
    const std::vector<double> &sorted_values, double pct) {
    if (sorted_values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (sorted_values.size() == 1) {
        return sorted_values.front();
    }
    pct = std::min(100.0, std::max(0.0, pct));
    const double pos =
        (pct / 100.0) * static_cast<double>(sorted_values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}

inline void add_rtcdiag_scan_double(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment, netCDF::NcDim n_scans_dim,
    const std::vector<std::size_t> &scan_chunks,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, n_scans_dim);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, scan_chunks, 1);
    v.putVar(values.data());
}

inline void add_rtcdiag_scan_array_double(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment,
    const std::vector<netCDF::NcDim> &scan_array_dims,
    const std::vector<std::size_t> &scan_array_chunks,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, scan_array_dims);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, scan_array_chunks, 1);
    v.putVar(values.data());
}

struct RtcDiagScanDoubleValues {
    const std::vector<double> &scan_duration_s;
    const std::vector<double> &scan_speed_p50_arcsec_s;
    const std::vector<double> &scan_speed_p95_arcsec_s;
    const std::vector<double> &scan_speed_p995_arcsec_s;
};

template <class AddDouble>
void add_rtcdiag_scan_summary_vars(
    const AddDouble &add_double, const RtcDiagScanDoubleValues &values) {
    add_double("scan_duration_s", "s",
               "inner scan duration used for scan-speed diagnostics",
               values.scan_duration_s);
    add_double("scan_speed_altaz_p50_arcsec_s", "arcsec/s",
               "per-scan median boresight speed in the delta-source altaz frame",
               values.scan_speed_p50_arcsec_s);
    add_double("scan_speed_altaz_p95_arcsec_s", "arcsec/s",
               "per-scan 95th percentile boresight speed in the delta-source altaz frame",
               values.scan_speed_p95_arcsec_s);
    add_double("scan_speed_altaz_p995_arcsec_s", "arcsec/s",
               "per-scan robust peak (99.5th percentile) boresight speed in the delta-source altaz frame",
               values.scan_speed_p995_arcsec_s);
}

struct RtcDiagScanArrayDoubleValues {
    const std::vector<double> &source_power_half_bandwidth_hz;
    const std::vector<double> &tod_lowpass_to_source_power_half_ratio;
};

template <class AddDouble>
void add_rtcdiag_scan_array_summary_vars(
    const AddDouble &add_double,
    const RtcDiagScanArrayDoubleValues &values) {
    add_double("scan_source_power_half_bandwidth_hz", "Hz",
               "Gaussian compact-source temporal power half-bandwidth from scan_speed_altaz_p995_arcsec_s and array mean FWHM",
               values.source_power_half_bandwidth_hz);
    add_double("scan_tod_lowpass_to_source_power_half_ratio", "N/A",
               "configured RTC FIR low-pass cutoff divided by scan_source_power_half_bandwidth_hz; values much larger than 1 indicate extra high-frequency noise admitted relative to compact-source half-power bandwidth",
               values.tod_lowpass_to_source_power_half_ratio);
}

inline void add_rtcdiag_det_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, det_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_det_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, det_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

template <class AddInt, class AddDouble>
void add_rtcdiag_detector_core_diag(const AddInt &add_int,
                                    const AddDouble &add_double) {
    add_int("rtc_despike_raw_exceed_count",
            "per-detector count of raw-sample MAD-threshold exceedances before despike expansion");
    add_int("rtc_despike_local_raw_candidate_count",
            "per-detector count of locally detrended raw candidate events considered by the compact-raw gate");
    add_int("rtc_despike_local_raw_accepted_event_count",
            "per-detector count of locally detrended raw candidate events accepted by the compact-raw gate");
    add_int("rtc_despike_local_flagged_sample_count",
            "per-detector count of samples flagged by accepted compact-raw local-residual events");
    add_int("rtc_despike_local_exceed_count",
            "legacy alias for rtc_despike_local_flagged_sample_count");
    add_int("rtc_despike_local_raw_reject_count",
            "per-detector count of locally detrended raw candidate events rejected by the compact-raw gate");
    add_int("rtc_despike_delta_spike_count",
            "per-detector count of delta-domain spikes identified by the RTC despiker");
    add_int("rtc_despike_local_delta_candidate_count",
            "per-detector count of locally detrended delta candidate events considered by the compact-delta gate");
    add_int("rtc_despike_local_delta_accepted_event_count",
            "per-detector count of locally detrended delta candidate events accepted by the compact-delta gate");
    add_int("rtc_despike_local_delta_exceed_count",
            "legacy alias for rtc_despike_local_delta_accepted_event_count");
    add_int("rtc_despike_local_delta_reject_count",
            "per-detector count of locally detrended delta candidate events rejected by the compact-delta gate");
    add_double("rtc_despike_added_flagged_frac",
               "fraction of samples newly flagged by RTC despiking, excluding pre-existing flags");
    add_int("rtc_despike_added_region_count",
            "count of newly flagged contiguous sample regions added by RTC despiking");
    add_double("rtc_despike_added_region_len_median",
               "median length of newly flagged contiguous sample regions added by RTC despiking");
    add_int("rtc_despike_added_region_len_max",
            "maximum length of newly flagged contiguous sample regions added by RTC despiking");
    add_double("rtc_despike_max_raw_abs_z",
               "maximum absolute raw-sample deviation in robust-sigma units before despiking");
    add_double("rtc_despike_max_local_abs_z",
               "maximum absolute locally detrended raw-sample deviation in robust-sigma units before despiking");
    add_double("rtc_despike_max_delta_abs_z",
               "maximum absolute adjacent-sample delta deviation in sigma units before despiking");
    add_double("rtc_despike_max_local_delta_abs_z",
               "maximum absolute locally detrended adjacent-sample delta deviation in sigma units before despiking");
    add_double("rtc_final_flagged_frac",
               "final per-detector flagged-sample fraction in the RTC product actually written");
    add_int("rtc_final_region_count",
            "final count of flagged contiguous sample regions in the RTC product actually written");
    add_double("rtc_final_region_len_median",
               "final median flagged-region length in the RTC product actually written");
    add_int("rtc_final_region_len_max",
            "final maximum flagged-region length in the RTC product actually written");
    add_double("rtc_step_score",
               "per-detector step-like pre/post window jump score on the RTC output");
    add_int("rtc_step_sample",
            "sample index of the strongest per-detector RTC step-like jump; -2147483647 means unavailable");
    add_double("rtc_impulsive_peak_abs_z",
               "maximum absolute per-sample deviation in robust-sigma units on the RTC output");
    add_int("rtc_impulsive_peak_abs_sample",
            "sample index of the maximum absolute per-sample deviation; -2147483647 means unavailable");
    add_double("rtc_impulsive_peak_delta_abs_z",
               "maximum absolute adjacent-sample delta deviation in robust-sigma units on the RTC output");
    add_int("rtc_impulsive_peak_delta_abs_sample",
            "sample index of the strongest adjacent-sample delta excursion; -2147483647 means unavailable");
    add_int("rtc_impulsive_near_abs_count",
            "count of RTC samples exceeding near_event_z in absolute robust-z units");
    add_int("rtc_impulsive_near_delta_count",
            "count of RTC adjacent-sample delta excursions exceeding near_event_z");
    add_double("rtc_impulsive_event_score",
               "per-detector impulsive event score, max of raw and delta robust-z peaks");
    add_int("rtc_impulsive_event_sample",
            "sample index of the strongest per-detector impulsive event; -2147483647 means unavailable");
    add_int("rtc_impulsive_event_kind",
            "0=raw-sample peak, 1=delta peak, -2147483647 means unavailable");
    add_int("rtc_detector_notch_n_applied",
            "per-detector count of post-filter detector-local RTC notches applied");
    add_double("rtc_detector_notch_primary_freq_hz",
               "frequency of the strongest detector-local post-filter RTC notch applied");
    add_double("rtc_detector_notch_primary_width_hz",
               "bandwidth of the strongest detector-local post-filter RTC notch applied");
    add_double("rtc_detector_notch_primary_prominence",
               "PSD prominence of the strongest detector-local post-filter RTC notch applied");
    add_double("rtc_detector_notch_primary_line_power_frac",
               "line-power fraction of the strongest detector-local post-filter RTC notch applied");
    add_double("rtc_detector_notch_rms_before",
               "robust RMS of the detector RTC timestream before detector-local post-filter notching");
    add_double("rtc_detector_notch_rms_after",
               "robust RMS of the detector RTC timestream after detector-local post-filter notching");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_detector_invvar_window_diag(const AddInt &add_int,
                                             const AddDouble &add_double) {
    add_double("rtc_invvar_window_valid_fraction",
               "fraction of remove_bad_dets diagnostic windows with enough unflagged samples to estimate inverse variance in the RTC timestream");
    add_double("rtc_invvar_window_median",
               "median per-window inverse variance used for RTC remove_bad_dets diagnostics");
    add_double("rtc_invvar_window_q10",
               "10th percentile of per-window inverse variance used for RTC remove_bad_dets diagnostics");
    add_double("rtc_invvar_window_q90",
               "90th percentile of per-window inverse variance used for RTC remove_bad_dets diagnostics");
    add_double("rtc_invvar_window_flagged_frac_median",
               "median flagged fraction across remove_bad_dets diagnostic windows in the RTC timestream");
    add_double("rtc_invvar_window_flagged_frac_max",
               "maximum flagged fraction across remove_bad_dets diagnostic windows in the RTC timestream");
    add_double("rtc_invvar_window_heavy_flagged_fraction",
               "fraction of remove_bad_dets diagnostic windows in the RTC timestream with at least 50 percent flagged samples");
    add_int("rtc_invvar_window_n_total",
            "total number of fixed windows evaluated for RTC remove_bad_dets diagnostics");
    add_int("rtc_invvar_window_n_valid",
            "number of fixed windows with a finite inverse-variance estimate for RTC remove_bad_dets diagnostics");
}

inline void add_rtcdiag_network_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, nw_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, nw_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_network_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, nw_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, nw_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

template <class AddInt>
void add_rtcdiag_network_detector_count_diag(const AddInt &add_int) {
    add_int("rtc_network_n_det_input",
            "input detector count in each RTC network block");
    add_int("rtc_network_n_det_used",
            "detectors passing the step-mask valid-sample threshold and finite robust scale");
    add_int("rtc_network_impulsive_n_det_used",
            "detectors passing the impulsive-coincidence valid-sample threshold and finite robust scale");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_line_audit_base_diag(const AddInt &add_int,
                                              const AddDouble &add_double) {
    add_int("rtc_network_line_audit_n_det_used",
            "detectors analyzed by the pre-filter RTC line audit in each network block");
    add_double("rtc_network_line_audit_shared_freq_hz",
               "frequency of the strongest shared narrowband RTC line family in each network block");
    add_int("rtc_network_line_audit_shared_detector_count",
            "number of detectors participating in the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_detector_frac",
               "fraction of audited detectors participating in the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_median_prominence",
               "median detector-level PSD prominence of the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_max_prominence",
               "maximum detector-level PSD prominence of the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_width_hz",
               "median linewidth of the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_line_power_frac",
               "median detector-level line-power fraction of the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_common_mode_freq_hz",
               "matched common-mode line frequency for the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_common_mode_prominence",
               "matched common-mode PSD prominence for the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_notch_score",
               "shared-line notch score, detector fraction times median prominence");
    add_int("rtc_network_line_audit_shared_recommend_notch",
            "1 if the strongest shared narrowband RTC line family met the current notch-candidate criteria");
    add_int("rtc_network_line_audit_n_applied_notches",
            "number of chunk-level shared-line RTC notches actually applied to this scan");
    add_int("rtc_network_line_audit_shared_applied_notch",
            "1 if the strongest shared narrowband RTC line family in this network matched an applied chunk-level RTC notch");
    add_double("rtc_network_line_audit_shared_applied_freq_hz",
               "center frequency of the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_double("rtc_network_line_audit_shared_applied_width_hz",
               "full-width bandwidth of the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_int("rtc_network_line_audit_shared_applied_support_network_count",
            "number of networks supporting the applied chunk-level RTC notch matched to the strongest shared narrowband RTC line family");
    add_int("rtc_network_line_audit_detector_candidate_uid",
            "UID of the strongest detector-local RTC line candidate in each network block; -2147483647 means none");
    add_double("rtc_network_line_audit_detector_candidate_freq_hz",
               "frequency of the strongest detector-local RTC line candidate");
    add_double("rtc_network_line_audit_detector_candidate_prominence",
               "PSD prominence of the strongest detector-local RTC line candidate");
    add_double("rtc_network_line_audit_detector_candidate_line_power_frac",
               "line-power fraction of the strongest detector-local RTC line candidate");
    add_double("rtc_network_line_audit_detector_candidate_cluster_detector_frac",
               "shared-cluster detector fraction associated with the strongest detector-local RTC line candidate");
    add_int("rtc_network_line_audit_detector_candidate_recommend_flag",
            "1 if the strongest detector-local RTC line candidate met the current bad-detector criteria");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_line_audit_diag(
    const AddInt &add_int, const AddDouble &add_double,
    const std::string &prefix, const std::string &stage) {
    add_int(prefix + "_n_det_used",
            "detectors analyzed by the " + stage +
                " RTC line audit in each network block");
    add_double(prefix + "_shared_freq_hz",
               "frequency of the strongest shared narrowband " + stage +
                   " RTC line family in each network block");
    add_int(prefix + "_shared_detector_count",
            "number of detectors participating in the strongest shared narrowband " +
                stage + " RTC line family");
    add_double(prefix + "_shared_detector_frac",
               "fraction of audited detectors participating in the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_median_prominence",
               "median detector-level PSD prominence of the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_max_prominence",
               "maximum detector-level PSD prominence of the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_width_hz",
               "median linewidth of the strongest shared narrowband " + stage +
                   " RTC line family");
    add_double(prefix + "_shared_line_power_frac",
               "median detector-level line-power fraction of the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_common_mode_freq_hz",
               "matched common-mode line frequency for the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_common_mode_prominence",
               "matched common-mode PSD prominence for the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_notch_score",
               "shared-line notch score, detector fraction times median prominence");
    add_int(prefix + "_shared_recommend_notch",
            "1 if the strongest shared narrowband " + stage +
                " RTC line family met the current notch-candidate criteria");
    add_int(prefix + "_n_applied_notches",
            "number of chunk-level shared-line RTC notches actually applied in the " +
                stage + " stage");
    add_int(prefix + "_shared_applied_notch",
            "1 if the strongest shared narrowband " + stage +
                " RTC line family in this network matched an applied chunk-level RTC notch");
    add_double(prefix + "_shared_applied_freq_hz",
               "center frequency of the applied chunk-level RTC notch matched to the strongest shared narrowband " +
                   stage + " RTC line family");
    add_double(prefix + "_shared_applied_width_hz",
               "full-width bandwidth of the applied chunk-level RTC notch matched to the strongest shared narrowband " +
                   stage + " RTC line family");
    add_int(prefix + "_shared_applied_support_network_count",
            "number of networks supporting the applied chunk-level RTC notch matched to the strongest shared narrowband " +
                stage + " RTC line family");
    add_int(prefix + "_detector_candidate_uid",
            "UID of the strongest detector-local " + stage +
                " RTC line candidate in each network block; -2147483647 means none");
    add_double(prefix + "_detector_candidate_freq_hz",
               "frequency of the strongest detector-local " + stage +
                   " RTC line candidate");
    add_double(prefix + "_detector_candidate_prominence",
               "PSD prominence of the strongest detector-local " + stage +
                   " RTC line candidate");
    add_double(prefix + "_detector_candidate_line_power_frac",
               "line-power fraction of the strongest detector-local " + stage +
                   " RTC line candidate");
    add_double(prefix + "_detector_candidate_cluster_detector_frac",
               "shared-cluster detector fraction associated with the strongest detector-local " +
                   stage + " RTC line candidate");
    add_int(prefix + "_detector_candidate_recommend_flag",
            "1 if the strongest detector-local " + stage +
                " RTC line candidate met the current bad-detector criteria");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_step_summary_diag(const AddInt &add_int,
                                           const AddDouble &add_double) {
    add_double("rtc_network_step_score_median",
               "median detector step score within each RTC network block");
    add_double("rtc_network_step_score_max",
               "maximum detector step score within each RTC network block");
    add_double("rtc_network_step_det_frac",
               "fraction of diagnostic-used detectors with strong step-like score in each RTC network block");
    add_double("rtc_network_step_alignment_frac",
               "fraction of strong-step detectors aligned in the dominant step-time cluster");
    add_int("rtc_network_step_dominant_sample",
            "dominant aligned step sample within each RTC network block; -2147483647 means unavailable");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_impulsive_summary_diag(
    const AddInt &add_int, const AddDouble &add_double) {
    add_double("rtc_network_impulsive_score_median",
               "median detector impulsive-event score within each RTC network block");
    add_double("rtc_network_impulsive_score_max",
               "maximum detector impulsive-event score within each RTC network block");
    add_double("rtc_network_impulsive_det_frac",
               "fraction of diagnostic-used detectors with impulsive-event score above the impulsive coincidence threshold");
    add_double("rtc_network_impulsive_alignment_frac",
               "fraction of impulsive-active detectors aligned in the dominant impulsive time cluster");
    add_int("rtc_network_impulsive_dominant_sample",
            "dominant aligned impulsive sample within each RTC network block; -2147483647 means unavailable");
}

template <class AddDouble>
void add_rtcdiag_network_common_mode_diag(const AddDouble &add_double) {
    add_double("rtc_network_cm_low_mid_ratio",
               "low-band to mid-band common-mode power ratio for each RTC network block");
    add_double("rtc_network_cm_peak_freq_hz",
               "frequency of the strongest common-mode spectral peak for each RTC network block");
    add_double("rtc_network_cm_peak_prominence",
               "prominence of the strongest common-mode spectral peak for each RTC network block");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_step_mask_diag(const AddInt &add_int,
                                        const AddDouble &add_double) {
    add_int("rtc_network_step_mask_applied",
            "1 if network_step_mask flagged a time window for this RTC network block, else 0");
    add_int("rtc_network_step_mask_start_sample",
            "inclusive starting sample of the applied network_step_mask window; -2147483647 means none");
    add_int("rtc_network_step_mask_end_sample",
            "inclusive ending sample of the applied network_step_mask window; -2147483647 means none");
    add_int("rtc_network_step_mask_window_samples",
            "number of RTC time samples in the applied network_step_mask window");
    add_int("rtc_network_step_mask_n_det_masked",
            "number of detectors included in the applied network_step_mask window");
    add_int("rtc_network_step_mask_n_det_samples_flagged",
            "number of previously good detector-samples newly flagged by network_step_mask");
    add_double("rtc_network_step_mask_flagged_fraction",
               "fraction of previously good detector-samples in the network block newly flagged by network_step_mask");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_impulsive_mask_window_diag(
    const AddInt &add_int, const AddDouble &add_double) {
    add_int("rtc_network_impulsive_mask_applied",
            "1 if impulsive_coincidence_mask flagged a time window for this RTC network block, else 0");
    add_int("rtc_network_impulsive_mask_start_sample",
            "inclusive starting sample of the applied impulsive_coincidence_mask window; -2147483647 means none");
    add_int("rtc_network_impulsive_mask_end_sample",
            "inclusive ending sample of the applied impulsive_coincidence_mask window; -2147483647 means none");
    add_int("rtc_network_impulsive_mask_window_samples",
            "number of RTC time samples in the applied impulsive_coincidence_mask window");
    add_int("rtc_network_impulsive_mask_n_det_masked",
            "number of detectors included in the applied impulsive_coincidence_mask window");
    add_int("rtc_network_impulsive_mask_n_det_samples_flagged",
            "number of previously good detector-samples newly flagged by impulsive_coincidence_mask");
    add_double("rtc_network_impulsive_mask_flagged_fraction",
               "fraction of previously good detector-samples in the network block newly flagged by impulsive_coincidence_mask");
}

template <class AddInt>
void add_rtcdiag_network_impulsive_mask_trigger_diag(
    const AddInt &add_int) {
    add_int("rtc_network_impulsive_mask_candidate_available",
            "1 if impulsive_coincidence_mask found a candidate for this RTC network block, else 0");
    add_int("rtc_network_impulsive_mask_local_trigger",
            "1 if the selected impulsive candidate satisfied the within-network trigger thresholds, else 0");
    add_int("rtc_network_impulsive_mask_cross_network_trigger",
            "1 if the selected impulsive candidate satisfied a cross-network alignment trigger, else 0");
    add_int("rtc_network_impulsive_mask_high_score_override_trigger",
            "1 if the selected impulsive candidate satisfied the looser high-score cross-network override, else 0");
    add_int("rtc_network_impulsive_mask_rejected_max_fraction",
            "1 if the selected impulsive candidate was rejected only because its proposed flagged fraction exceeded the configured limit");
}

template <class AddInt, class AddDouble>
void add_rtcdiag_network_impulsive_mask_candidate_diag(
    const AddInt &add_int, const AddDouble &add_double) {
    add_int("rtc_network_impulsive_mask_candidate_center_sample",
            "center sample of the selected impulsive candidate before any cross-network recentering; -2147483647 means unavailable");
    add_int("rtc_network_impulsive_mask_cluster_center_sample",
            "median aligned sample of the selected cross-network impulsive cluster; -2147483647 means unavailable");
    add_int("rtc_network_impulsive_mask_cluster_network_count",
            "number of distinct networks participating in the selected impulsive candidate cluster");
    add_int("rtc_network_impulsive_mask_cluster_active_count",
            "number of detector-level impulsive events in the selected within-network cluster");
    add_int("rtc_network_impulsive_mask_total_active_count",
            "total number of detector-level impulsive events above threshold in the selected network block");
    add_double("rtc_network_impulsive_mask_cluster_peak_score",
               "maximum impulsive-event score found within the selected cross-network impulsive cluster");
    add_double("rtc_network_impulsive_mask_override_score",
               "score used by the high-score override path after combining the selected cluster peak with the strongest candidate score seen in participating networks");
    add_int("rtc_network_impulsive_mask_override_uses_network_peak",
            "1 if rtc_network_impulsive_mask_override_score came from a participating network's strongest candidate rather than the selected cluster peak");
    add_double("rtc_network_impulsive_mask_proposed_flagged_fraction",
               "fraction of previously good detector-samples that the selected impulsive mask window would newly flag before any rejection");
}

inline std::vector<int> rtcdiag_impulsive_snippet_offsets(
    std::size_t n_snippet, std::size_t snippet_pre, int fill_value) {
    std::vector<int> offsets(n_snippet, fill_value);
    for (std::size_t i=0; i<n_snippet; ++i) {
        offsets[i] = static_cast<int>(i) - static_cast<int>(snippet_pre);
    }
    return offsets;
}

inline void add_rtcdiag_impulsive_slot_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &slot_dims,
    const std::vector<std::size_t> &slot_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, slot_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, slot_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_impulsive_slot_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &slot_dims,
    const std::vector<std::size_t> &slot_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, slot_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, slot_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_impulsive_snippet_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment,
    const std::vector<netCDF::NcDim> &snippet_dims,
    const std::vector<std::size_t> &snippet_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, snippet_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, snippet_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_impulsive_snippet_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment,
    const std::vector<netCDF::NcDim> &snippet_dims,
    const std::vector<std::size_t> &snippet_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, snippet_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, snippet_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

struct RtcDiagImpulsiveCaptureComments {
    std::string peak_abs_z;
    std::string peak_delta_abs_z;
    std::string added_flagged_frac;
    std::string raw_exceed_count;
    std::string local_raw_candidate_count;
    std::string local_raw_accepted_event_count;
    std::string local_flagged_sample_count;
    std::string local_raw_reject_count;
    std::string delta_spike_count;
    std::string local_delta_candidate_count;
    std::string local_delta_accepted_event_count;
    std::string local_delta_reject_count;
    std::string snippet_flag;
};

inline RtcDiagImpulsiveCaptureComments
rtcdiag_impulsive_capture_stream_comments() {
    return {
        "maximum per-sample absolute robust-z for a captured scan/network detector slot",
        "maximum adjacent-sample delta robust-z for a captured scan/network detector slot",
        "fraction of samples newly flagged by RTC despiking for a captured detector slot",
        "count of raw-sample MAD exceedances for a captured detector slot",
        "count of locally detrended raw candidate events considered by the compact-raw gate for a captured detector slot",
        "count of locally detrended raw candidate events accepted by the compact-raw gate for a captured detector slot",
        "count of samples flagged by accepted compact-raw local-residual events for a captured detector slot",
        "count of locally detrended raw candidate events rejected by the compact-raw gate for a captured detector slot",
        "count of delta-domain spikes for a captured detector slot",
        "count of locally detrended delta candidate events considered by the compact-delta gate for a captured detector slot",
        "count of locally detrended delta candidate events accepted by the compact-delta gate for a captured detector slot",
        "count of locally detrended delta candidate events rejected by the compact-delta gate for a captured detector slot",
        "final RTC flag state for each sample in a captured impulsive snippet",
    };
}

inline RtcDiagImpulsiveCaptureComments
rtcdiag_impulsive_capture_file_comments() {
    return {
        "absolute robust-z peak of a captured impulsive RTC event",
        "absolute delta robust-z peak of a captured impulsive RTC event",
        "newly added flagged-sample fraction for the captured detector",
        "native raw-threshold exceedance count for the captured detector",
        "compact-raw local candidate count for the captured detector",
        "accepted compact-raw local-event count for the captured detector",
        "samples flagged by accepted compact-raw local events for the captured detector",
        "rejected compact-raw local-event count for the captured detector",
        "native delta-spike count for the captured detector",
        "compact-delta local candidate count for the captured detector",
        "accepted compact-delta local-event count for the captured detector",
        "rejected compact-delta local-event count for the captured detector",
        "RTC flag state for each sample in the captured impulsive-event snippet",
    };
}

template <class AddSlotInt, class AddSlotDouble, class AddSnippetDouble,
          class AddSnippetInt>
void add_rtcdiag_impulsive_capture_diag(
    const AddSlotInt &add_slot_int, const AddSlotDouble &add_slot_double,
    const AddSnippetDouble &add_snippet_double,
    const AddSnippetInt &add_snippet_int,
    const RtcDiagImpulsiveCaptureComments &comments) {
    add_slot_int("rtc_impulsive_slot_det_index",
                 "detector index of a captured impulsive RTC event for each scan/network/slot");
    add_slot_int("rtc_impulsive_slot_event_sample",
                 "sample index of a captured impulsive RTC event; -2147483647 means unavailable");
    add_slot_int("rtc_impulsive_slot_event_kind",
                 "0=raw-sample peak, 1=delta peak, -2147483647 means unavailable");
    add_slot_double("rtc_impulsive_slot_event_score",
                    "impulsive event score for a captured scan/network detector slot");
    add_slot_double("rtc_impulsive_slot_peak_abs_z", comments.peak_abs_z);
    add_slot_double("rtc_impulsive_slot_peak_delta_abs_z",
                    comments.peak_delta_abs_z);
    add_slot_double("rtc_impulsive_slot_added_flagged_frac",
                    comments.added_flagged_frac);
    add_slot_int("rtc_impulsive_slot_raw_exceed_count",
                 comments.raw_exceed_count);
    add_slot_int("rtc_impulsive_slot_local_raw_candidate_count",
                 comments.local_raw_candidate_count);
    add_slot_int("rtc_impulsive_slot_local_raw_accepted_event_count",
                 comments.local_raw_accepted_event_count);
    add_slot_int("rtc_impulsive_slot_local_flagged_sample_count",
                 comments.local_flagged_sample_count);
    add_slot_int("rtc_impulsive_slot_local_exceed_count",
                 "legacy alias for rtc_impulsive_slot_local_flagged_sample_count");
    add_slot_int("rtc_impulsive_slot_local_raw_reject_count",
                 comments.local_raw_reject_count);
    add_slot_int("rtc_impulsive_slot_delta_spike_count",
                 comments.delta_spike_count);
    add_slot_int("rtc_impulsive_slot_local_delta_candidate_count",
                 comments.local_delta_candidate_count);
    add_slot_int("rtc_impulsive_slot_local_delta_accepted_event_count",
                 comments.local_delta_accepted_event_count);
    add_slot_int("rtc_impulsive_slot_local_delta_exceed_count",
                 "legacy alias for rtc_impulsive_slot_local_delta_accepted_event_count");
    add_slot_int("rtc_impulsive_slot_local_delta_reject_count",
                 comments.local_delta_reject_count);
    add_snippet_double("rtc_impulsive_slot_snippet_z",
                       "standardized RTC snippet around each captured impulsive event");
    add_snippet_int("rtc_impulsive_slot_snippet_flag",
                    comments.snippet_flag);
}

}  // namespace citlali::pipeline
