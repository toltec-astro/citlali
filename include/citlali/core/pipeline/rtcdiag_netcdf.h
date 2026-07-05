#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/utils/netcdf_io.h>

namespace citlali::pipeline {

inline double rtcdiag_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr int rtcdiag_fill_int() {
    return -2147483647;
}

template <class RtcProc>
double rtc_tod_stream_sample_rate(const RtcProc &rtcproc, double fsmp,
                                  double downsampled_fsmp) {
    return rtcproc.run_downsample ? downsampled_fsmp : fsmp;
}

struct RtcDiagDims {
    netCDF::NcDim n_scans;
    netCDF::NcDim n_dets;
    netCDF::NcDim n_arrays;
    netCDF::NcDim n_nws;
    std::vector<netCDF::NcDim> scan_array;
    std::vector<netCDF::NcDim> det;
    std::vector<netCDF::NcDim> nw;
    std::vector<std::size_t> scan_chunks;
    std::vector<std::size_t> scan_array_chunks;
    std::vector<std::size_t> det_chunks;
    std::vector<std::size_t> nw_chunks;
    std::size_t n_scan_values;
    std::size_t n_array_values;
    std::size_t n_scan_array_values;
    std::size_t n_det_values;
    std::size_t n_nw_values;
};

inline RtcDiagDims add_rtcdiag_dims(netCDF::NcFile &fo,
                                    Eigen::Index n_scans,
                                    Eigen::Index n_dets,
                                    Eigen::Index n_arrays,
                                    Eigen::Index n_nws) {
    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_scans);
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);
    netCDF::NcDim n_arrays_dim = fo.addDim("n_arrays", n_arrays);
    netCDF::NcDim n_nws_dim = fo.addDim("n_nws_rtcdiag", n_nws);

    const auto n_scan_values = static_cast<std::size_t>(n_scans);
    const auto n_array_values = static_cast<std::size_t>(n_arrays);
    const auto n_det_values = static_cast<std::size_t>(n_dets);
    const auto n_nw_values = static_cast<std::size_t>(n_nws);
    const std::vector<netCDF::NcDim> scan_array_dims = {
        n_scans_dim, n_arrays_dim};
    const std::vector<netCDF::NcDim> det_dims = {
        n_scans_dim, n_dets_dim};
    const std::vector<netCDF::NcDim> nw_dims = {
        n_scans_dim, n_nws_dim};
    const std::vector<std::size_t> scan_chunks = {
        static_cast<std::size_t>(std::max<Eigen::Index>(n_scans, 1))};
    const std::vector<std::size_t> scan_array_chunks = {
        1, static_cast<std::size_t>(std::max<Eigen::Index>(n_arrays, 1))};
    const std::vector<std::size_t> det_chunks = {
        1, n_det_values};
    const std::vector<std::size_t> nw_chunks = {
        1, n_nw_values};

    return {
        n_scans_dim,
        n_dets_dim,
        n_arrays_dim,
        n_nws_dim,
        scan_array_dims,
        det_dims,
        nw_dims,
        scan_chunks,
        scan_array_chunks,
        det_chunks,
        nw_chunks,
        n_scan_values,
        n_array_values,
        n_scan_values * n_array_values,
        n_scan_values * n_det_values,
        n_scan_values * n_nw_values};
}

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

template <class LocalResidual>
void add_rtc_local_despike_config_vars(
    netCDF::NcFile &fo, const LocalResidual &local_residual) {
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.ENABLED",
                   local_residual.enabled);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.WINDOW_SEC",
                   local_residual.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.SIGMA_SCALE",
                   local_residual.sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_SIGMA_SCALE",
                   local_residual.delta_sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.EXPAND_WITH_FILTER",
                   local_residual.expand_with_filter);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.EVENT_PADDING_SEC",
                   local_residual.event_padding_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.HIGH_SCORE_EVENT_OVERRIDE",
                   local_residual.high_score_event_override);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.MAX_ADDED_FLAGGED_FRAC",
                   local_residual.max_added_flagged_fraction);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.ENABLED",
                   local_residual.compact_raw_gate.enabled);
    add_netcdf_var(
        fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_REL_SIGMA_SCALE",
        local_residual.compact_raw_gate.candidate_rel_sigma_scale);
    add_netcdf_var(
        fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.CAND_SIGMA_SCALE",
        local_residual.compact_raw_gate.candidate_rel_sigma_scale *
            local_residual.sigma_scale);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.WINDOW_SEC",
                   local_residual.compact_raw_gate.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.HALF_PEAK_FRAC",
                   local_residual.compact_raw_gate.half_peak_frac);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_WIDTH_SEC",
                   local_residual.compact_raw_gate.max_width_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.RAW_GATE.MAX_STEP_SHIFT_Z",
                   local_residual.compact_raw_gate.max_step_shift_z);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.ENABLED",
                   local_residual.compact_delta_gate.enabled);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.WINDOW_SEC",
                   local_residual.compact_delta_gate.window_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.HALF_PEAK_FRAC",
                   local_residual.compact_delta_gate.half_peak_frac);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_WIDTH_SEC",
                   local_residual.compact_delta_gate.max_width_sec);
    add_netcdf_var(fo, "CONFIG.DESPIKE.LOCAL.DELTA_GATE.MAX_STEP_SHIFT_Z",
                   local_residual.compact_delta_gate.max_step_shift_z);
}

template <class StepMask>
void add_rtc_step_mask_config_vars(netCDF::NcFile &fo,
                                   const StepMask &step_mask) {
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.ENABLED",
                   step_mask.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.STEP_WINDOW_SEC",
                   step_mask.step_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.STEP_SCORE_THRESH",
                   step_mask.step_score_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_GOOD_FRAC",
                   step_mask.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_DET_USED",
                   step_mask.min_det_used);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_STEP_DET_FRAC",
                   step_mask.min_step_det_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MIN_ALIGNMENT_FRAC",
                   step_mask.min_alignment_frac);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.CLUSTER_TOL_SEC",
                   step_mask.cluster_tol_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.HALF_WIDTH_SEC",
                   step_mask.mask_half_width_sec);
    add_netcdf_var(fo, "CONFIG.RTC.STEP_MASK.MAX_FLAGGED_FRAC",
                   step_mask.max_flagged_fraction);
}

template <class ImpulsiveCapture>
void add_rtc_impulsive_capture_config_vars(
    netCDF::NcFile &fo, const ImpulsiveCapture &impulsive_capture) {
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.ENABLED",
                   impulsive_capture.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MIN_GOOD_FRAC",
                   impulsive_capture.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MIN_EVENT_Z",
                   impulsive_capture.min_event_z);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.NEAR_EVENT_Z",
                   impulsive_capture.near_event_z);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.MAX_EVENTS",
                   impulsive_capture.max_events_per_network);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.PRE_WINDOW_SEC",
                   impulsive_capture.snippet_pre_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE.POST_WINDOW_SEC",
                   impulsive_capture.snippet_post_window_sec);
}

template <class ImpulsiveCoincidence>
void add_rtc_impulsive_coincidence_config_vars(
    netCDF::NcFile &fo, const ImpulsiveCoincidence &impulsive_coincidence) {
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.ENABLED",
                   impulsive_coincidence.enabled);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_GOOD_FRAC",
                   impulsive_coincidence.min_good_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.EVENT_SCORE_THRESH",
                   impulsive_coincidence.event_score_thresh);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_USED",
                   impulsive_coincidence.min_det_used);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_DET_FRAC",
                   impulsive_coincidence.min_impulsive_det_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_ALIGNMENT_FRAC",
                   impulsive_coincidence.min_alignment_frac);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MIN_NETWORKS_ALIGNED",
                   impulsive_coincidence.min_networks_aligned);
    add_netcdf_var(
        fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_OVERRIDE_THRESH",
        impulsive_coincidence.high_score_override_thresh);
    add_netcdf_var(
        fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.HIGH_SCORE_MIN_NETWORKS",
        impulsive_coincidence.high_score_min_networks_aligned);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.CLUSTER_TOL_SEC",
                   impulsive_coincidence.cluster_tol_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.PRE_WINDOW_SEC",
                   impulsive_coincidence.mask_pre_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.POST_WINDOW_SEC",
                   impulsive_coincidence.mask_post_window_sec);
    add_netcdf_var(fo, "CONFIG.RTC.IMPULSIVE_COINCIDENCE.MAX_FLAGGED_FRAC",
                   impulsive_coincidence.max_flagged_fraction);
}

template <class RtcProc>
void add_rtc_event_mask_config_vars(netCDF::NcFile &fo,
                                    const RtcProc &rtcproc) {
    add_rtc_step_mask_config_vars(fo, rtcproc.network_step_mask);
    add_rtc_impulsive_capture_config_vars(fo, rtcproc.impulsive_capture);
    add_rtc_impulsive_coincidence_config_vars(
        fo, rtcproc.impulsive_coincidence);
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

template <class LineAudit>
void add_rtc_line_audit_config_vars_if_absent(
    netCDF::NcFile &fo, const LineAudit &line_audit) {
    if (fo.getVar("CONFIG.RTC.LINE_AUDIT.ENABLED").isNull()) {
        add_rtc_line_audit_config_vars(fo, line_audit);
    }
}

inline double rtcdiag_percentile_sorted(
    const std::vector<double> &sorted_values, double pct) {
    if (sorted_values.empty()) {
        return rtcdiag_fill_double();
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

struct RtcDiagScanSummaryData {
    std::vector<double> scan_duration_s;
    std::vector<double> scan_speed_p50_arcsec_s;
    std::vector<double> scan_speed_p95_arcsec_s;
    std::vector<double> scan_speed_p995_arcsec_s;
};

template <class Telescope, class Logger>
RtcDiagScanSummaryData calculate_rtcdiag_scan_summary(
    const Telescope &telescope, Eigen::Index n_scans,
    std::size_t n_scan_values, double rad_to_arcsec, double fill_double,
    const Logger &logger) {
    RtcDiagScanSummaryData values{
        std::vector<double>(n_scan_values, fill_double),
        std::vector<double>(n_scan_values, fill_double),
        std::vector<double>(n_scan_values, fill_double),
        std::vector<double>(n_scan_values, fill_double)};
    constexpr double max_tel_sample_step_s = 0.1;
    constexpr double max_pointing_step_rad = 0.01;

    const auto tel_time_it = telescope.tel_data.find("TelTime");
    const auto az_it = telescope.tel_data.find("az_phys");
    const auto alt_it = telescope.tel_data.find("alt_phys");
    const bool has_telescope_motion_data =
        tel_time_it != telescope.tel_data.end() &&
        az_it != telescope.tel_data.end() &&
        alt_it != telescope.tel_data.end();
    if (!has_telescope_motion_data) {
        logger->warn(
            "rtcdiag scan-speed diagnostics skipped: missing TelTime, "
            "az_phys, or alt_phys telescope data");
        return values;
    }

    const auto &tel_time = tel_time_it->second;
    const auto &az_phys = az_it->second;
    const auto &alt_phys = alt_it->second;
    const Eigen::Index n_tel =
        std::min({tel_time.size(), az_phys.size(), alt_phys.size()});
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        const auto scan_index = static_cast<std::size_t>(scan);
        const Eigen::Index start =
            std::max<Eigen::Index>(0, telescope.scan_indices(0, scan));
        const Eigen::Index stop =
            std::min<Eigen::Index>(n_tel - 1,
                                   telescope.scan_indices(1, scan));
        const bool has_valid_scan_bounds =
            stop > start && start >= 0 && stop < n_tel;
        if (!has_valid_scan_bounds) {
            continue;
        }
        const double duration = tel_time(stop) - tel_time(start);
        if (std::isfinite(duration) && duration > 0.0) {
            values.scan_duration_s[scan_index] = duration;
        }
        const auto n_scan_samples = std::max<Eigen::Index>(stop - start, 0);
        std::vector<double> speed_arcsec_s;
        speed_arcsec_s.reserve(static_cast<std::size_t>(n_scan_samples));
        for (Eigen::Index i = start; i < stop; ++i) {
            const double dt = tel_time(i + 1) - tel_time(i);
            const double daz = az_phys(i + 1) - az_phys(i);
            const double dalt = alt_phys(i + 1) - alt_phys(i);
            if (!std::isfinite(dt) || !std::isfinite(daz) ||
                !std::isfinite(dalt) || dt <= 0.0 ||
                dt > max_tel_sample_step_s ||
                std::abs(daz) > max_pointing_step_rad ||
                std::abs(dalt) > max_pointing_step_rad) {
                continue;
            }
            speed_arcsec_s.push_back(
                std::hypot(daz, dalt) / dt * rad_to_arcsec);
        }
        if (!speed_arcsec_s.empty()) {
            std::sort(speed_arcsec_s.begin(), speed_arcsec_s.end());
            values.scan_speed_p50_arcsec_s[scan_index] =
                rtcdiag_percentile_sorted(speed_arcsec_s, 50.0);
            values.scan_speed_p95_arcsec_s[scan_index] =
                rtcdiag_percentile_sorted(speed_arcsec_s, 95.0);
            values.scan_speed_p995_arcsec_s[scan_index] =
                rtcdiag_percentile_sorted(speed_arcsec_s, 99.5);
        }
    }
    return values;
}

inline void add_rtcdiag_scan_summary_outputs(
    netCDF::NcFile &fo, netCDF::NcDim n_scans_dim,
    const std::vector<std::size_t> &scan_chunks,
    const RtcDiagScanSummaryData &values) {
    auto add_scan_double = [&](const std::string &name,
                               const std::string &units,
                               const std::string &comment,
                               const std::vector<double> &data) {
        add_rtcdiag_scan_double(
            fo, name, units, comment, n_scans_dim, scan_chunks, data);
    };
    add_rtcdiag_scan_summary_vars(
        add_scan_double,
        {values.scan_duration_s,
         values.scan_speed_p50_arcsec_s,
         values.scan_speed_p95_arcsec_s,
         values.scan_speed_p995_arcsec_s});
}

struct RtcDiagScanArraySummaryData {
    std::vector<double> source_power_half_bandwidth_hz;
    std::vector<double> tod_lowpass_to_source_power_half_ratio;
};

template <class Calib, class RtcProc>
RtcDiagScanArraySummaryData calculate_rtcdiag_scan_array_summary(
    const Calib &calib, const RtcProc &rtcproc,
    const std::vector<double> &scan_speed_p995_arcsec_s,
    Eigen::Index n_scans, std::size_t n_array_values,
    std::size_t n_scan_array_values, double pi_value, double fwhm_to_std,
    double fill_double) {
    RtcDiagScanArraySummaryData values{
        std::vector<double>(n_scan_array_values, fill_double),
        std::vector<double>(n_scan_array_values, fill_double)};

    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        const auto scan_index = static_cast<std::size_t>(scan);
        const double speed = scan_speed_p995_arcsec_s[scan_index];
        if (!std::isfinite(speed) || speed <= 0.0) {
            continue;
        }
        for (Eigen::Index arr_i = 0; arr_i < calib.n_arrays; ++arr_i) {
            const Eigen::Index array = calib.arrays(arr_i);
            const auto fwhm_it = calib.array_fwhms.find(array);
            if (fwhm_it == calib.array_fwhms.end()) {
                continue;
            }
            const double fwhm_arcsec =
                0.5 * (std::get<0>(fwhm_it->second) +
                       std::get<1>(fwhm_it->second));
            if (!std::isfinite(fwhm_arcsec) || fwhm_arcsec <= 0.0) {
                continue;
            }
            const double f_half_hz =
                (std::sqrt(std::log(2.0)) /
                 (2.0 * pi_value * fwhm_arcsec * fwhm_to_std)) *
                speed;
            const auto flat_i = scan_index * n_array_values +
                                static_cast<std::size_t>(arr_i);
            values.source_power_half_bandwidth_hz[flat_i] = f_half_hz;
            const bool has_lowpass_ratio =
                rtcproc.run_tod_filter &&
                rtcproc.filter.freq_high_Hz > 0.0 && f_half_hz > 0.0;
            if (has_lowpass_ratio) {
                values.tod_lowpass_to_source_power_half_ratio[flat_i] =
                    rtcproc.filter.freq_high_Hz / f_half_hz;
            }
        }
    }
    return values;
}

inline void add_rtcdiag_scan_array_summary_outputs(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &scan_array_dims,
    const std::vector<std::size_t> &scan_array_chunks,
    const RtcDiagScanArraySummaryData &values) {
    auto add_scan_array_double = [&](const std::string &name,
                                     const std::string &units,
                                     const std::string &comment,
                                     const std::vector<double> &data) {
        add_rtcdiag_scan_array_double(
            fo, name, units, comment, scan_array_dims, scan_array_chunks,
            data);
    };
    add_rtcdiag_scan_array_summary_vars(
        add_scan_array_double,
        {values.source_power_half_bandwidth_hz,
         values.tod_lowpass_to_source_power_half_ratio});
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

inline void add_rtcdiag_standard_detector_outputs(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_det_values,
    int fill_int, double fill_double) {
    auto add_rtc_det_double = [&](const std::string &name,
                                  const std::string &comment) {
        add_rtcdiag_det_double(
            fo, name, comment, det_dims, det_chunks, n_det_values,
            fill_double);
    };
    auto add_rtc_det_int = [&](const std::string &name,
                               const std::string &comment) {
        add_rtcdiag_det_int(
            fo, name, comment, det_dims, det_chunks, n_det_values,
            fill_int);
    };

    add_rtcdiag_detector_core_diag(add_rtc_det_int, add_rtc_det_double);
    add_rtcdiag_detector_invvar_window_diag(
        add_rtc_det_int, add_rtc_det_double);
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

template <class AddInt, class AddDouble>
void add_rtcdiag_standard_network_diag(const AddInt &add_int,
                                       const AddDouble &add_double) {
    add_rtcdiag_network_detector_count_diag(add_int);
    add_rtcdiag_network_line_audit_base_diag(add_int, add_double);
    add_rtcdiag_network_line_audit_diag(
        add_int, add_double, "rtc_network_post_line_audit", "post-filter");
    add_rtcdiag_network_step_summary_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_summary_diag(add_int, add_double);
    add_rtcdiag_network_common_mode_diag(add_double);
    add_rtcdiag_network_step_mask_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_mask_window_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_mask_trigger_diag(add_int);
    add_rtcdiag_network_impulsive_mask_candidate_diag(add_int, add_double);
}

inline void add_rtcdiag_standard_network_outputs(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_nw_values,
    int fill_int, double fill_double) {
    auto add_rtc_nw_double = [&](const std::string &name,
                                 const std::string &comment) {
        add_rtcdiag_network_double(
            fo, name, comment, nw_dims, nw_chunks, n_nw_values,
            fill_double);
    };
    auto add_rtc_nw_int = [&](const std::string &name,
                              const std::string &comment) {
        add_rtcdiag_network_int(
            fo, name, comment, nw_dims, nw_chunks, n_nw_values,
            fill_int);
    };

    add_rtcdiag_standard_network_diag(add_rtc_nw_int, add_rtc_nw_double);
}

inline std::vector<int> rtcdiag_impulsive_snippet_offsets(
    std::size_t n_snippet, std::size_t snippet_pre, int fill_value) {
    std::vector<int> offsets(n_snippet, fill_value);
    for (std::size_t i=0; i<n_snippet; ++i) {
        offsets[i] = static_cast<int>(i) - static_cast<int>(snippet_pre);
    }
    return offsets;
}

inline std::size_t rtcdiag_impulsive_window_samples(
    double window_sec, double sample_rate_hz) {
    return static_cast<std::size_t>(
        std::max(0.0, std::round(window_sec * sample_rate_hz)));
}

inline std::size_t rtcdiag_impulsive_snippet_sample_count(
    std::size_t snippet_pre, std::size_t snippet_post) {
    return snippet_pre + snippet_post + 1;
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

template <class Calib, class Rtcproc>
void add_rtcdiag_tod_stream_diag(netCDF::NcFile &fo, const Calib &calib,
                                 const Rtcproc &rtcproc,
                                 netCDF::NcDim n_scans_dim,
                                 netCDF::NcDim n_dets_dim,
                                 Eigen::Index n_scans,
                                 double sample_rate_hz,
                                 int fill_int,
                                 double fill_double) {
    const std::vector<std::size_t> no_chunks;
    std::vector<netCDF::NcDim> rtc_det_dims = {n_scans_dim, n_dets_dim};
    const auto n_det_values =
        static_cast<std::size_t>(n_scans) *
        static_cast<std::size_t>(calib.n_dets);

    auto add_det_double = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_det_double(
            fo, name, comment, rtc_det_dims, no_chunks,
            n_det_values, fill_double);
    };
    auto add_det_int = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_det_int(
            fo, name, comment, rtc_det_dims, no_chunks,
            n_det_values, fill_int);
    };

    add_rtcdiag_detector_core_diag(add_det_int, add_det_double);

    netCDF::NcDim n_nws_rtcdiag_dim =
        fo.addDim("n_nws_rtcdiag", calib.n_nws);
    add_rtcdiag_network_ids(fo, calib, n_nws_rtcdiag_dim, fill_int);

    std::vector<netCDF::NcDim> rtc_nw_dims = {
        n_scans_dim, n_nws_rtcdiag_dim};
    const auto n_nw_values =
        static_cast<std::size_t>(n_scans) *
        static_cast<std::size_t>(calib.n_nws);
    auto add_nw_double = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_network_double(
            fo, name, comment, rtc_nw_dims, no_chunks,
            n_nw_values, fill_double);
    };
    auto add_nw_int = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_network_int(
            fo, name, comment, rtc_nw_dims, no_chunks,
            n_nw_values, fill_int);
    };

    add_rtcdiag_standard_network_diag(add_nw_int, add_nw_double);

    if (!rtcproc.impulsive_capture.enabled) {
        return;
    }

    const auto n_slots = static_cast<std::size_t>(
        std::max<Eigen::Index>(rtcproc.impulsive_capture.max_events_per_network, 1));
    const auto snippet_pre =
        rtcdiag_impulsive_window_samples(
            rtcproc.impulsive_capture.snippet_pre_window_sec,
            sample_rate_hz);
    const auto snippet_post =
        rtcdiag_impulsive_window_samples(
            rtcproc.impulsive_capture.snippet_post_window_sec,
            sample_rate_hz);
    const auto n_snippet =
        rtcdiag_impulsive_snippet_sample_count(snippet_pre, snippet_post);
    netCDF::NcDim n_rtc_impulsive_slots_dim =
        fo.addDim("n_rtc_impulsive_slots", n_slots);
    netCDF::NcDim n_rtc_impulsive_samples_dim =
        fo.addDim("n_rtc_impulsive_samples", n_snippet);

    netCDF::NcVar offset_v =
        fo.addVar("rtc_impulsive_snippet_offset_samples", netCDF::ncInt,
                  n_rtc_impulsive_samples_dim);
    offset_v.putAtt("units", "samples");
    offset_v.putAtt(
        "comment", "sample offsets relative to rtc_impulsive_slot_event_sample");
    const auto offsets =
        rtcdiag_impulsive_snippet_offsets(n_snippet, snippet_pre, fill_int);
    offset_v.putVar(offsets.data());

    std::vector<netCDF::NcDim> slot_dims = {
        n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim};
    std::vector<netCDF::NcDim> snippet_dims = {
        n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim,
        n_rtc_impulsive_samples_dim};
    const auto n_slot_values = n_nw_values * n_slots;
    const auto n_snippet_values = n_slot_values * n_snippet;

    auto add_slot_double = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_impulsive_slot_double(
            fo, name, comment, slot_dims, no_chunks,
            n_slot_values, fill_double);
    };
    auto add_slot_int = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_impulsive_slot_int(
            fo, name, comment, slot_dims, no_chunks,
            n_slot_values, fill_int);
    };
    auto add_snippet_double = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_impulsive_snippet_double(
            fo, name, comment, snippet_dims, no_chunks,
            n_snippet_values, fill_double);
    };
    auto add_snippet_int = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_impulsive_snippet_int(
            fo, name, comment, snippet_dims, no_chunks,
            n_snippet_values, fill_int);
    };

    add_rtcdiag_impulsive_capture_diag(
        add_slot_int, add_slot_double, add_snippet_double, add_snippet_int,
        rtcdiag_impulsive_capture_stream_comments());
}

}  // namespace citlali::pipeline
