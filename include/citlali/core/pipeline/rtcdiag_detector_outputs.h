#pragma once

// Included by rtcdiag_netcdf.h inside namespace citlali::pipeline.

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

