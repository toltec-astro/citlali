#pragma once

// Included by rtcdiag_netcdf.h inside namespace citlali::pipeline.

struct RtcSamplingHwprState {
    enum class AnalysisMode {
        total_intensity,
        hwpr_dependent,
    };
    AnalysisMode analysis_mode = AnalysisMode::total_intensity;

    bool supported() const {
        return analysis_mode == AnalysisMode::total_intensity;
    }
};

inline constexpr std::string_view to_string(
    RtcSamplingHwprState::AnalysisMode mode) {
    return mode == RtcSamplingHwprState::AnalysisMode::total_intensity
        ? "total_intensity" : "hwpr_dependent";
}

struct RtcSamplingCadenceState {
    double requested_output_hz = std::numeric_limits<double>::quiet_NaN();
    double effective_native_hz = std::numeric_limits<double>::quiet_NaN();
    double effective_output_hz = std::numeric_limits<double>::quiet_NaN();
    double realized_native_hz = std::numeric_limits<double>::quiet_NaN();
    double realized_output_hz = std::numeric_limits<double>::quiet_NaN();
    int requested_factor = 1;
    int effective_factor = 1;
    int realized_factor = 1;
    bool realized_downsample_enabled = false;
    bool realized_valid = false;
    std::string requested_effective_consistency{"unavailable_missing"};
    std::string effective_realized_consistency{"unavailable_missing"};
    RtcSamplingReasonCode realized_reason =
        RtcSamplingReasonCode::missing_realized_cadence;
};

inline bool rtc_sampling_cadence_equal(double a, double b) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <=
               32.0 * std::numeric_limits<double>::epsilon() *
                   std::max({1.0, std::abs(a), std::abs(b)});
}

inline std::string rtc_sampling_cadence_consistency(
    double left_hz, int left_factor, double right_hz, int right_factor) {
    if (!std::isfinite(left_hz) || !std::isfinite(right_hz)) {
        return "unavailable_nonfinite";
    }
    if (left_hz <= 0.0 || right_hz <= 0.0 || left_factor <= 0 ||
        right_factor <= 0) {
        return "unavailable_nonpositive";
    }
    return rtc_sampling_cadence_equal(left_hz, right_hz) &&
                   left_factor == right_factor
        ? "consistent" : "mismatch";
}

inline void measure_rtc_sampling_realized_cadence(
    RtcSamplingCadenceState &cadence, const Eigen::VectorXd &time_grid) {
    cadence.realized_valid = false;
    cadence.realized_reason =
        RtcSamplingReasonCode::missing_realized_cadence;
    if (time_grid.size() < 2) {
        return;
    }
    std::vector<double> intervals;
    intervals.reserve(static_cast<std::size_t>(time_grid.size() - 1));
    for (Eigen::Index i = 1; i < time_grid.size(); ++i) {
        const double interval = time_grid(i) - time_grid(i - 1);
        if (!std::isfinite(interval)) {
            cadence.realized_reason =
                RtcSamplingReasonCode::nonfinite_realized_cadence;
            return;
        }
        if (interval <= 0.0) {
            cadence.realized_reason =
                RtcSamplingReasonCode::nonpositive_realized_cadence;
            return;
        }
        intervals.push_back(interval);
    }
    std::sort(intervals.begin(), intervals.end());
    const double median = intervals[intervals.size() / 2];
    const double tolerance = std::max(
        1.0e-9, 64.0 * std::numeric_limits<double>::epsilon() * median);
    if (std::any_of(intervals.begin(), intervals.end(), [&](double interval) {
            return std::abs(interval - median) > tolerance;
        })) {
        cadence.realized_reason =
            RtcSamplingReasonCode::irregular_realized_cadence;
        return;
    }
    cadence.realized_native_hz = 1.0 / median;
    cadence.realized_factor = cadence.realized_downsample_enabled
        ? cadence.realized_factor : 1;
    if (cadence.realized_factor <= 0 ||
        !std::isfinite(cadence.realized_native_hz) ||
        cadence.realized_native_hz <= 0.0) {
        cadence.realized_reason =
            RtcSamplingReasonCode::nonpositive_realized_cadence;
        return;
    }
    cadence.realized_output_hz =
        cadence.realized_native_hz / cadence.realized_factor;
    cadence.realized_valid = true;
    cadence.realized_reason = RtcSamplingReasonCode::none;
    cadence.requested_effective_consistency =
        rtc_sampling_cadence_consistency(
            cadence.requested_output_hz, cadence.requested_factor,
            cadence.effective_output_hz, cadence.effective_factor);
    cadence.effective_realized_consistency =
        rtc_sampling_cadence_consistency(
            cadence.effective_output_hz, cadence.effective_factor,
            cadence.realized_output_hz, cadence.realized_factor);
}

struct RtcSamplingFilterState {
    bool requested_enabled = false;
    bool effective_enabled = false;
    bool realized_enabled = false;
    double requested_a_gibbs = std::numeric_limits<double>::quiet_NaN();
    double effective_a_gibbs = std::numeric_limits<double>::quiet_NaN();
    double realized_a_gibbs = std::numeric_limits<double>::quiet_NaN();
    double requested_low_hz = std::numeric_limits<double>::quiet_NaN();
    double effective_low_hz = std::numeric_limits<double>::quiet_NaN();
    double realized_low_hz = std::numeric_limits<double>::quiet_NaN();
    double requested_high_hz = std::numeric_limits<double>::quiet_NaN();
    double effective_high_hz = std::numeric_limits<double>::quiet_NaN();
    double realized_high_hz = std::numeric_limits<double>::quiet_NaN();
    int requested_n_terms = 0;
    int effective_n_terms = 0;
    int realized_n_terms = 0;
    std::vector<double> realized_coefficients;
};

inline void add_rtcdiag_scan_double(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment, netCDF::NcDim dim,
    const std::vector<std::size_t> &chunks,
    const std::vector<double> &values) {
    auto var = fo.addVar(name, netCDF::ncDouble, dim);
    var.putAtt("units", units);
    var.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(var, chunks, 1);
    var.putVar(values.data());
}

inline void add_rtcdiag_scan_int(
    netCDF::NcFile &fo, const std::string &name, const std::string &comment,
    netCDF::NcDim dim, const std::vector<std::size_t> &chunks,
    const std::vector<int> &values) {
    auto var = fo.addVar(name, netCDF::ncInt, dim);
    var.putAtt("units", "N/A");
    var.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(var, chunks, 1);
    var.putVar(values.data());
}

struct RtcDiagScanSummaryData {
    std::vector<double> scan_duration_s;
    std::vector<double> scan_speed_p95_arcsec_s;
    std::vector<double> scan_speed_p99_arcsec_s;
    std::vector<double> scan_speed_p995_arcsec_s;
    std::vector<double> scan_speed_max_arcsec_s;
    std::vector<double> scan_valid_motion_duration_s;
    std::vector<double> scan_eligible_motion_duration_s;
    std::vector<double> scan_low_velocity_excluded_duration_s;
    std::vector<double> scan_eligible_motion_fraction;
    std::vector<int> scan_motion_status;
    std::vector<int> scan_motion_reason;
    std::vector<int> scan_source_interval_count;
    std::vector<int> scan_valid_interval_count;
    std::vector<int> scan_rejected_interval_count;
    std::vector<int> scan_eligible_interval_count;
    std::vector<int> scan_low_velocity_excluded_count;
    std::vector<int> scan_overlapping_interval_count;
    std::vector<int> scan_boundary_guard_excluded_count;
    std::vector<int> scan_partial_overlap_count;
    std::vector<double> scan_boundary_guard_excluded_duration_s;
    std::vector<double> scan_partial_overlap_duration_s;
    std::vector<RtcSamplingScanMotion> scan_motion;
    std::vector<std::vector<unsigned char>> eligible_grid_by_scan;
};

template <class Telescope, class Logger>
RtcDiagScanSummaryData calculate_rtcdiag_scan_summary(
    const Telescope &telescope,
    const RtcSamplingSourceMotionSupport &source_support,
    const RtcSamplingHwprState &hwpr, Eigen::Index n_scans,
    std::size_t n_scan_values, double fill_double, int fill_int,
    const Logger &logger) {
    auto doubles = [&]() { return std::vector<double>(n_scan_values, fill_double); };
    auto ints = [&]() { return std::vector<int>(n_scan_values, fill_int); };
    RtcDiagScanSummaryData values;
    values.scan_duration_s = doubles();
    values.scan_speed_p95_arcsec_s = doubles();
    values.scan_speed_p99_arcsec_s = doubles();
    values.scan_speed_p995_arcsec_s = doubles();
    values.scan_speed_max_arcsec_s = doubles();
    values.scan_valid_motion_duration_s = doubles();
    values.scan_eligible_motion_duration_s = doubles();
    values.scan_low_velocity_excluded_duration_s = doubles();
    values.scan_eligible_motion_fraction = doubles();
    values.scan_boundary_guard_excluded_duration_s = doubles();
    values.scan_partial_overlap_duration_s = doubles();
    values.scan_motion_status = ints();
    values.scan_motion_reason = ints();
    values.scan_source_interval_count = ints();
    values.scan_valid_interval_count = ints();
    values.scan_rejected_interval_count = ints();
    values.scan_eligible_interval_count = ints();
    values.scan_low_velocity_excluded_count = ints();
    values.scan_overlapping_interval_count = ints();
    values.scan_boundary_guard_excluded_count = ints();
    values.scan_partial_overlap_count = ints();
    values.scan_motion.resize(n_scan_values);
    values.eligible_grid_by_scan.resize(n_scan_values);
    const auto time_it = telescope.tel_data.find("TelTime");
    if (time_it == telescope.tel_data.end() || time_it->second.size() == 0) {
        logger->warn("rtcdiag Stage A skipped: assigned telescope time grid is missing");
        return values;
    }
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        const std::size_t i = static_cast<std::size_t>(scan);
        const Eigen::Index start = telescope.scan_indices(0, scan);
        const Eigen::Index stop = telescope.scan_indices(1, scan);
        if (start < 0 || stop <= start || stop >= time_it->second.size()) {
            values.scan_motion_status[i] =
                static_cast<int>(
                    RtcSamplingStatusCode::prerequisite_unavailable);
            values.scan_motion_reason[i] =
                static_cast<int>(RtcSamplingReasonCode::invalid_output_grid);
            continue;
        }
        values.scan_duration_s[i] = time_it->second(stop) - time_it->second(start);
        if (!hwpr.supported()) {
            RtcSamplingScanMotion unavailable;
            unavailable.duration_s = values.scan_duration_s[i];
            unavailable.reason =
                RtcSamplingReasonCode::unsupported_hwpr;
            values.scan_motion[i] = unavailable;
        }
        else {
            values.scan_motion[i] = calculate_rtc_sampling_scan_motion(
                source_support, time_it->second(start), time_it->second(stop));
        }
        const auto &motion = values.scan_motion[i];
        values.eligible_grid_by_scan[i] = rtc_sampling_eligible_grid_mask(
            source_support, time_it->second, motion);
        values.scan_motion_status[i] = static_cast<int>(motion.status);
        values.scan_motion_reason[i] = static_cast<int>(motion.reason);
        values.scan_source_interval_count[i] = motion.source_interval_count;
        values.scan_valid_interval_count[i] = motion.valid_interval_count;
        values.scan_rejected_interval_count[i] = motion.rejected_interval_count;
        values.scan_eligible_interval_count[i] = motion.eligible_interval_count;
        values.scan_low_velocity_excluded_count[i] =
            motion.low_velocity_excluded_count;
        values.scan_overlapping_interval_count[i] =
            motion.overlapping_interval_count;
        values.scan_boundary_guard_excluded_count[i] =
            motion.boundary_guard_excluded_count;
        values.scan_partial_overlap_count[i] = motion.partial_overlap_count;
        values.scan_valid_motion_duration_s[i] = motion.valid_duration_s;
        values.scan_eligible_motion_duration_s[i] = motion.eligible_duration_s;
        values.scan_low_velocity_excluded_duration_s[i] =
            motion.low_velocity_excluded_duration_s;
        values.scan_boundary_guard_excluded_duration_s[i] =
            motion.boundary_guard_excluded_duration_s;
        values.scan_partial_overlap_duration_s[i] =
            motion.partial_overlap_duration_s;
        values.scan_eligible_motion_fraction[i] = motion.eligible_fraction;
        if (motion.status ==
            RtcSamplingStatusCode::prerequisite_available) {
            values.scan_speed_p95_arcsec_s[i] = motion.speed_p95_arcsec_s;
            values.scan_speed_p99_arcsec_s[i] = motion.speed_p99_arcsec_s;
            values.scan_speed_p995_arcsec_s[i] = motion.speed_p995_arcsec_s;
            values.scan_speed_max_arcsec_s[i] = motion.speed_max_arcsec_s;
        }
    }
    return values;
}

inline void add_rtcdiag_scan_summary_outputs(
    netCDF::NcFile &fo, netCDF::NcDim scan_dim,
    const std::vector<std::size_t> &chunks,
    const RtcDiagScanSummaryData &values) {
    auto add_double = [&](const char *name, const char *units,
                          const char *comment, const auto &data) {
        add_rtcdiag_scan_double(fo, name, units, comment, scan_dim, chunks, data);
    };
    auto add_int = [&](const char *name, const char *comment, const auto &data) {
        add_rtcdiag_scan_int(fo, name, comment, scan_dim, chunks, data);
    };
    add_double("scan_duration_s", "s", "science scan duration on the assigned detector grid", values.scan_duration_s);
    add_double("rtc_sampling_motion_v95_arcsec_s", "arcsec/s", "empirical v95 of valid eligible pre-interpolation source-row intervals; not an upper bound", values.scan_speed_p95_arcsec_s);
    add_double("rtc_sampling_motion_p99_arcsec_s", "arcsec/s", "empirical p99 diagnostic", values.scan_speed_p99_arcsec_s);
    add_double("rtc_sampling_motion_p995_arcsec_s", "arcsec/s", "empirical p99.5 diagnostic", values.scan_speed_p995_arcsec_s);
    add_double("rtc_sampling_motion_max_arcsec_s", "arcsec/s", "raw maximum of valid eligible intervals", values.scan_speed_max_arcsec_s);
    add_double("rtc_sampling_motion_valid_duration_s", "s", "summed valid source-row interval duration", values.scan_valid_motion_duration_s);
    add_double("rtc_sampling_motion_eligible_duration_s", "s", "summed v>=1 arcsec/s eligible duration", values.scan_eligible_motion_duration_s);
    add_double("rtc_sampling_motion_low_velocity_excluded_duration_s", "s", "summed valid duration excluded below 1 arcsec/s", values.scan_low_velocity_excluded_duration_s);
    add_double("rtc_sampling_motion_eligible_fraction", "N/A", "eligible duration divided by valid duration", values.scan_eligible_motion_fraction);
    add_double("rtc_sampling_motion_boundary_guard_excluded_duration_s", "s", "overlap duration excluded by the one-native-source-row guard on each scan boundary", values.scan_boundary_guard_excluded_duration_s);
    add_double("rtc_sampling_motion_partial_overlap_duration_s", "s", "duration of native source intervals that only partially overlap the scan", values.scan_partial_overlap_duration_s);
    add_int("rtc_sampling_motion_status", "status code; see RTC_SAMPLING_STATUS_REASON_VOCABULARY", values.scan_motion_status);
    add_int("rtc_sampling_motion_reason", "reason code; see RTC_SAMPLING_STATUS_REASON_VOCABULARY", values.scan_motion_reason);
    add_int("rtc_sampling_motion_source_interval_count", "source intervals assigned to this scan", values.scan_source_interval_count);
    add_int("rtc_sampling_motion_valid_interval_count", "valid source intervals", values.scan_valid_interval_count);
    add_int("rtc_sampling_motion_rejected_interval_count", "invalid source intervals", values.scan_rejected_interval_count);
    add_int("rtc_sampling_motion_eligible_interval_count", "eligible source intervals", values.scan_eligible_interval_count);
    add_int("rtc_sampling_motion_low_velocity_excluded_count", "valid source intervals excluded below 1 arcsec/s", values.scan_low_velocity_excluded_count);
    add_int("rtc_sampling_motion_overlapping_interval_count", "all native source intervals with positive overlap with this scan before the boundary guard", values.scan_overlapping_interval_count);
    add_int("rtc_sampling_motion_boundary_guard_excluded_count", "overlapping intervals excluded by the one-native-source-row guard", values.scan_boundary_guard_excluded_count);
    add_int("rtc_sampling_motion_partial_overlap_count", "native source intervals partially overlapping a scan boundary", values.scan_partial_overlap_count);
}

struct RtcDiagScanArraySummaryData {
    std::vector<int> candidate_factors;
    std::vector<int> candidate_phases;
    std::vector<double> fir_coefficients;
    std::string fir_digest;
    RtcSamplingStatusCode fir_status =
        RtcSamplingStatusCode::plan_transfer_unavailable;
    RtcSamplingReasonCode fir_reason = RtcSamplingReasonCode::missing_fir;
    bool filter_requested_enabled = false;
    bool filter_effective_enabled = false;
    bool filter_realized_enabled = false;
    double filter_requested_a_gibbs = std::numeric_limits<double>::quiet_NaN();
    double filter_effective_a_gibbs = std::numeric_limits<double>::quiet_NaN();
    double filter_realized_a_gibbs = std::numeric_limits<double>::quiet_NaN();
    double filter_requested_low_hz = std::numeric_limits<double>::quiet_NaN();
    double filter_effective_low_hz = std::numeric_limits<double>::quiet_NaN();
    double filter_realized_low_hz = std::numeric_limits<double>::quiet_NaN();
    double filter_requested_high_hz = std::numeric_limits<double>::quiet_NaN();
    double filter_effective_high_hz = std::numeric_limits<double>::quiet_NaN();
    double filter_realized_high_hz = std::numeric_limits<double>::quiet_NaN();
    int filter_requested_n_terms = 0;
    int filter_effective_n_terms = 0;
    int filter_realized_n_terms = 0;
    std::vector<int> prerequisite_status;
    std::vector<int> prerequisite_reason;
    std::vector<int> candidate_mmax;
    std::vector<int> candidate_range_status;
    std::vector<int> candidate_range_reason;
    std::vector<int> applied_scan_status;
    std::vector<int> applied_scan_reason;
    std::vector<double> beam_fwhm_arcsec;
    std::vector<double> temporal_sigma_s;
    std::vector<int> candidate_status;
    std::vector<int> candidate_reason;
    std::vector<int> candidate_plan_transfer_status;
    std::vector<int> candidate_plan_transfer_reason;
    std::vector<int> candidate_alias_status;
    std::vector<int> candidate_alias_reason;
    std::vector<int> candidate_amplitude_status;
    std::vector<int> candidate_amplitude_reason;
    std::vector<int> candidate_phase_status;
    std::vector<int> candidate_phase_reason;
    std::vector<int> candidate_power_status;
    std::vector<int> candidate_power_reason;
    std::vector<int> candidate_distortion_status;
    std::vector<int> candidate_distortion_reason;
    std::vector<int> candidate_stopband_status;
    std::vector<int> candidate_stopband_reason;
    std::vector<double> output_sample_rate_hz;
    std::vector<double> output_nyquist_hz;
    std::vector<double> samples_per_fwhm;
    std::vector<double> relative_amplitude_at_dc;
    std::vector<double> relative_phase_at_dc_rad;
    std::vector<double> relative_power_at_dc;
    std::vector<double> relative_distortion_at_dc;
    std::vector<double> alias_amplitude_max_lower;
    std::vector<double> alias_amplitude_max_upper;
    std::vector<double> alias_lipschitz_bound;
    std::vector<int> alias_evaluations;
    std::vector<double> relative_amplitude_max_lower;
    std::vector<double> relative_amplitude_max_upper;
    std::vector<double> relative_amplitude_error_enclosure;
    std::vector<double> relative_amplitude_lipschitz_bound;
    std::vector<int> relative_amplitude_evaluations;
    std::vector<double> relative_phase_abs_max_lower_rad;
    std::vector<double> relative_phase_abs_max_upper_rad;
    std::vector<double> relative_phase_error_enclosure_rad;
    std::vector<double> relative_phase_lipschitz_bound;
    std::vector<int> relative_phase_evaluations;
    std::vector<double> relative_power_max_lower;
    std::vector<double> relative_power_max_upper;
    std::vector<double> relative_power_error_enclosure;
    std::vector<double> relative_power_lipschitz_bound;
    std::vector<int> relative_power_evaluations;
    std::vector<double> relative_distortion_max_lower;
    std::vector<double> relative_distortion_max_upper;
    std::vector<double> relative_distortion_error_enclosure;
    std::vector<double> relative_distortion_lipschitz_bound;
    std::vector<int> relative_distortion_evaluations;
    std::vector<double> alias_error_enclosure;
    std::vector<double> stopband_amplitude_max_lower;
    std::vector<double> stopband_amplitude_max_upper;
    std::vector<double> stopband_rejection_db_lower;
    std::vector<double> stopband_rejection_db_upper;
    std::vector<double> stopband_error_enclosure;
    std::vector<double> stopband_lipschitz_bound;
    std::vector<int> stopband_evaluations;
    std::vector<int> numerical_evaluations;
    std::vector<int> tap_count;
    std::vector<int> left_context;
    std::vector<int> right_context;
    std::vector<int> eligible_input_support;
    std::vector<int> candidate_output_count;
    std::vector<int> full_output_count;
    std::vector<int> incomplete_boundary_count;
    std::vector<int> incomplete_gap_count;
    std::vector<int> incomplete_other_count;
    std::vector<int> longest_full_run;
    std::vector<double> full_duration_s;
    std::vector<double> full_fraction;
    RtcSamplingStatusCode candidate_table_status =
        RtcSamplingStatusCode::candidate_table_unavailable_resource_limit;
    RtcSamplingReasonCode candidate_table_reason =
        RtcSamplingReasonCode::candidate_range_resource_limit;
    bool candidate_table_available = false;
    std::size_t estimated_candidate_rows = 0;
    std::size_t estimated_rectangular_storage_cells = 0;
    std::size_t estimated_complex_evaluations = 0;
    std::size_t estimated_context_work_units = 0;
    std::size_t estimated_actual_work_units = 0;
    std::size_t estimated_auxiliary_storage_bytes = 0;
    std::size_t estimated_rtcdiag_bytes = 0;
};

template <class Calib>
auto rtc_sampling_array_detector_count(const Calib &calib,
                                       Eigen::Index array_index, int)
    -> decltype(calib.apt["array"], calib.n_dets, std::size_t{}) {
    std::size_t count = 0;
    const int array_id = calib.arrays(array_index);
    for (Eigen::Index detector = 0; detector < calib.n_dets; ++detector) {
        if (static_cast<int>(calib.apt["array"](detector)) == array_id) {
            ++count;
        }
    }
    return count;
}

template <class Calib>
std::size_t rtc_sampling_array_detector_count(const Calib &,
                                              Eigen::Index, long) {
    return 1;
}

template <class Calib, class Telescope>
RtcDiagScanArraySummaryData calculate_rtcdiag_scan_array_summary(
    const Calib &calib, const RtcSamplingFilterState &filter_state,
    const Telescope &telescope, const RtcDiagScanSummaryData &scan_summary,
    const RtcSamplingHwprState &hwpr,
    const RtcSamplingCadenceState &cadence, Eigen::Index n_scans,
    std::size_t n_array_values, std::size_t n_scan_array_values,
    double fill_double, int fill_int) {
    RtcDiagScanArraySummaryData values;
    if (!filter_state.realized_enabled) {
        values.fir_coefficients = {1.0};
        values.fir_status = RtcSamplingStatusCode::plan_transfer_available;
        values.fir_reason = RtcSamplingReasonCode::none;
    }
    else if (filter_state.realized_coefficients.empty()) {
        values.fir_coefficients = {
            std::numeric_limits<double>::quiet_NaN()};
        values.fir_status = RtcSamplingStatusCode::plan_transfer_unavailable;
        values.fir_reason = RtcSamplingReasonCode::missing_fir;
    }
    else {
        values.fir_coefficients = filter_state.realized_coefficients;
        const bool finite = std::all_of(
            values.fir_coefficients.begin(), values.fir_coefficients.end(),
            [](double coefficient) { return std::isfinite(coefficient); });
        values.fir_status = finite
            ? RtcSamplingStatusCode::plan_transfer_available
            : RtcSamplingStatusCode::plan_transfer_unavailable;
        values.fir_reason = finite ? RtcSamplingReasonCode::none
                                   : RtcSamplingReasonCode::invalid_fir;
    }
    values.fir_digest = rtc_sampling_fir_digest(values.fir_coefficients);
    values.filter_requested_enabled = filter_state.requested_enabled;
    values.filter_effective_enabled = filter_state.effective_enabled;
    values.filter_realized_enabled = filter_state.realized_enabled;
    values.filter_requested_a_gibbs = filter_state.requested_a_gibbs;
    values.filter_effective_a_gibbs = filter_state.effective_a_gibbs;
    values.filter_realized_a_gibbs = filter_state.realized_a_gibbs;
    values.filter_requested_low_hz = filter_state.requested_low_hz;
    values.filter_effective_low_hz = filter_state.effective_low_hz;
    values.filter_realized_low_hz = filter_state.realized_low_hz;
    values.filter_requested_high_hz = filter_state.requested_high_hz;
    values.filter_effective_high_hz = filter_state.effective_high_hz;
    values.filter_realized_high_hz = filter_state.realized_high_hz;
    values.filter_requested_n_terms = filter_state.requested_n_terms;
    values.filter_effective_n_terms = filter_state.effective_n_terms;
    values.filter_realized_n_terms = filter_state.realized_n_terms;
    auto scan_array_int = [&]() {
        return std::vector<int>(n_scan_array_values, fill_int);
    };
    auto scan_array_double = [&]() {
        return std::vector<double>(n_scan_array_values, fill_double);
    };
    values.prerequisite_status = scan_array_int();
    values.prerequisite_reason = scan_array_int();
    values.candidate_mmax = scan_array_int();
    values.candidate_range_status = scan_array_int();
    values.candidate_range_reason = scan_array_int();
    values.applied_scan_status = scan_array_int();
    values.applied_scan_reason = scan_array_int();
    values.beam_fwhm_arcsec = scan_array_double();
    values.temporal_sigma_s = scan_array_double();

    std::vector<unsigned char> include_in_resource_preflight(
        n_scan_array_values, 0);
    std::vector<std::size_t> resource_tap_counts(
        n_scan_array_values, values.fir_coefficients.size());
    std::vector<std::size_t> resource_detector_counts(
        n_scan_array_values, 1);
    std::vector<std::size_t> resource_native_sample_counts(
        n_scan_array_values, 0);
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        for (Eigen::Index arr_i = 0; arr_i < calib.n_arrays; ++arr_i) {
            const std::size_t flat = static_cast<std::size_t>(scan) *
                                         n_array_values +
                                     static_cast<std::size_t>(arr_i);
            const auto beam = rtc_sampling_beam_authority(calib.arrays(arr_i));
            resource_detector_counts[flat] =
                rtc_sampling_array_detector_count(calib, arr_i, 0);
            const auto scan_start = telescope.scan_indices(0, scan);
            const auto scan_stop = telescope.scan_indices(1, scan);
            if (scan_start >= 0 && scan_stop >= scan_start) {
                resource_native_sample_counts[flat] =
                    static_cast<std::size_t>(scan_stop - scan_start + 1);
            }
            values.beam_fwhm_arcsec[flat] = beam.fwhm_arcsec;
            const auto &motion = scan_summary.scan_motion[
                static_cast<std::size_t>(scan)];
            if (!cadence.realized_valid) {
                values.applied_scan_status[flat] =
                    static_cast<int>(
                        RtcSamplingStatusCode::applied_operator_not_applicable);
                values.applied_scan_reason[flat] = static_cast<int>(
                    cadence.realized_reason);
            }
            else if (values.fir_status !=
                     RtcSamplingStatusCode::plan_transfer_available) {
                values.applied_scan_status[flat] =
                    static_cast<int>(
                        RtcSamplingStatusCode::applied_operator_not_applicable);
                values.applied_scan_reason[flat] =
                    static_cast<int>(values.fir_reason);
            }
            else if (!hwpr.supported()) {
                values.applied_scan_status[flat] =
                    static_cast<int>(
                        RtcSamplingStatusCode::applied_operator_not_applicable);
                values.applied_scan_reason[flat] = static_cast<int>(
                    RtcSamplingReasonCode::unsupported_hwpr);
            }
            else if (motion.status !=
                     RtcSamplingStatusCode::prerequisite_available) {
                values.applied_scan_status[flat] =
                    static_cast<int>(
                        RtcSamplingStatusCode::applied_operator_not_applicable);
                values.applied_scan_reason[flat] =
                    static_cast<int>(motion.reason);
            }
            else {
                const int applied_factor =
                    cadence.realized_downsample_enabled
                        ? std::max(1, cadence.realized_factor)
                                               : 1;
                const auto applied_context =
                    calculate_rtc_sampling_complete_context(
                        scan_summary.eligible_grid_by_scan[
                            static_cast<std::size_t>(scan)],
                        telescope.scan_indices(2, scan),
                        telescope.scan_indices(3, scan),
                        telescope.scan_indices(0, scan),
                        telescope.scan_indices(1, scan), applied_factor, 0,
                        values.fir_coefficients.size(),
                        cadence.realized_native_hz);
                values.applied_scan_status[flat] = static_cast<int>(
                    applied_context.full_output_count > 0
                        ? RtcSamplingStatusCode::scan_usable_for_applied_rtc_operator
                        : RtcSamplingStatusCode::scan_unusable_for_applied_rtc_operator);
                values.applied_scan_reason[flat] = static_cast<int>(
                    applied_context.full_output_count > 0
                        ? RtcSamplingReasonCode::none
                        : RtcSamplingReasonCode::no_complete_context);
            }
            if (!cadence.realized_valid) {
                values.prerequisite_status[flat] = static_cast<int>(
                    RtcSamplingStatusCode::prerequisite_unavailable);
                values.prerequisite_reason[flat] = static_cast<int>(
                    cadence.realized_reason);
                continue;
            }
            if (values.fir_status !=
                RtcSamplingStatusCode::plan_transfer_available) {
                values.prerequisite_status[flat] = static_cast<int>(
                    RtcSamplingStatusCode::prerequisite_unavailable);
                values.prerequisite_reason[flat] =
                    static_cast<int>(values.fir_reason);
                continue;
            }
            if (!beam.available) {
                values.prerequisite_status[flat] = static_cast<int>(
                    RtcSamplingStatusCode::prerequisite_unavailable);
                values.prerequisite_reason[flat] =
                    static_cast<int>(beam.reason);
                continue;
            }
            if (!hwpr.supported()) {
                values.prerequisite_status[flat] = static_cast<int>(
                    RtcSamplingStatusCode::prerequisite_unavailable);
                values.prerequisite_reason[flat] = static_cast<int>(
                    RtcSamplingReasonCode::unsupported_hwpr);
                values.candidate_mmax[flat] = 1;
                continue;
            }
            if (motion.status !=
                RtcSamplingStatusCode::prerequisite_available) {
                values.prerequisite_status[flat] = static_cast<int>(
                    RtcSamplingStatusCode::prerequisite_unavailable);
                values.prerequisite_reason[flat] =
                    static_cast<int>(motion.reason);
                continue;
            }
            const int mmax = rtc_sampling_candidate_mmax(
                beam.fwhm_arcsec, cadence.realized_native_hz,
                motion.speed_p95_arcsec_s);
            if (mmax < 0) {
                values.prerequisite_status[flat] = static_cast<int>(
                    RtcSamplingStatusCode::prerequisite_unavailable);
                values.prerequisite_reason[flat] =
                    static_cast<int>(RtcSamplingReasonCode::invalid_cadence);
                continue;
            }
            values.candidate_mmax[flat] = mmax;
            values.temporal_sigma_s[flat] = rtc_sampling_temporal_sigma_s(
                beam.fwhm_arcsec, motion.speed_p95_arcsec_s);
            values.prerequisite_status[flat] = static_cast<int>(
                RtcSamplingStatusCode::prerequisite_available);
            values.prerequisite_reason[flat] =
                static_cast<int>(RtcSamplingReasonCode::none);
            include_in_resource_preflight[flat] = 1;
        }
    }

    const auto resource = rtc_sampling_resource_preflight(
        values.candidate_mmax, include_in_resource_preflight,
        resource_tap_counts, resource_detector_counts,
        resource_native_sample_counts);
    for (std::size_t i = 0; i < n_scan_array_values; ++i) {
        values.candidate_range_status[i] =
            static_cast<int>(resource.range_status[i]);
        values.candidate_range_reason[i] =
            static_cast<int>(resource.range_reason[i]);
        if (resource.range_status[i] ==
            RtcSamplingStatusCode::candidate_not_evaluated_prerequisite) {
            values.candidate_range_reason[i] = values.prerequisite_reason[i];
        }
    }
    values.candidate_table_status = resource.table_status;
    values.candidate_table_reason = resource.table_reason;
    values.candidate_table_available = resource.table_available;
    values.estimated_candidate_rows = resource.logical_candidate_rows;
    values.estimated_rectangular_storage_cells =
        resource.rectangular_storage_cells;
    values.estimated_complex_evaluations =
        resource.estimated_complex_evaluations;
    values.estimated_context_work_units =
        resource.estimated_context_work_units;
    values.estimated_actual_work_units =
        resource.estimated_actual_work_units;
    values.estimated_auxiliary_storage_bytes =
        resource.estimated_auxiliary_storage_bytes;
    values.estimated_rtcdiag_bytes = resource.estimated_rtcdiag_bytes;
    if (!values.candidate_table_available) {
        return values;
    }

    values.candidate_factors.resize(resource.candidate_axis_size);
    std::iota(values.candidate_factors.begin(), values.candidate_factors.end(), 1);
    values.candidate_phases.assign(resource.candidate_axis_size, 0);

    const std::size_t table_size = resource.rectangular_storage_cells;
    auto table_int = [&]() { return std::vector<int>(table_size, fill_int); };
    auto table_double = [&]() {
        return std::vector<double>(table_size, fill_double);
    };
    values.candidate_status = table_int(); values.candidate_reason = table_int();
    values.candidate_plan_transfer_status = table_int();
    values.candidate_plan_transfer_reason = table_int();
    values.candidate_alias_status = table_int(); values.candidate_alias_reason = table_int();
    values.candidate_amplitude_status = table_int(); values.candidate_amplitude_reason = table_int();
    values.candidate_phase_status = table_int(); values.candidate_phase_reason = table_int();
    values.candidate_power_status = table_int(); values.candidate_power_reason = table_int();
    values.candidate_distortion_status = table_int(); values.candidate_distortion_reason = table_int();
    values.candidate_stopband_status = table_int(); values.candidate_stopband_reason = table_int();
    values.output_sample_rate_hz = table_double(); values.output_nyquist_hz = table_double();
    values.samples_per_fwhm = table_double(); values.relative_amplitude_at_dc = table_double();
    values.relative_phase_at_dc_rad = table_double(); values.relative_power_at_dc = table_double();
    values.relative_distortion_at_dc = table_double(); values.alias_amplitude_max_lower = table_double();
    values.alias_amplitude_max_upper = table_double(); values.relative_amplitude_max_upper = table_double();
    values.alias_lipschitz_bound = table_double(); values.alias_evaluations = table_int();
    values.relative_amplitude_max_lower = table_double();
    values.relative_amplitude_error_enclosure = table_double();
    values.relative_amplitude_lipschitz_bound = table_double();
    values.relative_amplitude_evaluations = table_int();
    values.relative_phase_abs_max_lower_rad = table_double();
    values.relative_phase_abs_max_upper_rad = table_double(); values.relative_power_max_upper = table_double();
    values.relative_phase_error_enclosure_rad = table_double();
    values.relative_phase_lipschitz_bound = table_double();
    values.relative_phase_evaluations = table_int();
    values.relative_power_max_lower = table_double();
    values.relative_power_error_enclosure = table_double();
    values.relative_power_lipschitz_bound = table_double();
    values.relative_power_evaluations = table_int();
    values.relative_distortion_max_lower = table_double();
    values.relative_distortion_max_upper = table_double(); values.alias_error_enclosure = table_double();
    values.relative_distortion_error_enclosure = table_double();
    values.relative_distortion_lipschitz_bound = table_double();
    values.relative_distortion_evaluations = table_int();
    values.stopband_amplitude_max_lower = table_double(); values.stopband_amplitude_max_upper = table_double();
    values.stopband_rejection_db_lower = table_double(); values.stopband_rejection_db_upper = table_double();
    values.stopband_error_enclosure = table_double();
    values.stopband_lipschitz_bound = table_double();
    values.stopband_evaluations = table_int();
    values.numerical_evaluations = table_int();
    values.tap_count = table_int(); values.left_context = table_int(); values.right_context = table_int();
    values.eligible_input_support = table_int(); values.candidate_output_count = table_int();
    values.full_output_count = table_int(); values.incomplete_boundary_count = table_int();
    values.incomplete_gap_count = table_int(); values.incomplete_other_count = table_int();
    values.longest_full_run = table_int(); values.full_duration_s = table_double();
    values.full_fraction = table_double();

    const std::size_t candidate_count = values.candidate_factors.size();
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        for (Eigen::Index arr_i = 0; arr_i < calib.n_arrays; ++arr_i) {
            const std::size_t flat = static_cast<std::size_t>(scan) * n_array_values + arr_i;
            for (std::size_t ci = 0; ci < candidate_count; ++ci) {
                const std::size_t table = flat * candidate_count + ci;
                const int factor = values.candidate_factors[ci];
                if (resource.range_status[flat] ==
                    RtcSamplingStatusCode::candidate_range_resource_limit) {
                    continue;
                }
                if (resource.range_status[flat] ==
                    RtcSamplingStatusCode::candidate_range_available &&
                    factor > std::max(1, values.candidate_mmax[flat])) {
                    continue;
                }
                if (values.prerequisite_status[flat] != static_cast<int>(
                        RtcSamplingStatusCode::prerequisite_available)) {
                    if (factor != 1) {
                        continue;
                    }
                    const int reason = values.prerequisite_reason[flat];
                    values.candidate_status[table] = static_cast<int>(
                        RtcSamplingStatusCode::candidate_not_evaluated_prerequisite);
                    values.candidate_reason[table] = reason;
                    values.candidate_plan_transfer_status[table] =
                        static_cast<int>(
                            RtcSamplingStatusCode::plan_transfer_unavailable);
                    values.candidate_plan_transfer_reason[table] = reason;
                    const int failed = static_cast<int>(
                        RtcSamplingStatusCode::numerical_failed);
                    values.candidate_alias_status[table] = failed;
                    values.candidate_alias_reason[table] = reason;
                    values.candidate_amplitude_status[table] = failed;
                    values.candidate_amplitude_reason[table] = reason;
                    values.candidate_phase_status[table] = failed;
                    values.candidate_phase_reason[table] = reason;
                    values.candidate_power_status[table] = failed;
                    values.candidate_power_reason[table] = reason;
                    values.candidate_distortion_status[table] = failed;
                    values.candidate_distortion_reason[table] = reason;
                    values.candidate_stopband_status[table] = failed;
                    values.candidate_stopband_reason[table] = reason;
                    continue;
                }
                const auto metrics = calculate_rtc_sampling_candidate_metrics(
                    factor, cadence.realized_native_hz,
                    values.fir_coefficients,
                    values.temporal_sigma_s[flat]);
                const auto context = calculate_rtc_sampling_complete_context(
                    scan_summary.eligible_grid_by_scan[
                        static_cast<std::size_t>(scan)],
                    telescope.scan_indices(2, scan),
                    telescope.scan_indices(3, scan), telescope.scan_indices(0, scan),
                    telescope.scan_indices(1, scan), factor, 0,
                    values.fir_coefficients.size(),
                    cadence.realized_native_hz);
                values.candidate_status[table] = static_cast<int>(context.candidate_status);
                values.candidate_reason[table] = static_cast<int>(context.candidate_reason);
                values.candidate_plan_transfer_status[table] =
                    static_cast<int>(
                        RtcSamplingStatusCode::plan_transfer_available);
                values.candidate_plan_transfer_reason[table] =
                    static_cast<int>(RtcSamplingReasonCode::none);
                values.candidate_alias_status[table] = static_cast<int>(metrics.alias_status);
                values.candidate_alias_reason[table] = static_cast<int>(metrics.alias_reason);
                values.candidate_amplitude_status[table] = static_cast<int>(metrics.amplitude_status);
                values.candidate_amplitude_reason[table] = static_cast<int>(metrics.amplitude_reason);
                values.candidate_phase_status[table] = static_cast<int>(metrics.phase_status);
                values.candidate_phase_reason[table] = static_cast<int>(metrics.phase_reason);
                values.candidate_power_status[table] = static_cast<int>(metrics.power_status);
                values.candidate_power_reason[table] = static_cast<int>(metrics.power_reason);
                values.candidate_distortion_status[table] = static_cast<int>(metrics.distortion_status);
                values.candidate_distortion_reason[table] = static_cast<int>(metrics.distortion_reason);
                values.candidate_stopband_status[table] = static_cast<int>(metrics.stopband_status);
                values.candidate_stopband_reason[table] = static_cast<int>(metrics.stopband_reason);
                values.output_sample_rate_hz[table] = metrics.output_sample_rate_hz;
                values.output_nyquist_hz[table] = metrics.output_nyquist_hz;
                values.samples_per_fwhm[table] = metrics.samples_per_fwhm;
                values.relative_amplitude_at_dc[table] = metrics.relative_amplitude_at_dc;
                values.relative_phase_at_dc_rad[table] = metrics.relative_phase_at_dc_rad;
                values.relative_power_at_dc[table] = metrics.relative_power_at_dc;
                values.relative_distortion_at_dc[table] = metrics.relative_distortion_at_dc;
                values.alias_amplitude_max_lower[table] = metrics.alias_amplitude_max_lower;
                values.alias_amplitude_max_upper[table] = metrics.alias_amplitude_max_upper;
                values.alias_lipschitz_bound[table] = metrics.alias_lipschitz_bound;
                values.alias_evaluations[table] = static_cast<int>(metrics.alias_evaluations);
                values.relative_amplitude_max_lower[table] = metrics.relative_amplitude_max_lower;
                values.relative_amplitude_max_upper[table] = metrics.relative_amplitude_max_upper;
                values.relative_amplitude_error_enclosure[table] = metrics.relative_amplitude_error_enclosure;
                values.relative_amplitude_lipschitz_bound[table] = metrics.relative_amplitude_lipschitz_bound;
                values.relative_amplitude_evaluations[table] = static_cast<int>(metrics.relative_amplitude_evaluations);
                values.relative_phase_abs_max_lower_rad[table] = metrics.relative_phase_abs_max_lower_rad;
                values.relative_phase_abs_max_upper_rad[table] = metrics.relative_phase_abs_max_upper_rad;
                values.relative_phase_error_enclosure_rad[table] = metrics.relative_phase_error_enclosure_rad;
                values.relative_phase_lipschitz_bound[table] = metrics.relative_phase_lipschitz_bound;
                values.relative_phase_evaluations[table] = static_cast<int>(metrics.relative_phase_evaluations);
                values.relative_power_max_lower[table] = metrics.relative_power_max_lower;
                values.relative_power_max_upper[table] = metrics.relative_power_max_upper;
                values.relative_power_error_enclosure[table] = metrics.relative_power_error_enclosure;
                values.relative_power_lipschitz_bound[table] = metrics.relative_power_lipschitz_bound;
                values.relative_power_evaluations[table] = static_cast<int>(metrics.relative_power_evaluations);
                values.relative_distortion_max_lower[table] = metrics.relative_distortion_max_lower;
                values.relative_distortion_max_upper[table] = metrics.relative_distortion_max_upper;
                values.relative_distortion_error_enclosure[table] = metrics.relative_distortion_error_enclosure;
                values.relative_distortion_lipschitz_bound[table] = metrics.relative_distortion_lipschitz_bound;
                values.relative_distortion_evaluations[table] = static_cast<int>(metrics.relative_distortion_evaluations);
                values.alias_error_enclosure[table] = metrics.alias_error_enclosure;
                values.stopband_amplitude_max_lower[table] = metrics.stopband_amplitude_max_lower;
                values.stopband_amplitude_max_upper[table] = metrics.stopband_amplitude_max_upper;
                values.stopband_rejection_db_lower[table] = metrics.stopband_rejection_db_lower;
                values.stopband_rejection_db_upper[table] = metrics.stopband_rejection_db_upper;
                values.stopband_error_enclosure[table] = metrics.stopband_error_enclosure;
                values.stopband_lipschitz_bound[table] = metrics.stopband_lipschitz_bound;
                values.stopband_evaluations[table] = static_cast<int>(metrics.stopband_evaluations);
                values.numerical_evaluations[table] = metrics.numerical_evaluations;
                values.tap_count[table] = context.tap_count;
                values.left_context[table] = context.left_context;
                values.right_context[table] = context.right_context;
                values.eligible_input_support[table] = context.eligible_input_support;
                values.candidate_output_count[table] = context.candidate_output_count;
                values.full_output_count[table] = context.full_output_count;
                values.incomplete_boundary_count[table] = context.incomplete_boundary_count;
                values.incomplete_gap_count[table] = context.incomplete_gap_count;
                values.incomplete_other_count[table] = context.incomplete_other_count;
                values.longest_full_run[table] = context.longest_full_run;
                values.full_duration_s[table] = context.full_duration_s;
                values.full_fraction[table] = context.full_fraction;
            }
        }
    }
    return values;
}

inline void add_rtc_sampling_table_double(netCDF::NcFile &fo,
    const std::string &name, const std::string &units, const std::string &comment,
    const std::vector<netCDF::NcDim> &dims, const std::vector<std::size_t> &chunks,
    const std::vector<double> &data) {
    auto var = fo.addVar(name, netCDF::ncDouble, dims); var.putAtt("units", units);
    var.putAtt("comment", comment); set_netcdf_chunking_and_compression(var, chunks, 1);
    var.putVar(data.data());
}

inline void add_rtc_sampling_table_int(netCDF::NcFile &fo,
    const std::string &name, const std::string &comment,
    const std::vector<netCDF::NcDim> &dims, const std::vector<std::size_t> &chunks,
    const std::vector<int> &data) {
    auto var = fo.addVar(name, netCDF::ncInt, dims); var.putAtt("units", "N/A");
    var.putAtt("comment", comment); set_netcdf_chunking_and_compression(var, chunks, 1);
    var.putVar(data.data());
}

inline void add_rtcdiag_scan_array_summary_outputs(netCDF::NcFile &fo,
    const std::vector<netCDF::NcDim> &scan_array_dims,
    const std::vector<std::size_t> &scan_array_chunks,
    const RtcDiagScanArraySummaryData &values,
    const RtcSamplingHwprState &hwpr, const RtcSamplingCadenceState &cadence,
    const RtcSamplingSourceMotionSupport &source_support,
    const std::string &raw_manifest_reference, const std::string &commit) {
    add_netcdf_var(fo, "RTC_DIAG_SCHEMA_VERSION", std::string{rtc_sampling_schema_version});
    add_netcdf_var(fo, "RTC_SAMPLING_ALGORITHM_VERSION", std::string{rtc_sampling_algorithm_version});
    add_netcdf_var(fo, "RTC_SAMPLING_STATUS_REASON_VOCABULARY", rtc_sampling_status_reason_vocabulary());
    add_netcdf_var(fo, "RTC_SAMPLING_METRICS_NOTICE", std::string{"Observe-only diagnostic characterization; no factor is ranked, recommended, selected, or applied."});
    add_netcdf_var(fo, "RTC_SAMPLING_BEAM_MODEL", std::string{rtc_sampling_beam_model});
    add_netcdf_var(fo, "RTC_SAMPLING_BEAM_FWHM_AUTHORITY", std::string{rtc_sampling_beam_fwhm_authority});
    add_netcdf_var(fo, "RTC_SAMPLING_ALIAS_CONVENTION", std::string{rtc_sampling_alias_convention});
    add_netcdf_var(fo, "RTC_SAMPLING_NUMERICAL_METHOD", std::string{rtc_sampling_numerical_method});
    add_netcdf_var(fo, "RTC_SAMPLING_NUMERICAL_DOMAIN", std::string{"alias and relative response on [-fs/(2M),fs/(2M)) using nextafter at the upper endpoint; FIR stopband on [fs/(2M),fs/2]"});
    add_netcdf_var(fo, "RTC_SAMPLING_NUMERICAL_ENCLOSURE", std::string{"uniform closed subintervals; sampled lower bound plus global analytic Lipschitz bound times half partition width; nonzero enclosure is bounded_not_converged"});
    add_netcdf_var(fo, "RTC_SAMPLING_NUMERICAL_PARTITIONS", static_cast<int>(rtc_sampling_numerical_partitions));
    add_netcdf_var(fo, "RTC_SAMPLING_FIR_DIGEST_CONVENTION", std::string{rtc_sampling_fir_digest_convention});
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_GUARD_VERSION", std::string{rtc_sampling_source_guard_version});
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_MAX_INTERVAL_S", rtc_sampling_source_max_interval_s);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_MAX_POINTING_STEP_RAD", rtc_sampling_source_max_pointing_step_rad);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_BOUNDARY_GUARD_ROWS", static_cast<int>(rtc_sampling_source_boundary_guard_rows));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_SCAN_BOUNDARY_CONVENTION", std::string{"native source rows are assigned by row timestamps; for each scan, the first row at-or-after scan start and last row at-or-before scan stop are found, exactly one native row is excluded at each end, and only intervals whose two endpoint rows remain inside that guarded inclusive row range contribute; partial overlaps remain diagnosed but do not contribute"});
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_SPEED_VALID_MAX_ARCSEC_S", rtc_sampling_source_max_speed_arcsec_s);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_SPEED_ELIGIBLE_MIN_ARCSEC_S", rtc_sampling_source_min_eligible_speed_arcsec_s);
    add_netcdf_var(fo, "RTC_SAMPLING_FIR_DIGEST", values.fir_digest);
    add_netcdf_var(fo, "RTC_SAMPLING_FIR_STATUS", static_cast<int>(values.fir_status));
    add_netcdf_var(fo, "RTC_SAMPLING_FIR_REASON", static_cast<int>(values.fir_reason));
    add_netcdf_var(fo, "RTC_SAMPLING_COUNTERFACTUAL_BINDING", std::string{"(M,phase=0,H_RTC_realized); same exact realized FIR for every unranked factor; no factor-specific synthesis"});
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REQUESTED_ENABLED", values.filter_requested_enabled);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_EFFECTIVE_ENABLED", values.filter_effective_enabled);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REALIZED_ENABLED", values.filter_realized_enabled);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REQUESTED_A_GIBBS", values.filter_requested_a_gibbs);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_EFFECTIVE_A_GIBBS", values.filter_effective_a_gibbs);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REALIZED_A_GIBBS", values.filter_realized_a_gibbs);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REQUESTED_LOW_HZ", values.filter_requested_low_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_EFFECTIVE_LOW_HZ", values.filter_effective_low_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REALIZED_LOW_HZ", values.filter_realized_low_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REQUESTED_HIGH_HZ", values.filter_requested_high_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_EFFECTIVE_HIGH_HZ", values.filter_effective_high_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REALIZED_HIGH_HZ", values.filter_realized_high_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REQUESTED_N_TERMS", values.filter_requested_n_terms);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_EFFECTIVE_N_TERMS", values.filter_effective_n_terms);
    add_netcdf_var(fo, "RTC_SAMPLING_FILTER_REALIZED_N_TERMS", values.filter_realized_n_terms);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_SUPPORT_AUTHORITY", source_support.authority);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_SUPPORT_COORDINATE_IDENTITY", source_support.coordinate_identity);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_SUPPORT_STATUS", source_support.status);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_SUPPORT_REASON", source_support.reason);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_OBSERVATION_IDENTITY_AVAILABLE",
                   source_support.observation_identity_available);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_OBSERVATION_INDEX",
                   static_cast<long long>(source_support.observation_index));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_OBSNUM",
                   source_support.observation_obsnum);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_TELESCOPE_PATH",
                   source_support.telescope_source_path);
    add_netcdf_var(
        fo, "RTC_SAMPLING_SOURCE_SUPPORT_REASON_CODE",
        static_cast<int>(rtc_sampling_source_support_reason_code(
            source_support.reason)));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_ROW_COUNT", static_cast<int>(source_support.source_row_count));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_INTERVAL_COUNT", static_cast<int>(source_support.interval_count));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_VALID_INTERVAL_COUNT", static_cast<int>(source_support.valid_interval_count));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_REJECTED_INTERVAL_COUNT", static_cast<int>(source_support.rejected_interval_count));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_ELIGIBLE_INTERVAL_COUNT", static_cast<int>(source_support.eligible_interval_count));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_LOW_VELOCITY_EXCLUDED_COUNT", static_cast<int>(source_support.low_velocity_excluded_count));
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_VALID_DURATION_S", source_support.valid_duration_s);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_ELIGIBLE_DURATION_S", source_support.eligible_duration_s);
    add_netcdf_var(fo, "RTC_SAMPLING_SOURCE_LOW_VELOCITY_EXCLUDED_DURATION_S", source_support.low_velocity_excluded_duration_s);
    add_netcdf_var(
        fo, "RTC_SAMPLING_SOURCE_ELIGIBLE_FRACTION",
        source_support.valid_duration_s > 0.0
            ? source_support.eligible_duration_s /
                  source_support.valid_duration_s
            : std::numeric_limits<double>::quiet_NaN());
    if (!source_support.intervals.empty()) {
        const auto source_interval_dim = fo.addDim(
            "n_rtc_sampling_source_intervals",
            source_support.intervals.size());
        const std::vector<netCDF::NcDim> source_interval_dims{
            source_interval_dim};
        const std::vector<std::size_t> source_interval_chunks{
            std::min<std::size_t>(source_support.intervals.size(), 4096)};
        std::vector<double> start(source_support.intervals.size());
        std::vector<double> stop(source_support.intervals.size());
        std::vector<double> duration(source_support.intervals.size());
        std::vector<double> speed(source_support.intervals.size());
        std::vector<int> start_row(source_support.intervals.size());
        std::vector<int> stop_row(source_support.intervals.size());
        std::vector<int> valid(source_support.intervals.size());
        std::vector<int> eligible(source_support.intervals.size());
        std::vector<int> reason(source_support.intervals.size());
        for (std::size_t i = 0; i < source_support.intervals.size(); ++i) {
            const auto &interval = source_support.intervals[i];
            start[i] = interval.start_time_s;
            stop[i] = interval.stop_time_s;
            duration[i] = interval.duration_s;
            speed[i] = interval.speed_arcsec_s;
            start_row[i] = static_cast<int>(std::min(
                interval.start_row_index,
                static_cast<std::size_t>(std::numeric_limits<int>::max())));
            stop_row[i] = static_cast<int>(std::min(
                interval.stop_row_index,
                static_cast<std::size_t>(std::numeric_limits<int>::max())));
            valid[i] = interval.valid ? 1 : 0;
            eligible[i] = interval.eligible ? 1 : 0;
            reason[i] = static_cast<int>(
                rtc_sampling_source_interval_reason_code(interval.reason));
        }
        add_rtc_sampling_table_double(
            fo, "rtc_sampling_source_interval_start_time_s", "s",
            "authoritative pre-interpolation source interval start",
            source_interval_dims, source_interval_chunks, start);
        add_rtc_sampling_table_double(
            fo, "rtc_sampling_source_interval_stop_time_s", "s",
            "authoritative pre-interpolation source interval stop",
            source_interval_dims, source_interval_chunks, stop);
        add_rtc_sampling_table_double(
            fo, "rtc_sampling_source_interval_duration_s", "s",
            "source interval duration", source_interval_dims,
            source_interval_chunks, duration);
        add_rtc_sampling_table_double(
            fo, "rtc_sampling_source_interval_speed_arcsec_s", "arcsec/s",
            "source-telescope tangent-plane speed; valid only when interval_valid is one",
            source_interval_dims, source_interval_chunks, speed);
        add_rtc_sampling_table_int(
            fo, "rtc_sampling_source_interval_start_row",
            "zero-based native pre-interpolation source-row index",
            source_interval_dims, source_interval_chunks, start_row);
        add_rtc_sampling_table_int(
            fo, "rtc_sampling_source_interval_stop_row",
            "zero-based native pre-interpolation source-row index",
            source_interval_dims, source_interval_chunks, stop_row);
        add_rtc_sampling_table_int(
            fo, "rtc_sampling_source_interval_valid",
            "one when the source interval passes validity bounds",
            source_interval_dims, source_interval_chunks, valid);
        add_rtc_sampling_table_int(
            fo, "rtc_sampling_source_interval_eligible",
            "one when valid and speed is at least 1 arcsec/s",
            source_interval_dims, source_interval_chunks, eligible);
        add_rtc_sampling_table_int(
            fo, "rtc_sampling_source_interval_reason",
            "stable reason code; see RTC_SAMPLING_STATUS_REASON_VOCABULARY",
            source_interval_dims, source_interval_chunks, reason);
    }
    add_netcdf_var(fo, "RTC_SAMPLING_RAW_MANIFEST_REFERENCE", raw_manifest_reference);
    add_netcdf_var(fo, "RTC_SAMPLING_CITLALI_COMMIT", commit);
    add_netcdf_var(fo, "RTC_SAMPLING_ANALYSIS_MODE",
                   std::string{to_string(hwpr.analysis_mode)});
    add_netcdf_var(fo, "RTC_SAMPLING_HWPR_STATUS",
                   std::string{hwpr.supported()
                       ? "prerequisite_available" : "unsupported_hwpr"});
    add_netcdf_var(fo, "RTC_SAMPLING_HWPR_REASON",
                   std::string{hwpr.supported() ? "none" : "unsupported_hwpr"});
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_REQUESTED_OUTPUT_HZ", cadence.requested_output_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_EFFECTIVE_NATIVE_HZ", cadence.effective_native_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_EFFECTIVE_OUTPUT_HZ", cadence.effective_output_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_REALIZED_NATIVE_HZ", cadence.realized_native_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_REALIZED_OUTPUT_HZ", cadence.realized_output_hz);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_REQUESTED_FACTOR", cadence.requested_factor);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_EFFECTIVE_FACTOR", cadence.effective_factor);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_REALIZED_FACTOR", cadence.realized_factor);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_REALIZED_VALID",
                   cadence.realized_valid);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_REALIZED_REASON",
                   std::string{to_string(cadence.realized_reason)});
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_REQUESTED_EFFECTIVE_CONSISTENCY",
                   cadence.requested_effective_consistency);
    add_netcdf_var(fo, "RTC_SAMPLING_CADENCE_EFFECTIVE_REALIZED_CONSISTENCY",
                   cadence.effective_realized_consistency);
    add_netcdf_var(fo, "RTC_SAMPLING_MAX_CANDIDATES_PER_SCAN_ARRAY", static_cast<double>(rtc_sampling_max_candidates));
    add_netcdf_var(fo, "RTC_SAMPLING_MAX_TOTAL_CANDIDATE_ROWS", static_cast<double>(rtc_sampling_max_candidate_rows));
    add_netcdf_var(fo, "RTC_SAMPLING_MAX_ACTUAL_WORK_UNITS", static_cast<double>(rtc_sampling_max_actual_work_units));
    add_netcdf_var(fo, "RTC_SAMPLING_MAX_ESTIMATED_RTCDIAG_BYTES", static_cast<double>(rtc_sampling_max_estimated_rtcdiag_bytes));
    add_netcdf_var(fo, "RTC_SAMPLING_ESTIMATED_CANDIDATE_ROW_BYTES", static_cast<double>(rtc_sampling_estimated_candidate_row_bytes));
    add_netcdf_var(fo, "RTC_SAMPLING_CANDIDATE_TABLE_STATUS", static_cast<int>(values.candidate_table_status));
    add_netcdf_var(fo, "RTC_SAMPLING_CANDIDATE_TABLE_REASON", static_cast<int>(values.candidate_table_reason));
    add_netcdf_var(fo, "RTC_SAMPLING_CANDIDATE_TABLE_AVAILABLE", values.candidate_table_available);
    add_netcdf_var(fo, "RTC_SAMPLING_CANDIDATE_COUNT",
                   static_cast<long long>(values.candidate_table_available
                       ? values.candidate_factors.size() : 0));
    add_netcdf_var(fo, "RTC_SAMPLING_ESTIMATED_CANDIDATE_ROWS", static_cast<double>(values.estimated_candidate_rows));
    add_netcdf_var(fo, "RTC_SAMPLING_ESTIMATED_RECTANGULAR_STORAGE_CELLS", static_cast<double>(values.estimated_rectangular_storage_cells));
    add_netcdf_var(fo, "RTC_SAMPLING_ESTIMATED_COMPLEX_EVALUATIONS", static_cast<double>(values.estimated_complex_evaluations));
    add_netcdf_var(fo, "RTC_SAMPLING_ESTIMATED_CONTEXT_WORK_UNITS", static_cast<double>(values.estimated_context_work_units));
    add_netcdf_var(fo, "RTC_SAMPLING_ESTIMATED_ACTUAL_WORK_UNITS", static_cast<double>(values.estimated_actual_work_units));
    add_netcdf_var(fo, "RTC_SAMPLING_ESTIMATED_AUXILIARY_STORAGE_BYTES", static_cast<double>(values.estimated_auxiliary_storage_bytes));
    add_netcdf_var(fo, "RTC_SAMPLING_ESTIMATED_RTCDIAG_BYTES", static_cast<double>(values.estimated_rtcdiag_bytes));

    auto add_sa_d = [&](const char *n, const char *u, const char *c, const auto &d) {
        add_rtc_sampling_table_double(fo, n, u, c, scan_array_dims, scan_array_chunks, d);
    };
    auto add_sa_i = [&](const char *n, const char *c, const auto &d) {
        add_rtc_sampling_table_int(fo, n, c, scan_array_dims, scan_array_chunks, d);
    };
    add_sa_i("rtc_sampling_prerequisite_status", "scan-array prerequisite status", values.prerequisite_status);
    add_sa_i("rtc_sampling_prerequisite_reason", "scan-array prerequisite reason", values.prerequisite_reason);
    add_sa_i("rtc_sampling_candidate_mmax", "full scientific range endpoint floor(theta*fs/v95)", values.candidate_mmax);
    add_sa_i("rtc_sampling_candidate_range_status", "per-scan-array full-range availability; a resource-limited range preserves Mmax and has no candidate rows", values.candidate_range_status);
    add_sa_i("rtc_sampling_candidate_range_reason", "per-scan-array candidate-range reason", values.candidate_range_reason);
    add_sa_i("rtc_sampling_applied_scan_status", "separate observe-only status for the actually applied RTC operator", values.applied_scan_status);
    add_sa_i("rtc_sampling_applied_scan_reason", "cause for applied-operator scan status; no_complete_context is report-only in Stage A", values.applied_scan_reason);
    add_sa_d("rtc_sampling_beam_fwhm_arcsec", "arcsec", "fixed diffraction-derived FWHM used only as the circular-Gaussian scale", values.beam_fwhm_arcsec);
    add_sa_d("rtc_sampling_temporal_sigma_s", "s", "theta/(2 sqrt(2 ln2) v95)", values.temporal_sigma_s);
    add_netcdf_var(
        fo, "RTC_SAMPLING_CONTEXT_CATEGORY_PRECEDENCE",
        std::string{
            "motion_internal_gap,motion_low_velocity,motion_invalid_or_overlimit,"
            "per_detector_invalid,realized_filter_guard,science_flag_without_guard,"
            "nonfinite_input,fully_supported; mutually-exclusive sum-to-total-v1"});
    const std::vector<int> zero_scan_array(values.prerequisite_status.size(), 0);
    add_sa_i("rtc_sampling_input_total_detector_cells",
             "total production detector-time cells; category counts sum exactly to this value",
             zero_scan_array);
    for (const auto &[name, comment] : std::array<std::pair<const char *, const char *>, 10>{
             {{"rtc_sampling_input_fully_supported_count", "fully supported production input cells"},
              {"rtc_sampling_input_boundary_context_count", "boundary/context exclusions (input-domain value is zero; candidate-domain boundary is separate)"},
              {"rtc_sampling_input_internal_gap_count", "cells excluded by missing internal source-motion support"},
              {"rtc_sampling_input_low_velocity_motion_count", "cells excluded by valid source motion below 1 arcsec/s"},
              {"rtc_sampling_input_invalid_or_overlimit_motion_count", "cells excluded by invalid or over-limit source motion"},
              {"rtc_sampling_input_per_detector_invalid_count", "cells excluded by final per-detector validity"},
              {"rtc_sampling_input_science_flag_count", "pre-guard or residual science flags; realized guard is explicitly subtracted"},
              {"rtc_sampling_input_nonfinite_input_count", "cells excluded by non-finite signal or time"},
              {"rtc_sampling_input_realized_filter_guard_count", "cells in the separately captured realized filter-guard mask"},
              {"rtc_sampling_input_unclassified_count", "must remain zero"}}}) {
        add_sa_i(name, comment, zero_scan_array);
    }

    const auto fir_dim = fo.addDim("n_rtc_sampling_fir_coefficients", values.fir_coefficients.size());
    auto fir_var = fo.addVar("rtc_sampling_realized_fir_coefficients", netCDF::ncDouble, fir_dim);
    fir_var.putAtt("units", "N/A"); fir_var.putAtt("comment", "exact realized centered FIR; [1] when disabled");
    fir_var.putVar(values.fir_coefficients.data());
    if (!values.candidate_table_available) {
        return;
    }

    const auto candidate_dim = fo.addDim("n_rtc_sampling_candidates", values.candidate_factors.size());
    auto factor_var = fo.addVar("rtc_sampling_candidate_factor", netCDF::ncInt, candidate_dim);
    factor_var.putAtt("units", "N/A");
    factor_var.putAtt("comment", "unranked rectangular candidate axis; cells above a scan-array Mmax are fill values, and a resource-limited scan-array has no evaluated cells");
    factor_var.putVar(values.candidate_factors.data());
    auto phase_var = fo.addVar("rtc_sampling_candidate_phase", netCDF::ncInt,
                               candidate_dim);
    phase_var.putAtt("units", "N/A");
    phase_var.putAtt("comment", "fixed coherent phase-zero authority");
    phase_var.putVar(values.candidate_phases.data());
    auto dims = scan_array_dims; dims.push_back(candidate_dim);
    auto chunks = scan_array_chunks; chunks.push_back(values.candidate_factors.size());
    auto add_i = [&](const char *n, const char *c, const auto &d) { add_rtc_sampling_table_int(fo, n, c, dims, chunks, d); };
    auto add_d = [&](const char *n, const char *u, const char *c, const auto &d) { add_rtc_sampling_table_double(fo, n, u, c, dims, chunks, d); };
    add_i("rtc_sampling_candidate_status", "candidate applicability status", values.candidate_status);
    add_i("rtc_sampling_candidate_reason", "candidate applicability reason", values.candidate_reason);
    add_i("rtc_sampling_plan_transfer_status", "interior plan-transfer status independent of finite-scan applicability", values.candidate_plan_transfer_status);
    add_i("rtc_sampling_plan_transfer_reason", "interior plan-transfer reason", values.candidate_plan_transfer_reason);
    add_i("rtc_sampling_alias_status", "alias metric validity", values.candidate_alias_status);
    add_i("rtc_sampling_alias_reason", "alias metric reason", values.candidate_alias_reason);
    add_i("rtc_sampling_amplitude_status", "relative-amplitude metric validity", values.candidate_amplitude_status);
    add_i("rtc_sampling_amplitude_reason", "relative-amplitude metric reason", values.candidate_amplitude_reason);
    add_i("rtc_sampling_phase_status", "relative-phase metric validity", values.candidate_phase_status);
    add_i("rtc_sampling_phase_reason", "relative-phase metric reason", values.candidate_phase_reason);
    add_i("rtc_sampling_power_status", "relative-power metric validity", values.candidate_power_status);
    add_i("rtc_sampling_power_reason", "relative-power metric reason", values.candidate_power_reason);
    add_i("rtc_sampling_distortion_status", "complex-distortion metric validity", values.candidate_distortion_status);
    add_i("rtc_sampling_distortion_reason", "complex-distortion metric reason", values.candidate_distortion_reason);
    add_i("rtc_sampling_stopband_status", "stopband metric validity", values.candidate_stopband_status);
    add_i("rtc_sampling_stopband_reason", "stopband metric reason", values.candidate_stopband_reason);
    add_d("rtc_sampling_output_sample_rate_hz", "Hz", "fs/M", values.output_sample_rate_hz);
    add_d("rtc_sampling_output_nyquist_hz", "Hz", "fs/(2M)", values.output_nyquist_hz);
    add_d("rtc_sampling_samples_per_fwhm", "sample/FWHM", "theta*fs/(M*v95); diagnostic endpoint only", values.samples_per_fwhm);
    add_d("rtc_sampling_relative_amplitude_at_dc", "N/A", "coherent folded amplitude relative to unaliased DC", values.relative_amplitude_at_dc);
    add_d("rtc_sampling_relative_phase_at_dc_rad", "rad", "coherent folded phase relative to unaliased DC", values.relative_phase_at_dc_rad);
    add_d("rtc_sampling_relative_power_at_dc", "N/A", "squared relative amplitude at DC", values.relative_power_at_dc);
    add_d("rtc_sampling_relative_distortion_at_dc", "N/A", "complex distance from unity at DC", values.relative_distortion_at_dc);
    add_d("rtc_sampling_alias_amplitude_max_lower", "N/A", "certified lower bound on maximum coherent alias amplitude", values.alias_amplitude_max_lower);
    add_d("rtc_sampling_alias_amplitude_max_upper", "N/A", "certified upper bound on maximum coherent alias amplitude", values.alias_amplitude_max_upper);
    add_d("rtc_sampling_alias_lipschitz_bound", "N/A/Hz", "global analytic Lipschitz bound used for the alias enclosure", values.alias_lipschitz_bound);
    add_i("rtc_sampling_alias_evaluations", "complex-response evaluations used by the alias enclosure", values.alias_evaluations);
    add_d("rtc_sampling_relative_amplitude_max_lower", "N/A", "sampled certified lower bound on maximum relative amplitude", values.relative_amplitude_max_lower);
    add_d("rtc_sampling_relative_amplitude_max_upper", "N/A", "certified upper enclosure for relative amplitude", values.relative_amplitude_max_upper);
    add_d("rtc_sampling_relative_amplitude_error_enclosure", "N/A", "relative-amplitude upper minus lower", values.relative_amplitude_error_enclosure);
    add_d("rtc_sampling_relative_amplitude_lipschitz_bound", "N/A/Hz", "global analytic Lipschitz bound used for relative amplitude", values.relative_amplitude_lipschitz_bound);
    add_i("rtc_sampling_relative_amplitude_evaluations", "complex-response evaluations used by relative amplitude", values.relative_amplitude_evaluations);
    add_d("rtc_sampling_relative_phase_abs_max_lower_rad", "rad", "sampled certified lower bound on maximum absolute relative phase", values.relative_phase_abs_max_lower_rad);
    add_d("rtc_sampling_relative_phase_abs_max_upper_rad", "rad", "certified upper enclosure for absolute relative phase", values.relative_phase_abs_max_upper_rad);
    add_d("rtc_sampling_relative_phase_error_enclosure_rad", "rad", "relative-phase upper minus lower", values.relative_phase_error_enclosure_rad);
    add_d("rtc_sampling_relative_phase_lipschitz_bound", "rad/Hz", "global analytic Lipschitz bound used for relative phase", values.relative_phase_lipschitz_bound);
    add_i("rtc_sampling_relative_phase_evaluations", "complex-response evaluations used by relative phase", values.relative_phase_evaluations);
    add_d("rtc_sampling_relative_power_max_lower", "N/A", "sampled certified lower bound on maximum relative power", values.relative_power_max_lower);
    add_d("rtc_sampling_relative_power_max_upper", "N/A", "certified upper enclosure for relative power", values.relative_power_max_upper);
    add_d("rtc_sampling_relative_power_error_enclosure", "N/A", "relative-power upper minus lower", values.relative_power_error_enclosure);
    add_d("rtc_sampling_relative_power_lipschitz_bound", "N/A/Hz", "global analytic Lipschitz bound used for relative power", values.relative_power_lipschitz_bound);
    add_i("rtc_sampling_relative_power_evaluations", "complex-response evaluations used by relative power", values.relative_power_evaluations);
    add_d("rtc_sampling_relative_distortion_max_lower", "N/A", "sampled certified lower bound on maximum complex distortion", values.relative_distortion_max_lower);
    add_d("rtc_sampling_relative_distortion_max_upper", "N/A", "certified upper enclosure for complex distortion", values.relative_distortion_max_upper);
    add_d("rtc_sampling_relative_distortion_error_enclosure", "N/A", "complex-distortion upper minus lower", values.relative_distortion_error_enclosure);
    add_d("rtc_sampling_relative_distortion_lipschitz_bound", "N/A/Hz", "global analytic Lipschitz bound used for complex distortion", values.relative_distortion_lipschitz_bound);
    add_i("rtc_sampling_relative_distortion_evaluations", "complex-response evaluations used by complex distortion", values.relative_distortion_evaluations);
    add_d("rtc_sampling_alias_error_enclosure", "N/A", "alias maximum upper minus lower", values.alias_error_enclosure);
    add_d("rtc_sampling_stopband_amplitude_max_lower", "N/A", "certified lower bound on maximum FIR stopband amplitude", values.stopband_amplitude_max_lower);
    add_d("rtc_sampling_stopband_amplitude_max_upper", "N/A", "certified upper bound on maximum FIR stopband amplitude", values.stopband_amplitude_max_upper);
    add_d("rtc_sampling_stopband_rejection_db_lower", "dB", "conservative lower bound on rejection relative to DC", values.stopband_rejection_db_lower);
    add_d("rtc_sampling_stopband_rejection_db_upper", "dB", "upper enclosure on rejection relative to DC", values.stopband_rejection_db_upper);
    add_d("rtc_sampling_stopband_error_enclosure", "N/A", "stopband amplitude upper minus lower", values.stopband_error_enclosure);
    add_d("rtc_sampling_stopband_lipschitz_bound", "N/A/Hz", "global analytic FIR Lipschitz bound used for stopband amplitude", values.stopband_lipschitz_bound);
    add_i("rtc_sampling_stopband_evaluations", "complex-response evaluations used by the stopband enclosure", values.stopband_evaluations);
    add_i("rtc_sampling_numerical_evaluations", "deterministic complex-response evaluations", values.numerical_evaluations);
    add_i("rtc_sampling_fir_tap_count", "realized coefficient count", values.tap_count);
    add_i("rtc_sampling_left_context_samples", "exact centered FIR left context", values.left_context);
    add_i("rtc_sampling_right_context_samples", "exact centered FIR right context", values.right_context);
    add_i("rtc_sampling_eligible_input_support", "eligible assigned-grid samples in the observation", values.eligible_input_support);
    add_i("rtc_sampling_candidate_output_count", "phase-zero outputs in the science scan", values.candidate_output_count);
    add_i("rtc_sampling_full_output_count", "outputs with complete realized-FIR eligible context", values.full_output_count);
    add_i("rtc_sampling_incomplete_boundary_count", "outputs missing observation/outer boundary context", values.incomplete_boundary_count);
    add_i("rtc_sampling_incomplete_gap_count", "outputs intersecting invalid/ineligible source support", values.incomplete_gap_count);
    add_i("rtc_sampling_incomplete_other_count", "other incomplete outputs", values.incomplete_other_count);
    const std::vector<int> zero_candidate_context(values.candidate_status.size(), 0);
    add_i("rtc_sampling_detector_output_cell_count",
          "exact detector-output cells classified for this factor and phase",
          zero_candidate_context);
    for (const auto &[name, comment] :
         std::array<std::pair<const char *, const char *>, 10>{
             {{"rtc_sampling_detector_output_fully_supported_count", "detector-output cells with complete production-valid context"},
              {"rtc_sampling_detector_output_boundary_context_count", "detector-output cells excluded by observation or scan boundary context"},
              {"rtc_sampling_detector_output_internal_gap_count", "detector-output cells excluded by an internal source-motion gap"},
              {"rtc_sampling_detector_output_low_velocity_motion_count", "detector-output cells excluded by valid low-velocity motion"},
              {"rtc_sampling_detector_output_invalid_or_overlimit_motion_count", "detector-output cells excluded by invalid or over-limit motion"},
              {"rtc_sampling_detector_output_per_detector_invalid_count", "detector-output cells excluded by exact detector validity"},
              {"rtc_sampling_detector_output_science_flag_count", "detector-output cells excluded by residual pre-guard science flags"},
              {"rtc_sampling_detector_output_nonfinite_input_count", "detector-output cells excluded by non-finite signal or time"},
              {"rtc_sampling_detector_output_realized_filter_guard_count", "detector-output cells excluded by the separately captured realized guard"},
              {"rtc_sampling_detector_output_unclassified_count", "must remain zero"}}}) {
        add_i(name, comment, zero_candidate_context);
    }
    add_i("rtc_sampling_longest_full_run", "longest contiguous run of complete outputs", values.longest_full_run);
    add_d("rtc_sampling_full_duration_s", "s", "N_full*M/fs", values.full_duration_s);
    add_d("rtc_sampling_full_fraction", "N/A", "N_full/N_candidate_outputs", values.full_fraction);
}
