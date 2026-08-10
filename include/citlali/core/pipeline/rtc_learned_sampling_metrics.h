#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include <citlali/core/pipeline/timestream_alignment_state.h>
#include <citlali/core/utils/sha256.h>

namespace citlali::pipeline {

inline constexpr double rtc_sampling_pi =
    3.141592653589793238462643383279502884;
inline constexpr const char *rtc_sampling_schema_version =
    "rtcdiag-v2";
inline constexpr const char *rtc_sampling_algorithm_version =
    "rtc-learned-sampling-stage-a-v2";
inline constexpr const char *rtc_sampling_beam_model =
    "circular-gaussian-temporal-intensity-v1";
inline constexpr const char *rtc_sampling_beam_fwhm_authority =
    "fixed-diffraction-derived-airy-intensity-fwhm-1.028-lambda-over-50m";
inline constexpr const char *rtc_sampling_alias_convention =
    "phase-zero-unit-complex-tone-coherent-half-open-M-images-v1";
inline constexpr const char *rtc_sampling_numerical_method =
    "uniform-partition-global-lipschitz-enclosure-v1";
inline constexpr const char *rtc_sampling_fir_digest_convention =
    "sha256-u64le-count-then-ieee754-binary64le-realized-order-v1";
inline constexpr std::size_t rtc_sampling_numerical_partitions = 256;
inline constexpr std::size_t rtc_sampling_max_candidates = 8192;
inline constexpr std::size_t rtc_sampling_max_candidate_rows = 8000000;
inline constexpr std::size_t rtc_sampling_max_complex_evaluations = 50000000;
inline constexpr std::size_t rtc_sampling_max_estimated_rtcdiag_bytes =
    536870912;
inline constexpr std::size_t rtc_sampling_estimated_candidate_row_bytes = 512;

enum class RtcSamplingStatusCode : int {
    prerequisite_available = 0,
    prerequisite_unavailable = 1,
    candidate_range_available = 2,
    candidate_range_resource_limit = 3,
    candidate_not_evaluated_prerequisite = 4,
    candidate_evaluable = 5,
    candidate_unusable_no_complete_context = 6,
    plan_transfer_available = 7,
    plan_transfer_unavailable = 8,
    scan_usable_for_applied_rtc_operator = 9,
    scan_unusable_for_applied_rtc_operator = 10,
    applied_operator_not_applicable = 11,
    numerical_converged = 12,
    numerical_bounded_not_converged = 13,
    numerical_failed = 14,
    not_applicable_no_decimation = 15,
    candidate_table_available = 16,
    candidate_table_unavailable_resource_limit = 17,
};

enum class RtcSamplingReasonCode : int {
    none = 0,
    missing_cadence = 1,
    invalid_cadence = 2,
    cadence_state_mismatch = 3,
    missing_source_motion = 4,
    invalid_source_motion = 5,
    unavailable_low_velocity = 6,
    unavailable_hwpr_sampling_contract = 7,
    unknown_array = 8,
    missing_beam = 9,
    missing_fir = 10,
    invalid_fir = 11,
    missing_output_grid = 12,
    invalid_output_grid = 13,
    candidate_range_resource_limit = 14,
    prerequisite_unavailable = 15,
    no_complete_context = 16,
    numerical_resource_limit = 17,
    numerical_nonfinite = 18,
    numerical_singular_reference = 19,
    not_applicable_no_decimation = 20,
    hwpr_state_unavailable = 21,
    insufficient_source_motion_rows = 22,
    invalid_source_nonfinite_interval = 23,
    invalid_source_nonpositive_interval = 24,
    invalid_source_gap = 25,
    invalid_source_pointing_step = 26,
    invalid_source_speed_above_bound = 27,
    excluded_low_velocity = 28,
    unequal_source_column_lengths = 29,
    arithmetic_overflow = 30,
    candidate_table_storage_limit = 31,
    bounded_enclosure_nonzero = 32,
    no_guarded_source_motion_support = 33,
};

inline constexpr std::string_view to_string(RtcSamplingStatusCode value) {
    switch (value) {
        case RtcSamplingStatusCode::prerequisite_available:
            return "prerequisite_available";
        case RtcSamplingStatusCode::prerequisite_unavailable:
            return "prerequisite_unavailable";
        case RtcSamplingStatusCode::candidate_range_available:
            return "candidate_range_available";
        case RtcSamplingStatusCode::candidate_range_resource_limit:
            return "candidate_range_resource_limit";
        case RtcSamplingStatusCode::candidate_not_evaluated_prerequisite:
            return "candidate_not_evaluated_prerequisite";
        case RtcSamplingStatusCode::candidate_evaluable: return "candidate_evaluable";
        case RtcSamplingStatusCode::candidate_unusable_no_complete_context:
            return "candidate_unusable_no_complete_context";
        case RtcSamplingStatusCode::plan_transfer_available:
            return "plan_transfer_available";
        case RtcSamplingStatusCode::plan_transfer_unavailable:
            return "plan_transfer_unavailable";
        case RtcSamplingStatusCode::scan_usable_for_applied_rtc_operator:
            return "scan_usable_for_applied_rtc_operator";
        case RtcSamplingStatusCode::scan_unusable_for_applied_rtc_operator:
            return "scan_unusable_for_applied_rtc_operator";
        case RtcSamplingStatusCode::applied_operator_not_applicable:
            return "applied_operator_not_applicable";
        case RtcSamplingStatusCode::numerical_converged:
            return "numerical_converged";
        case RtcSamplingStatusCode::numerical_bounded_not_converged:
            return "numerical_bounded_not_converged";
        case RtcSamplingStatusCode::numerical_failed:
            return "numerical_failed";
        case RtcSamplingStatusCode::not_applicable_no_decimation:
            return "not_applicable_no_decimation";
        case RtcSamplingStatusCode::candidate_table_available:
            return "candidate_table_available";
        case RtcSamplingStatusCode::candidate_table_unavailable_resource_limit:
            return "candidate_table_unavailable_resource_limit";
    }
    return "numerical_failed";
}

inline constexpr std::string_view to_string(RtcSamplingReasonCode value) {
    switch (value) {
        case RtcSamplingReasonCode::none: return "none";
        case RtcSamplingReasonCode::missing_cadence: return "missing_cadence";
        case RtcSamplingReasonCode::invalid_cadence: return "invalid_cadence";
        case RtcSamplingReasonCode::cadence_state_mismatch: return "cadence_state_mismatch";
        case RtcSamplingReasonCode::missing_source_motion: return "missing_source_motion";
        case RtcSamplingReasonCode::invalid_source_motion: return "invalid_source_motion";
        case RtcSamplingReasonCode::unavailable_low_velocity: return "unavailable_low_velocity";
        case RtcSamplingReasonCode::unavailable_hwpr_sampling_contract:
            return "unavailable_hwpr_sampling_contract";
        case RtcSamplingReasonCode::unknown_array: return "unknown_array";
        case RtcSamplingReasonCode::missing_beam: return "missing_beam";
        case RtcSamplingReasonCode::missing_fir: return "missing_fir";
        case RtcSamplingReasonCode::invalid_fir: return "invalid_fir";
        case RtcSamplingReasonCode::missing_output_grid: return "missing_output_grid";
        case RtcSamplingReasonCode::invalid_output_grid: return "invalid_output_grid";
        case RtcSamplingReasonCode::candidate_range_resource_limit:
            return "candidate_range_resource_limit";
        case RtcSamplingReasonCode::prerequisite_unavailable:
            return "prerequisite_unavailable";
        case RtcSamplingReasonCode::no_complete_context: return "no_complete_context";
        case RtcSamplingReasonCode::numerical_resource_limit:
            return "numerical_resource_limit";
        case RtcSamplingReasonCode::numerical_nonfinite: return "numerical_nonfinite";
        case RtcSamplingReasonCode::numerical_singular_reference:
            return "numerical_singular_reference";
        case RtcSamplingReasonCode::not_applicable_no_decimation:
            return "not_applicable_no_decimation";
        case RtcSamplingReasonCode::hwpr_state_unavailable:
            return "hwpr_state_unavailable";
        case RtcSamplingReasonCode::insufficient_source_motion_rows:
            return "insufficient_source_motion_rows";
        case RtcSamplingReasonCode::invalid_source_nonfinite_interval:
            return "invalid_source_nonfinite_interval";
        case RtcSamplingReasonCode::invalid_source_nonpositive_interval:
            return "invalid_source_nonpositive_interval";
        case RtcSamplingReasonCode::invalid_source_gap:
            return "invalid_source_gap";
        case RtcSamplingReasonCode::invalid_source_pointing_step:
            return "invalid_source_pointing_step";
        case RtcSamplingReasonCode::invalid_source_speed_above_bound:
            return "invalid_source_speed_above_bound";
        case RtcSamplingReasonCode::excluded_low_velocity:
            return "excluded_low_velocity";
        case RtcSamplingReasonCode::unequal_source_column_lengths:
            return "unequal_source_column_lengths";
        case RtcSamplingReasonCode::arithmetic_overflow:
            return "arithmetic_overflow";
        case RtcSamplingReasonCode::candidate_table_storage_limit:
            return "candidate_table_storage_limit";
        case RtcSamplingReasonCode::bounded_enclosure_nonzero:
            return "bounded_enclosure_nonzero";
        case RtcSamplingReasonCode::no_guarded_source_motion_support:
            return "no_guarded_source_motion_support";
    }
    return "numerical_nonfinite";
}

inline std::string rtc_sampling_status_reason_vocabulary() {
    return
        "status:0=prerequisite_available,1=prerequisite_unavailable,"
        "2=candidate_range_available,3=candidate_range_resource_limit,"
        "4=candidate_not_evaluated_prerequisite,5=candidate_evaluable,"
        "6=candidate_unusable_no_complete_context,"
        "7=plan_transfer_available,8=plan_transfer_unavailable,"
        "9=scan_usable_for_applied_rtc_operator,"
        "10=scan_unusable_for_applied_rtc_operator,"
        "11=applied_operator_not_applicable,12=numerical_converged,"
        "13=numerical_bounded_not_converged,14=numerical_failed,"
        "15=not_applicable_no_decimation,16=candidate_table_available,"
        "17=candidate_table_unavailable_resource_limit;"
        "reason:0=none,1=missing_cadence,2=invalid_cadence,"
        "3=cadence_state_mismatch,4=missing_source_motion,"
        "5=invalid_source_motion,6=unavailable_low_velocity,"
        "7=unavailable_hwpr_sampling_contract,8=unknown_array,9=missing_beam,"
        "10=missing_fir,11=invalid_fir,12=missing_output_grid,"
        "13=invalid_output_grid,14=candidate_range_resource_limit,"
        "15=prerequisite_unavailable,16=no_complete_context,"
        "17=numerical_resource_limit,18=numerical_nonfinite,"
        "19=numerical_singular_reference,20=not_applicable_no_decimation,"
        "21=hwpr_state_unavailable,22=insufficient_source_motion_rows,"
        "23=invalid_source_nonfinite_interval,"
        "24=invalid_source_nonpositive_interval,25=invalid_source_gap,"
        "26=invalid_source_pointing_step,"
        "27=invalid_source_speed_above_bound,28=excluded_low_velocity,"
        "29=unequal_source_column_lengths,30=arithmetic_overflow,"
        "31=candidate_table_storage_limit,"
        "32=bounded_enclosure_nonzero,"
        "33=no_guarded_source_motion_support";
}

inline RtcSamplingReasonCode rtc_sampling_source_interval_reason_code(
    std::string_view reason) {
    if (reason == "none") return RtcSamplingReasonCode::none;
    if (reason == "invalid_nonfinite_source_interval") {
        return RtcSamplingReasonCode::invalid_source_nonfinite_interval;
    }
    if (reason == "invalid_nonpositive_source_interval") {
        return RtcSamplingReasonCode::invalid_source_nonpositive_interval;
    }
    if (reason == "invalid_source_gap") {
        return RtcSamplingReasonCode::invalid_source_gap;
    }
    if (reason == "invalid_source_pointing_step") {
        return RtcSamplingReasonCode::invalid_source_pointing_step;
    }
    if (reason == "invalid_source_speed_above_bound") {
        return RtcSamplingReasonCode::invalid_source_speed_above_bound;
    }
    if (reason == "excluded_low_velocity") {
        return RtcSamplingReasonCode::excluded_low_velocity;
    }
    return RtcSamplingReasonCode::invalid_source_motion;
}

inline RtcSamplingReasonCode rtc_sampling_source_support_reason_code(
    std::string_view reason) {
    if (reason == "none") return RtcSamplingReasonCode::none;
    if (reason == "missing_source_motion_columns") {
        return RtcSamplingReasonCode::missing_source_motion;
    }
    if (reason == "insufficient_source_motion_rows") {
        return RtcSamplingReasonCode::insufficient_source_motion_rows;
    }
    if (reason == "unequal_source_column_lengths") {
        return RtcSamplingReasonCode::unequal_source_column_lengths;
    }
    if (reason == "unavailable_low_velocity") {
        return RtcSamplingReasonCode::unavailable_low_velocity;
    }
    return RtcSamplingReasonCode::invalid_source_motion;
}

struct RtcSamplingBeamAuthority {
    bool available = false;
    std::string array_name;
    double fwhm_arcsec = std::numeric_limits<double>::quiet_NaN();
    RtcSamplingReasonCode reason = RtcSamplingReasonCode::unknown_array;
};

inline RtcSamplingBeamAuthority rtc_sampling_beam_authority(int array_id) {
    switch (array_id) {
        case 0: return {true, "a1100", 4.66, RtcSamplingReasonCode::none};
        case 1: return {true, "a1400", 5.94, RtcSamplingReasonCode::none};
        case 2: return {true, "a2000", 8.48, RtcSamplingReasonCode::none};
        default: return {};
    }
}

inline double rtc_sampling_temporal_sigma_s(double fwhm_arcsec,
                                             double speed_arcsec_s) {
    if (!std::isfinite(fwhm_arcsec) || !std::isfinite(speed_arcsec_s) ||
        fwhm_arcsec <= 0.0 || speed_arcsec_s <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return fwhm_arcsec /
           (2.0 * std::sqrt(2.0 * std::log(2.0)) * speed_arcsec_s);
}

inline double rtc_sampling_percentile_sorted(
    const std::vector<double> &sorted_values, double percentile) {
    if (sorted_values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (sorted_values.size() == 1) {
        return sorted_values.front();
    }
    const double p = std::clamp(percentile, 0.0, 100.0);
    const double position = p / 100.0 * (sorted_values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - lower;
    return sorted_values[lower] * (1.0 - fraction) +
           sorted_values[upper] * fraction;
}

struct RtcSamplingScanMotion {
    RtcSamplingStatusCode status =
        RtcSamplingStatusCode::prerequisite_unavailable;
    RtcSamplingReasonCode reason = RtcSamplingReasonCode::missing_source_motion;
    double duration_s = std::numeric_limits<double>::quiet_NaN();
    double speed_p95_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    double speed_p99_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    double speed_p995_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    double speed_max_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    std::size_t source_interval_count = 0;
    std::size_t overlapping_interval_count = 0;
    std::size_t boundary_guard_excluded_count = 0;
    std::size_t partial_overlap_count = 0;
    std::size_t valid_interval_count = 0;
    std::size_t rejected_interval_count = 0;
    std::size_t eligible_interval_count = 0;
    std::size_t low_velocity_excluded_count = 0;
    double valid_duration_s = 0.0;
    double eligible_duration_s = 0.0;
    double low_velocity_excluded_duration_s = 0.0;
    double boundary_guard_excluded_duration_s = 0.0;
    double partial_overlap_duration_s = 0.0;
    double eligible_fraction = 0.0;
    std::size_t guarded_first_row_index =
        std::numeric_limits<std::size_t>::max();
    std::size_t guarded_last_row_index =
        std::numeric_limits<std::size_t>::max();
};

inline RtcSamplingScanMotion calculate_rtc_sampling_scan_motion(
    const RtcSamplingSourceMotionSupport &support, double scan_start_time_s,
    double scan_stop_time_s) {
    RtcSamplingScanMotion result;
    if (!std::isfinite(scan_start_time_s) || !std::isfinite(scan_stop_time_s) ||
        scan_stop_time_s <= scan_start_time_s) {
        result.reason = RtcSamplingReasonCode::invalid_output_grid;
        return result;
    }
    result.duration_s = scan_stop_time_s - scan_start_time_s;
    if (support.source_row_count < 2 || support.intervals.size() + 1 !=
            support.source_row_count) {
        result.reason = rtc_sampling_source_support_reason_code(support.reason);
        return result;
    }
    const auto row_time = [&](std::size_t row) {
        return row == 0 ? support.intervals.front().start_time_s
                        : support.intervals[row - 1].stop_time_s;
    };
    std::size_t first_row = support.source_row_count;
    std::size_t last_row = support.source_row_count;
    for (std::size_t row = 0; row < support.source_row_count; ++row) {
        const double time = row_time(row);
        if (std::isfinite(time) && time >= scan_start_time_s) {
            first_row = row;
            break;
        }
    }
    for (std::size_t row = support.source_row_count; row-- > 0;) {
        const double time = row_time(row);
        if (std::isfinite(time) && time <= scan_stop_time_s) {
            last_row = row;
            break;
        }
    }
    if (first_row == support.source_row_count ||
        last_row == support.source_row_count || last_row <= first_row ||
        last_row - first_row <=
            2 * rtc_sampling_source_boundary_guard_rows) {
        result.reason = RtcSamplingReasonCode::no_guarded_source_motion_support;
        return result;
    }
    result.guarded_first_row_index =
        first_row + rtc_sampling_source_boundary_guard_rows;
    result.guarded_last_row_index =
        last_row - rtc_sampling_source_boundary_guard_rows;

    std::vector<double> eligible_speeds;
    for (const auto &interval : support.intervals) {
        const double overlap_start = std::max(
            interval.start_time_s, scan_start_time_s);
        const double overlap_stop = std::min(
            interval.stop_time_s, scan_stop_time_s);
        if (!std::isfinite(overlap_start) || !std::isfinite(overlap_stop) ||
            overlap_stop <= overlap_start) {
            continue;
        }
        result.overlapping_interval_count++;
        const double overlap_duration = overlap_stop - overlap_start;
        if (overlap_start != interval.start_time_s ||
            overlap_stop != interval.stop_time_s) {
            result.partial_overlap_count++;
            result.partial_overlap_duration_s += overlap_duration;
        }
        if (interval.start_row_index < result.guarded_first_row_index ||
            interval.stop_row_index > result.guarded_last_row_index) {
            result.boundary_guard_excluded_count++;
            result.boundary_guard_excluded_duration_s += overlap_duration;
            continue;
        }
        result.source_interval_count++;
        if (!interval.valid) {
            result.rejected_interval_count++;
            continue;
        }
        result.valid_interval_count++;
        result.valid_duration_s += interval.duration_s;
        if (interval.eligible) {
            result.eligible_interval_count++;
            result.eligible_duration_s += interval.duration_s;
            eligible_speeds.push_back(interval.speed_arcsec_s);
        }
        else {
            result.low_velocity_excluded_count++;
            result.low_velocity_excluded_duration_s += interval.duration_s;
        }
    }
    if (result.valid_duration_s > 0.0) {
        result.eligible_fraction = result.eligible_duration_s /
                                   result.valid_duration_s;
    }
    if (eligible_speeds.empty()) {
        result.reason = result.source_interval_count == 0
            ? RtcSamplingReasonCode::no_guarded_source_motion_support
            : result.valid_interval_count > 0
            ? RtcSamplingReasonCode::unavailable_low_velocity
            : RtcSamplingReasonCode::invalid_source_motion;
        return result;
    }
    std::sort(eligible_speeds.begin(), eligible_speeds.end());
    result.speed_p95_arcsec_s =
        rtc_sampling_percentile_sorted(eligible_speeds, 95.0);
    result.speed_p99_arcsec_s =
        rtc_sampling_percentile_sorted(eligible_speeds, 99.0);
    result.speed_p995_arcsec_s =
        rtc_sampling_percentile_sorted(eligible_speeds, 99.5);
    result.speed_max_arcsec_s = eligible_speeds.back();
    result.status = RtcSamplingStatusCode::prerequisite_available;
    result.reason = RtcSamplingReasonCode::none;
    return result;
}

inline std::vector<unsigned char> rtc_sampling_eligible_grid_mask(
    const RtcSamplingSourceMotionSupport &support,
    const Eigen::VectorXd &grid_time_s,
    const RtcSamplingScanMotion &motion) {
    std::vector<unsigned char> mask(static_cast<std::size_t>(grid_time_s.size()), 0);
    std::size_t interval_index = 0;
    for (Eigen::Index i = 0; i < grid_time_s.size(); ++i) {
        const double time = grid_time_s(i);
        while (interval_index < support.intervals.size() &&
               support.intervals[interval_index].stop_time_s < time) {
            ++interval_index;
        }
        if (interval_index < support.intervals.size()) {
            const auto &interval = support.intervals[interval_index];
            mask[static_cast<std::size_t>(i)] =
                interval.eligible && time >= interval.start_time_s &&
                time <= interval.stop_time_s &&
                interval.start_row_index >= motion.guarded_first_row_index &&
                interval.stop_row_index <= motion.guarded_last_row_index;
        }
    }
    return mask;
}

inline std::complex<double> rtc_sampling_fir_response(
    const std::vector<double> &coefficients, double frequency_hz,
    double native_sample_rate_hz) {
    if (coefficients.empty() || !std::isfinite(frequency_hz) ||
        !std::isfinite(native_sample_rate_hz) || native_sample_rate_hz <= 0.0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    const double center = 0.5 * (coefficients.size() - 1);
    std::complex<double> response{0.0, 0.0};
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        const double phase = -2.0 * rtc_sampling_pi * frequency_hz *
                             (static_cast<double>(i) - center) /
                             native_sample_rate_hz;
        response += coefficients[i] *
                    std::complex<double>{std::cos(phase), std::sin(phase)};
    }
    return response;
}

inline double rtc_sampling_gaussian_beam_amplitude(double frequency_hz,
                                                    double temporal_sigma_s) {
    if (!std::isfinite(frequency_hz) || !std::isfinite(temporal_sigma_s) ||
        temporal_sigma_s <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::exp(-2.0 * rtc_sampling_pi * rtc_sampling_pi *
                    temporal_sigma_s * temporal_sigma_s *
                    frequency_hz * frequency_hz);
}

inline std::complex<double> rtc_sampling_composed_transfer(
    const std::vector<double> &coefficients, double frequency_hz,
    double native_sample_rate_hz, double temporal_sigma_s) {
    return rtc_sampling_gaussian_beam_amplitude(frequency_hz,
                                                 temporal_sigma_s) *
           rtc_sampling_fir_response(coefficients, frequency_hz,
                                     native_sample_rate_hz);
}

struct RtcSamplingCoherentResponse {
    std::complex<double> unaliased{0.0, 0.0};
    std::complex<double> alias{0.0, 0.0};
    std::complex<double> folded{0.0, 0.0};
    std::complex<double> relative{0.0, 0.0};
    bool relative_valid = false;
};

inline RtcSamplingCoherentResponse rtc_sampling_phase_zero_coherent_response_at(
    const std::vector<double> &coefficients, double output_frequency_hz,
    double native_sample_rate_hz, int factor, double temporal_sigma_s) {
    RtcSamplingCoherentResponse result;
    if (factor <= 0 || !std::isfinite(native_sample_rate_hz) ||
        native_sample_rate_hz <= 0.0 ||
        !std::isfinite(output_frequency_hz) || temporal_sigma_s <= 0.0) {
        return result;
    }
    result.unaliased = rtc_sampling_composed_transfer(
        coefficients, output_frequency_hz, native_sample_rate_hz,
        temporal_sigma_s);
    if (factor == 1) {
        result.folded = result.unaliased;
        result.alias = {0.0, 0.0};
        result.relative = {1.0, 0.0};
        result.relative_valid = true;
        return result;
    }
    const double output_rate = native_sample_rate_hz / factor;
    const double native_low = -0.5 * native_sample_rate_hz;
    const int first_image = static_cast<int>(std::ceil(
        (native_low - output_frequency_hz) / output_rate));
    for (int image = 0; image < factor; ++image) {
        const int k = first_image + image;
        const double source_frequency =
            output_frequency_hz + static_cast<double>(k) * output_rate;
        const auto response = rtc_sampling_composed_transfer(
            coefficients, source_frequency, native_sample_rate_hz,
            temporal_sigma_s);
        result.folded += response;  // unit complex amplitude per admitted tone
        if (k != 0) {
            result.alias += response;
        }
    }
    if (std::abs(result.unaliased) > 0.0) {
        result.relative = result.folded / result.unaliased;
        result.relative_valid = std::isfinite(result.relative.real()) &&
                                std::isfinite(result.relative.imag());
    }
    return result;
}

inline std::string rtc_sampling_fir_digest(
    const std::vector<double> &coefficients) {
    citlali::utils::Sha256 digest;
    const auto update_u64le = [&](std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (unsigned i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
        }
        digest.update(bytes.data(), bytes.size());
    };
    update_u64le(static_cast<std::uint64_t>(coefficients.size()));
    for (const double value : coefficients) {
        update_u64le(std::bit_cast<std::uint64_t>(value));
    }
    return "sha256:" + digest.finish();
}

inline double rtc_sampling_transfer_derivative_bound(
    const std::vector<double> &coefficients, double native_sample_rate_hz,
    double temporal_sigma_s) {
    if (coefficients.empty() || native_sample_rate_hz <= 0.0 ||
        temporal_sigma_s <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double h_abs = 0.0;
    double h_derivative = 0.0;
    const double center = 0.5 * (coefficients.size() - 1);
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        h_abs += std::abs(coefficients[i]);
        h_derivative += std::abs(coefficients[i]) *
                        std::abs(static_cast<double>(i) - center);
    }
    h_derivative *= 2.0 * rtc_sampling_pi / native_sample_rate_hz;
    const double a = 2.0 * rtc_sampling_pi * rtc_sampling_pi *
                     temporal_sigma_s * temporal_sigma_s;
    const double beam_derivative = std::sqrt(2.0 * a / std::exp(1.0));
    return beam_derivative * h_abs + h_derivative;
}

struct RtcSamplingBoundedMaximum {
    bool valid = false;
    double lower = std::numeric_limits<double>::quiet_NaN();
    double upper = std::numeric_limits<double>::quiet_NaN();
    double error_enclosure = std::numeric_limits<double>::quiet_NaN();
    std::size_t evaluations = 0;
    RtcSamplingReasonCode reason = RtcSamplingReasonCode::numerical_nonfinite;
};

template <class Function>
RtcSamplingBoundedMaximum rtc_sampling_bounded_maximum(
    const Function &function, double lower_frequency_hz,
    double upper_frequency_hz, double lipschitz_bound,
    std::size_t partitions = rtc_sampling_numerical_partitions) {
    RtcSamplingBoundedMaximum result;
    if (!std::isfinite(lower_frequency_hz) ||
        !std::isfinite(upper_frequency_hz) ||
        upper_frequency_hz < lower_frequency_hz ||
        !std::isfinite(lipschitz_bound) || lipschitz_bound < 0.0 ||
        partitions == 0) {
        return result;
    }
    const double width = upper_frequency_hz - lower_frequency_hz;
    double sampled_max = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i <= partitions; ++i) {
        const double frequency = lower_frequency_hz +
            width * static_cast<double>(i) / static_cast<double>(partitions);
        const double value = function(frequency);
        result.evaluations++;
        if (!std::isfinite(value)) {
            return result;
        }
        sampled_max = std::max(sampled_max, value);
    }
    const double radius = partitions > 0
        ? width / (2.0 * static_cast<double>(partitions)) : 0.0;
    result.valid = true;
    result.lower = sampled_max;
    result.upper = sampled_max + lipschitz_bound * radius;
    result.error_enclosure = result.upper - result.lower;
    result.reason = RtcSamplingReasonCode::none;
    return result;
}

struct RtcSamplingCompleteContext {
    RtcSamplingStatusCode candidate_status =
        RtcSamplingStatusCode::candidate_unusable_no_complete_context;
    RtcSamplingReasonCode candidate_reason =
        RtcSamplingReasonCode::no_complete_context;
    int factor = 0;
    int phase = 0;
    std::size_t tap_count = 0;
    Eigen::Index left_context = 0;
    Eigen::Index right_context = 0;
    std::size_t eligible_input_support = 0;
    std::size_t candidate_output_count = 0;
    std::size_t full_output_count = 0;
    std::size_t incomplete_boundary_count = 0;
    std::size_t incomplete_gap_count = 0;
    std::size_t incomplete_other_count = 0;
    std::size_t longest_full_run = 0;
    double full_duration_s = 0.0;
    double full_fraction = std::numeric_limits<double>::quiet_NaN();
};

inline RtcSamplingCompleteContext calculate_rtc_sampling_complete_context(
    const std::vector<unsigned char> &eligible_grid, Eigen::Index outer_start,
    Eigen::Index outer_stop, Eigen::Index scan_start, Eigen::Index scan_stop,
    int factor, int phase, std::size_t tap_count,
    double native_sample_rate_hz) {
    RtcSamplingCompleteContext result;
    result.factor = factor;
    result.phase = phase;
    result.tap_count = tap_count;
    result.left_context = tap_count > 0
        ? static_cast<Eigen::Index>((tap_count - 1) / 2) : 0;
    result.right_context = tap_count > 0
        ? static_cast<Eigen::Index>(tap_count - 1) - result.left_context : 0;
    result.eligible_input_support = static_cast<std::size_t>(
        std::count(eligible_grid.begin(), eligible_grid.end(),
                   static_cast<unsigned char>(1)));
    if (scan_stop < scan_start) {
        return result;
    }
    if (factor <= 0 || phase < 0 || phase >= factor || scan_start < 0 ||
        outer_start > scan_start ||
        outer_stop < scan_stop || native_sample_rate_hz <= 0.0) {
        result.candidate_reason = RtcSamplingReasonCode::invalid_output_grid;
        return result;
    }
    std::size_t current_run = 0;
    for (Eigen::Index output = scan_start + phase; output <= scan_stop;
         output += factor) {
        result.candidate_output_count++;
        const Eigen::Index first = output - result.left_context;
        const Eigen::Index last = output + result.right_context;
        if (first < outer_start || last > outer_stop || first < 0 ||
            last >= static_cast<Eigen::Index>(eligible_grid.size())) {
            result.incomplete_boundary_count++;
            current_run = 0;
            continue;
        }
        bool full = true;
        for (Eigen::Index i = first; i <= last; ++i) {
            if (eligible_grid[static_cast<std::size_t>(i)] == 0) {
                full = false;
                break;
            }
        }
        if (!full) {
            result.incomplete_gap_count++;
            current_run = 0;
            continue;
        }
        result.full_output_count++;
        current_run++;
        result.longest_full_run = std::max(result.longest_full_run, current_run);
    }
    if (result.candidate_output_count > 0) {
        result.full_fraction = static_cast<double>(result.full_output_count) /
                               result.candidate_output_count;
    }
    result.full_duration_s = result.full_output_count * factor /
                             native_sample_rate_hz;
    if (result.full_output_count > 0) {
        result.candidate_status = RtcSamplingStatusCode::candidate_evaluable;
        result.candidate_reason = RtcSamplingReasonCode::none;
    }
    return result;
}

inline int rtc_sampling_candidate_mmax(double fwhm_arcsec,
                                       double native_sample_rate_hz,
                                       double speed_p95_arcsec_s) {
    if (!std::isfinite(fwhm_arcsec) ||
        !std::isfinite(native_sample_rate_hz) ||
        !std::isfinite(speed_p95_arcsec_s) || fwhm_arcsec <= 0.0 ||
        native_sample_rate_hz <= 0.0 || speed_p95_arcsec_s <= 0.0) {
        return -1;
    }
    const double value = std::floor(
        fwhm_arcsec * native_sample_rate_hz / speed_p95_arcsec_s);
    if (!std::isfinite(value) || value > std::numeric_limits<int>::max()) {
        return -1;
    }
    return static_cast<int>(value);
}

inline std::vector<int> rtc_sampling_supported_factors(
    double fwhm_arcsec, double native_sample_rate_hz,
    double speed_p95_arcsec_s) {
    const int mmax = rtc_sampling_candidate_mmax(
        fwhm_arcsec, native_sample_rate_hz, speed_p95_arcsec_s);
    if (mmax < 0) {
        return {};
    }
    std::vector<int> factors(
        static_cast<std::size_t>(std::max(1, mmax)));
    std::iota(factors.begin(), factors.end(), 1);
    return factors;
}

inline bool rtc_sampling_checked_add(std::size_t a, std::size_t b,
                                     std::size_t &result) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        return false;
    }
    result = a + b;
    return true;
}

inline bool rtc_sampling_checked_multiply(std::size_t a, std::size_t b,
                                          std::size_t &result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

struct RtcSamplingResourcePreflight {
    std::vector<RtcSamplingStatusCode> range_status;
    std::vector<RtcSamplingReasonCode> range_reason;
    RtcSamplingStatusCode table_status =
        RtcSamplingStatusCode::candidate_table_unavailable_resource_limit;
    RtcSamplingReasonCode table_reason =
        RtcSamplingReasonCode::candidate_range_resource_limit;
    bool table_available = false;
    std::size_t candidate_axis_size = 0;
    std::size_t logical_candidate_rows = 0;
    std::size_t rectangular_storage_cells = 0;
    std::size_t estimated_complex_evaluations = 0;
    std::size_t estimated_rtcdiag_bytes = 0;
};

inline RtcSamplingResourcePreflight rtc_sampling_resource_preflight(
    const std::vector<int> &derived_mmax,
    const std::vector<unsigned char> &prerequisite_available,
    std::size_t evaluations_per_candidate =
        8 * (rtc_sampling_numerical_partitions + 1),
    std::size_t estimated_candidate_row_bytes =
        rtc_sampling_estimated_candidate_row_bytes) {
    RtcSamplingResourcePreflight result;
    const auto n = derived_mmax.size();
    result.range_status.assign(
        n, RtcSamplingStatusCode::candidate_not_evaluated_prerequisite);
    result.range_reason.assign(
        n, RtcSamplingReasonCode::prerequisite_unavailable);
    if (prerequisite_available.size() != n) {
        result.table_reason = RtcSamplingReasonCode::arithmetic_overflow;
        return result;
    }
    bool arithmetic_ok = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (prerequisite_available[i] == 0) {
            continue;
        }
        const std::size_t range = static_cast<std::size_t>(
            std::max(1, derived_mmax[i]));
        if (range > rtc_sampling_max_candidates) {
            result.range_status[i] =
                RtcSamplingStatusCode::candidate_range_resource_limit;
            result.range_reason[i] =
                RtcSamplingReasonCode::candidate_range_resource_limit;
            continue;
        }
        result.range_status[i] =
            RtcSamplingStatusCode::candidate_range_available;
        result.range_reason[i] = RtcSamplingReasonCode::none;
        result.candidate_axis_size = std::max(
            result.candidate_axis_size, range);
        std::size_t updated_rows = 0;
        if (!rtc_sampling_checked_add(result.logical_candidate_rows, range,
                                      updated_rows)) {
            arithmetic_ok = false;
            break;
        }
        result.logical_candidate_rows = updated_rows;
    }
    if (!arithmetic_ok) {
        result.table_reason = RtcSamplingReasonCode::arithmetic_overflow;
        return result;
    }
    // Preserve a factor-1 provenance/reference slot when no scientific range
    // can be derived. It is not counted as an evaluated candidate row and its
    // cells remain candidate_not_evaluated_prerequisite.
    if (result.candidate_axis_size == 0 && n > 0) {
        result.candidate_axis_size = 1;
    }
    if (result.candidate_axis_size == 0) {
        return result;
    }
    if (result.logical_candidate_rows > rtc_sampling_max_candidate_rows) {
        result.table_reason =
            RtcSamplingReasonCode::candidate_range_resource_limit;
        return result;
    }
    if (!rtc_sampling_checked_multiply(
            result.logical_candidate_rows, evaluations_per_candidate,
            result.estimated_complex_evaluations)) {
        result.table_reason = RtcSamplingReasonCode::arithmetic_overflow;
        return result;
    }
    if (result.estimated_complex_evaluations >
        rtc_sampling_max_complex_evaluations) {
        result.table_reason = RtcSamplingReasonCode::numerical_resource_limit;
        return result;
    }
    if (!rtc_sampling_checked_multiply(
            n, result.candidate_axis_size,
            result.rectangular_storage_cells) ||
        !rtc_sampling_checked_multiply(
            result.rectangular_storage_cells,
            estimated_candidate_row_bytes,
            result.estimated_rtcdiag_bytes)) {
        result.table_reason = RtcSamplingReasonCode::arithmetic_overflow;
        return result;
    }
    if (result.estimated_rtcdiag_bytes >
        rtc_sampling_max_estimated_rtcdiag_bytes) {
        result.table_reason =
            RtcSamplingReasonCode::candidate_table_storage_limit;
        return result;
    }
    result.table_available = true;
    result.table_status = RtcSamplingStatusCode::candidate_table_available;
    result.table_reason = RtcSamplingReasonCode::none;
    return result;
}

struct RtcSamplingCandidateMetrics {
    int factor = 0;
    int phase = 0;
    double output_sample_rate_hz = std::numeric_limits<double>::quiet_NaN();
    double output_nyquist_hz = std::numeric_limits<double>::quiet_NaN();
    double samples_per_fwhm = std::numeric_limits<double>::quiet_NaN();
    bool alias_valid = false;
    double relative_amplitude_at_dc = std::numeric_limits<double>::quiet_NaN();
    double relative_phase_at_dc_rad = std::numeric_limits<double>::quiet_NaN();
    double relative_power_at_dc = std::numeric_limits<double>::quiet_NaN();
    double relative_distortion_at_dc = std::numeric_limits<double>::quiet_NaN();
    double alias_amplitude_max_lower = std::numeric_limits<double>::quiet_NaN();
    double alias_amplitude_max_upper = std::numeric_limits<double>::quiet_NaN();
    double alias_lipschitz_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t alias_evaluations = 0;
    double relative_amplitude_max_lower = std::numeric_limits<double>::quiet_NaN();
    double relative_amplitude_max_upper = std::numeric_limits<double>::quiet_NaN();
    double relative_amplitude_error_enclosure = std::numeric_limits<double>::quiet_NaN();
    double relative_amplitude_lipschitz_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t relative_amplitude_evaluations = 0;
    double relative_phase_abs_max_lower_rad = std::numeric_limits<double>::quiet_NaN();
    double relative_phase_abs_max_upper_rad = std::numeric_limits<double>::quiet_NaN();
    double relative_phase_error_enclosure_rad = std::numeric_limits<double>::quiet_NaN();
    double relative_phase_lipschitz_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t relative_phase_evaluations = 0;
    double relative_power_max_lower = std::numeric_limits<double>::quiet_NaN();
    double relative_power_max_upper = std::numeric_limits<double>::quiet_NaN();
    double relative_power_error_enclosure = std::numeric_limits<double>::quiet_NaN();
    double relative_power_lipschitz_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t relative_power_evaluations = 0;
    double relative_distortion_max_lower = std::numeric_limits<double>::quiet_NaN();
    double relative_distortion_max_upper = std::numeric_limits<double>::quiet_NaN();
    double relative_distortion_error_enclosure = std::numeric_limits<double>::quiet_NaN();
    double relative_distortion_lipschitz_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t relative_distortion_evaluations = 0;
    double alias_error_enclosure = std::numeric_limits<double>::quiet_NaN();
    RtcSamplingStatusCode alias_status = RtcSamplingStatusCode::numerical_failed;
    RtcSamplingReasonCode alias_reason = RtcSamplingReasonCode::numerical_nonfinite;
    RtcSamplingStatusCode amplitude_status = RtcSamplingStatusCode::numerical_failed;
    RtcSamplingReasonCode amplitude_reason = RtcSamplingReasonCode::numerical_nonfinite;
    RtcSamplingStatusCode phase_status = RtcSamplingStatusCode::numerical_failed;
    RtcSamplingReasonCode phase_reason = RtcSamplingReasonCode::numerical_nonfinite;
    RtcSamplingStatusCode power_status = RtcSamplingStatusCode::numerical_failed;
    RtcSamplingReasonCode power_reason = RtcSamplingReasonCode::numerical_nonfinite;
    RtcSamplingStatusCode distortion_status = RtcSamplingStatusCode::numerical_failed;
    RtcSamplingReasonCode distortion_reason = RtcSamplingReasonCode::numerical_nonfinite;
    bool stopband_valid = false;
    double stopband_amplitude_max_lower = std::numeric_limits<double>::quiet_NaN();
    double stopband_amplitude_max_upper = std::numeric_limits<double>::quiet_NaN();
    double stopband_rejection_db_lower = std::numeric_limits<double>::quiet_NaN();
    double stopband_rejection_db_upper = std::numeric_limits<double>::quiet_NaN();
    double stopband_error_enclosure = std::numeric_limits<double>::quiet_NaN();
    double stopband_lipschitz_bound = std::numeric_limits<double>::quiet_NaN();
    std::size_t stopband_evaluations = 0;
    RtcSamplingStatusCode stopband_status = RtcSamplingStatusCode::numerical_failed;
    RtcSamplingReasonCode stopband_reason = RtcSamplingReasonCode::numerical_nonfinite;
    std::size_t numerical_evaluations = 0;
};

inline RtcSamplingStatusCode rtc_sampling_bounded_status(double error) {
    return error == 0.0
        ? RtcSamplingStatusCode::numerical_converged
        : RtcSamplingStatusCode::numerical_bounded_not_converged;
}

inline RtcSamplingReasonCode rtc_sampling_bounded_reason(double error) {
    return error == 0.0 ? RtcSamplingReasonCode::none
                        : RtcSamplingReasonCode::bounded_enclosure_nonzero;
}

inline RtcSamplingCandidateMetrics calculate_rtc_sampling_candidate_metrics(
    int factor, double native_sample_rate_hz,
    const std::vector<double> &coefficients, double temporal_sigma_s,
    std::size_t partitions = rtc_sampling_numerical_partitions) {
    RtcSamplingCandidateMetrics result;
    result.factor = factor;
    if (factor <= 0 || native_sample_rate_hz <= 0.0 ||
        coefficients.empty() || temporal_sigma_s <= 0.0 || partitions == 0) {
        return result;
    }
    result.output_sample_rate_hz = native_sample_rate_hz / factor;
    result.output_nyquist_hz = 0.5 * result.output_sample_rate_hz;
    const double fwhm_s = 2.0 * std::sqrt(2.0 * std::log(2.0)) *
                          temporal_sigma_s;
    result.samples_per_fwhm = result.output_sample_rate_hz * fwhm_s;

    const auto dc = rtc_sampling_phase_zero_coherent_response_at(
        coefficients, 0.0, native_sample_rate_hz, factor, temporal_sigma_s);
    if (dc.relative_valid) {
        result.relative_amplitude_at_dc = std::abs(dc.relative);
        result.relative_phase_at_dc_rad = std::arg(dc.relative);
        result.relative_power_at_dc = std::norm(dc.relative);
        result.relative_distortion_at_dc = std::abs(dc.relative -
                                                     std::complex<double>{1.0, 0.0});
    }
    if (factor == 1) {
        result.alias_valid = true;
        result.alias_amplitude_max_lower = 0.0;
        result.alias_amplitude_max_upper = 0.0;
        result.alias_lipschitz_bound = 0.0;
        result.relative_amplitude_max_lower = 1.0;
        result.relative_amplitude_max_upper = 1.0;
        result.relative_amplitude_error_enclosure = 0.0;
        result.relative_amplitude_lipschitz_bound = 0.0;
        result.relative_phase_abs_max_lower_rad = 0.0;
        result.relative_phase_abs_max_upper_rad = 0.0;
        result.relative_phase_error_enclosure_rad = 0.0;
        result.relative_phase_lipschitz_bound = 0.0;
        result.relative_power_max_lower = 1.0;
        result.relative_power_max_upper = 1.0;
        result.relative_power_error_enclosure = 0.0;
        result.relative_power_lipschitz_bound = 0.0;
        result.relative_distortion_max_lower = 0.0;
        result.relative_distortion_max_upper = 0.0;
        result.relative_distortion_error_enclosure = 0.0;
        result.relative_distortion_lipschitz_bound = 0.0;
        result.alias_error_enclosure = 0.0;
        result.alias_status = RtcSamplingStatusCode::numerical_converged;
        result.alias_reason = RtcSamplingReasonCode::none;
        result.amplitude_status = RtcSamplingStatusCode::numerical_converged;
        result.amplitude_reason = RtcSamplingReasonCode::none;
        result.phase_status = RtcSamplingStatusCode::numerical_converged;
        result.phase_reason = RtcSamplingReasonCode::none;
        result.power_status = RtcSamplingStatusCode::numerical_converged;
        result.power_reason = RtcSamplingReasonCode::none;
        result.distortion_status = RtcSamplingStatusCode::numerical_converged;
        result.distortion_reason = RtcSamplingReasonCode::none;
        result.stopband_status =
            RtcSamplingStatusCode::not_applicable_no_decimation;
        result.stopband_reason =
            RtcSamplingReasonCode::not_applicable_no_decimation;
        return result;
    }

    const double transfer_lipschitz = rtc_sampling_transfer_derivative_bound(
        coefficients, native_sample_rate_hz, temporal_sigma_s);
    const double low = -result.output_nyquist_hz;
    const double high = std::nextafter(result.output_nyquist_hz, low);
    const auto alias_bound = rtc_sampling_bounded_maximum(
        [&](double f) {
            return std::abs(rtc_sampling_phase_zero_coherent_response_at(
                coefficients, f, native_sample_rate_hz, factor,
                temporal_sigma_s).alias);
        }, low, high, (factor - 1) * transfer_lipschitz, partitions);
    result.alias_lipschitz_bound = (factor - 1) * transfer_lipschitz;
    result.alias_evaluations = alias_bound.evaluations;
    result.numerical_evaluations += alias_bound.evaluations;
    if (alias_bound.valid) {
        result.alias_valid = true;
        result.alias_amplitude_max_lower = alias_bound.lower;
        result.alias_amplitude_max_upper = alias_bound.upper;
        result.alias_error_enclosure = alias_bound.error_enclosure;
        result.alias_status = rtc_sampling_bounded_status(
            alias_bound.error_enclosure);
        result.alias_reason = rtc_sampling_bounded_reason(
            alias_bound.error_enclosure);

        const auto base_min_negative = rtc_sampling_bounded_maximum(
            [&](double f) {
                return -std::abs(rtc_sampling_composed_transfer(
                    coefficients, f, native_sample_rate_hz,
                    temporal_sigma_s));
            }, low, high, transfer_lipschitz, partitions);
        result.numerical_evaluations += base_min_negative.evaluations;
        const double base_min_lower = base_min_negative.valid
            ? std::max(0.0, -base_min_negative.upper) : 0.0;
        if (base_min_lower > 0.0) {
            const double fir_l1 = std::accumulate(
                coefficients.begin(), coefficients.end(), 0.0,
                [](double total, double coefficient) {
                    return total + std::abs(coefficient);
                });
            const double folded_magnitude_upper = factor * fir_l1;
            const double relative_lipschitz =
                factor * transfer_lipschitz / base_min_lower +
                (folded_magnitude_upper * transfer_lipschitz) /
                    (base_min_lower * base_min_lower);
            auto bound_relative = [&](const auto &metric,
                                      double metric_lipschitz) {
                const auto bounded = rtc_sampling_bounded_maximum(
                    [&](double f) {
                        const auto response =
                            rtc_sampling_phase_zero_coherent_response_at(
                                coefficients, f, native_sample_rate_hz,
                                factor, temporal_sigma_s);
                        return response.relative_valid
                            ? metric(response.relative)
                            : std::numeric_limits<double>::quiet_NaN();
                    }, low, high, metric_lipschitz, partitions);
                result.numerical_evaluations += bounded.evaluations;
                return bounded;
            };
            const auto amplitude = bound_relative(
                [](auto value) { return std::abs(value); },
                relative_lipschitz);
            result.relative_amplitude_lipschitz_bound = relative_lipschitz;
            result.relative_amplitude_evaluations = amplitude.evaluations;
            const auto power = bound_relative(
                [](auto value) { return std::norm(value); },
                2.0 * folded_magnitude_upper / base_min_lower *
                    relative_lipschitz);
            result.relative_power_lipschitz_bound =
                2.0 * folded_magnitude_upper / base_min_lower *
                relative_lipschitz;
            result.relative_power_evaluations = power.evaluations;
            const auto distortion = bound_relative(
                [](auto value) {
                    return std::abs(value - std::complex<double>{1.0, 0.0});
                }, relative_lipschitz);
            result.relative_distortion_lipschitz_bound = relative_lipschitz;
            result.relative_distortion_evaluations = distortion.evaluations;
            const auto relative_min_negative = bound_relative(
                [](auto value) { return -std::abs(value); },
                relative_lipschitz);
            if (amplitude.valid) {
                result.relative_amplitude_max_lower = amplitude.lower;
                result.relative_amplitude_max_upper = amplitude.upper;
                result.relative_amplitude_error_enclosure =
                    amplitude.error_enclosure;
                result.amplitude_status = rtc_sampling_bounded_status(
                    amplitude.error_enclosure);
                result.amplitude_reason = rtc_sampling_bounded_reason(
                    amplitude.error_enclosure);
            }
            if (power.valid) {
                result.relative_power_max_lower = power.lower;
                result.relative_power_max_upper = power.upper;
                result.relative_power_error_enclosure = power.error_enclosure;
                result.power_status = rtc_sampling_bounded_status(
                    power.error_enclosure);
                result.power_reason = rtc_sampling_bounded_reason(
                    power.error_enclosure);
            }
            if (distortion.valid) {
                result.relative_distortion_max_lower = distortion.lower;
                result.relative_distortion_max_upper = distortion.upper;
                result.relative_distortion_error_enclosure =
                    distortion.error_enclosure;
                result.distortion_status = rtc_sampling_bounded_status(
                    distortion.error_enclosure);
                result.distortion_reason = rtc_sampling_bounded_reason(
                    distortion.error_enclosure);
            }
            const double relative_min_lower = relative_min_negative.valid
                ? std::max(0.0, -relative_min_negative.upper) : 0.0;
            if (relative_min_lower > 0.0) {
                const double phase_lipschitz =
                    relative_lipschitz / relative_min_lower;
                const auto phase = bound_relative(
                    [](auto value) { return std::abs(std::arg(value)); },
                    phase_lipschitz);
                result.relative_phase_lipschitz_bound = phase_lipschitz;
                result.relative_phase_evaluations = phase.evaluations;
                if (phase.valid) {
                    result.relative_phase_abs_max_lower_rad = phase.lower;
                    result.relative_phase_abs_max_upper_rad =
                        std::min(rtc_sampling_pi, phase.upper);
                    result.relative_phase_error_enclosure_rad =
                        result.relative_phase_abs_max_upper_rad - phase.lower;
                    result.phase_status = rtc_sampling_bounded_status(
                        result.relative_phase_error_enclosure_rad);
                    result.phase_reason = rtc_sampling_bounded_reason(
                        result.relative_phase_error_enclosure_rad);
                }
            }
            else if (relative_min_negative.valid) {
                result.phase_reason =
                    RtcSamplingReasonCode::numerical_singular_reference;
            }
        }
        else if (base_min_negative.valid) {
            result.amplitude_reason =
                RtcSamplingReasonCode::numerical_singular_reference;
            result.phase_reason =
                RtcSamplingReasonCode::numerical_singular_reference;
            result.power_reason =
                RtcSamplingReasonCode::numerical_singular_reference;
            result.distortion_reason =
                RtcSamplingReasonCode::numerical_singular_reference;
        }
    }

    const double dc_gain = std::abs(rtc_sampling_fir_response(
        coefficients, 0.0, native_sample_rate_hz));
    double fir_lipschitz = 0.0;
    const double center = 0.5 * (coefficients.size() - 1);
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        fir_lipschitz += std::abs(coefficients[i]) *
                         std::abs(static_cast<double>(i) - center);
    }
    fir_lipschitz *= 2.0 * rtc_sampling_pi / native_sample_rate_hz;
    const auto stopband = rtc_sampling_bounded_maximum(
        [&](double f) {
            return std::abs(rtc_sampling_fir_response(
                coefficients, f, native_sample_rate_hz));
        }, result.output_nyquist_hz, 0.5 * native_sample_rate_hz,
        fir_lipschitz, partitions);
    result.stopband_lipschitz_bound = fir_lipschitz;
    result.stopband_evaluations = stopband.evaluations;
    result.numerical_evaluations += stopband.evaluations;
    if (stopband.valid && stopband.lower > 0.0 && dc_gain > 0.0) {
        result.stopband_valid = true;
        result.stopband_amplitude_max_lower = stopband.lower;
        result.stopband_amplitude_max_upper = stopband.upper;
        result.stopband_rejection_db_lower =
            -20.0 * std::log10(stopband.upper / dc_gain);
        result.stopband_rejection_db_upper =
            -20.0 * std::log10(stopband.lower / dc_gain);
        result.stopband_error_enclosure = stopband.error_enclosure;
        result.stopband_status = rtc_sampling_bounded_status(
            stopband.error_enclosure);
        result.stopband_reason = rtc_sampling_bounded_reason(
            stopband.error_enclosure);
    }
    return result;
}

}  // namespace citlali::pipeline
