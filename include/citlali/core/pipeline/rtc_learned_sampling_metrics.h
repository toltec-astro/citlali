#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace citlali::pipeline {

inline constexpr double rtc_sampling_pi =
    3.141592653589793238462643383279502884;

struct RtcSamplingMotionInterval {
    double speed_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    double direction_rad = std::numeric_limits<double>::quiet_NaN();
};

struct RtcSamplingScanMotion {
    double duration_s = std::numeric_limits<double>::quiet_NaN();
    double speed_max_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    double speed_p50_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    double speed_p95_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    double speed_p995_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    std::size_t valid_interval_count = 0;
    std::size_t rejected_interval_count = 0;
    std::vector<RtcSamplingMotionInterval> intervals;
};

inline double rtc_sampling_percentile_sorted(
    const std::vector<double> &sorted_values, double percentile) {
    if (sorted_values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (sorted_values.size() == 1) {
        return sorted_values.front();
    }
    percentile = std::clamp(percentile, 0.0, 100.0);
    const double position =
        percentile / 100.0 * static_cast<double>(sorted_values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted_values[lower] * (1.0 - fraction) +
           sorted_values[upper] * fraction;
}

template <class Vector>
RtcSamplingScanMotion calculate_rtc_sampling_scan_motion(
    const Vector &time_s, const Vector &az_rad, const Vector &alt_rad,
    Eigen::Index scan_start, Eigen::Index scan_stop, double rad_to_arcsec,
    double max_sample_step_s = 0.1, double max_pointing_step_rad = 0.01,
    Eigen::Index boundary_guard_samples = 1) {
    RtcSamplingScanMotion result;
    const Eigen::Index size =
        std::min({time_s.size(), az_rad.size(), alt_rad.size()});
    if (size <= 1 || scan_start < 0 || scan_stop >= size ||
        scan_stop <= scan_start || boundary_guard_samples < 0) {
        return result;
    }

    const double duration = time_s(scan_stop) - time_s(scan_start);
    if (std::isfinite(duration) && duration > 0.0) {
        result.duration_s = duration;
    }

    const Eigen::Index first = scan_start + boundary_guard_samples;
    const Eigen::Index last = scan_stop - boundary_guard_samples;
    if (last <= first) {
        return result;
    }

    std::vector<double> sorted_speeds;
    sorted_speeds.reserve(static_cast<std::size_t>(last - first));
    result.intervals.reserve(static_cast<std::size_t>(last - first));
    for (Eigen::Index i = first; i < last; ++i) {
        const double dt = time_s(i + 1) - time_s(i);
        const double daz = az_rad(i + 1) - az_rad(i);
        const double dalt = alt_rad(i + 1) - alt_rad(i);
        if (!std::isfinite(dt) || !std::isfinite(daz) ||
            !std::isfinite(dalt) || dt <= 0.0 || dt > max_sample_step_s ||
            std::abs(daz) > max_pointing_step_rad ||
            std::abs(dalt) > max_pointing_step_rad) {
            ++result.rejected_interval_count;
            continue;
        }
        const double speed = std::hypot(daz, dalt) / dt * rad_to_arcsec;
        if (!std::isfinite(speed) || speed <= 0.0) {
            ++result.rejected_interval_count;
            continue;
        }
        result.intervals.push_back({speed, std::atan2(dalt, daz)});
        sorted_speeds.push_back(speed);
    }

    result.valid_interval_count = sorted_speeds.size();
    if (sorted_speeds.empty()) {
        return result;
    }
    std::sort(sorted_speeds.begin(), sorted_speeds.end());
    result.speed_max_arcsec_s = sorted_speeds.back();
    result.speed_p50_arcsec_s =
        rtc_sampling_percentile_sorted(sorted_speeds, 50.0);
    result.speed_p95_arcsec_s =
        rtc_sampling_percentile_sorted(sorted_speeds, 95.0);
    result.speed_p995_arcsec_s =
        rtc_sampling_percentile_sorted(sorted_speeds, 99.5);
    return result;
}

struct RtcSamplingProjectedBeam {
    double major_fwhm_arcsec = std::numeric_limits<double>::quiet_NaN();
    double minor_fwhm_arcsec = std::numeric_limits<double>::quiet_NaN();
    double position_angle_rad = std::numeric_limits<double>::quiet_NaN();
    double limiting_projected_fwhm_arcsec =
        std::numeric_limits<double>::quiet_NaN();
    double limiting_speed_arcsec_s =
        std::numeric_limits<double>::quiet_NaN();
    double limiting_crossing_time_s =
        std::numeric_limits<double>::quiet_NaN();
    double temporal_sigma_s = std::numeric_limits<double>::quiet_NaN();
};

inline double rtc_sampling_projected_fwhm_arcsec(
    double major_fwhm_arcsec, double minor_fwhm_arcsec,
    double position_angle_rad, double scan_direction_rad) {
    if (!std::isfinite(major_fwhm_arcsec) ||
        !std::isfinite(minor_fwhm_arcsec) ||
        !std::isfinite(position_angle_rad) ||
        !std::isfinite(scan_direction_rad) || major_fwhm_arcsec <= 0.0 ||
        minor_fwhm_arcsec <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double relative = scan_direction_rad - position_angle_rad;
    const double major_component =
        std::cos(relative) / major_fwhm_arcsec;
    const double minor_component =
        std::sin(relative) / minor_fwhm_arcsec;
    return 1.0 / std::hypot(major_component, minor_component);
}

inline RtcSamplingProjectedBeam calculate_rtc_sampling_projected_beam(
    const RtcSamplingScanMotion &motion, double axis_a_fwhm_arcsec,
    double axis_b_fwhm_arcsec, double position_angle_rad) {
    RtcSamplingProjectedBeam result;
    result.major_fwhm_arcsec =
        std::max(axis_a_fwhm_arcsec, axis_b_fwhm_arcsec);
    result.minor_fwhm_arcsec =
        std::min(axis_a_fwhm_arcsec, axis_b_fwhm_arcsec);
    result.position_angle_rad = position_angle_rad;
    if (!std::isfinite(result.major_fwhm_arcsec) ||
        !std::isfinite(result.minor_fwhm_arcsec) ||
        result.major_fwhm_arcsec <= 0.0 || result.minor_fwhm_arcsec <= 0.0 ||
        !std::isfinite(position_angle_rad)) {
        return result;
    }

    double minimum_crossing_time = std::numeric_limits<double>::infinity();
    for (const auto &interval : motion.intervals) {
        const double projected = rtc_sampling_projected_fwhm_arcsec(
            result.major_fwhm_arcsec, result.minor_fwhm_arcsec,
            position_angle_rad, interval.direction_rad);
        if (!std::isfinite(projected) || !std::isfinite(interval.speed_arcsec_s) ||
            interval.speed_arcsec_s <= 0.0) {
            continue;
        }
        const double crossing_time = projected / interval.speed_arcsec_s;
        if (crossing_time < minimum_crossing_time) {
            minimum_crossing_time = crossing_time;
            result.limiting_projected_fwhm_arcsec = projected;
            result.limiting_speed_arcsec_s = interval.speed_arcsec_s;
        }
    }
    if (std::isfinite(minimum_crossing_time) && minimum_crossing_time > 0.0) {
        result.limiting_crossing_time_s = minimum_crossing_time;
        result.temporal_sigma_s =
            minimum_crossing_time / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    }
    return result;
}

inline std::complex<double> rtc_sampling_fir_response(
    const std::vector<double> &coefficients, double frequency_hz,
    double native_sample_rate_hz) {
    if (coefficients.empty() || !std::isfinite(frequency_hz) ||
        !std::isfinite(native_sample_rate_hz) || native_sample_rate_hz <= 0.0) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }
    const double center =
        0.5 * static_cast<double>(coefficients.size() - 1);
    std::complex<double> response{0.0, 0.0};
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        const double centered_index = static_cast<double>(i) - center;
        const double phase = -2.0 * rtc_sampling_pi * frequency_hz *
                             centered_index / native_sample_rate_hz;
        response += coefficients[i] *
                    std::complex<double>(std::cos(phase), std::sin(phase));
    }
    return response;
}

inline double rtc_sampling_gaussian_beam_amplitude(double frequency_hz,
                                                   double temporal_sigma_s) {
    if (!std::isfinite(frequency_hz) || !std::isfinite(temporal_sigma_s) ||
        temporal_sigma_s <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double x = 2.0 * rtc_sampling_pi * frequency_hz * temporal_sigma_s;
    return std::exp(-0.5 * x * x);
}

inline std::complex<double> rtc_sampling_composed_transfer(
    const std::vector<double> &coefficients, double frequency_hz,
    double native_sample_rate_hz, double temporal_sigma_s) {
    return rtc_sampling_gaussian_beam_amplitude(
               frequency_hz, temporal_sigma_s) *
           rtc_sampling_fir_response(
               coefficients, frequency_hz, native_sample_rate_hz);
}

struct RtcSamplingAliasPower {
    double desired = std::numeric_limits<double>::quiet_NaN();
    double aliased = std::numeric_limits<double>::quiet_NaN();
};

inline RtcSamplingAliasPower rtc_sampling_phase_zero_alias_power_at(
    const std::vector<double> &coefficients, double output_frequency_hz,
    double native_sample_rate_hz, int factor, double temporal_sigma_s) {
    RtcSamplingAliasPower result{0.0, 0.0};
    if (factor <= 0 || native_sample_rate_hz <= 0.0 ||
        !std::isfinite(output_frequency_hz) || temporal_sigma_s <= 0.0) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }
    const double output_rate = native_sample_rate_hz / factor;
    const double native_nyquist = 0.5 * native_sample_rate_hz;
    const int k_min = static_cast<int>(std::ceil(
        (-native_nyquist - output_frequency_hz) / output_rate));
    const int k_max = static_cast<int>(std::floor(
        (native_nyquist - output_frequency_hz) / output_rate));
    for (int k = k_min; k <= k_max; ++k) {
        const double source_frequency =
            output_frequency_hz + static_cast<double>(k) * output_rate;
        const double power = std::norm(rtc_sampling_composed_transfer(
            coefficients, source_frequency, native_sample_rate_hz,
            temporal_sigma_s));
        if (k == 0) {
            result.desired += power;
        }
        else {
            result.aliased += power;
        }
    }
    return result;
}

inline double rtc_sampling_filtered_gaussian_profile(
    const std::vector<double> &coefficients, double time_s,
    double native_sample_rate_hz, double temporal_sigma_s) {
    if (coefficients.empty() || native_sample_rate_hz <= 0.0 ||
        temporal_sigma_s <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double center =
        0.5 * static_cast<double>(coefficients.size() - 1);
    double value = 0.0;
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        const double sample_time =
            (static_cast<double>(i) - center) / native_sample_rate_hz;
        const double x = (time_s - sample_time) / temporal_sigma_s;
        value += coefficients[i] * std::exp(-0.5 * x * x);
    }
    return value;
}

inline double rtc_sampling_filtered_gaussian_fwhm_s(
    const std::vector<double> &coefficients, double native_sample_rate_hz,
    double temporal_sigma_s) {
    const double peak = rtc_sampling_filtered_gaussian_profile(
        coefficients, 0.0, native_sample_rate_hz, temporal_sigma_s);
    if (!std::isfinite(peak) || peak <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double target = 0.5 * peak;
    const double input_fwhm =
        2.0 * std::sqrt(2.0 * std::log(2.0)) * temporal_sigma_s;
    const double support_s =
        0.5 * static_cast<double>(coefficients.size() - 1) /
        native_sample_rate_hz;
    const double upper_limit = 4.0 * input_fwhm + support_s;
    double lower = 0.0;
    double upper = std::max(input_fwhm, 1.0 / native_sample_rate_hz);
    while (upper < upper_limit &&
           rtc_sampling_filtered_gaussian_profile(
               coefficients, upper, native_sample_rate_hz,
               temporal_sigma_s) > target) {
        upper *= 2.0;
    }
    if (rtc_sampling_filtered_gaussian_profile(
            coefficients, upper, native_sample_rate_hz,
            temporal_sigma_s) > target) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double midpoint = 0.5 * (lower + upper);
        if (rtc_sampling_filtered_gaussian_profile(
                coefficients, midpoint, native_sample_rate_hz,
                temporal_sigma_s) > target) {
            lower = midpoint;
        }
        else {
            upper = midpoint;
        }
    }
    return lower + upper;
}

inline std::vector<int> rtc_sampling_supported_factors(
    double native_sample_rate_hz, double configured_fir_high_hz) {
    if (!std::isfinite(native_sample_rate_hz) || native_sample_rate_hz <= 0.0 ||
        !std::isfinite(configured_fir_high_hz) ||
        configured_fir_high_hz <= 0.0) {
        return {};
    }
    const double maximum =
        std::floor(native_sample_rate_hz / (2.0 * configured_fir_high_hz));
    if (!std::isfinite(maximum) || maximum < 1.0 ||
        maximum > static_cast<double>(std::numeric_limits<int>::max())) {
        return {};
    }
    std::vector<int> factors(static_cast<std::size_t>(maximum));
    std::iota(factors.begin(), factors.end(), 1);
    return factors;
}

struct RtcSamplingCandidateMetrics {
    int factor = 0;
    double output_sample_rate_hz = std::numeric_limits<double>::quiet_NaN();
    double output_nyquist_hz = std::numeric_limits<double>::quiet_NaN();
    double samples_per_fwhm = std::numeric_limits<double>::quiet_NaN();
    double beam_peak_attenuation_fraction =
        std::numeric_limits<double>::quiet_NaN();
    double beam_half_power_fir_attenuation_db =
        std::numeric_limits<double>::quiet_NaN();
    double beam_broadening_fraction =
        std::numeric_limits<double>::quiet_NaN();
    double astronomical_alias_power_ratio =
        std::numeric_limits<double>::quiet_NaN();
    double fir_stopband_rejection_db =
        std::numeric_limits<double>::quiet_NaN();
    double fir_transition_margin_hz =
        std::numeric_limits<double>::quiet_NaN();
    double fir_raw_group_delay_s =
        std::numeric_limits<double>::quiet_NaN();
    double software_group_delay_s =
        std::numeric_limits<double>::quiet_NaN();
};

inline RtcSamplingCandidateMetrics calculate_rtc_sampling_candidate_metrics(
    int factor, double native_sample_rate_hz, double configured_fir_high_hz,
    const std::vector<double> &coefficients, double temporal_sigma_s,
    std::size_t integration_points = 512) {
    RtcSamplingCandidateMetrics result;
    result.factor = factor;
    if (factor <= 0 || native_sample_rate_hz <= 0.0 ||
        configured_fir_high_hz <= 0.0 || coefficients.empty() ||
        temporal_sigma_s <= 0.0 || integration_points < 2) {
        return result;
    }
    result.output_sample_rate_hz = native_sample_rate_hz / factor;
    result.output_nyquist_hz = 0.5 * result.output_sample_rate_hz;
    const double input_fwhm_s =
        2.0 * std::sqrt(2.0 * std::log(2.0)) * temporal_sigma_s;
    result.samples_per_fwhm = result.output_sample_rate_hz * input_fwhm_s;
    result.fir_transition_margin_hz =
        result.output_nyquist_hz - configured_fir_high_hz;
    result.fir_raw_group_delay_s =
        0.5 * static_cast<double>(coefficients.size() - 1) /
        native_sample_rate_hz;

    bool symmetric = true;
    for (std::size_t i = 0; i < coefficients.size() / 2; ++i) {
        const double scale = std::max(
            {1.0, std::abs(coefficients[i]),
             std::abs(coefficients[coefficients.size() - 1 - i])});
        if (std::abs(coefficients[i] -
                     coefficients[coefficients.size() - 1 - i]) >
            32.0 * std::numeric_limits<double>::epsilon() * scale) {
            symmetric = false;
            break;
        }
    }
    if (symmetric) {
        result.software_group_delay_s = 0.0;
    }

    const double peak = rtc_sampling_filtered_gaussian_profile(
        coefficients, 0.0, native_sample_rate_hz, temporal_sigma_s);
    if (std::isfinite(peak)) {
        result.beam_peak_attenuation_fraction = 1.0 - peak;
    }
    const double filtered_fwhm = rtc_sampling_filtered_gaussian_fwhm_s(
        coefficients, native_sample_rate_hz, temporal_sigma_s);
    if (std::isfinite(filtered_fwhm)) {
        result.beam_broadening_fraction = filtered_fwhm / input_fwhm_s - 1.0;
    }

    const double beam_half_power_hz =
        std::sqrt(std::log(2.0)) /
        (2.0 * rtc_sampling_pi * temporal_sigma_s);
    const double dc = std::abs(rtc_sampling_fir_response(
        coefficients, 0.0, native_sample_rate_hz));
    const double at_half = std::abs(rtc_sampling_fir_response(
        coefficients, beam_half_power_hz, native_sample_rate_hz));
    if (dc > 0.0 && at_half > 0.0) {
        result.beam_half_power_fir_attenuation_db =
            -20.0 * std::log10(at_half / dc);
    }

    double desired_integral = 0.0;
    double alias_integral = 0.0;
    for (std::size_t i = 0; i < integration_points; ++i) {
        const double fraction =
            static_cast<double>(i) /
            static_cast<double>(integration_points - 1);
        const double frequency = fraction * result.output_nyquist_hz;
        const auto power = rtc_sampling_phase_zero_alias_power_at(
            coefficients, frequency, native_sample_rate_hz, factor,
            temporal_sigma_s);
        const double weight =
            (i == 0 || i + 1 == integration_points) ? 0.5 : 1.0;
        desired_integral += weight * power.desired;
        alias_integral += weight * power.aliased;
    }
    if (desired_integral > 0.0) {
        result.astronomical_alias_power_ratio =
            alias_integral / desired_integral;
    }

    if (factor == 1) {
        result.fir_stopband_rejection_db =
            std::numeric_limits<double>::infinity();
    }
    else if (dc > 0.0) {
        double maximum_stopband = 0.0;
        constexpr std::size_t stopband_points = 2048;
        const double native_nyquist = 0.5 * native_sample_rate_hz;
        for (std::size_t i = 0; i < stopband_points; ++i) {
            const double fraction =
                static_cast<double>(i) /
                static_cast<double>(stopband_points - 1);
            const double frequency = result.output_nyquist_hz +
                fraction * (native_nyquist - result.output_nyquist_hz);
            maximum_stopband = std::max(
                maximum_stopband,
                std::abs(rtc_sampling_fir_response(
                    coefficients, frequency, native_sample_rate_hz)));
        }
        if (maximum_stopband > 0.0) {
            result.fir_stopband_rejection_db =
                -20.0 * std::log10(maximum_stopband / dc);
        }
    }
    return result;
}

}  // namespace citlali::pipeline
