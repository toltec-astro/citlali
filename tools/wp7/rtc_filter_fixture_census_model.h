#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string_view>

namespace citlali::wp7::rtc_filter_fixture {

inline constexpr std::string_view numerical_policy_id =
    "wp7-rtc-scan-array-numerical-policy-v2";
inline constexpr std::string_view speed_admission_policy_id =
    "wp7-rtc-occurrence-speed-admission-v1";
inline constexpr double speed_of_light_m_per_sec = 299792458.0;
inline constexpr double aperture_diameter_m = 50.0;
inline constexpr double airy_fwhm_coefficient = 1.028993969962188;
inline constexpr double velocity_margin = 1.05;
inline constexpr double cadence_margin = 0.9999;
inline constexpr double minimum_samples_per_airy_fwhm = 4.0;
inline constexpr double minimum_science_speed_arcsec_per_sec = 1.0;
inline constexpr int minimum_factor = 1;
inline constexpr int maximum_factor = 256;

enum class Array : std::int64_t {
    a1100 = 0,
    a1400 = 1,
    a2000 = 2,
};

constexpr std::string_view array_name(Array array) {
    switch (array) {
    case Array::a1100: return "a1100";
    case Array::a1400: return "a1400";
    case Array::a2000: return "a2000";
    }
    throw std::invalid_argument("unsupported TolTEC array identity");
}

constexpr double array_frequency_hz(Array array) {
    switch (array) {
    case Array::a1100: return 272.0e9;
    case Array::a1400: return 214.0e9;
    case Array::a2000: return 150.0e9;
    }
    throw std::invalid_argument("unsupported TolTEC array identity");
}

inline Array array_from_index(std::int64_t value) {
    switch (value) {
    case 0: return Array::a1100;
    case 1: return Array::a1400;
    case 2: return Array::a2000;
    default:
        throw std::invalid_argument("unsupported TolTEC array identity");
    }
}

enum class StructuralCeilingConstraint {
    science_band_nyquist,
    beam_sampling,
};

constexpr std::string_view constraint_name(
    StructuralCeilingConstraint constraint) noexcept {
    switch (constraint) {
    case StructuralCeilingConstraint::science_band_nyquist:
        return "science_band_nyquist";
    case StructuralCeilingConstraint::beam_sampling:
        return "beam_sampling";
    }
    return "unknown";
}

struct StructuralModeEvidence {
    Array array = Array::a1100;
    int factor = 1;
    double input_sample_rate_hz = 0.0;
    double safe_input_sample_rate_hz = 0.0;
    double output_sample_rate_hz = 0.0;
    double safe_output_sample_rate_hz = 0.0;
    double safe_output_nyquist_hz = 0.0;
    double wavelength_m = 0.0;
    double airy_fwhm_arcsec = 0.0;
    double science_band_ceiling_arcsec_per_sec = 0.0;
    double beam_sampling_ceiling_arcsec_per_sec = 0.0;
    double upper_speed_ceiling_arcsec_per_sec = 0.0;
    StructuralCeilingConstraint governing_constraint =
        StructuralCeilingConstraint::beam_sampling;

    bool has_science_speed_domain() const noexcept {
        return upper_speed_ceiling_arcsec_per_sec >=
            minimum_science_speed_arcsec_per_sec;
    }
};

inline StructuralModeEvidence evaluate_structural_mode(
    Array array, double input_sample_rate_hz, int factor) {
    if (!std::isfinite(input_sample_rate_hz) ||
        input_sample_rate_hz <= 0.0 || factor < minimum_factor ||
        factor > maximum_factor) {
        throw std::invalid_argument(
            "RTC filter structural-mode inputs are invalid");
    }

    StructuralModeEvidence result;
    result.array = array;
    result.factor = factor;
    result.input_sample_rate_hz = input_sample_rate_hz;
    result.safe_input_sample_rate_hz =
        cadence_margin * input_sample_rate_hz;
    result.output_sample_rate_hz = input_sample_rate_hz / factor;
    result.safe_output_sample_rate_hz =
        result.safe_input_sample_rate_hz / factor;
    result.safe_output_nyquist_hz =
        result.safe_output_sample_rate_hz / 2.0;
    result.wavelength_m =
        speed_of_light_m_per_sec / array_frequency_hz(array);
    constexpr double arcsec_per_rad =
        180.0 * 3600.0 / std::numbers::pi_v<double>;
    result.airy_fwhm_arcsec =
        airy_fwhm_coefficient * result.wavelength_m /
        aperture_diameter_m * arcsec_per_rad;
    result.science_band_ceiling_arcsec_per_sec =
        result.safe_output_nyquist_hz * result.wavelength_m /
        aperture_diameter_m * arcsec_per_rad / velocity_margin;
    result.beam_sampling_ceiling_arcsec_per_sec =
        result.safe_output_sample_rate_hz * result.airy_fwhm_arcsec /
        minimum_samples_per_airy_fwhm / velocity_margin;
    if (result.science_band_ceiling_arcsec_per_sec <
        result.beam_sampling_ceiling_arcsec_per_sec) {
        result.governing_constraint =
            StructuralCeilingConstraint::science_band_nyquist;
        result.upper_speed_ceiling_arcsec_per_sec =
            result.science_band_ceiling_arcsec_per_sec;
    } else {
        result.governing_constraint =
            StructuralCeilingConstraint::beam_sampling;
        result.upper_speed_ceiling_arcsec_per_sec =
            result.beam_sampling_ceiling_arcsec_per_sec;
    }
    return result;
}

inline bool upper_speed_admitted(
    double speed_arcsec_per_sec,
    double upper_speed_ceiling_arcsec_per_sec) noexcept {
    return std::isfinite(speed_arcsec_per_sec) &&
        std::isfinite(upper_speed_ceiling_arcsec_per_sec) &&
        speed_arcsec_per_sec <= upper_speed_ceiling_arcsec_per_sec;
}

struct OccurrenceAdmissionSummary {
    std::size_t occurrence_count = 0;
    std::size_t ast_unavailable_count = 0;
    std::size_t below_minimum_science_speed_count = 0;
    std::size_t base_admitted_count = 0;
    std::size_t upper_speed_admitted_count = 0;
    std::size_t scan_speed_above_mode_support_count = 0;
    std::size_t retained_run_count = 0;
    std::size_t longest_retained_run_occurrences = 0;
};

// A non-finite mapped speed denotes unavailable AST motion. Entry zero of
// continues_previous is ignored; each later nonzero entry says that the native
// packet occurrence continues the preceding delivered occurrence.
inline OccurrenceAdmissionSummary summarize_occurrence_admission(
    std::span<const double> mapped_speeds_arcsec_per_sec,
    std::span<const std::uint8_t> continues_previous,
    double upper_speed_ceiling_arcsec_per_sec) {
    if (mapped_speeds_arcsec_per_sec.empty() ||
        mapped_speeds_arcsec_per_sec.size() != continues_previous.size() ||
        !std::isfinite(upper_speed_ceiling_arcsec_per_sec) ||
        upper_speed_ceiling_arcsec_per_sec < 0.0) {
        throw std::invalid_argument(
            "RTC occurrence-admission inputs are invalid");
    }

    OccurrenceAdmissionSummary result;
    result.occurrence_count = mapped_speeds_arcsec_per_sec.size();
    bool preceding_retained = false;
    std::size_t current_run_length = 0;
    for (std::size_t index = 0;
         index < mapped_speeds_arcsec_per_sec.size(); ++index) {
        const double speed = mapped_speeds_arcsec_per_sec[index];
        bool retained = false;
        if (!std::isfinite(speed)) {
            ++result.ast_unavailable_count;
        } else if (speed < minimum_science_speed_arcsec_per_sec) {
            ++result.below_minimum_science_speed_count;
        } else {
            ++result.base_admitted_count;
            if (upper_speed_admitted(
                    speed, upper_speed_ceiling_arcsec_per_sec)) {
                ++result.upper_speed_admitted_count;
                retained = true;
            } else {
                ++result.scan_speed_above_mode_support_count;
            }
        }

        if (retained && preceding_retained &&
            continues_previous[index] != 0U) {
            ++current_run_length;
        } else if (retained) {
            ++result.retained_run_count;
            current_run_length = 1;
        } else {
            current_run_length = 0;
        }
        result.longest_retained_run_occurrences = std::max(
            result.longest_retained_run_occurrences,
            current_run_length);
        preceding_retained = retained;
    }
    return result;
}

}  // namespace citlali::wp7::rtc_filter_fixture
