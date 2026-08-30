#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string_view>

namespace citlali::wp7::rtc_filter_fixture {

inline constexpr std::string_view policy_id =
    "wp7-rtc-scan-array-numerical-policy-v2";
inline constexpr double speed_of_light_m_per_sec = 299792458.0;
inline constexpr double aperture_diameter_m = 50.0;
inline constexpr double airy_fwhm_coefficient = 1.028993969962188;
inline constexpr double velocity_margin = 1.05;
inline constexpr double cadence_margin = 0.9999;
inline constexpr double minimum_samples_per_airy_fwhm = 4.0;
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

inline double minimum_safe_output_sample_rate_hz(
    Array array, double measured_maximum_speed_arcsec_per_sec) {
    if (!std::isfinite(measured_maximum_speed_arcsec_per_sec) ||
        measured_maximum_speed_arcsec_per_sec <= 0.0) {
        throw std::invalid_argument(
            "RTC filter structural-eligibility speed is invalid");
    }
    const double planned_speed =
        velocity_margin * measured_maximum_speed_arcsec_per_sec;
    const double wavelength =
        speed_of_light_m_per_sec / array_frequency_hz(array);
    const double airy_fwhm_arcsec =
        airy_fwhm_coefficient * wavelength / aperture_diameter_m *
        (180.0 * 3600.0 / std::numbers::pi_v<double>);
    const double science_cutoff =
        planned_speed * (std::numbers::pi_v<double> / (180.0 * 3600.0)) *
        aperture_diameter_m / wavelength;
    return std::max(2.0 * science_cutoff,
                    minimum_samples_per_airy_fwhm * planned_speed /
                        airy_fwhm_arcsec);
}

struct FactorEvidence {
    Array array = Array::a1100;
    int factor = 1;
    double input_sample_rate_hz = 0.0;
    double safe_input_sample_rate_hz = 0.0;
    double output_sample_rate_hz = 0.0;
    double safe_output_sample_rate_hz = 0.0;
    double safe_output_nyquist_hz = 0.0;
    double measured_maximum_speed_arcsec_per_sec = 0.0;
    double planned_speed_arcsec_per_sec = 0.0;
    double wavelength_m = 0.0;
    double airy_fwhm_arcsec = 0.0;
    double science_cutoff_hz = 0.0;
    double output_samples_per_airy_fwhm = 0.0;
    bool science_band_sampling_adequate = false;
    bool beam_sampling_adequate = false;

    bool sampling_eligible() const noexcept {
        return science_band_sampling_adequate && beam_sampling_adequate;
    }
};

inline FactorEvidence evaluate_factor(
    Array array, double input_sample_rate_hz,
    double measured_maximum_speed_arcsec_per_sec, int factor) {
    if (!std::isfinite(input_sample_rate_hz) ||
        input_sample_rate_hz <= 0.0 ||
        !std::isfinite(measured_maximum_speed_arcsec_per_sec) ||
        measured_maximum_speed_arcsec_per_sec <= 0.0 ||
        factor < minimum_factor || factor > maximum_factor) {
        throw std::invalid_argument(
            "RTC filter structural-eligibility inputs are invalid");
    }

    FactorEvidence result;
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
    result.measured_maximum_speed_arcsec_per_sec =
        measured_maximum_speed_arcsec_per_sec;
    result.planned_speed_arcsec_per_sec =
        velocity_margin * measured_maximum_speed_arcsec_per_sec;
    result.wavelength_m =
        speed_of_light_m_per_sec / array_frequency_hz(array);
    result.airy_fwhm_arcsec =
        airy_fwhm_coefficient * result.wavelength_m /
        aperture_diameter_m *
        (180.0 * 3600.0 / std::numbers::pi_v<double>);
    const double planned_speed_rad_per_sec =
        result.planned_speed_arcsec_per_sec *
        (std::numbers::pi_v<double> / (180.0 * 3600.0));
    result.science_cutoff_hz =
        planned_speed_rad_per_sec * aperture_diameter_m /
        result.wavelength_m;
    result.output_samples_per_airy_fwhm =
        result.safe_output_sample_rate_hz *
        result.airy_fwhm_arcsec /
        result.planned_speed_arcsec_per_sec;
    result.science_band_sampling_adequate =
        result.safe_output_nyquist_hz >= result.science_cutoff_hz;
    result.beam_sampling_adequate =
        result.output_samples_per_airy_fwhm >=
        minimum_samples_per_airy_fwhm;
    return result;
}

}  // namespace citlali::wp7::rtc_filter_fixture
