#pragma once
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>

namespace citlali::pipeline {
// Accepted ADR-0020/v2 reference inherited unchanged from v1: unobscured
// circular 50 m aperture, exact array center frequencies. No empirical APT fit.
enum class RtcOpticalArray : std::uint8_t { a1100, a1400, a2000 };
inline double rtc_optical_frequency_hz(RtcOpticalArray array) {
  switch (array) {
  case RtcOpticalArray::a1100:
    return 272e9;
  case RtcOpticalArray::a1400:
    return 214e9;
  case RtcOpticalArray::a2000:
    return 150e9;
  }
  throw std::invalid_argument("RTC optical model requires an approved array");
}
struct RtcOpticalScale {
  double airy_fwhm_arcsec, temporal_support_hz;
};
inline RtcOpticalScale rtc_optical_scale(RtcOpticalArray array,
                                         double speed_arcsec_per_sec) {
  if (!std::isfinite(speed_arcsec_per_sec) || speed_arcsec_per_sec < 0)
    throw std::invalid_argument(
        "RTC optical scale requires finite nonnegative speed");
  constexpr double radians_per_arcsec = std::numbers::pi / (180.0 * 3600.0);
  const double lambda = 299792458.0 / rtc_optical_frequency_hz(array);
  const double fwhm = 1.028993969962188 * lambda / 50.0 / radians_per_arcsec;
  const double band = speed_arcsec_per_sec * radians_per_arcsec * 50.0 / lambda;
  return {fwhm, band};
}
} // namespace citlali::pipeline
