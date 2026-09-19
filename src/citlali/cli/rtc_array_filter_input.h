#pragma once
// Application input binding for explicit, precomputed development artifacts.
// Numerical construction stays offline; existing RTC Consider owns admission.
#include <citlali/core/pipeline/timestream_rtc_optical_model.h>
#include <citlali/core/utils/sha256.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::cli::detail {
struct RtcArrayFilterInput {
    pipeline::RtcOpticalArray array;
    unsigned factor;
    double nominal_interval, cadence_bound, speed_ceiling;
    std::string identity;
    std::vector<double> coefficients;
    void require_cadence(double nominal, double measured) const {
        if(nominal!=nominal_interval || !std::isfinite(measured) ||
           std::abs(measured/nominal_interval-1)>cadence_bound)
            throw std::invalid_argument("RTC explicit low-pass cadence binding mismatch");
    }
};
inline RtcArrayFilterInput read_rtc_array_filter(const YAML::Node &n,int apt_array) {
    constexpr std::array<const char*,3> names{"a1100","a1400","a2000"};
    if(apt_array<0 || apt_array>2 || !n.IsMap() ||
       n["schema"].as<std::string>()!="rtc-explicit-array-lowpass-v1" ||
       n["array"].as<std::string>()!=names[apt_array] ||
       n["use"].as<std::string>()!="explicit-development-only" ||
       n["automatic_selection"].as<bool>() || n["production_certified"].as<bool>())
        throw std::invalid_argument("RTC explicit low-pass array/use binding mismatch");
    RtcArrayFilterInput out{static_cast<pipeline::RtcOpticalArray>(apt_array),
        n["factor"].as<unsigned>(),n["nominal_interval_seconds"].as<double>(),
        n["cadence_relative_bound"].as<double>(),n["speed_ceiling_arcsec_per_sec"].as<double>(),
        n["identity"].as<std::string>(),n["coefficients"].as<std::vector<double>>()};
    if(out.factor!=(apt_array==2?2U:1U) || out.nominal_interval!=.008192 ||
       out.cadence_bound!=.0001 || out.speed_ceiling!=235 || out.identity.empty() ||
       n["reference_frequency_hz"].as<double>()!=pipeline::rtc_optical_frequency_hz(out.array) ||
       n["beam"].as<std::string>()!="50m-unobscured-circular-Airy")
        throw std::invalid_argument("RTC explicit low-pass domain is outside the selected development profiles");
    // The bounded request selects these measured artifacts, not an arbitrary
    // symmetric FIR supplied with a self-consistent digest.
    constexpr std::array<const char*,3> selected_coefficients{
        "686965f3e485a09ed592996a8068a4868f019cfbf9b6db8ba034bf7b3032ad0d",
        "cd41c869fcf4c8af190c8dab742cb6c3e9e8288a2c58ce0d9d69c5c9bc4042bd",
        "e25377075b9b20147bfc11b9c4b9ab792dd60576166f6c25dbc8b9aaf75e8967"};
    constexpr std::array<const char*,3> selected_identities{
        "offline-explicit-a1100-235arcsec-per-sec-Kaiser-F1-20260919",
        "offline-explicit-a1400-235arcsec-per-sec-Kaiser-F1-20260919",
        "offline-explicit-a2000-235arcsec-per-sec-80dB-Kaiser-F2"};
    if(out.identity!=selected_identities[apt_array] ||
       n["coefficients_sha256"].as<std::string>()!=selected_coefficients[apt_array])
        throw std::invalid_argument("RTC low-pass coefficients are not a selected development artifact");
    const auto &h=out.coefficients;
    if(h.empty() || h.size()%2!=1 ||
       !std::all_of(h.begin(),h.end(),[](double v){return std::isfinite(v);}) ||
       !std::equal(h.begin(),h.end(),h.rbegin()) ||
       std::abs(std::accumulate(h.begin(),h.end(),0.)-1)>1e-12)
        throw std::invalid_argument("RTC explicit low-pass requires finite symmetric unit-DC coefficients");
    if(std::endian::native!=std::endian::little)
        throw std::invalid_argument("RTC development coefficient digest requires little-endian binary64");
    citlali::utils::Sha256 digest;
    digest.update(reinterpret_cast<const std::uint8_t*>(h.data()),h.size()*sizeof(double));
    if(digest.finish()!=n["coefficients_sha256"].as<std::string>())
        throw std::invalid_argument("RTC explicit low-pass coefficient digest mismatch");
    if(pipeline::rtc_optical_scale(out.array,out.speed_ceiling).temporal_support_hz>=
       .5/(out.nominal_interval*out.factor)*(1-out.cadence_bound))
        throw std::invalid_argument("RTC explicit low-pass cannot contain the optical domain");
    return out;
}
} // namespace citlali::cli::detail
