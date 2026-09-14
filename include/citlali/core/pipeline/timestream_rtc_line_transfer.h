#pragma once
#include <array>
#include <citlali/core/pipeline/timestream_rtc_line_power.h>
#include <citlali/core/pipeline/timestream_rtc_optical_model.h>
#include <complex>
#include <optional>

namespace citlali::pipeline {

enum class RtcNotchResponseDirection { causal, forward_reverse };
struct RtcNotchResponseSection {
  std::string identity;
  // Exact binary64 coefficients, a0=1; H(z)=sum(b_j*z^-j)/sum(a_j*z^-j).
  std::array<double, 3> b{1, 0, 0}, a{1, 0, 0};
  RtcNotchResponseDirection direction =
      RtcNotchResponseDirection::forward_reverse;
};
struct RtcTransferScienceDomain {
  std::string identity;
  RtcOpticalArray array = RtcOpticalArray::a1100;
  double trial_speed_arcsec_per_sec = NAN;
  // Explicit hypothetical evaluation domain, NOT an AST speed authority,
  // array-association admission, or a physical bound over observed support.
};
struct RtcLineTransferSpecification {
  std::string identity, lowpass_identity, state_support_identity;
  double input_interval_seconds = NAN;
  unsigned factor =
      1; // explicitly supplied trial; never automatically selected
  std::vector<RtcNotchResponseSection> notches;
  // Centered, odd-length, exactly symmetric FIR. No normalization or design.
  std::vector<double> centered_lowpass{1};
  std::optional<RtcTransferScienceDomain> science_domain;
};
struct RtcLineTransferBin {
  double input_hz = NAN, folded_output_hz = NAN;
  std::complex<double> notch_response{}, lowpass_response{},
      combined_response{};
  bool above_output_nyquist = false;
  // Native stored-bin integrals; no fabricated rectangular output spectrum.
  double input_power = NAN, lowpass_only_power = NAN, combined_power = NAN;
};
struct RtcLineTransferRegion {
  std::size_t source_region = 0;
  double input_positive_excess_power = NAN;
  double lowpass_only_positive_excess_power = NAN;
  double combined_positive_excess_power = NAN;
  double folded_combined_positive_excess_power = NAN;
};
struct RtcLineTransferBand {
  RtcLineBandMeasurement input;
  double combined_stored_power = 0, combined_background_power = 0;
  double combined_signed_residual_power = 0;
};
struct RtcTransferScienceSummary {
  double airy_fwhm_arcsec = NAN, full_temporal_support_hz = NAN;
  bool exceeds_native_nyquist = false;
  bool exceeds_output_nyquist = false;
  std::size_t sampled_bins = 0;
  double maximum_sampled_magnitude_error = NAN,
         maximum_sampled_complex_error = NAN;
  // Sampled steady-state diagnostics, not continuous-band bounds, point-source
  // response qualification, or an authorization to discard in-band signal.
};
enum class RtcLineTransferCause {
  available,
  spectral_unavailable,
  arithmetic_unavailable
};

namespace rtc_line_transfer_detail {
inline void validate(const RtcLineTransferSpecification &s) {
  if (s.identity.empty() || s.lowpass_identity.empty() ||
      s.state_support_identity.empty() ||
      !std::isfinite(s.input_interval_seconds) ||
      s.input_interval_seconds <= 0 ||
      !std::isfinite(1 / s.input_interval_seconds) || s.factor < 1 ||
      s.factor > 256 || !std::isfinite(s.input_interval_seconds * s.factor) ||
      !(1 / (s.input_interval_seconds * s.factor) > 0) ||
      s.centered_lowpass.empty() || s.centered_lowpass.size() % 2 != 1)
    throw std::invalid_argument("RTC transfer requires exact trial identities, "
                                "cadence, factor and centered FIR");
  for (std::size_t i = 0; i < s.centered_lowpass.size(); ++i)
    if (!std::isfinite(s.centered_lowpass[i]) ||
        s.centered_lowpass[i] !=
            s.centered_lowpass[s.centered_lowpass.size() - 1 - i])
      throw std::invalid_argument(
          "RTC trial FIR must be finite and exactly symmetric");
  for (std::size_t i = 0; i < s.notches.size(); ++i) {
    const auto &n = s.notches[i];
    if (n.identity.empty() || n.a[0] != 1 ||
        (n.direction != RtcNotchResponseDirection::causal &&
         n.direction != RtcNotchResponseDirection::forward_reverse))
      throw std::invalid_argument("RTC notch response convention missing");
    for (auto v : n.a)
      if (!std::isfinite(v))
        throw std::invalid_argument("RTC notch denominator nonfinite");
    for (auto v : n.b)
      if (!std::isfinite(v))
        throw std::invalid_argument("RTC notch numerator nonfinite");
    // Exact real second-order Schur/Jury inequalities. Strict stability;
    // no inferred settling tolerance, support or state-carry authorization.
    if (!(1.L + n.a[1] + n.a[2] > 0 && 1.L - n.a[1] + n.a[2] > 0 &&
          1.L - n.a[2] > 0))
      throw std::invalid_argument("RTC trial notch is not strictly stable");
    for (std::size_t j = 0; j < i; ++j)
      if (s.notches[j].identity == n.identity)
        throw std::invalid_argument(
            "RTC trial notch identities must be distinct");
  }
  if (s.science_domain) {
    if (s.science_domain->identity.empty())
      throw std::invalid_argument("RTC trial science domain lacks identity");
    auto scale = rtc_optical_scale(
        s.science_domain->array, s.science_domain->trial_speed_arcsec_per_sec);
    if (!std::isfinite(scale.temporal_support_hz))
      throw std::invalid_argument("RTC trial optical support nonfinite");
  }
}
inline bool finite(std::complex<double> z) {
  return std::isfinite(z.real()) && std::isfinite(z.imag());
}
inline std::pair<std::complex<double>, std::complex<double>>
response(const RtcLineTransferSpecification &s, double hz) {
  // Caller supplies either a checked arbitrary frequency or an exact bin from
  // the bound Learn grid. Do not reject its rounded Nyquist using a second
  // independently rounded calculation of the same boundary.
  const double w = 2 * std::numbers::pi * (hz * s.input_interval_seconds);
  const std::complex<double> z{std::cos(w), -std::sin(w)};
  std::complex<double> notch{1, 0};
  for (const auto &n : s.notches) {
    const auto denominator = n.a[0] + n.a[1] * z + n.a[2] * z * z;
    if (!finite(denominator) || std::abs(denominator) == 0)
      throw std::overflow_error("RTC transfer denominator unavailable");
    auto h = (n.b[0] + n.b[1] * z + n.b[2] * z * z) / denominator;
    if (n.direction == RtcNotchResponseDirection::forward_reverse)
      h = {std::norm(h), 0};
    notch *= h;
    if (!finite(notch))
      throw std::overflow_error("RTC notch transfer arithmetic unavailable");
  }
  const auto m = s.centered_lowpass.size() / 2;
  long double fir = s.centered_lowpass[m];
  for (std::size_t j = 1; j <= m; ++j)
    fir += 2.L * s.centered_lowpass[m + j] * std::cos(w * j);
  std::complex<double> lowpass{static_cast<double>(fir), 0};
  if (!finite(lowpass) || !finite(notch * lowpass))
    throw std::overflow_error("RTC combined response unavailable");
  return {notch, lowpass};
}
inline double fold(double frequency, double output_rate) {
  // Signed/mirrored image mapping for real samples; do not clip high input
  // frequencies to the output Nyquist or use a fixed audit frequency cutoff.
  return std::abs(std::remainder(frequency, output_rate));
}
} // namespace rtc_line_transfer_detail

class RtcLineTransferCandidate {
public:
  static std::shared_ptr<const RtcLineTransferCandidate>
  bind(std::shared_ptr<const RtcLinePowerEvidence> lines,
       TimestreamNetworkId network, std::uint32_t detector,
       RtcLineTransferSpecification specification) {
    if (!lines)
      throw std::invalid_argument("RTC transfer requires exact line evidence");
    rtc_line_transfer_detail::validate(specification);
    (void)lines->coordinate(network, detector, NativeReadoutCoordinate::x);
    (void)lines->coordinate(network, detector, NativeReadoutCoordinate::r);
    const auto &n = lines->spectral_handle()->network(network);
    if (n.cadence_available &&
        n.interval_seconds != specification.input_interval_seconds)
      throw std::invalid_argument(
          "RTC transfer trial cadence does not match measured native evidence");
    return std::shared_ptr<const RtcLineTransferCandidate>(
        new RtcLineTransferCandidate{std::move(lines), network, detector,
                                     std::move(specification)});
  }
  const auto &line_handle() const noexcept { return lines_; }
  const auto &specification() const noexcept { return specification_; }
  auto network() const noexcept { return network_; }
  auto detector() const noexcept { return detector_; }
  static constexpr bool filter_bank_certified = false, selected_factor = false;
  // Continuous-frequency arithmetic for controlled response checks. It is not
  // a realized finite-record filter, even when the supplied coefficients pass.
  std::complex<double> combined_response(double hz) const {
    if (!std::isfinite(hz) ||
        std::abs(hz) > .5 / specification_.input_interval_seconds)
      throw std::invalid_argument(
          "RTC transfer frequency outside native Nyquist");
    auto [n, l] = rtc_line_transfer_detail::response(specification_, hz);
    return n * l;
  }

private:
  RtcLineTransferCandidate(std::shared_ptr<const RtcLinePowerEvidence> l,
                           TimestreamNetworkId n, std::uint32_t d,
                           RtcLineTransferSpecification s)
      : lines_{std::move(l)}, network_{n}, detector_{d},
        specification_{std::move(s)} {}
  std::shared_ptr<const RtcLinePowerEvidence> lines_;
  TimestreamNetworkId network_;
  std::uint32_t detector_;
  RtcLineTransferSpecification specification_;
};

struct RtcLineTransferCoordinate {
  NativeReadoutCoordinate coordinate = NativeReadoutCoordinate::x;
  RtcLineTransferCause cause = RtcLineTransferCause::spectral_unavailable;
  std::vector<RtcLineTransferBin> bins;
  std::vector<RtcLineTransferRegion> regions;
  double incoherent_folded_power_proxy = NAN;
  std::optional<RtcTransferScienceSummary> science;
  bool available() const noexcept {
    return cause == RtcLineTransferCause::available;
  }
};

// Runtime Consider's numerical assessment product. This is deliberately not an
// accepted treatment plan. Exact Learn and transient handles remain reachable;
// unavailable science/admission evidence cannot become an implicit approval.
class RtcLineTransferAssessment {
public:
  static std::shared_ptr<const RtcLineTransferAssessment>
  consider(std::shared_ptr<const RtcLineTransferCandidate> candidate,
           std::shared_ptr<const RtcLinePowerConsideration> joint,
           std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t attempt) {
    if (!candidate || !joint || !snapshot || !attempt ||
        joint->line_handle().get() != candidate->line_handle().get() ||
        snapshot.get() != candidate->line_handle()->snapshot_handle().get())
      throw std::invalid_argument("RTC transfer consideration requires exact "
                                  "line/transient/VAL bindings");
    auto out = std::shared_ptr<RtcLineTransferAssessment>(
        new RtcLineTransferAssessment{std::move(candidate), std::move(joint),
                                      std::move(snapshot), attempt});
    out->coordinates_[0] = out->measure(NativeReadoutCoordinate::x);
    out->coordinates_[1] = out->measure(NativeReadoutCoordinate::r);
    return out;
  }
  const auto &candidate_handle() const noexcept { return candidate_; }
  const auto &joint_handle() const noexcept { return joint_; }
  const auto &snapshot_handle() const noexcept { return snapshot_; }
  const auto &coordinates() const noexcept { return coordinates_; }
  auto attempt() const noexcept { return attempt_; }
  static constexpr bool apply_authorized = false, interference_admitted = false;
  static constexpr bool finite_record_response_qualified = false,
                        coherent_alias_cross_terms_available = false;
  static constexpr std::string_view response_convention =
      "binary64-steady-state;ordered-notches-then-centered-FIR;phase-zero-"
      "trial-decimation";
  static constexpr std::string_view power_convention =
      "unchanged-stored-one-sided-native-bin-integrals;incoherent-proxy;no-"
      "noise-denominator";
  RtcLineTransferBand measure_band(NativeReadoutCoordinate coordinate,
                                   std::string identity, double low,
                                   double high) const {
    const auto &s = coordinate == NativeReadoutCoordinate::x ? coordinates_[0]
                                                             : coordinates_[1];
    if (coordinate != NativeReadoutCoordinate::x &&
        coordinate != NativeReadoutCoordinate::r)
      throw std::invalid_argument("RTC transfer coordinate invalid");
    if (!s.available())
      throw std::invalid_argument("RTC transferred band is unavailable");
    RtcLineTransferBand result;
    result.input = candidate_->line_handle()->measure_band(
        candidate_->network(), candidate_->detector(), coordinate,
        std::move(identity), low, high);
    const auto &line = candidate_->line_handle()->coordinate(
        candidate_->network(), candidate_->detector(), coordinate);
    const auto &psd = candidate_->line_handle()
                          ->spectral_handle()
                          ->spectrum(candidate_->network(),
                                     candidate_->detector(), coordinate)
                          .psd;
    long double total = 0, background = 0, signed_power = 0;
    for (auto i = result.input.first_bin; i < result.input.past_last_bin; ++i) {
      const long double gain = std::norm(s.bins[i].combined_response);
      total += s.bins[i].combined_power;
      const long double bg =
          static_cast<long double>(line.background[i]) * line.bin_increment_hz;
      background += bg * gain;
      signed_power += static_cast<long double>(psd[i] - line.background[i]) *
                      line.bin_increment_hz * gain;
    }
    result.combined_stored_power = static_cast<double>(total);
    result.combined_background_power = static_cast<double>(background);
    result.combined_signed_residual_power = static_cast<double>(signed_power);
    if (!std::isfinite(result.combined_stored_power) ||
        !std::isfinite(result.combined_background_power) ||
        !std::isfinite(result.combined_signed_residual_power))
      throw std::overflow_error("RTC transfer band arithmetic unavailable");
    return result;
  }

private:
  RtcLineTransferAssessment(std::shared_ptr<const RtcLineTransferCandidate> c,
                            std::shared_ptr<const RtcLinePowerConsideration> j,
                            std::shared_ptr<const ValSnapshot> s,
                            std::uint64_t a)
      : candidate_{std::move(c)}, joint_{std::move(j)}, snapshot_{std::move(s)},
        attempt_{a} {}
  RtcLineTransferCoordinate measure(NativeReadoutCoordinate coordinate) const {
    RtcLineTransferCoordinate result;
    result.coordinate = coordinate;
    const auto &lines = *candidate_->line_handle();
    const auto &spec = candidate_->specification();
    const auto &line = lines.coordinate(candidate_->network(),
                                        candidate_->detector(), coordinate);
    if (!line.available())
      return result;
    const auto &network =
        lines.spectral_handle()->network(candidate_->network());
    const auto &psd = lines.spectral_handle()
                          ->spectrum(candidate_->network(),
                                     candidate_->detector(), coordinate)
                          .psd;
    const double output_rate = 1 / (spec.input_interval_seconds * spec.factor);
    try {
      long double alias = 0;
      for (std::size_t i = 0; i < psd.size(); ++i) {
        auto [n, l] =
            rtc_line_transfer_detail::response(spec, network.frequency_hz[i]);
        auto h = n * l;
        const double p = psd[i] * line.bin_increment_hz, lp = p * std::norm(l),
                     cp = p * std::norm(h);
        if (!std::isfinite(p) || !std::isfinite(lp) || !std::isfinite(cp))
          throw std::overflow_error("RTC transferred power unavailable");
        // k/(N*dt) > 1/(2*factor*dt), using exact bin membership. This avoids
        // classifying a rounded Nyquist endpoint as out of band.
        const bool folded = i > network.fft_samples / (2 * spec.factor);
        if (folded)
          alias += cp;
        result.bins.push_back(
            {network.frequency_hz[i],
             folded ? rtc_line_transfer_detail::fold(network.frequency_hz[i],
                                                     output_rate)
                    : network.frequency_hz[i],
             n, l, h, folded, p, lp, cp});
      }
      result.incoherent_folded_power_proxy = static_cast<double>(alias);
      if (!std::isfinite(result.incoherent_folded_power_proxy))
        throw std::overflow_error("RTC folded power unavailable");
      for (std::size_t k = 0; k < line.regions.size(); ++k) {
        const auto &region = line.regions[k];
        long double lp = 0, cp = 0, folded = 0;
        for (auto i = region.first_bin; i < region.past_last_bin; ++i) {
          const long double p =
              static_cast<long double>(psd[i] - line.background[i]) *
              line.bin_increment_hz;
          lp += p * std::norm(result.bins[i].lowpass_response);
          const auto transmitted =
              p * std::norm(result.bins[i].combined_response);
          cp += transmitted;
          if (result.bins[i].above_output_nyquist)
            folded += transmitted;
        }
        if (!std::isfinite(static_cast<double>(lp)) ||
            !std::isfinite(static_cast<double>(cp)) ||
            !std::isfinite(static_cast<double>(folded)))
          throw std::overflow_error("RTC line-transfer region unavailable");
        result.regions.push_back(
            {k, region.positive_excess_power, static_cast<double>(lp),
             static_cast<double>(cp), static_cast<double>(folded)});
      }
      if (spec.science_domain && coordinate == NativeReadoutCoordinate::x) {
        // Astronomical protection is evaluated for x; this does not
        // assign an optical calibration or independent science claim to r.
        auto scale =
            rtc_optical_scale(spec.science_domain->array,
                              spec.science_domain->trial_speed_arcsec_per_sec);
        RtcTransferScienceSummary science{
            scale.airy_fwhm_arcsec,
            scale.temporal_support_hz,
            scale.temporal_support_hz > .5 / spec.input_interval_seconds,
            scale.temporal_support_hz > output_rate / 2,
            0,
            0,
            0};
        for (const auto &bin : result.bins)
          if (bin.input_hz <= scale.temporal_support_hz) {
            ++science.sampled_bins;
            science.maximum_sampled_magnitude_error =
                std::max(science.maximum_sampled_magnitude_error,
                         std::abs(std::abs(bin.combined_response) - 1));
            science.maximum_sampled_complex_error = std::max(
                science.maximum_sampled_complex_error,
                std::abs(bin.combined_response - std::complex<double>{1, 0}));
          }
        result.science = science;
      }
      result.cause = RtcLineTransferCause::available;
    } catch (const std::overflow_error &) {
      result.bins.clear();
      result.regions.clear();
      result.science.reset();
      result.incoherent_folded_power_proxy = NAN;
      result.cause = RtcLineTransferCause::arithmetic_unavailable;
    }
    return result;
  }
  std::shared_ptr<const RtcLineTransferCandidate> candidate_;
  std::shared_ptr<const RtcLinePowerConsideration> joint_;
  std::shared_ptr<const ValSnapshot> snapshot_;
  std::uint64_t attempt_;
  std::array<RtcLineTransferCoordinate, 2> coordinates_;
};
} // namespace citlali::pipeline
