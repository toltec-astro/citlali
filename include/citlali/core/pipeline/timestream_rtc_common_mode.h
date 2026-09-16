#pragma once

#include <citlali/core/pipeline/ast_scan_motion_alignment.h>
#include <citlali/core/pipeline/timestream_rtc_jump_exclusion.h>
#include <citlali/core/pipeline/timestream_rtc_line_power.h>
#include <citlali/core/pipeline/timestream_rtc_optical_model.h>

namespace citlali::pipeline {

// Bounded owner diagnostic, 2026-09-16. These guards do not admit/reject data.
struct RtcCommonModePolicy {
  static constexpr std::string_view identity =
      "rtc-original-common-mode-health-v1";
  static constexpr std::size_t minimum_contributors = 3, minimum_samples = 64;
  static constexpr double huber = 1.345, mad_scale = 1.4826, tolerance = 1e-8;
  static constexpr std::size_t maximum_iterations = 256,
                               maximum_self_checks = 8;
};
struct RtcCommonModeMember {
  std::string detector_occurrence;
  bool reference_eligible = false;
  double flxscale = NAN; // existing multiplicative mJy/beam/xs, signed
};
struct RtcCommonModeDomain {
  TimestreamNetworkId network = -1;
  std::shared_ptr<const RtcExistingScanBinding> scans;
  std::shared_ptr<const AstScanMotionNetworkView> motion;
  std::string population_authority;
  std::vector<RtcCommonModeMember> members; // exact stored-column order
  RtcOpticalArray array = RtcOpticalArray::a2000;
  double nominal_interval_seconds = NAN, speed_ceiling_arcsec_per_sec = NAN;
  std::uint32_t output_factor = 0;
  double speed_margin_fraction = .05, cadence_margin_fraction = .0001;
};
enum class RtcCommonModeCause : std::uint8_t {
  available,
  insufficient_support,
  weak_reference,
  input_nonfinite,
  fit_failed
};
// Reference reason bitset: 1 supplied population exclusion, 2 missing pair,
// 4 unresolved candidate edge, 8 unavailable paired noise screen, 16 too short.
struct RtcCommonModeInterval {
  std::uint64_t scan = 0;
  RtcEventRange rows;
  std::vector<std::uint8_t> reference_reasons;
  std::vector<std::size_t> paired_rows, target_eligible_rows;
  std::vector<double> baseline; // original x units; NaN for noncontributors
  std::size_t contributors = 0, speed_eligible_rows = 0;
  double reference_scatter = NAN;
};
struct RtcCommonModeFit {
  std::uint32_t detector = 0;
  std::size_t interval = 0;
  RtcEventRange rows; // contiguous native support; never compacted
  RtcCommonModeCause cause = RtcCommonModeCause::insufficient_support;
  double gain = NAN, offset = NAN, correlation = NAN, residual_scatter = NAN;
  double reference_scatter = NAN, calibrated_relative_gain = NAN;
  std::size_t iterations = 0;
  bool available() const { return cause == RtcCommonModeCause::available; }
};
class RtcCommonModeEvidence;
struct RtcCommonModeSelfCheck {
  std::shared_ptr<const RtcCommonModeEvidence> evidence;
  std::uint32_t detector = 0;
  std::vector<double> reference; // same full native axis, NaN off support
  std::vector<RtcCommonModeFit> fits;
};
class RtcCommonModeEvidence
    : public std::enable_shared_from_this<RtcCommonModeEvidence> {
public:
  static std::shared_ptr<const RtcCommonModeEvidence>
  learn(std::shared_ptr<const RtcSpikeEvidence>, RtcCommonModeDomain,
        std::uint64_t attempt);
  const auto &original_handle() const { return original_; }
  const auto &domain() const { return domain_; }
  auto attempt() const { return attempt_; }
  const auto &intervals() const { return intervals_; }
  const auto &reference() const { return reference_; }
  const auto &fits() const { return fits_; }
  const auto &speed_admitted() const { return speed_admitted_; }
  double sampling_speed_limit() const { return speed_limit_; }
  std::size_t logical_owned_bytes() const;
  // Compact residual product: aliases immutable originals, reference and fit;
  // no new science plane or replacement measurements. Row must be in this fit.
  double residual(std::size_t fit, TimestreamNativeRow row) const;
  std::vector<RtcCommonModeSelfCheck>
  self_excluded_checks(std::span<const std::uint32_t> detectors) const;

private:
  std::shared_ptr<const RtcSpikeEvidence> original_;
  RtcCommonModeDomain domain_;
  std::uint64_t attempt_ = 0;
  double speed_limit_ = NAN;
  std::vector<bool> speed_admitted_;
  std::vector<double> reference_;
  std::vector<RtcCommonModeInterval> intervals_;
  std::vector<RtcCommonModeFit> fits_;
};
struct RtcCommonModeDetectorSummary {
  std::uint32_t detector = 0;
  std::size_t fitted_rows = 0, available_segments = 0, unavailable_segments = 0;
  std::size_t original_candidates = 0, unavailable_noise_blocks = 0;
  double gain = NAN, gain_mad = NAN, relative_gain = NAN,
         negative_relative_fraction = NAN;
  double correlation = NAN, residual_scatter = NAN;
  double spectral_peak_hz = NAN, spectral_excess_fraction = NAN;
};
// Diagnostic comparison only. No VAL delta, rejection decision or Apply
// adapter.
class RtcCommonModeConsideration {
public:
  static RtcCommonModeConsideration
      compare(std::shared_ptr<const RtcCommonModeEvidence>,
              std::shared_ptr<const RtcLinePowerEvidence>);
  const auto &evidence_handle() const { return evidence_; }
  const auto &spectral_handle() const { return lines_; }
  const auto &detectors() const { return detectors_; }
  const auto &inspection_targets() const { return targets_; }
  static constexpr bool rejection_authorized = false,
                        formal_uncertainty_available = false;

private:
  std::shared_ptr<const RtcCommonModeEvidence> evidence_;
  std::shared_ptr<const RtcLinePowerEvidence> lines_;
  std::vector<RtcCommonModeDetectorSummary> detectors_;
  std::vector<std::uint32_t> targets_;
};
} // namespace citlali::pipeline
