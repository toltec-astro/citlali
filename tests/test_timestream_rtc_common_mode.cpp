#include "timestream_successor_identity_test_support.h"
#include <citlali/core/pipeline/timestream_rtc_common_mode.h>
#include <gtest/gtest.h>
#include <random>
namespace {
using namespace citlali::pipeline;
namespace helper = citlali::test::timestream_successor;
struct Case {
  static constexpr std::size_t n = 2400, nd = 5;
  NativePairedReadoutMatrix x{n, nd}, r{n, nd};
  std::vector<NativeReadoutCoordinateState> xs = helper::valid_states(n * nd),
                                            rs = xs;
  std::vector<double> times;
  std::vector<TimestreamPacketCounter> counters;
  std::shared_ptr<const NativePairedReadoutObservation> parent;
  std::shared_ptr<const RtcSpikeEvidence> spikes;
  RtcCommonModeDomain domain;
  Case() {
    std::mt19937 generator(1729);
    std::normal_distribution<double> noise(0, .02);
    for (std::size_t i = 0; i < n; ++i) {
      times.push_back(1000 + .008192 * i);
      counters.push_back(2000 + i);
      const double c =
          2 * std::sin(.008192 * i * 3.2) + .3 * std::cos(.008192 * i * 1.2);
      for (std::size_t d = 0; d < nd; ++d) {
        x(i, d) =
            10 * d + std::array{1., 2., .5, -1., 1.}[d] * c + noise(generator);
        r(i, d) = noise(generator);
      }
    }
  }
  void bind(double speed = 10) {
    auto timing = std::make_shared<const NativeNetworkAlignment>(
        0, 100, helper::time_vector(times), counters);
    std::vector<NativePairedReadoutOccurrenceBinding> occurrences;
    for (std::size_t i = 0; i < n; ++i)
      occurrences.push_back({static_cast<std::int64_t>(i + 10000),
                             static_cast<std::int64_t>(i + 20000),
                             {times[i] - .004096, times[i] + .004096}});
    auto axis = std::make_shared<const NativePairedReadoutOccurrenceAxis>(
        timing, 100, std::move(occurrences));
    std::vector<NativePairedReadoutNetwork> nets;
    nets.push_back(NativePairedReadoutNetwork::admit(
        axis, helper::detector_axis(0, nd),
        helper::mapping_authority(0, "health-fixture"), x, r, xs, rs));
    parent = helper::make_observation(std::move(nets), {0});
    auto val = ValSnapshot::initial(parent);
    spikes = learn_rtc_spike_candidates(
        NativePairedReadoutView::full(parent), val,
        RtcSpikeSourceProtection::admit(parent, "test-unknown-retained",
                                        RtcSpikeProtection::unavailable),
        1);
    domain.network = 0;
    domain.population_authority = "test-APT-signed-calibration";
    for (std::size_t d = 0; d < nd; ++d)
      domain.members.push_back(
          {parent->network(0).detectors()[d].detector_occurrence_id, d != 3,
           std::array{1., .5, 2., 1., 1.}[d]});
    domain.scans = RtcExistingScanBinding::admit(
        parent, "existing-test-processing", "exact-native", "known",
        RtcExistingScanSupportState::conservative_native_support_bound,
        {{0, {0, 100, 1300}}, {1, {0, 1300, 2500}}});
    auto t = Eigen::VectorXd::LinSpaced(1500, 999, 1028.98);
    Eigen::VectorXd ra = (t.array() - 999) * speed * std::numbers::pi /
                         (180 * 3600),
                    dec = Eigen::VectorXd::Zero(t.size());
    AstScanMotionSourceMetadata metadata{
        AstScanMotionProducerKind::real_toltec,
        "Science",
        "Lissajous",
        1,
        2000,
        0,
        50,
        AstScanMotionFieldRegistry::source_ra_act_source_dec_act_j2000_radians,
        "health-controlled-motion"};
    auto source = AstScanMotionSource::admit(parent->scope(), parent->scope(),
                                             0, metadata, t, ra, dec);
    domain.motion = AstScanMotionNetworkView::admit(
        build_ast_scan_motion_product(source, {1, 2, 3, 4}), timing);
    domain.nominal_interval_seconds = .008192;
    domain.speed_ceiling_arcsec_per_sec = 20;
    domain.output_factor = 2;
  }
  auto learn() { return RtcCommonModeEvidence::learn(spikes, domain, 2); }
};
TEST(RtcCommonMode, FreeSignedGainsOffsetsCalibrationAndSelfExclusion) {
  Case c;
  c.bind();
  c.domain.members[3].reference_eligible = true;
  const auto ev = c.learn();
  ASSERT_EQ(ev->intervals().size(), 2);
  ASSERT_EQ(ev->intervals()[0].contributors, 5);
  for (const auto &f : ev->fits())
    if (f.available()) {
      const double g = std::array{1., 2., .5, -1., 1.}[f.detector];
      EXPECT_NEAR(f.gain, g, .035);
      EXPECT_NEAR(f.calibrated_relative_gain, f.detector == 3 ? -1. : 1., .035);
      // Independent target and median-reference noise both contribute.
      EXPECT_LT(f.residual_scatter, .05);
    }
  const auto checks =
      ev->self_excluded_checks(std::array<std::uint32_t, 2>{3, 0});
  for (const auto &f : checks[0].fits)
    if (f.available()) {
      EXPECT_LT(f.gain, -.95);
      EXPECT_LT(f.correlation, -.99);
    }
  for (const auto &f : checks[1].fits)
    if (f.available())
      EXPECT_GT(f.correlation, .99);
  EXPECT_EQ(c.parent->network(0).value(NativeReadoutCoordinate::x, 140, 0),
            c.x(40, 0));
  EXPECT_EQ(c.spikes->val_snapshot_handle()->generation().value, 0);
  for (std::size_t i = 0; i < ev->fits().size(); ++i) {
    const auto &f = ev->fits()[i];
    if (f.available())
      EXPECT_NEAR(ev->residual(i, f.rows.first),
                  c.parent->network(0).value(NativeReadoutCoordinate::x,
                                             f.rows.first, f.detector) -
                      f.offset - f.gain * ev->reference()[f.rows.first - 100],
                  1e-14);
  }
}
TEST(RtcCommonMode, NoisyUnstableAndMissingCalibrationStillHaveRawFits) {
  Case c;
  for (std::size_t i = 0; i < c.n; ++i) {
    if (i >= 1200)
      c.x(i, 3) = 30 + 2 * (c.x(i, 0));
    c.x(i, 4) += .5 * std::sin(i * 1.735);
  }
  c.bind();
  c.domain.members[3].flxscale = NAN;
  const auto ev = c.learn();
  bool first = false, second = false, noisy = false;
  for (const auto &f : ev->fits())
    if (f.available()) {
      if (f.detector == 3) {
        EXPECT_TRUE(std::isnan(f.calibrated_relative_gain));
        if (f.interval == 0) {
          first = true;
          EXPECT_LT(f.gain, 0);
        } else {
          second = true;
          EXPECT_GT(f.gain, 0);
        }
      }
      if (f.detector == 4 && f.residual_scatter > .2)
        noisy = true;
    }
  EXPECT_TRUE(first && second && noisy);
}
TEST(RtcCommonMode, SharedStructureRemainsInReferenceAndOriginalEvidence) {
  Case c;
  for (std::size_t i = 0; i < c.n; ++i)
    for (std::size_t d = 0; d < c.nd; ++d)
      c.x(i, d) += std::array{1., 2., .5, -1., 1.}[d] * .3 *
                   std::sin(2 * std::numbers::pi * 11 * .008192 * i);
  c.bind();
  const auto ev = c.learn();
  double projection = 0, norm = 0;
  for (std::size_t i = 0; i < c.n; ++i)
    if (std::isfinite(ev->reference()[i])) {
      const double tone = std::sin(2 * std::numbers::pi * 11 * .008192 * i);
      projection += ev->reference()[i] * tone;
      norm += tone * tone;
    }
  EXPECT_GT(projection / norm, .25);
  EXPECT_FALSE(RtcCommonModeConsideration::rejection_authorized);
  EXPECT_EQ(ev->original_handle().get(), c.spikes.get());
}
TEST(RtcCommonMode, ChangingMembershipAndPairedInvalidityAreExplicit) {
  Case c;
  c.rs[700 * c.nd] =
      NativeReadoutCoordinateState::measured(true, false, true, true);
  c.bind();
  auto ev = c.learn();
  EXPECT_TRUE(ev->intervals()[0].reference_reasons[0] & 2);
  EXPECT_EQ(ev->intervals()[0].contributors, 3);
  EXPECT_EQ(ev->intervals()[1].contributors, 4);
  for (const auto &f : ev->fits())
    if (f.detector == 0)
      EXPECT_FALSE(f.rows.first <= 800 && f.rows.past_last > 800);
  auto checks = ev->self_excluded_checks(std::array<std::uint32_t, 1>{1});
  for (const auto &f : checks[0].fits)
    if (f.interval == 0)
      EXPECT_FALSE(f.available());
}
TEST(RtcCommonMode,
     InsufficientReferenceIsInconclusiveAndIdentityCannotRebind) {
  Case c;
  c.bind();
  c.domain.members[1].reference_eligible = false;
  c.domain.members[2].reference_eligible = false;
  const auto ev = c.learn();
  EXPECT_TRUE(ev->fits().empty());
  for (auto v : ev->reference())
    EXPECT_TRUE(std::isnan(v));
  c.domain.members[0].detector_occurrence = "wrong";
  EXPECT_THROW(c.learn(), std::invalid_argument);
}
TEST(RtcCommonMode, CandidateBoundaryIsNeverFittedAcrossOrAccepted) {
  Case c;
  for (std::size_t i = 650; i < c.n; ++i)
    c.x(i, 0) += 3;
  c.bind();
  auto ev = c.learn();
  ASSERT_FALSE(c.spikes->candidates().empty());
  EXPECT_TRUE(ev->intervals()[0].reference_reasons[0] & 4);
  for (const auto &f : ev->fits())
    if (f.detector == 0)
      EXPECT_FALSE(f.rows.first < 750 && f.rows.past_last > 750);
}
TEST(RtcCommonMode, PhysicalGapAndSpeedExclusionsRemainLearningBoundaries) {
  Case c;
  for (std::size_t i = 700; i < c.n; ++i) {
    c.counters[i] += 2;
    c.times[i] += .016384;
  }
  c.bind();
  const auto ev = c.learn();
  for (const auto &f : ev->fits())
    EXPECT_FALSE(f.rows.first < 800 && f.rows.past_last > 800);
  Case slow;
  slow.bind(.5);
  auto no = slow.learn();
  EXPECT_TRUE(no->fits().empty());
  for (bool admitted : no->speed_admitted())
    EXPECT_FALSE(admitted);
}
TEST(RtcCommonMode,
     ConsiderRequiresSameOriginalSpectralAndVALAndRetainsSignedOutlier) {
  Case c;
  c.bind();
  auto ev = c.learn();
  auto val = c.spikes->val_snapshot_handle();
  auto subject =
      ValNativeRealization::create(c.parent, {ValProducer::align, 1}, 1,
                                   ValNativeProductRole::original_input, 0);
  auto id = RtcSpectralInputIdentity::bind(
      subject, val, c.spikes->input_handle()->span(0),
      RtcSpectralInputStage::original_reference, "test-original", 1);
  auto spectral = RtcNativeSpectralEvidence::learn_initial(
      c.spikes, {id}, {{0, "test", .008192, 1e-7}}, 3);
  auto lines = RtcLinePowerEvidence::learn(
      spectral, val, RtcLinePowerProfile::initial_2_hz, 4);
  auto result = RtcCommonModeConsideration::compare(ev, lines);
  EXPECT_LT(result.detectors()[3].relative_gain, -.95);
  EXPECT_GT(result.detectors()[3].negative_relative_fraction, .99);
  EXPECT_FALSE(result.formal_uncertainty_available);
  Case other;
  other.bind();
  EXPECT_THROW(RtcCommonModeConsideration::compare(other.learn(), lines),
               std::invalid_argument);
}

TEST(RtcCommonMode, NegativeRawGainWithMatchingSignedCalibrationIsNotAnomaly) {
  Case c;
  c.bind();
  c.domain.members[3].reference_eligible = true;
  c.domain.members[3].flxscale = -1.;
  const auto ev = c.learn();
  std::size_t checked = 0;
  for (const auto &f : ev->fits())
    if (f.detector == 3 && f.available()) {
      EXPECT_LT(f.gain, -.95);
      EXPECT_NEAR(f.calibrated_relative_gain, 1., .035);
      ++checked;
    }
  EXPECT_GE(checked, 2);
}
} // namespace
