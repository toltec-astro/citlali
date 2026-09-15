#include "timestream_rtc_reassessment_test_support.h"
#include <citlali/core/pipeline/timestream_rtc_notch_recovery.h>
#include <gtest/gtest.h>

namespace {
using namespace citlali::pipeline;
using citlali::test::rtc_reassessment::Fixture;
using citlali::test::rtc_reassessment::Input;
struct Trial : Fixture {
  std::shared_ptr<const RtcTransientExclusionPlan> transient;
  std::shared_ptr<const RtcLinePowerEvidence> lines;
  std::shared_ptr<const RtcLinePowerConsideration> joint;
  RtcNotchRecoveryDomain domain;
  explicit Trial(const Input &in, double speed = 10)
      : Fixture(in, RtcSpikeProtection::unavailable) {
    auto support = RtcJumpSupportEvidence::learn(transition, 9);
    auto refit = RtcJumpRefitEvidence::learn(
        RtcJumpRefitRequest::consider(support, val, 10), 11);
    auto measured = RtcJumpReassessmentEvidence::learn(
        RtcJumpRemeasureRequest::consider(refit, val, 12), 13);
    auto admission = RtcJumpAdmissionDecision::consider(
        RtcJumpReassessmentDecision::consider(measured, val, 14), val, 15);
    auto jumps = RtcJumpExclusionPlan::consider(admission, nullptr, val, 16);
    transient = RtcTransientExclusionPlan::consider(
        review->original_screening_handle(), jumps, val, 17);
    auto native =
        ValNativeRealization::create(parent, {ValProducer::align, 1}, 1,
                                     ValNativeProductRole::original_input, 0);
    auto identity = RtcSpectralInputIdentity::bind(
        native, val, spikes->input_handle()->span(0),
        RtcSpectralInputStage::original_reference, "finite-control", 1);
    auto spectral = RtcNativeSpectralEvidence::learn_initial(
        spikes, {identity}, {{0, "control-cadence", .008192, 1e-7}}, 18);
    lines = RtcLinePowerEvidence::learn(spectral, val,
                                        RtcLinePowerProfile::initial_2_hz, 19);
    joint = RtcLinePowerConsideration::rank(
        lines,
        RtcSpectralTransientConsideration::consider(spectral, val, review, val,
                                                    20),
        21);
    auto times = Eigen::VectorXd::LinSpaced(5000, 999, 1098.98);
    Eigen::VectorXd ra = (times.array() - 999) * speed * std::numbers::pi /
                         (180 * 3600),
                    dec = Eigen::VectorXd::Zero(times.size());
    AstScanMotionSourceMetadata metadata{
        AstScanMotionProducerKind::real_toltec,
        "Science",
        "Lissajous",
        1,
        2000,
        0,
        50,
        AstScanMotionFieldRegistry::source_ra_act_source_dec_act_j2000_radians,
        "controlled-constant-motion"};
    auto source = AstScanMotionSource::admit(parent->scope(), parent->scope(),
                                             0, metadata, times, ra, dec);
    domain.motion = AstScanMotionNetworkView::admit(
        build_ast_scan_motion_product(source, {1, 2, 3, 4}),
        parent->network(0).occurrence_axis().native_timing_handle());
    domain.identity = "controlled-domain";
    domain.detector_array_association =
        parent->network(0).detector(0).detector_association_record_id;
    domain.speed_ceiling_arcsec_per_sec = 20;
    domain.nominal_interval_seconds = .008192;
  }
  auto plan(bool notch = false, bool reject = false,
            std::vector<double> finite = {}) const {
    RtcLineTransferSpecification s;
    s.identity = "controlled-plan";
    s.lowpass_identity = "test-only-centered3";
    s.state_support_identity = "mature-kernel";
    s.input_interval_seconds =
        lines->spectral_handle()->network(0).interval_seconds;
    s.factor = 2;
    s.centered_lowpass = {.25, .5, .25};
    if (!finite.empty()) {
      s.finite_notch_identity = "finite-control";
      s.centered_notch = std::move(finite);
    }
    auto d = domain;
    d.reject = reject;
    if (notch) {
      timestream::Filter f;
      f.w0s = {11};
      f.qs = {22};
      f.make_notch_filter(1 / s.input_interval_seconds);
      RtcNotchResponseSection n;
      n.identity = "11Hz";
      for (int i = 0; i < 3; ++i) {
        n.a[i] = f.notch_a[0][i];
        n.b[i] = f.notch_b[0][i];
      }
      s.notches = {n};
      d.notch_guard_samples = 1100;
    }
    return RtcNotchRecoveryPlan::consider(
        RtcLineTransferAssessment::consider(
            RtcLineTransferCandidate::bind(lines, 0, 0, s), joint, val, 22),
        transient, val, d, 23);
  }
  auto apply(const std::shared_ptr<const RtcNotchRecoveryPlan> &p,
             const RtcRecoveryInjection *inj = nullptr) const {
    const std::array parts{spikes->input_handle()};
    return RtcNotchRecoveryResult::apply(p, spikes->input_handle(), val, parts,
                                         inj);
  }
};

TEST(rtc_notch_recovery, centered_fir_exact_native_phase_and_original_replay) {
  Input in(6000);
  Trial t(in);
  auto p = t.plan();
  auto a = t.apply(p), b = t.apply(p);
  ASSERT_FALSE(a->output_native_rows().empty());
  for (auto row : a->output_native_rows()) {
    EXPECT_EQ((row - 100) % 2, 0);
    const auto i = row - 100;
    EXPECT_NEAR(a->filtered_native_pair()(i, 0),
                .25 * in.x(i - 1, 0) + .5 * in.x(i, 0) + .25 * in.x(i + 1, 0),
                1e-12);
    EXPECT_DOUBLE_EQ(a->filtered_native_pair()(i, 0),
                     b->filtered_native_pair()(i, 0));
    EXPECT_DOUBLE_EQ(
        t.parent->network(0).value(NativeReadoutCoordinate::x, row, 0),
        in.x(i, 0));
  }
}
TEST(rtc_notch_recovery,
     physical_gap_and_declared_invalid_pair_split_support_without_compression) {
  Input in(6000);
  in.xs[3000 * 3] =
      NativeReadoutCoordinateState::measured(true, false, true, true);
  for (std::size_t i = 4000; i < in.times.size(); ++i) {
    in.times[i] += .5;
    in.counters[i] += 30;
  }
  Trial t(in);
  auto a = t.apply(t.plan());
  EXPECT_EQ(a->causes()[3000], RtcNotchRecoveryCause::producer_invalid);
  EXPECT_EQ(a->causes()[2999], RtcNotchRecoveryCause::boundary_guard);
  EXPECT_EQ(a->causes()[4000], RtcNotchRecoveryCause::boundary_guard);
  EXPECT_EQ(a->causes()[3999], RtcNotchRecoveryCause::boundary_guard);
  EXPECT_GT(a->output_native_rows().back(), 5000);
}
TEST(rtc_notch_recovery, paired_injection_same_plan_linear_response_and_flags) {
  Input in(6000);
  Trial t(in);
  auto p = t.plan(true);
  auto a = t.apply(p);
  RtcRecoveryInjection inj{
      p, "paired-controlled-source",
      Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(6000, 2)};
  for (int i = 0; i < 6000; ++i) {
    double z = (i - 3000) / 7.;
    inj.delta(i, 0) = std::exp(-z * z / 2);
    inj.delta(i, 1) = 2 * inj.delta(i, 0);
  }
  auto b = t.apply(p, &inj);
  EXPECT_EQ(a->causes(), b->causes());
  EXPECT_EQ(a->output_native_rows(), b->output_native_rows());
  const auto once = inj.delta;
  inj.delta *= 3;
  auto c = t.apply(p, &inj);
  for (auto row : a->output_native_rows())
    for (int k = 0; k < 2; ++k) {
      auto i = row - 100;
      EXPECT_NEAR(c->filtered_native_pair()(i, k) -
                      a->filtered_native_pair()(i, k),
                  3 * (b->filtered_native_pair()(i, k) -
                       a->filtered_native_pair()(i, k)),
                  1e-9);
    }
  EXPECT_FALSE(b->map_input_authorized);
  EXPECT_FALSE(b->independent_measurements);
}
TEST(rtc_notch_recovery, rejection_retains_original_and_gives_no_outputs) {
  Input in(6000);
  Trial t(in);
  auto a = t.apply(t.plan(false, true));
  EXPECT_TRUE(a->output_native_rows().empty());
  EXPECT_EQ(a->causes()[3000], RtcNotchRecoveryCause::rejection);
  EXPECT_DOUBLE_EQ(
      t.parent->network(0).value(NativeReadoutCoordinate::x, 3100, 0),
      in.x(3000, 0));
}
TEST(rtc_notch_recovery, stale_parent_val_and_injection_fail_closed) {
  Input in(6000);
  Trial t(in);
  auto p = t.plan();
  const std::array parts{t.spikes->input_handle()};
  EXPECT_THROW(RtcNotchRecoveryResult::apply(
                   p, NativePairedReadoutView::full(t.parent), t.val, parts),
               std::invalid_argument);
  EXPECT_THROW(RtcNotchRecoveryResult::apply(p, t.spikes->input_handle(),
                                             ValSnapshot::initial(t.parent),
                                             parts),
               StaleRtcValGeneration);
  RtcRecoveryInjection inj{
      t.plan(), "wrong-plan",
      Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(6000, 2)};
  EXPECT_THROW(t.apply(p, &inj), std::invalid_argument);
}
TEST(
    rtc_notch_recovery,
    unexpected_nonfinite_rejected_at_ingress_and_declared_invalid_splits_support) {
  Input in(6000);
  in.x(3000, 0) = NAN;
  EXPECT_THROW({ Trial unexpected(in); }, std::invalid_argument);
  in.xs[3000 * 3] =
      NativeReadoutCoordinateState::measured(true, false, true, false);
  Trial declared(in);
  auto a = declared.apply(declared.plan());
  EXPECT_EQ(a->causes()[3000], RtcNotchRecoveryCause::producer_invalid);
}
TEST(
    rtc_notch_recovery,
    finite_sine_transfer_matches_explicit_notch_and_preserves_coordinate_pairing) {
  Input in(6000);
  Trial t(in);
  auto p = t.plan(true);
  auto a = t.apply(p);
  RtcRecoveryInjection inj{
      p, "known-two-frequency-response",
      Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(6000, 2)};
  const auto dt = p->assessment_handle()
                      ->candidate_handle()
                      ->specification()
                      .input_interval_seconds;
  for (int i = 0; i < 6000; ++i) {
    inj.delta(i, 0) = std::sin(2 * std::numbers::pi * 11 * dt * i);
    inj.delta(i, 1) = std::sin(2 * std::numbers::pi * 3 * dt * i);
  }
  auto b = t.apply(p, &inj);
  double line_max = 0, pass_max = 0;
  double gain = std::real(
      p->assessment_handle()->candidate_handle()->combined_response(3));
  for (auto row : a->output_native_rows()) {
    auto i = row - 100;
    line_max = std::max(line_max, std::abs(b->filtered_native_pair()(i, 0) -
                                           a->filtered_native_pair()(i, 0)));
    pass_max = std::max(pass_max, std::abs(b->filtered_native_pair()(i, 1) -
                                           a->filtered_native_pair()(i, 1) -
                                           gain * inj.delta(i, 1)));
  }
  EXPECT_LT(line_max, 1e-5);
  EXPECT_LT(pass_max, 1e-5);
}
TEST(rtc_notch_recovery, engineering_partition_does_not_reset_filter_state) {
  Input in(6000);
  Trial t(in);
  auto p = t.plan(true);
  auto a = t.apply(p);
  std::vector<std::shared_ptr<const NativePairedReadoutView>> parts{
      NativePairedReadoutView::admit(t.parent, {{0, 100, 3001}}),
      NativePairedReadoutView::admit(t.parent, {{0, 3001, 6100}})};
  auto b =
      RtcNotchRecoveryResult::apply(p, t.spikes->input_handle(), t.val, parts);
  EXPECT_EQ(a->causes(), b->causes());
  EXPECT_EQ(a->output_native_rows(), b->output_native_rows());
  for (auto row : a->output_native_rows())
    EXPECT_EQ(a->filtered_native_pair().row(row - 100),
              b->filtered_native_pair().row(row - 100));
  parts.pop_back();
  EXPECT_THROW(
      RtcNotchRecoveryResult::apply(p, t.spikes->input_handle(), t.val, parts),
      std::invalid_argument);
}
TEST(rtc_notch_recovery,
     mismatched_motion_and_unsupported_optical_domain_fail_closed) {
  Input in(6000);
  Trial t(in);
  t.domain.speed_ceiling_arcsec_per_sec = 1000;
  EXPECT_THROW(t.plan(), std::invalid_argument);
  t.domain.speed_ceiling_arcsec_per_sec = 20;
  t.domain.detector_array_association = "different-array-relation";
  EXPECT_THROW(t.plan(), std::invalid_argument);
}

TEST(rtc_notch_recovery,
     short_finite_record_retains_unavailable_boundary_disposition) {
  Input in(1000);
  Trial t(in);
  auto a = t.apply(t.plan(true));
  EXPECT_TRUE(a->output_native_rows().empty());
  EXPECT_EQ(a->causes()[500], RtcNotchRecoveryCause::boundary_guard);
}

TEST(rtc_notch_recovery,
     inclusive_minimum_and_sampling_ceiling_use_existing_authority) {
  EXPECT_FALSE(ast_scan_motion_speed_admitted(std::nextafter(1., 0.)));
  EXPECT_TRUE(ast_scan_motion_speed_admitted(1.));
  Input in(6000);
  Trial slow(in, .5);
  auto low = slow.apply(slow.plan());
  EXPECT_TRUE(low->output_native_rows().empty());
  EXPECT_EQ(low->causes()[3000], RtcNotchRecoveryCause::below_minimum_speed);
  Trial fast(in, 150);
  fast.domain.speed_ceiling_arcsec_per_sec = 235;
  auto plan = fast.plan();
  auto high = fast.apply(plan);
  EXPECT_NEAR(plan->sampling_speed_limit_arcsec_per_sec(), 123.277762, 1e-5);
  EXPECT_TRUE(high->output_native_rows().empty());
  EXPECT_EQ(high->causes()[3000],
            RtcNotchRecoveryCause::insufficient_output_sampling);
  Trial ordinary(in);
  EXPECT_TRUE(ordinary.plan()->finite_five_second_footprint());
  EXPECT_FALSE(ordinary.plan(true)->finite_five_second_footprint());
}

TEST(rtc_notch_recovery, finite_two_stage_impulse_has_exact_support_and_phase) {
  Input in(6000);
  Trial t(in);
  auto p = t.plan(false, false, {.25, .5, .25});
  auto a = t.apply(p);
  RtcRecoveryInjection inj{
      p, "finite-impulse",
      Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(6000, 2)};
  inj.delta(3000, 0) = 1;
  inj.delta(3000, 1) = 2;
  auto b = t.apply(p, &inj);
  const std::array expected{.0625, .25, .375, .25, .0625};
  for (int i = 2990; i <= 3010; ++i)
    for (int c = 0; c < 2; ++c) {
      const auto wanted =
          i >= 2998 && i <= 3002 ? expected[i - 2998] * (c + 1) : 0;
      EXPECT_NEAR(b->filtered_native_pair()(i, c) -
                      a->filtered_native_pair()(i, c),
                  wanted, 1e-12);
    }
  EXPECT_TRUE(p->finite_five_second_footprint());
  EXPECT_EQ(a->causes(), b->causes());
  EXPECT_EQ(a->output_native_rows(), b->output_native_rows());
}
TEST(rtc_notch_recovery, finite_notch_dc_and_tone_follow_frozen_response) {
  Input in(6000);
  Trial t(in);
  const double dt = t.lines->spectral_handle()->network(0).interval_seconds;
  const double outer = 1 / (2 * (1 - std::cos(2 * std::numbers::pi * 11 * dt)));
  auto p = t.plan(false, false, {outer, 1 - 2 * outer, outer});
  auto a = t.apply(p);
  RtcRecoveryInjection inj{
      p, "finite-dc-and-line",
      Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(6000, 2)};
  for (int i = 0; i < 6000; ++i) {
    inj.delta(i, 0) = std::sin(2 * std::numbers::pi * 11 * dt * i);
    inj.delta(i, 1) = 2;
  }
  auto b = t.apply(p, &inj);
  EXPECT_NEAR(
      std::abs(
          p->assessment_handle()->candidate_handle()->combined_response(11)),
      0, 1e-12);
  for (auto row : a->output_native_rows()) {
    auto i = row - 100;
    EXPECT_NEAR(b->filtered_native_pair()(i, 0) -
                    a->filtered_native_pair()(i, 0),
                0, 1e-10);
    EXPECT_NEAR(b->filtered_native_pair()(i, 1) -
                    a->filtered_native_pair()(i, 1),
                2, 1e-12);
  }
}
TEST(rtc_notch_recovery,
     finite_chain_never_bridges_exclusions_or_uses_padding) {
  Input in(6000);
  in.xs[3000 * 3] =
      NativeReadoutCoordinateState::measured(true, false, true, true);
  Trial t(in);
  auto a = t.apply(t.plan(false, false, {.25, .5, .25}));
  EXPECT_EQ(a->causes()[3000], RtcNotchRecoveryCause::producer_invalid);
  for (int i : {2998, 2999, 3001, 3002})
    EXPECT_EQ(a->causes()[i], RtcNotchRecoveryCause::boundary_guard);
  EXPECT_TRUE(std::isnan(a->conditioned_native_pair()(2999, 0)));
  EXPECT_TRUE(std::isfinite(a->conditioned_native_pair()(2998, 0)));
  EXPECT_TRUE(std::isfinite(a->filtered_native_pair()(2997, 0)));
  EXPECT_TRUE(std::isfinite(a->filtered_native_pair()(3003, 0)));
}
TEST(rtc_notch_recovery,
     finite_trials_reject_mixed_invalid_or_overlong_operators) {
  Input in(6000);
  Trial t(in);
  EXPECT_THROW(t.plan(true, false, {.25, .5, .25}), std::invalid_argument);
  EXPECT_THROW(t.plan(false, false, {.5, .5}), std::invalid_argument);
  EXPECT_THROW(t.plan(false, false, {.25, .5, .2}), std::invalid_argument);
  EXPECT_THROW(t.plan(false, false, {.25, NAN, .25}), std::invalid_argument);
  EXPECT_THROW(t.plan(false, false, {.25, 1., .25}), std::invalid_argument);
  std::vector<double> long_filter(1301, 0);
  long_filter[650] = 1;
  EXPECT_THROW(t.plan(false, false, long_filter), std::invalid_argument);
}
} // namespace
