#include "timestream_rtc_reassessment_test_support.h"
#include <citlali/core/pipeline/timestream_rtc_line_population.h>
#include <citlali/core/pipeline/timestream_rtc_notch_recovery.h>
#include <citlali/core/pipeline/timestream_rtc_pipeline.h>
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
  explicit Trial(const Input &in, double speed = 10,
                 RtcSpikeProtection protection = RtcSpikeProtection::unavailable)
      : Fixture(in, protection) {
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
            std::vector<double> finite = {}, std::uint32_t detector = 0,
            std::vector<std::shared_ptr<const RtcDonorFillPlan>> donors = {}) const {
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
    if (detector != 0)
      d.detector_array_association =
          parent->network(0).detector(detector).detector_association_record_id;
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
            RtcLineTransferCandidate::bind(lines, 0, detector, s), joint, val,
            22),
        transient, val, d, 23, std::move(donors));
  }
  auto apply(const std::shared_ptr<const RtcNotchRecoveryPlan> &p,
             const RtcRecoveryInjection *inj = nullptr) const {
    const std::array parts{spikes->input_handle()};
    return RtcNotchRecoveryResult::apply(p, spikes->input_handle(), val, parts,
                                         inj);
  }
};

auto population_members(const Trial &t) {
  std::vector<RtcLinePopulationMember> members;
  for (std::uint32_t d = 0; d < 3; ++d) {
    const auto &b = t.parent->network(0).detector(d);
    members.push_back(
        {0, d, b.detector_occurrence_id, b.detector_association_record_id,
         "controlled-science", RtcOpticalArray::a2000,
         d == 2 ? std::nullopt : std::optional<double>(d == 0 ? 1. : 4.),
         "controlled-prior-APT", "fixed-inverse-square-sensitivity"});
  }
  return members;
}

TEST(rtc_line_population,
     learn_preserves_overlapping_causes_and_exact_original_context) {
  Input in(6000);
  in.xs[3000 * 3] =
      NativeReadoutCoordinateState::measured(true, false, true, true);
  in.rs[3000 * 3] = in.xs[3000 * 3];
  for (std::size_t i = 4000; i < in.times.size(); ++i) {
    in.times[i] += .5;
    in.counters[i] += 30;
  }
  Trial t(in);
  auto p = RtcLinePopulationEvidence::learn(t.lines, t.transient,
                                            population_members(t), 99);
  ASSERT_EQ(p->detectors().size(), 3);
  EXPECT_EQ(p->line_handle().get(), t.lines.get());
  EXPECT_EQ(p->line_handle()->snapshot_handle().get(), t.val.get());
  EXPECT_FALSE(p->treatment_selected);
  const auto &d = p->detectors()[0];
  EXPECT_EQ(d.paired_original_cells, 5999);
  bool found = false;
  for (const auto &s : d.support) {
    if (s.rows.first <= 3100 && s.rows.past_last > 3100) {
      EXPECT_EQ(s.cause_bits & 3, 3);
      found = true;
    }
    EXPECT_FALSE(s.rows.first < 4100 && s.rows.past_last > 4100);
  }
  EXPECT_TRUE(found);
  EXPECT_FALSE(p->detectors()[2].member.reference_weight.has_value());
  EXPECT_DOUBLE_EQ(
      t.parent->network(0).value(NativeReadoutCoordinate::x, 3100, 0),
      in.x(3000, 0));
}

TEST(rtc_line_population,
     incomplete_duplicate_foreign_or_unbound_weight_inputs_fail) {
  Input in(6000);
  Trial t(in), other(in);
  auto m = population_members(t);
  auto incomplete = m;
  incomplete.pop_back();
  EXPECT_THROW(
      RtcLinePopulationEvidence::learn(t.lines, t.transient, incomplete, 1),
      std::invalid_argument);
  auto duplicate = m;
  duplicate[1] = duplicate[0];
  EXPECT_THROW(
      RtcLinePopulationEvidence::learn(t.lines, t.transient, duplicate, 1),
      std::invalid_argument);
  auto bad = m;
  bad[0].reference_weight = NAN;
  EXPECT_THROW(RtcLinePopulationEvidence::learn(t.lines, t.transient, bad, 1),
               std::invalid_argument);
  bad = m;
  bad[0].weight_authority.clear();
  EXPECT_THROW(RtcLinePopulationEvidence::learn(t.lines, t.transient, bad, 1),
               std::invalid_argument);
  EXPECT_THROW(RtcLinePopulationEvidence::learn(t.lines, other.transient, m, 1),
               std::invalid_argument);
}

TEST(rtc_line_population,
     consider_requires_complete_denominator_and_retains_missing_recovery) {
  Input in(6000);
  Trial t(in);
  auto p = RtcLinePopulationEvidence::learn(t.lines, t.transient,
                                            population_members(t), 1);
  std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> b;
  for (std::uint32_t d = 0; d < 3; ++d)
    b.push_back(t.plan(false, false, {}, d));
  auto recovery = t.plan(false, false, {.1, -.2, 1.2, -.2, .1});
  auto c = RtcLinePopulationComparison::consider(p, b, {recovery}, 2);
  ASSERT_EQ(c->contributions().size(), 3);
  const auto &v = c->contributions();
  EXPECT_EQ(v[0].baseline_cells, 5998);
  EXPECT_EQ(*v[0].recovery_cells, 5994);
  EXPECT_DOUBLE_EQ(*v[1].baseline_weight_seconds,
                   4 * *v[0].baseline_weight_seconds);
  EXPECT_FALSE(v[1].recovery_cells.has_value());
  EXPECT_FALSE(v[2].baseline_weight_seconds.has_value());
  EXPECT_FALSE(c->scientific_recovery_admitted);
  auto applied = t.apply(recovery);
  EXPECT_EQ(*v[0].recovery_cells,
            std::count(applied->causes().begin(), applied->causes().end(),
                       RtcNotchRecoveryCause::retained));
  b.pop_back();
  EXPECT_THROW(RtcLinePopulationComparison::consider(p, b, {recovery}, 3),
               std::invalid_argument);
}

TEST(rtc_line_population,
     rejection_zero_is_distinct_from_unavailable_and_iir_is_not_finite) {
  Input in(6000);
  Trial t(in);
  auto p = RtcLinePopulationEvidence::learn(t.lines, t.transient,
                                            population_members(t), 1);
  std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> b;
  for (std::uint32_t d = 0; d < 3; ++d)
    b.push_back(t.plan(false, false, {}, d));
  auto c =
      RtcLinePopulationComparison::consider(p, b, {t.plan(false, true)}, 2);
  ASSERT_TRUE(c->contributions()[0].recovery_cells.has_value());
  EXPECT_EQ(*c->contributions()[0].recovery_cells, 0);
  EXPECT_THROW(RtcLinePopulationComparison::consider(p, b, {t.plan(true)}, 3),
               std::invalid_argument);
  EXPECT_THROW(t.plan(true)->finite_retained_runs(), std::invalid_argument);
}

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

std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> complete_plans(
    const Trial &t, std::vector<double> notch = {},
    std::vector<std::shared_ptr<const RtcDonorFillPlan>> donors = {}) {
  return {t.plan(false,false,notch,0,donors), t.plan(false,false,notch,1),
          t.plan(false,false,notch,2)};
}
auto complete_apply(const Trial &t, std::shared_ptr<const RtcPipelinePlan> p) {
  const std::array parts{t.spikes->input_handle()};
  return RtcPipelineResult::apply(p,t.spikes->input_handle(),t.val,parts);
}
auto conditioned_learn(const Trial &t, const std::shared_ptr<const RtcPipelineResult> &r,
                       bool after_lowpass = false) {
  return RtcNativeSpectralEvidence::learn_conditioned(r->native_product(after_lowpass,t.val),
      t.val,{{0,"control-cadence",.008192,1e-7}},40);
}
auto explicit_donor(const Trial &t, bool available = true) {
  std::vector<RtcDonorDetectorFacts> facts;
  const auto &axis=t.parent->network(0).occurrence_axis();
  const RtcEventRange full{axis.first_native_row(),axis.past_last_native_row()};
  for(std::uint32_t d=0;d<3;++d) facts.push_back({0,d,
      t.parent->network(0).detector(d).detector_occurrence_id,"fixture-prior-APT","fixture-flxscale",
      d==0&&!available?0.:1.,full,{full},{}});
  for(std::size_t i=0;i<t.assessment->events().size();++i)
    if(t.assessment->events()[i].detector==0) return RtcDonorFillPlan::consider(
        {t.assessment,"fixture-explicit-isolated-event",RtcDonorSelectionState::accepted_isolated_event,i,{600,601}},
        RtcDonorFillFacts::bind(t.assessment,"fixture-prior-APT","fixture-flxscale",
            "fixture-explicit-stable-segments","fixture-explicit-contamination",facts),t.transient,t.val,30);
  throw std::runtime_error("fixture has no isolated event");
}
TEST(rtc_pipeline, complete_plan_rejects_missing_duplicate_foreign_or_unbound_inputs) {
  Input in; Trial t(in), other(in); auto plans=complete_plans(t);
  EXPECT_THROW(RtcPipelinePlan::consider({},t.joint->joint_handle(),31),std::invalid_argument);
  plans.pop_back(); EXPECT_THROW(RtcPipelinePlan::consider(plans,t.joint->joint_handle(),31),std::invalid_argument);
  plans=complete_plans(t);plans[1]=plans[0];
  EXPECT_THROW(RtcPipelinePlan::consider(plans,t.joint->joint_handle(),31),std::invalid_argument);
  plans=complete_plans(t);plans[1]=other.plan(false,false,{},1);
  EXPECT_THROW(RtcPipelinePlan::consider(plans,t.joint->joint_handle(),31),std::invalid_argument);
  auto p=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  const std::array parts{t.spikes->input_handle()};
  EXPECT_THROW(RtcPipelineResult::apply(p,other.spikes->input_handle(),t.val,parts),std::invalid_argument);
  EXPECT_THROW(RtcPipelineResult::apply(p,t.spikes->input_handle(),ValSnapshot::initial(t.parent),parts),std::invalid_argument);
  EXPECT_THROW(RtcPipelineResult::apply(p,t.spikes->input_handle(),t.val,{}),std::invalid_argument);
}
TEST(rtc_pipeline, conditioned_learning_measures_actual_stage_and_preserves_source_and_original) {
  Input in(6000); const double f=44./(488*.008192);
  for(Eigen::Index i=0;i<in.x.rows();++i)for(int d=0;d<3;++d){
    in.x(i,d)=std::sin(2*std::numbers::pi*f*i*.008192);
    in.r(i,d)=.2*std::cos(2*std::numbers::pi*f*i*.008192);
  }
  Trial t(in,10,RtcSpikeProtection::protected_source);
  auto p=RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31);
  auto r=complete_apply(t,p);auto post_notch=conditioned_learn(t,r);auto post_lpf=conditioned_learn(t,r,true);
  ASSERT_TRUE(post_notch->spectrum(0,0,NativeReadoutCoordinate::x).available());
  const double h=.5+.5*std::cos(2*std::numbers::pi*f*.008192);
  const auto &original=t.lines->spectral_handle()->spectrum(0,0,NativeReadoutCoordinate::x);
  EXPECT_NEAR(post_notch->spectrum(0,0,NativeReadoutCoordinate::x).psd[44]/original.psd[44],h*h,2e-4);
  EXPECT_NEAR(post_lpf->spectrum(0,0,NativeReadoutCoordinate::x).psd[44]/original.psd[44],h*h*h*h,2e-4);
  EXPECT_EQ(post_notch->conditioned_handle()->columns()[0].values.get(),&r->detector_results()[0]->conditioned_native_pair());
  EXPECT_EQ(post_lpf->conditioned_handle()->columns()[0].values.get(),&r->detector_results()[0]->filtered_native_pair());
  EXPECT_NE(post_notch->use_policy(),t.lines->spectral_handle()->use_policy());
  for(const auto &window:post_notch->spectrum(0,0,NativeReadoutCoordinate::x).windows)
    EXPECT_EQ(window.source_counts[1],static_cast<std::size_t>(window.rows.past_last-window.rows.first));
  for(Eigen::Index i=0;i<in.x.rows();++i)for(int d=0;d<3;++d){
    EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x,100+i,d),in.x(i,d));
    EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::r,100+i,d),in.r(i,d));
  }
}
TEST(rtc_pipeline, donor_exception_propagates_exact_nonzero_support_without_erasing_x_or_grid) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto donor=explicit_donor(t);ASSERT_EQ(donor->cause(),RtcDonorFillCause::ready);
  auto p=RtcPipelinePlan::consider(complete_plans(t,{0,1,0},{donor}),t.joint->joint_handle(),31);
  auto r=complete_apply(t,p);const auto &target=r->detector_results()[0];
  EXPECT_TRUE(target->representative_replaced(600));EXPECT_FALSE(target->representative_replaced(602));
  for(auto row:{598,599,600,601,602}){
    EXPECT_TRUE(target->coordinate_stage_available(NativeReadoutCoordinate::x,row,true));
    EXPECT_EQ(target->coordinate_stage_available(NativeReadoutCoordinate::r,row,true),row==598||row==602);
    EXPECT_EQ(target->replacement_influence(row,true),row>=599&&row<=601);
  }
  EXPECT_TRUE(std::binary_search(target->output_native_rows().begin(),target->output_native_rows().end(),600));
  EXPECT_TRUE(std::isfinite(target->filtered_native_pair()(500,0)));
  EXPECT_TRUE(std::isnan(target->filtered_native_pair()(500,1)));
  auto spectral=conditioned_learn(t,r,true);std::size_t touched=0,replaced=0;
  for(const auto &window:spectral->spectrum(0,0,NativeReadoutCoordinate::x).windows){
    touched+=window.replacement_influenced_samples;replaced+=window.representative_replacements;
  }
  EXPECT_GT(touched,0);EXPECT_GT(replaced,0);
  EXPECT_EQ(spectral->conditioned_handle()->columns()[0].source->donor_results()[0]->plan_handle().get(),donor.get());
  EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x,600,0),in.x(500,0));
  EXPECT_FALSE(RtcConditionedNativeProduct::independent_measurements);
}
TEST(rtc_pipeline, failed_donor_remains_unavailable_through_both_finite_stages) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);auto donor=explicit_donor(t,false);
  ASSERT_NE(donor->cause(),RtcDonorFillCause::ready);
  auto r=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25},{donor}),t.joint->joint_handle(),31));
  const auto &out=r->detector_results()[0];
  for(auto row:{598,599,600,601,602})for(auto c:{NativeReadoutCoordinate::x,NativeReadoutCoordinate::r})
    EXPECT_FALSE(out->coordinate_stage_available(c,row,true));
  EXPECT_TRUE(out->coordinate_stage_available(NativeReadoutCoordinate::x,597,true));
  EXPECT_FALSE(out->representative_replaced(600));
  EXPECT_TRUE(out->requires_representative_exclusion(600));
  EXPECT_TRUE(out->unrepaired_influence(598,true));
  EXPECT_FALSE(out->replacement_influence(598,true));
  EXPECT_TRUE(std::binary_search(out->output_native_rows().begin(),out->output_native_rows().end(),600));
}
TEST(rtc_pipeline, relearning_then_reconsidering_replays_original_instead_of_accumulating_filters) {
  Input in(6000);Trial t(in);
  auto first=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31));
  auto spectral=conditioned_learn(t,first,true);
  auto joint=RtcSpectralTransientConsideration::consider(spectral,t.val,t.review,t.val,41);
  auto evidence=RtcPipelineReassessment::consider(first,joint,42);
  auto next=RtcPipelinePlan::reconsider(evidence,complete_plans(t),43);
  auto second=complete_apply(t,next);auto direct=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),44));
  for(std::size_t d=0;d<3;++d){
    const auto &a=second->detector_results()[d];const auto &b=direct->detector_results()[d];
    EXPECT_EQ(a->output_native_rows(),b->output_native_rows());EXPECT_EQ(a->causes(),b->causes());
    for(auto row:a->output_native_rows())for(int c=0;c<2;++c)
      EXPECT_DOUBLE_EQ(a->filtered_native_pair()(row-100,c),b->filtered_native_pair()(row-100,c));
  }
  EXPECT_EQ(next->reassessment_handle()->previous_handle().get(),first.get());
  auto other_realization=complete_apply(t,first->plan_handle());
  EXPECT_THROW(RtcPipelineReassessment::consider(other_realization,joint,50),std::invalid_argument);
  EXPECT_THROW(RtcPipelinePlan::reconsider(evidence,complete_plans(t),31),std::invalid_argument);
  EXPECT_THROW(RtcPipelineReassessment::consider(first,t.joint->joint_handle(),50),std::invalid_argument);
  EXPECT_FALSE(RtcPipelineReassessment::classification_authorized);
  EXPECT_FALSE(RtcPipelineReassessment::stopping_rule_selected);
}
TEST(rtc_pipeline, later_snapshot_is_a_new_exact_evidence_binding_and_cannot_rebind_old_evidence) {
  Input in;Trial t(in);auto out=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31));
  auto product=out->native_product(false,t.val);
  ValDeltaBuilder builder{t.val,{ValProducer::rtc,44}};
  builder.propose(t.val->address(0,100,0),ValFactCode{1},ValFactState{1},ValFactCause{1});
  auto later=ValSnapshot::commit(builder.freeze());auto later_product=out->native_product(false,later);
  EXPECT_NE(product->identities()[0].get(),later_product->identities()[0].get());
  EXPECT_EQ(product->snapshot_handle()->generation().value,0);
  EXPECT_EQ(later_product->snapshot_handle()->generation().value,1);
  EXPECT_THROW(RtcNativeSpectralEvidence::learn_conditioned(product,later,{{0,"fixture",.008192,1e-7}},50),std::invalid_argument);
  EXPECT_THROW(out->native_product(false,ValSnapshot::initial(t.parent)),std::invalid_argument);
  auto learned=RtcNativeSpectralEvidence::learn_conditioned(later_product,later,{{0,"fixture",.008192,1e-7}},51);
  auto considered=RtcSpectralTransientConsideration::consider(learned,later,t.review,t.val,52);
  auto later_review=RtcPipelineReassessment::consider(out,considered,53);
  EXPECT_THROW(RtcPipelinePlan::reconsider(later_review,complete_plans(t),54),StaleRtcValGeneration);
}
TEST(rtc_pipeline, declared_gaps_do_not_join_native_time_or_manufacture_spectral_support) {
  Input in(1600);
  for(std::size_t i=700;i<900;++i){in.xs[i*3]=NativeReadoutCoordinateState::measured(true,false,true,true);in.rs[i*3]=in.xs[i*3];}
  Trial t(in);auto out=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31));
  auto learned=conditioned_learn(t,out,true);
  for(const auto &w:learned->spectrum(0,0,NativeReadoutCoordinate::x).windows)
    EXPECT_TRUE(w.rows.past_last<=800||w.rows.first>=1000);
  EXPECT_EQ(out->native_product(true,t.val)->columns()[0].values->rows(),1600);
  Input short_in(300);Trial short_t(short_in);
  auto short_out=complete_apply(short_t,RtcPipelinePlan::consider(complete_plans(short_t),short_t.joint->joint_handle(),31));
  EXPECT_FALSE(conditioned_learn(short_t,short_out)->spectrum(0,0,NativeReadoutCoordinate::x).available());
}

TEST(rtc_pipeline, unexpected_nonfinite_conditioned_payload_is_a_run_failure_not_silent_sample_removal) {
  for(double fault:{NAN,INFINITY}) {
    Input in(1600);Trial t(in);
    auto result=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31));
    auto product=result->native_product(false,t.val);
    // Test-only storage corruption after the producer established availability.
    // Production exposes no mutation API; always restore the actual allocation.
    struct Fault {
      double &cell;double saved;
      Fault(const RtcConditionedNativeProduct::Column &c,double v)
          :cell(const_cast<Eigen::Matrix<double,Eigen::Dynamic,2>&>(*c.values)(500,0)),saved(cell){cell=v;}
      ~Fault(){cell=saved;}
    } injected{product->columns()[0],fault};
    auto learned=RtcNativeSpectralEvidence::learn_conditioned(product,t.val,{{0,"fixture",.008192,1e-7}},40);
    const auto &x=learned->spectrum(0,0,NativeReadoutCoordinate::x);
    EXPECT_FALSE(x.available());EXPECT_EQ(x.cause,RtcSpectralCause::input_consistency_failure);
    EXPECT_EQ(x.runs[0].unexpected_nonfinite_samples,1);EXPECT_TRUE(x.psd.empty());
    EXPECT_TRUE(learned->spectrum(0,0,NativeReadoutCoordinate::r).available());
    EXPECT_TRUE(learned->spectrum(0,1,NativeReadoutCoordinate::x).available());
  }
}
TEST(rtc_pipeline, frozen_complete_plan_source_pair_has_identical_support_and_diagonal_response) {
  Input in(1600);Trial t(in);
  auto p=RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31);
  auto plain=complete_apply(t,p);std::vector<RtcRecoveryInjection> injections;
  for(const auto &detector:p->detector_plans()) {
    RtcRecoveryInjection inj;inj.plan=detector;inj.identity="fixture-paired-gaussian";
    inj.delta.resize(1600,2);
    for(int i=0;i<1600;++i){const double u=(i-700)/15.;inj.delta(i,0)=2*std::exp(-u*u/2);inj.delta(i,1)=0;}
    injections.push_back(std::move(inj));
  }
  const std::array parts{t.spikes->input_handle()};
  auto injected=RtcPipelineResult::apply(p,t.spikes->input_handle(),t.val,parts,injections);
  const std::array<double,5> kernel{1./16,4./16,6./16,4./16,1./16};
  for(int d=0;d<3;++d){
    const auto &a=plain->detector_results()[d];const auto &b=injected->detector_results()[d];
    EXPECT_EQ(a->causes(),b->causes());EXPECT_EQ(a->output_native_rows(),b->output_native_rows());
    for(auto row:a->output_native_rows()){
      const auto local=row-100;double expected=0;
      for(int j=0;j<5;++j)expected+=kernel[j]*injections[d].delta(local+j-2,0);
      EXPECT_NEAR(b->filtered_native_pair()(local,0)-a->filtered_native_pair()(local,0),expected,5e-14);
      EXPECT_DOUBLE_EQ(b->filtered_native_pair()(local,1),a->filtered_native_pair()(local,1));
    }
  }
  injections[0].plan=p->detector_plans()[1];
  EXPECT_THROW(RtcPipelineResult::apply(p,t.spikes->input_handle(),t.val,parts,injections),std::invalid_argument);
}

TEST(rtc_pipeline, donor_replay_is_partition_invariant_and_retains_exact_stage_owners) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto p=RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25},{explicit_donor(t)}),t.joint->joint_handle(),31);
  auto a=complete_apply(t,p);
  std::vector<std::shared_ptr<const NativePairedReadoutView>> parts{
      NativePairedReadoutView::admit(t.parent,{{0,100,600}}),
      NativePairedReadoutView::admit(t.parent,{{0,600,601}}),
      NativePairedReadoutView::admit(t.parent,{{0,601,1200}})};
  auto b=RtcPipelineResult::apply(p,t.spikes->input_handle(),t.val,parts);
  const auto x=a->native_product(true,t.val),y=b->native_product(true,t.val);
  for(std::size_t d=0;d<3;++d){
    EXPECT_EQ(x->columns()[d].state,y->columns()[d].state);
    EXPECT_EQ(a->detector_results()[d]->output_native_rows(),b->detector_results()[d]->output_native_rows());
    for(std::size_t i=0;i<x->columns()[d].state.size();++i)for(int c=0;c<2;++c)
      if(x->columns()[d].state[i]&(1U<<c))EXPECT_DOUBLE_EQ((*x->columns()[d].values)(i,c),(*y->columns()[d].values)(i,c));
  }
  auto sx=conditioned_learn(t,a,true),sy=conditioned_learn(t,b,true);
  for(std::size_t i=0;i<sx->spectra().size();++i)EXPECT_EQ(sx->spectra()[i].psd,sy->spectra()[i].psd);
  std::reverse(parts.begin(),parts.end());
  EXPECT_THROW(RtcPipelineResult::apply(p,t.spikes->input_handle(),t.val,parts),std::invalid_argument);
}

TEST(rtc_pipeline, finite_support_limit_includes_donor_background_not_only_filter_taps) {
  Input in(6000);in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto donor=explicit_donor(t);ASSERT_EQ(donor->cause(),RtcDonorFillCause::ready);
  std::vector<double> wide(1001,1./1001);
  auto plain=t.plan(false,false,wide);EXPECT_TRUE(plain->finite_five_second_footprint());
  auto with_donor=t.plan(false,false,wide,0,{donor});EXPECT_FALSE(with_donor->finite_five_second_footprint());
  EXPECT_THROW(RtcPipelinePlan::consider({with_donor,t.plan(false,false,{},1),t.plan(false,false,{},2)},
      t.joint->joint_handle(),31),std::invalid_argument);
}
} // namespace
