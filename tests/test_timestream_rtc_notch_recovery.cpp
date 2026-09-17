#include "timestream_rtc_reassessment_test_support.h"
#include <citlali/core/pipeline/timestream_rtc_line_population.h>
#include <citlali/core/pipeline/timestream_rtc_notch_recovery.h>
#include <citlali/core/pipeline/timestream_rtc_pipeline.h>
#include <citlali/core/pipeline/timestream_rtc_output_grid.h>
#include "../tools/timestream_successor/rtc_multidetector_bindings.h"
#include <gtest/gtest.h>
#include <citlali/core/pipeline/timestream_processing_scan_native.h>
#include <citlali/core/pipeline/timestream_rtc_common_mode.h>

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
      d.speed_support = RtcSpeedSupportTreatment::comparison_reject_both;
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
  EXPECT_FALSE(low->output_native_rows().empty());
  EXPECT_FALSE(low->map_center_admitted(3100));
  EXPECT_EQ(low->plan_handle()->input_causes()[3000], RtcNotchRecoveryCause::below_minimum_speed);
  EXPECT_EQ(low->causes()[3000], RtcNotchRecoveryCause::retained);
  Trial fast(in, 150);
  fast.domain.speed_ceiling_arcsec_per_sec = 235;
  auto plan = fast.plan();
  auto high = fast.apply(plan);
  EXPECT_NEAR(plan->sampling_speed_limit_arcsec_per_sec(), 123.277762, 1e-5);
  EXPECT_FALSE(high->output_native_rows().empty());
  EXPECT_FALSE(high->map_center_admitted(3100));
  EXPECT_EQ(plan->input_causes()[3000],
            RtcNotchRecoveryCause::insufficient_output_sampling);
  fast.domain.speed_support=RtcSpeedSupportTreatment::comparison_reject_both;
  EXPECT_TRUE(fast.apply(fast.plan())->output_native_rows().empty());
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
auto output_align(const Trial &t, std::shared_ptr<const ValSnapshot> snapshot = {},
                  std::shared_ptr<const AstScanMotionProduct> motion = {}) {
  auto views = AstScanMotionNetworkViews::admit(t.parent->scope(),
      motion ? motion : t.domain.motion->raw_product_handle(),
      {t.parent->network(0).occurrence_axis().native_timing_handle()});
  return IdentityRouteAlignContext::admit(t.parent, views, snapshot ? snapshot : t.val);
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
TEST(rtc_pipeline, donor_background_cannot_bypass_a_resolved_pair_invalid_boundary) {
  Input in;in.spike();
  // The original x background remains fit-eligible, while paired processing
  // must exclude this occurrence because r is invalid. The old composition
  // let changes in x349 alter the retained reconstructed x600 through the fit.
  in.rs[249*3]=NativeReadoutCoordinateState::measured(true,false,true,true);
  Trial t(in,10,RtcSpikeProtection::outside_source);
  auto donor=explicit_donor(t);ASSERT_EQ(donor->cause(),RtcDonorFillCause::ready);
  auto plain=t.plan();EXPECT_EQ(plain->input_causes()[249],RtcNotchRecoveryCause::producer_invalid);
  EXPECT_THROW(t.plan(false,false,{},0,{donor}),std::invalid_argument);
}
TEST(rtc_pipeline, complete_plan_cannot_reuse_a_rejected_donors_original_median) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto donor=explicit_donor(t);ASSERT_EQ(donor->cause(),RtcDonorFillCause::ready);
  auto plans=complete_plans(t,{}, {donor});plans[1]=t.plan(false,true,{},1);
  EXPECT_THROW(RtcPipelinePlan::consider(plans,t.joint->joint_handle(),31),std::invalid_argument);
  plans=complete_plans(t,{}, {donor});
  auto restricted=plans[1]->domain();restricted.speed_ceiling_arcsec_per_sec=5;
  plans[1]=RtcNotchRecoveryPlan::consider(plans[1]->assessment_handle(),t.transient,t.val,restricted,32);
  EXPECT_EQ(plans[1]->input_causes()[500],RtcNotchRecoveryCause::motion_outside_domain);
  EXPECT_THROW(RtcPipelinePlan::consider(plans,t.joint->joint_handle(),33),std::invalid_argument);
}
TEST(rtc_pipeline, donor_background_cannot_bridge_a_new_ast_speed_boundary) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto donor=explicit_donor(t);ASSERT_EQ(donor->cause(),RtcDonorFillCause::ready);
  auto times=Eigen::VectorXd::LinSpaced(5000,999,1098.98);
  Eigen::VectorXd ra(times.size()),dec=Eigen::VectorXd::Zero(times.size());
  for(Eigen::Index i=0;i<times.size();++i){
    const double elapsed=times[i]-999;
    ra[i]=(.5*std::min(elapsed,3.5)+10*std::max(0.,elapsed-3.5))*std::numbers::pi/(180*3600);
  }
  AstScanMotionSourceMetadata metadata{AstScanMotionProducerKind::real_toltec,
      "Science","Lissajous",1,2000,0,50,
      AstScanMotionFieldRegistry::source_ra_act_source_dec_act_j2000_radians,"fixture-speed-boundary"};
  auto source=AstScanMotionSource::admit(t.parent->scope(),t.parent->scope(),0,metadata,times,ra,dec);
  t.domain.motion=AstScanMotionNetworkView::admit(build_ast_scan_motion_product(source,{1,2,3,4}),
      t.parent->network(0).occurrence_axis().native_timing_handle());
  auto plain=t.plan();
  EXPECT_EQ(plain->input_causes()[249],RtcNotchRecoveryCause::below_minimum_speed);
  EXPECT_EQ(plain->input_causes()[500],RtcNotchRecoveryCause::retained);
  EXPECT_THROW(t.plan(false,false,{},0,{donor}),std::invalid_argument);
}
TEST(rtc_pipeline, unadmitted_selected_support_is_not_a_realized_replacement) {
  Input in;in.spike();Trial t(in,.5,RtcSpikeProtection::outside_source);
  t.domain.speed_support=RtcSpeedSupportTreatment::comparison_reject_both;
  auto donor=explicit_donor(t);ASSERT_EQ(donor->cause(),RtcDonorFillCause::ready);
  // Standalone donor selection remains evidence; the numerical recovery did
  // not use it because the entire target run fails the AST speed admission.
  auto result=t.apply(t.plan(false,false,{},0,{donor}));
  EXPECT_TRUE(result->donor_results()[0]->filled());
  EXPECT_FALSE(result->representative_replaced(600));
  EXPECT_TRUE(result->requires_representative_exclusion(600));
  EXPECT_FALSE(result->replacement_influence(600,false));
  EXPECT_TRUE(std::isnan(result->conditioned_native_pair()(500,0)));
}

namespace caller = citlali::rtc_multidetector_tool;
YAML::Node explicit_review(const Trial &t) {
  YAML::Node n;
  n["schema"]="rtc-reviewed-selection-v1";n["learning_binding"]="exact-controlled-Learn";
  n["VAL_generation"]=0;n["approved"]=true;n["authority"]="controlled-injected-isolated-spike";
  n["stable_support_authority"]="controlled-injection-truth";
  n["contamination_authority"]="controlled-complete-contamination";
  const auto &axis=t.parent->network(0).occurrence_axis();
  for(int d=0;d<3;++d){
    YAML::Node r;r["channel"]=d;r["occurrence"]=t.parent->network(0).detector(d).detector_occurrence_id;
    r["stable_segments"]=YAML::Load("[["+std::to_string(axis.first_native_row())+","+std::to_string(axis.past_last_native_row())+"]]");
    r["contaminated"]=YAML::Node(YAML::NodeType::Sequence);n["detectors"].push_back(r);
  }
  const auto event=t.event_at(500);YAML::Node e;e["event"]=event;e["channel"]=0;
  e["seed_earlier_row"]=t.spikes->candidates()[t.assessment->events()[event].seed].earlier_row;
  e["disposition"]="accepted_isolated_event";e["affected"]=YAML::Load("[600,601]");
  n["events"].push_back(e);
  return n;
}
auto bind_review(const Trial &t,const YAML::Node &n,std::array<double,3> factors={1.,1.,1.}) {
  return caller::read_review(n,"exact-controlled-Learn",t.assessment,
      std::array{0,1,2},factors,0,"sha256:controlled-prior-APT");
}
TEST(rtc_multidetector_caller, no_mask_authority_is_scoped_not_a_missing_metadata_fallback) {
  EXPECT_NO_THROW(caller::require_no_mask_scope(NativeObservationScope{152390,0,2},std::string(caller::no_mask_authority)));
  EXPECT_THROW(caller::require_no_mask_scope(NativeObservationScope{152391,0,2},std::string(caller::no_mask_authority)),std::invalid_argument);
  EXPECT_THROW(caller::require_no_mask_scope(NativeObservationScope{152390,1,2},std::string(caller::no_mask_authority)),std::invalid_argument);
  EXPECT_THROW(caller::require_no_mask_scope(NativeObservationScope{152390,0,3},std::string(caller::no_mask_authority)),std::invalid_argument);
  EXPECT_THROW(caller::require_no_mask_scope(NativeObservationScope{152390,0,2},"unknown"),std::invalid_argument);
}
TEST(rtc_multidetector_caller, explicit_review_connects_factors_donors_and_complete_apply) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  const auto selected=bind_review(t,explicit_review(t));
  ASSERT_EQ(selected.events.size(),1);
  auto d=RtcDonorFillPlan::consider(selected.events[0],selected.facts,t.transient,t.val,50);
  ASSERT_EQ(d->cause(),RtcDonorFillCause::ready);
  auto plans=complete_plans(t,{}, {d});
  auto p=RtcPipelinePlan::consider(plans,t.joint->joint_handle(),60);
  auto r=complete_apply(t,p);
  EXPECT_TRUE(r->detector_results()[0]->representative_replaced(600));
  EXPECT_TRUE(r->detector_results()[0]->requires_representative_exclusion(600));
  EXPECT_FALSE(r->detector_results()[0]->coordinate_stage_available(NativeReadoutCoordinate::r,600,true));
  EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x,600,0),in.x(500,0));
  for(int donor=1;donor<3;++donor){
    EXPECT_FALSE(r->detector_results()[donor]->representative_replaced(600));
    EXPECT_TRUE(r->detector_results()[donor]->coordinate_stage_available(NativeReadoutCoordinate::r,600,true));
  }
  EXPECT_EQ(selected.facts->find(0,1)->prior_flxscale,1.);
  EXPECT_EQ(selected.facts->evidence_handle().get(),t.assessment.get());
}
TEST(rtc_multidetector_caller, unapproved_stale_or_foreign_review_cannot_authorize_apply) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto n=explicit_review(t);n["approved"]=false;EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["learning_binding"]="another-Learn";EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["VAL_generation"]=1;EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["authority"]="";EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["detectors"][1]["occurrence"]="another-occurrence";EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["detectors"][1]["channel"]=2;EXPECT_THROW(bind_review(t,n),std::invalid_argument);
}
TEST(rtc_multidetector_caller, candidates_and_event_ordinals_are_not_acceptance) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto n=explicit_review(t);n["events"][0]["disposition"]="recovered_candidate";
  EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["events"][0]["seed_earlier_row"]=123;EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["events"].push_back(YAML::Clone(n["events"][0]));EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["events"][0]["channel"]=1;EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["events"]=YAML::Node(YAML::NodeType::Sequence);
  EXPECT_TRUE(bind_review(t,n).events.empty());
}
TEST(rtc_multidetector_caller, missing_stable_support_and_factor_stay_unavailable) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto n=explicit_review(t);n["detectors"][0].remove("stable_segments");
  EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n=explicit_review(t);n["detectors"][0]["stable_segments"]=YAML::Node(YAML::NodeType::Sequence);
  auto selection=bind_review(t,n);
  EXPECT_EQ(RtcDonorFillPlan::consider(selection.events[0],selection.facts,t.transient,t.val,50)->cause(),RtcDonorFillCause::boundary_unavailable);
  n=explicit_review(t);selection=bind_review(t,n,{NAN,1.,1.});
  EXPECT_FALSE(selection.facts->find(0,0)->prior_flxscale.has_value());
  EXPECT_EQ(RtcDonorFillPlan::consider(selection.events[0],selection.facts,t.transient,t.val,50)->cause(),RtcDonorFillCause::target_transfer_unavailable);
}
TEST(rtc_multidetector_caller, actual_static_factor_ratio_and_contamination_reach_donor_owner) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto n=explicit_review(t);auto selection=bind_review(t,n,{2.,4.,4.});
  auto d=RtcDonorFillPlan::consider(selection.events[0],selection.facts,t.transient,t.val,50);
  ASSERT_EQ(d->cause(),RtcDonorFillCause::ready);
  ASSERT_FALSE(d->medians().empty());const auto &m=d->medians().front();
  EXPECT_DOUBLE_EQ(m.value,(t.parent->network(0).value(NativeReadoutCoordinate::x,m.row,1)+
                           t.parent->network(0).value(NativeReadoutCoordinate::x,m.row,2)));
  n["detectors"][1]["contaminated"]=YAML::Load("[[599,602]]");
  n["detectors"][2]["contaminated"]=YAML::Load("[[599,602]]");selection=bind_review(t,n);
  EXPECT_EQ(RtcDonorFillPlan::consider(selection.events[0],selection.facts,t.transient,t.val,50)->cause(),RtcDonorFillCause::no_usable_donor);
}
TEST(rtc_multidetector_caller, scan_binding_is_explicit_native_support_not_acquisition_scan_number) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto n=explicit_review(t);EXPECT_FALSE(bind_review(t,n).scans);
  n["existing_scans"]=YAML::Load("{state: unavailable, processing_generation: fake}");
  EXPECT_THROW(bind_review(t,n),std::invalid_argument);
  n["existing_scans"]=YAML::Load("{state: conservative_native_support_bound, processing_generation: fixture-scans, native_relation_authority: fixture-exact-native, timing_uncertainty_authority: fixture-exact-timing, support: [{scan: 77, rows: [100,1200]}]}");
  auto selection=bind_review(t,n);ASSERT_TRUE(selection.scans);ASSERT_EQ(selection.scans->supports().size(),1);
  EXPECT_EQ(selection.scans->supports()[0].scan,77);
  EXPECT_EQ(selection.scans->parent_handle().get(),t.parent.get());
}

TEST(rtc_treatment_outcome, identity_stage_is_exact_on_same_windows_and_keeps_source) {
  Input in(6000); Trial t(in,10,RtcSpikeProtection::protected_source);
  auto p=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  auto result=complete_apply(t,p);auto conditioned=conditioned_learn(t,result);
  auto e=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),conditioned,41);
  for(const auto &r:e->records()) {
    ASSERT_TRUE(r.available());EXPECT_EQ(r.original_matched.psd,r.conditioned_matched.psd);
    EXPECT_EQ(r.window_union_samples,r.common_eligible_samples);
    EXPECT_LE(r.window_union_seconds,6000*.008192+1e-8);
    for(const auto &w:r.conditioned_matched.windows)EXPECT_EQ(w.source_counts[1],w.rows.past_last-w.rows.first);
  }
  EXPECT_FALSE(e->classification_authorized);EXPECT_FALSE(e->stopping_rule_selected);
  EXPECT_FALSE(e->independent_noise_estimate);
  EXPECT_EQ(e->original_handle(),t.lines->spectral_handle());
}
TEST(rtc_treatment_outcome, analytic_line_power_uses_matched_footprints_and_preserves_inputs) {
  Input in(6000);const double f=44./(488*.008192);
  for(Eigen::Index i=0;i<in.x.rows();++i)for(int d=0;d<3;++d) {
    in.x(i,d)=std::sin(2*std::numbers::pi*f*i*.008192);
    in.r(i,d)=.2*std::cos(2*std::numbers::pi*f*i*.008192);
  }
  Trial t(in);auto p=RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31);
  auto result=complete_apply(t,p);auto original=t.lines->spectral_handle();const auto before=original->spectra()[0].psd;
  auto e=RtcTreatmentOutcomeEvidence::learn(original,conditioned_learn(t,result,true),41);
  const double h=.5+.5*std::cos(2*std::numbers::pi*f*.008192);
  auto band=e->band(0,0,NativeReadoutCoordinate::x,43,46);
  ASSERT_TRUE(band.conditioned_over_original);EXPECT_NEAR(*band.conditioned_over_original,std::pow(h,4),1e-10);
  EXPECT_LT(e->records()[0].common_eligible_samples,e->records()[0].original_eligible_samples);
  EXPECT_EQ(before,original->spectra()[0].psd);
  for(Eigen::Index i=0;i<in.x.rows();++i)for(int d=0;d<3;++d) {
    EXPECT_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x,100+i,d),in.x(i,d));
    EXPECT_EQ(t.parent->network(0).value(NativeReadoutCoordinate::r,100+i,d),in.r(i,d));
  }
}
TEST(rtc_treatment_outcome, unequal_masks_gaps_and_window_overlap_keep_exact_native_support) {
  Input in(6000);
  in.x(1000,0)=NAN;in.xs[3000]=NativeReadoutCoordinateState::measured(true,false,true,false);
  for(std::size_t i=3000;i<in.times.size();++i){in.times[i]+=.5;in.counters[i]+=30;}
  Trial t(in);auto p=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  auto result=complete_apply(t,p);auto e=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),conditioned_learn(t,result,true),41);
  const auto &r=e->record(0,0,NativeReadoutCoordinate::x);ASSERT_TRUE(r.available());
  EXPECT_LT(r.common_eligible_samples,r.original_eligible_samples);
  std::size_t overlapped=0;
  for(std::size_t i=0;i<r.original_matched.windows.size();++i) {
    const auto &a=r.original_matched.windows[i],&b=r.conditioned_matched.windows[i];
    EXPECT_EQ(a.rows.first,b.rows.first);EXPECT_EQ(a.rows.past_last,b.rows.past_last);
    EXPECT_TRUE(a.rows.past_last<=1100||a.rows.first>1100);
    EXPECT_TRUE(a.rows.past_last<=3100||a.rows.first>=3100);
    overlapped+=a.rows.past_last-a.rows.first;
  }
  EXPECT_GT(overlapped,r.window_union_samples);
  for(auto span:r.window_union)EXPECT_TRUE(span.past_last<=3100||span.first>=3100);
  EXPECT_EQ(r.original_matched.runs[0].unexpected_nonfinite_samples,0);
}
TEST(rtc_treatment_outcome, unexpected_nonfinite_outside_shared_support_is_not_hidden) {
  Input in(6000);Trial t(in);auto p=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  auto result=complete_apply(t,p);auto conditioned=conditioned_learn(t,result,true);
  // First row is outside the finite filtered footprint, but was admitted raw.
  auto *cell=const_cast<double*>(t.parent->network(0).values(NativeReadoutCoordinate::x).data());
  const double saved=*cell;*cell=NAN;
  auto e=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),conditioned,41);*cell=saved;
  const auto &r=e->record(0,0,NativeReadoutCoordinate::x);
  EXPECT_FALSE(r.available());EXPECT_EQ(r.original_matched.cause,RtcSpectralCause::input_consistency_failure);
  EXPECT_EQ(r.original_matched.runs[0].first_unexpected_nonfinite,100);
  EXPECT_TRUE(e->record(0,0,NativeReadoutCoordinate::r).available());
}
TEST(rtc_treatment_outcome, unavailable_support_is_not_zero_residual_power) {
  Input in(6000);Trial t(in);
  auto plans=complete_plans(t);plans[0]=t.plan(false,true,{},0);
  auto p=RtcPipelinePlan::consider(plans,t.joint->joint_handle(),31);auto result=complete_apply(t,p);
  auto e=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),conditioned_learn(t,result,true),41);
  EXPECT_FALSE(e->record(0,0,NativeReadoutCoordinate::x).available());
  EXPECT_FALSE(e->band(0,0,NativeReadoutCoordinate::x,1,10).conditioned_over_original);
  const auto &r=e->record(0,1,NativeReadoutCoordinate::x);ASSERT_TRUE(r.available());
  EXPECT_GT(r.power.original,0);EXPECT_GT(r.power.conditioned,0);
  EXPECT_THROW(e->band(0,1,NativeReadoutCoordinate::x,10,1),std::invalid_argument);
}
TEST(rtc_treatment_outcome, donor_history_and_local_r_unavailability_survive_comparison) {
  Input in(6000);in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);auto donor=explicit_donor(t);
  auto p=RtcPipelinePlan::consider(complete_plans(t,{}, {donor}),t.joint->joint_handle(),31);
  auto result=complete_apply(t,p);auto e=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),conditioned_learn(t,result,true),41);
  const auto &x=e->record(0,0,NativeReadoutCoordinate::x),&r=e->record(0,0,NativeReadoutCoordinate::r);
  ASSERT_TRUE(x.available());ASSERT_TRUE(r.available());
  EXPECT_GT(x.common_eligible_samples,r.common_eligible_samples);
  std::size_t replaced=0,influenced=0;
  for(const auto &w:x.conditioned_matched.windows){replaced+=w.representative_replacements;influenced+=w.replacement_influenced_samples;}
  EXPECT_GT(replaced,0);EXPECT_GT(influenced,0);
  for(const auto &w:r.conditioned_matched.windows)EXPECT_TRUE(w.rows.past_last<=599||w.rows.first>=602);
  EXPECT_TRUE(result->detector_results()[0]->requires_representative_exclusion(600));
}
TEST(rtc_treatment_outcome, reassessment_rejects_foreign_apply_stage_and_cadence_binding) {
  Input in(6000);Trial t(in),other(in);auto p=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  auto result=complete_apply(t,p);auto conditioned=conditioned_learn(t,result),later=conditioned_learn(t,result,true);
  auto e=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),conditioned,41);
  auto considered=RtcSpectralTransientConsideration::consider(conditioned,t.val,t.review,t.val,42);
  auto c=RtcPipelineReassessment::consider(result,considered,43,e);EXPECT_EQ(c->outcome_handle(),e);
  auto other_stage=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),later,44);
  EXPECT_THROW(RtcPipelineReassessment::consider(result,considered,45,other_stage),std::invalid_argument);
  EXPECT_THROW(RtcTreatmentOutcomeEvidence::learn(other.lines->spectral_handle(),conditioned,46),std::invalid_argument);
  EXPECT_THROW(RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),conditioned,0),std::invalid_argument);
  auto wrong=RtcNativeSpectralEvidence::learn_conditioned(result->native_product(false,t.val),t.val,{{0,"other-cadence",.008192,1e-7}},40);
  EXPECT_THROW(RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),wrong,47),std::invalid_argument);
  auto other_apply=complete_apply(t,p);EXPECT_THROW(RtcPipelineReassessment::consider(other_apply,considered,48,e),std::invalid_argument);
}


TEST(rtc_treatment_outcome, too_few_shared_windows_is_unavailable_not_suppression) {
  Input in(490);Trial t(in);
  auto p=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  auto result=complete_apply(t,p);auto conditioned=conditioned_learn(t,result,true);
  auto e=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),conditioned,41);
  const auto &r=e->record(0,0,NativeReadoutCoordinate::x);
  EXPECT_FALSE(r.available());EXPECT_FALSE(r.power.conditioned_over_original);
  EXPECT_LT(r.conditioned_matched.windows.size(),2);
}
TEST(rtc_treatment_outcome, later_VAL_is_explicit_and_cannot_rebind_original_or_select_old_plan) {
  Input in(6000);Trial t(in);auto p=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  auto result=complete_apply(t,p);
  ValDeltaBuilder b{t.val,{ValProducer::rtc,44}};
  b.propose(t.val->address(0,100,0),ValFactCode{1},ValFactState{1},ValFactCause{1});
  auto later=ValSnapshot::commit(b.freeze());auto product=result->native_product(true,later);
  auto spectral=RtcNativeSpectralEvidence::learn_conditioned(product,later,{{0,"control-cadence",.008192,1e-7}},45);
  auto e=RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),spectral,46);
  EXPECT_EQ(e->original_handle()->network(0).input->snapshot_handle(),t.val);
  EXPECT_EQ(e->conditioned_handle()->network(0).input->snapshot_handle(),later);
  EXPECT_THROW(RtcNativeSpectralEvidence::learn_conditioned(product,t.val,{{0,"control-cadence",.008192,1e-7}},45),std::invalid_argument);
  auto considered=RtcSpectralTransientConsideration::consider(spectral,later,t.review,t.val,47);
  auto reassess=RtcPipelineReassessment::consider(result,considered,48,e);
  EXPECT_THROW(RtcPipelinePlan::reconsider(reassess,complete_plans(t),49),StaleRtcValGeneration);
}

auto decision_evidence(const Trial &t,const std::shared_ptr<const RtcPipelineResult> &r,
                       std::uint64_t base=40,bool lowpass=true,bool matched=true,
                       std::shared_ptr<const ValSnapshot> snapshot={}) {
  if(!snapshot)snapshot=t.val;
  auto spectral=RtcNativeSpectralEvidence::learn_conditioned(r->native_product(lowpass,snapshot),snapshot,
      {{0,"control-cadence",.008192,1e-7}},base);
  auto joint=RtcSpectralTransientConsideration::consider(spectral,snapshot,t.review,t.val,base+1);
  auto outcome=matched ? RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),spectral,base+2) : nullptr;
  return RtcPipelineReassessment::consider(r,joint,base+3,outcome);
}
RtcPipelineSelection retain_selection(std::shared_ptr<const RtcPipelineReassessment> e) {
  return {std::move(e),RtcPipelineSelectionIntent::retain_development_candidate,
    "explicit-test-authority", "bounded-development-replay", "retain the explicitly frozen fixture for inspection",{},0};
}
auto advance_trial(const Trial &t,std::shared_ptr<const RtcPipelineResult> current,
                   std::shared_ptr<const RtcPipelineDecision> d) {
  const std::array parts{t.spikes->input_handle()};
  return RtcPipelineResult::advance(current,d,t.spikes->input_handle(),t.val,parts);
}
TEST(rtc_reassessment_decision, authorized_retain_keeps_exact_candidate_without_qualification) {
  Input in(6000);Trial t(in);
  auto current=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31));
  auto e=decision_evidence(t,current);auto selection=retain_selection(e);
  auto d=RtcPipelineDecision::consider(e,t.val,selection,44);
  auto step=advance_trial(t,current,d);
  EXPECT_EQ(d->disposition(),RtcPipelineDisposition::retain);EXPECT_EQ(d->selected_plan(),current->plan_handle());
  EXPECT_EQ(step.candidate,current);EXPECT_FALSE(step.revision_executed);
  EXPECT_EQ(d->qualification,RtcPipelineQualification::unresolved);EXPECT_FALSE(d->scientifically_qualified);
  EXPECT_FALSE(d->downstream_admission_authorized);EXPECT_FALSE(d->production_authorized);EXPECT_FALSE(d->stopping_rule_selected);
  EXPECT_EQ(d->selection()->positive_rationale,selection.positive_rationale);
  selection.authority="mutated caller request";EXPECT_EQ(d->selection()->authority,"explicit-test-authority");
  std::cout<<"decision_trace retain plan=31 next_apply=false qualification=unresolved authority=explicit-test-authority\n";
}
TEST(rtc_reassessment_decision, prescribed_revision_executes_complete_plan_afresh_and_differs_from_cumulative) {
  Input in(6000);const double f=44./(488*.008192);
  for(Eigen::Index i=0;i<in.x.rows();++i)for(int c=0;c<3;++c){
    in.x(i,c)=std::sin(2*std::numbers::pi*f*i*.008192);
    in.r(i,c)=.2*std::cos(2*std::numbers::pi*f*i*.008192);
  }
  Trial t(in);auto first=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31));
  auto e=decision_evidence(t,first);auto selection=retain_selection(e);
  selection.intent=RtcPipelineSelectionIntent::prescribed_finite_revision;
  selection.positive_rationale="test-only prescribed lowpass-only complete alternative; no optimization";
  selection.complete_revision=complete_plans(t);selection.next_attempt=45;
  auto d=RtcPipelineDecision::consider(e,t.val,selection,44);auto step=advance_trial(t,first,d);
  ASSERT_TRUE(step.revision_executed);EXPECT_EQ(d->disposition(),RtcPipelineDisposition::revise);
  auto direct=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),46));
  double cumulative_difference=0;
  for(std::size_t detector=0;detector<3;++detector){
    const auto &actual=*step.candidate->detector_results()[detector],&expected=*direct->detector_results()[detector];
    EXPECT_EQ(actual.output_native_rows(),expected.output_native_rows());EXPECT_EQ(actual.causes(),expected.causes());
    for(Eigen::Index row=0;row<in.x.rows();++row)for(int c=0;c<2;++c)
      EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.filtered_native_pair()(row,c)),std::bit_cast<std::uint64_t>(expected.filtered_native_pair()(row,c)));
    const auto &prior=first->detector_results()[detector]->filtered_native_pair();
    for(auto row:actual.output_native_rows()){
      const auto i=row-100;
      if(i>10&&i+10<in.x.rows())for(int c=0;c<2;++c){
        const double wrong=.25*prior(i-1,c)+.5*prior(i,c)+.25*prior(i+1,c);
        if(std::isfinite(wrong))cumulative_difference=std::max(cumulative_difference,std::abs(actual.filtered_native_pair()(i,c)-wrong));
      }
    }
  }
  EXPECT_GT(cumulative_difference,.05);EXPECT_EQ(step.candidate->plan_handle()->input_handle(),first->plan_handle()->input_handle());
  EXPECT_EQ(step.candidate->plan_handle()->reassessment_handle(),e);EXPECT_FALSE(d->scientifically_qualified);
  std::cout<<"decision_trace revise plan=31->45 direct_original=bitwise_equal cumulative_max_difference="<<cumulative_difference<<" qualification=unresolved\n";
}
TEST(rtc_reassessment_decision, missing_authority_rationale_and_required_qualification_never_pass) {
  Input in(6000);Trial t(in);auto current=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31));
  auto e=decision_evidence(t,current);
  for(int which=0;which<4;++which){
    std::optional<RtcPipelineSelection> selection=retain_selection(e);
    if(which==0)selection.reset();else if(which==1)selection->authority.clear();
    else if(which==2)selection->positive_rationale.clear();else selection->intent=RtcPipelineSelectionIntent::require_scientific_qualification;
    auto d=RtcPipelineDecision::consider(e,t.val,selection,44);auto step=advance_trial(t,current,d);
    EXPECT_EQ(d->disposition(),RtcPipelineDisposition::unavailable);EXPECT_FALSE(d->selected_plan());
    EXPECT_EQ(d->cause(),which==3?RtcPipelineDecisionCause::qualification_unavailable:RtcPipelineDecisionCause::missing_authority);
    EXPECT_EQ(step.candidate,current);EXPECT_FALSE(step.revision_executed);EXPECT_FALSE(d->scientifically_qualified);
  }
  std::cout<<"decision_trace unavailable reason=residual-line-acceptance-policy-unselected candidate_preserved=true next_apply=false\n";
}
TEST(rtc_reassessment_decision, missing_or_insufficient_outcome_names_affected_scope_and_preserves_product) {
  for(bool missing:{false,true}){
    Input in(missing?6000:490);Trial t(in);auto current=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31));
    auto e=decision_evidence(t,current,40,true,!missing);
    auto d=RtcPipelineDecision::consider(e,t.val,retain_selection(e),44);auto step=advance_trial(t,current,d);
    EXPECT_EQ(d->disposition(),RtcPipelineDisposition::unavailable);ASSERT_FALSE(d->issues().empty());
    EXPECT_EQ(d->cause(),missing?RtcPipelineDecisionCause::missing_outcome:RtcPipelineDecisionCause::unavailable_outcome);
    EXPECT_EQ(d->issues()[0].scope.has_value(),!missing);EXPECT_EQ(step.candidate,current);EXPECT_FALSE(step.revision_executed);
  }
}
TEST(rtc_reassessment_decision, foreign_stage_apply_parent_and_stale_attempt_are_rejected) {
  Input in(6000);Trial t(in),foreign(in);auto plan=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  auto current=complete_apply(t,plan);auto e=decision_evidence(t,current),other_stage=decision_evidence(t,current,50,false);
  auto choice=retain_selection(e);
  EXPECT_THROW(RtcPipelineDecision::consider(other_stage,t.val,choice,54),std::invalid_argument);
  EXPECT_THROW(RtcPipelineDecision::consider(e,t.val,choice,43),std::invalid_argument);
  auto d=RtcPipelineDecision::consider(e,t.val,choice,44);const std::array parts{t.spikes->input_handle()};
  EXPECT_THROW(RtcPipelineResult::advance(complete_apply(t,plan),d,t.spikes->input_handle(),t.val,parts),std::invalid_argument);
  EXPECT_THROW(RtcPipelineResult::advance(current,d,foreign.spikes->input_handle(),t.val,parts),std::invalid_argument);
  auto wrong=RtcNativeSpectralEvidence::learn_conditioned(current->native_product(true,t.val),t.val,{{0,"foreign-cadence",.008192,1e-7}},50);
  EXPECT_THROW(RtcTreatmentOutcomeEvidence::learn(t.lines->spectral_handle(),wrong,51),std::invalid_argument);
}
TEST(rtc_reassessment_decision, changed_VAL_invalidates_old_decision_and_later_bound_evidence_cannot_select_old_plan) {
  Input in(6000);Trial t(in);auto current=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31));
  auto e=decision_evidence(t,current);ValDeltaBuilder builder{t.val,{ValProducer::rtc,60}};
  builder.propose(t.val->address(0,100,0),ValFactCode{1},ValFactState{1},ValFactCause{1});auto later=ValSnapshot::commit(builder.freeze());
  EXPECT_THROW(RtcPipelineDecision::consider(e,later,retain_selection(e),44),StaleRtcValGeneration);
  auto d=RtcPipelineDecision::consider(e,t.val,retain_selection(e),44);const std::array parts{t.spikes->input_handle()};
  EXPECT_THROW(RtcPipelineResult::advance(current,d,t.spikes->input_handle(),later,parts),std::invalid_argument);
  auto later_e=decision_evidence(t,current,60,true,true,later);
  EXPECT_THROW(RtcPipelineDecision::consider(later_e,later,retain_selection(later_e),64),StaleRtcValGeneration);
}
TEST(rtc_reassessment_decision, no_op_repeated_plan_and_second_revision_terminate_without_false_success) {
  Input in(6000);Trial t(in);auto first=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31));
  auto e=decision_evidence(t,first);auto selection=retain_selection(e);
  selection.intent=RtcPipelineSelectionIntent::prescribed_finite_revision;selection.next_attempt=45;
  selection.complete_revision=complete_plans(t,{.25,.5,.25}); // new objects/ids, same execution
  auto noop=RtcPipelineDecision::consider(e,t.val,selection,44);
  EXPECT_EQ(noop->cause(),RtcPipelineDecisionCause::no_op_revision);EXPECT_FALSE(advance_trial(t,first,noop).revision_executed);
  selection.complete_revision=complete_plans(t);
  auto revised=advance_trial(t,first,RtcPipelineDecision::consider(e,t.val,selection,44));ASSERT_TRUE(revised.revision_executed);
  auto after=decision_evidence(t,revised.candidate,60);selection=retain_selection(after);
  selection.intent=RtcPipelineSelectionIntent::prescribed_finite_revision;selection.next_attempt=65;
  for(bool repeat:{true,false}){
    selection.complete_revision=complete_plans(t,repeat?std::vector<double>{.25,.5,.25}:std::vector<double>{.2,.6,.2});
    auto d=RtcPipelineDecision::consider(after,t.val,selection,64);auto step=advance_trial(t,revised.candidate,d);
    EXPECT_EQ(d->cause(),repeat?RtcPipelineDecisionCause::repeated_plan:RtcPipelineDecisionCause::revision_budget_exhausted);
    EXPECT_EQ(d->disposition(),RtcPipelineDisposition::unavailable);EXPECT_EQ(step.candidate,revised.candidate);
    EXPECT_FALSE(step.revision_executed);EXPECT_FALSE(d->scientifically_qualified);
  }
  selection.complete_revision=complete_plans(t,{1});
  auto identity=RtcPipelineDecision::consider(after,t.val,selection,64);
  EXPECT_EQ(identity->cause(),RtcPipelineDecisionCause::no_op_revision);
  // Exhausted revision budget does not prevent an explicitly authorized retain.
  auto retained=RtcPipelineDecision::consider(after,t.val,retain_selection(after),64);
  EXPECT_EQ(retained->disposition(),RtcPipelineDisposition::retain);
}
TEST(rtc_reassessment_decision, partial_plan_or_unapproved_control_change_cannot_be_revised) {
  Input in(6000);Trial t(in);auto current=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31));
  auto e=decision_evidence(t,current);auto s=retain_selection(e);s.intent=RtcPipelineSelectionIntent::prescribed_finite_revision;s.next_attempt=45;
  s.complete_revision=complete_plans(t,{.25,.5,.25});s.complete_revision.pop_back();
  EXPECT_THROW(RtcPipelineDecision::consider(e,t.val,s,44),std::invalid_argument);
  s.complete_revision=complete_plans(t,{.25,.5,.25});s.complete_revision[0]=t.plan(false,true,{},0);
  EXPECT_THROW(RtcPipelineDecision::consider(e,t.val,s,44),std::invalid_argument);
  s.complete_revision=complete_plans(t,{.25,.5,.25});
  auto spec=s.complete_revision[0]->assessment_handle()->candidate_handle()->specification();
  spec.science_domain=RtcTransferScienceDomain{"unapproved-response-context",RtcOpticalArray::a2000,10};
  s.complete_revision[0]=RtcNotchRecoveryPlan::consider(RtcLineTransferAssessment::consider(
      RtcLineTransferCandidate::bind(t.lines,0,0,spec),t.joint,t.val,50),t.transient,t.val,t.domain,51);
  EXPECT_THROW(RtcPipelineDecision::consider(e,t.val,s,44),std::invalid_argument);
}
TEST(rtc_reassessment_decision, retained_donor_support_keeps_original_r_absence_and_history) {
  Input in(6000);in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);auto donor=explicit_donor(t);
  auto first=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t,{}, {donor}),t.joint->joint_handle(),31));
  auto e=decision_evidence(t,first);auto d=RtcPipelineDecision::consider(e,t.val,retain_selection(e),44);
  auto step=advance_trial(t,first,d);ASSERT_EQ(d->disposition(),RtcPipelineDisposition::retain);
  EXPECT_EQ(step.candidate,first);
  EXPECT_TRUE(step.candidate->detector_results()[0]->coordinate_stage_available(NativeReadoutCoordinate::x,599,true));
  EXPECT_FALSE(step.candidate->detector_results()[0]->coordinate_stage_available(NativeReadoutCoordinate::r,599,true));
  EXPECT_TRUE(step.candidate->detector_results()[0]->replacement_influence(599,true));
  EXPECT_TRUE(step.candidate->detector_results()[0]->requires_representative_exclusion(600));
}


TEST(rtc_reassessment_decision, diagnostic_overlay_is_preserved_but_not_silently_replayed_as_original) {
  Input in(6000);Trial t(in);auto plan=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  std::vector<RtcRecoveryInjection> overlays;
  for(const auto &p:plan->detector_plans()) {
    RtcRecoveryInjection overlay{p,"diagnostic-overlay",Eigen::Matrix<double,Eigen::Dynamic,2>::Zero(6000,2)};
    overlay.delta(700,0)=.01;overlays.push_back(std::move(overlay));
  }
  const std::array parts{t.spikes->input_handle()};
  auto current=RtcPipelineResult::apply(plan,t.spikes->input_handle(),t.val,parts,overlays);
  auto e=decision_evidence(t,current);auto d=RtcPipelineDecision::consider(e,t.val,retain_selection(e),44);
  EXPECT_EQ(d->cause(),RtcPipelineDecisionCause::diagnostic_overlay_unbound);
  auto step=advance_trial(t,current,d);EXPECT_EQ(step.candidate,current);EXPECT_FALSE(step.revision_executed);
}
} // namespace

namespace {
auto resolved_factors(const Trial &t) {
  std::vector<RtcDonorDetectorFacts> out;
  for(std::uint32_t d=0;d<3;++d){
    RtcDonorDetectorFacts f;f.network=0;f.detector=d;
    f.detector_occurrence_id=t.parent->network(0).detector(d).detector_occurrence_id;
    f.factor_identity="known-prior";f.factor_convention="same-units";f.prior_flxscale=1.;
    f.factor_support={100,1200};out.push_back(f);
  }
  return out;
}
auto resolve_events(const Trial &t, bool truth=false) {
  std::optional<RtcDeclaredContaminant> model;
  if(truth)model=RtcDeclaredContaminant{t.parent,0,0,{600,601},"retained-unmodified-fixture","paired-additive-contaminant"};
  return RtcEventTreatmentDecision::consider(t.review,t.transient,resolved_factors(t),
    "known-prior","same-units",{},t.val,2000,model);
}
auto decision_plan(const Trial &t,std::shared_ptr<const RtcEventTreatmentDecision> d,bool continuity) {
  std::vector<std::shared_ptr<const RtcDonorFillPlan>> donors;
  if(continuity)for(const auto &r:d->records())if(r.donor && r.donor->cause()==RtcDonorFillCause::ready)donors.push_back(r.donor);
  auto base=t.plan();
  return RtcNotchRecoveryPlan::consider(base->assessment_handle(),t.transient,t.val,t.domain,3000,
    std::move(donors),std::move(d),continuity);
}
TEST(rtc_event_treatment, recovered_natural_candidate_is_unavailable_not_accepted) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto d=resolve_events(t);ASSERT_EQ(d->records().size(),t.assessment->events().size());
  const auto &r=d->records()[t.event_at(500)];
  EXPECT_EQ(r.disposition,RtcEventTreatmentClass::isolated_admission_unavailable);
  EXPECT_TRUE(r.proposed_isolation_prerequisites);EXPECT_FALSE(r.donor);
  auto result=t.apply(decision_plan(t,d,true));
  EXPECT_EQ(result->causes().at(500),RtcNotchRecoveryCause::event_policy_unavailable);
  EXPECT_FALSE(result->representative_replaced(600));
  EXPECT_FALSE(result->coordinate_stage_available(NativeReadoutCoordinate::x,599,true));
  EXPECT_TRUE(result->coordinate_stage_available(NativeReadoutCoordinate::x,500,true));
}
TEST(rtc_event_treatment, declared_pair_disturbance_exercises_genuine_exclusion_and_donor_paths) {
  Input in;in.spike();in.paired_identity="declared-contaminant:known-test-copy";
  Trial t(in,10,RtcSpikeProtection::outside_source);auto d=resolve_events(t,true);
  const auto &r=d->records()[t.event_at(500)];
  ASSERT_EQ(r.disposition,RtcEventTreatmentClass::accepted_declared_contaminant);
  ASSERT_TRUE(r.donor);ASSERT_EQ(r.donor->cause(),RtcDonorFillCause::ready);
  auto control=t.apply(decision_plan(t,d,false)),donor=t.apply(decision_plan(t,d,true));
  EXPECT_EQ(control->causes().at(500),RtcNotchRecoveryCause::accepted_event_excluded);
  EXPECT_FALSE(control->representative_replaced(600));
  EXPECT_TRUE(control->requires_representative_exclusion(600));
  EXPECT_TRUE(donor->representative_replaced(600));EXPECT_TRUE(donor->requires_representative_exclusion(600));
  EXPECT_TRUE(donor->coordinate_stage_available(NativeReadoutCoordinate::x,600,true));
  EXPECT_FALSE(donor->coordinate_stage_available(NativeReadoutCoordinate::r,600,true));
  EXPECT_TRUE(donor->replacement_influence(599,true));
  EXPECT_FALSE(donor->requires_representative_exclusion(599));
  EXPECT_FALSE(control->coordinate_stage_available(NativeReadoutCoordinate::x,599,true));
  EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x,600,0),in.x(500,0));
}
TEST(rtc_event_treatment, shared_or_protected_disturbance_is_not_admitted_by_test_truth) {
  Input in;in.spike();in.paired_identity="declared-contaminant:shared";
  in.x(500,1)+=40;in.r(500,1)+=25;
  Trial shared(in,10,RtcSpikeProtection::outside_source);auto d=resolve_events(shared,true);
  for(const auto &r:d->records())EXPECT_FALSE(r.donor);
  Trial protected_t(in,10,RtcSpikeProtection::protected_source);d=resolve_events(protected_t,true);
  for(const auto &r:d->records())EXPECT_FALSE(r.donor);
}
TEST(rtc_event_treatment, undeclared_parent_and_manual_support_are_refused) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  EXPECT_THROW(resolve_events(t,true),std::invalid_argument);
  auto f=resolved_factors(t);f[0].stable_segments={{100,1200}};
  EXPECT_THROW(RtcEventTreatmentDecision::consider(t.review,t.transient,f,"known-prior","same-units",{},t.val,2000),std::invalid_argument);
}
TEST(rtc_event_treatment, later_excluded_fit_support_cannot_supply_continuity) {
  Input in;in.spike();in.paired_identity="declared-contaminant:known-test-copy";
  Trial t(in,10,RtcSpikeProtection::outside_source);
  RtcDeclaredContaminant truth{t.parent,0,0,{600,601},"reference","model"};
  auto d=RtcEventTreatmentDecision::consider(t.review,t.transient,resolved_factors(t),"prior","same-units",{{0,{{400,401}}}},t.val,2000,truth);
  ASSERT_TRUE(d->records()[t.event_at(500)].donor);
  EXPECT_NE(d->records()[t.event_at(500)].donor->cause(),RtcDonorFillCause::ready);
  auto out=t.apply(decision_plan(t,d,true));EXPECT_FALSE(out->representative_replaced(600));
  EXPECT_EQ(out->causes()[500],RtcNotchRecoveryCause::accepted_event_excluded);
}
TEST(processing_scan_native, exact_offset_support_and_missing_slots_do_not_compact_native_time) {
  Input in;auto parent=in.freeze();Eigen::VectorXd common(5);
  std::vector<NativeSlotAssociation> associations;
  for(int i=0;i<5;++i){common[i]=in.times[i+10];associations.push_back({110+i});}
  Eigen::MatrixXI scans(4,1);scans<<1,3,0,4;
  auto out=project_processing_scans_to_native(parent,0,common,associations,scans,.004096,"existing-generation","existing-relation");
  ASSERT_EQ(out.scans[0].science_native.size(),1);EXPECT_EQ(out.scans[0].science_native[0].first,111);
  EXPECT_EQ(out.scans[0].science_native[0].past_last,114);
  associations[2]=NativeSlotAssociation{};
  out=project_processing_scans_to_native(parent,0,common,associations,scans,.004096,"existing-generation","existing-relation");
  EXPECT_EQ(out.scans[0].unmapped_science_slots,1);EXPECT_EQ(out.scans[0].science_native.size(),2);
  associations[2]={110};
  EXPECT_THROW(project_processing_scans_to_native(parent,0,common,associations,scans,.004096,"g","r"),std::invalid_argument);
  associations[2]={112};common[2]+=.01;
  EXPECT_THROW(project_processing_scans_to_native(parent,0,common,associations,scans,.004096,"g","r"),std::invalid_argument);
}
}

namespace {
TEST(rtc_event_treatment, unavailable_paired_donor_support_remains_an_exclusion) {
  Input in;in.spike();in.paired_identity="declared-contaminant:missing-paired-donors";
  for(int d=1;d<3;++d)in.rs[500*3+d]=NativeReadoutCoordinateState::measured(true,false,true,true);
  Trial t(in,10,RtcSpikeProtection::outside_source);auto decision=resolve_events(t,true);
  const auto &r=decision->records()[t.event_at(500)];
  ASSERT_TRUE(r.donor);EXPECT_EQ(r.donor->cause(),RtcDonorFillCause::no_usable_donor);
  auto result=t.apply(decision_plan(t,decision,true));
  EXPECT_FALSE(result->representative_replaced(600));
  EXPECT_EQ(result->causes()[500],RtcNotchRecoveryCause::accepted_event_excluded);
}
TEST(rtc_event_treatment, decided_ready_donor_cannot_be_omitted_from_continuity_arm) {
  Input in;in.spike();in.paired_identity="declared-contaminant:exact-decision";
  Trial t(in,10,RtcSpikeProtection::outside_source);auto decision=resolve_events(t,true);
  EXPECT_THROW(RtcNotchRecoveryPlan::consider(t.plan()->assessment_handle(),t.transient,t.val,t.domain,3000,{},decision,true),std::invalid_argument);
}
TEST(rtc_event_treatment, admitted_shift_excludes_its_existing_scan_and_never_fills) {
  Input in;in.step();Fixture t(in);
  auto support=RtcJumpSupportEvidence::learn(t.transition,9);
  auto refit=RtcJumpRefitEvidence::learn(RtcJumpRefitRequest::consider(support,t.val,10),11);
  auto measured=RtcJumpReassessmentEvidence::learn(RtcJumpRemeasureRequest::consider(refit,t.val,12),13);
  auto admitted=RtcJumpAdmissionDecision::consider(RtcJumpReassessmentDecision::consider(measured,t.val,14),t.val,15);
  auto scans=RtcExistingScanBinding::admit(t.parent,"fixture-processing","exact-native-membership","exact-test-clock",
    RtcExistingScanSupportState::conservative_native_support_bound,{{0,{0,100,500}},{1,{0,500,900}},{2,{0,900,1200}}});
  auto jumps=RtcJumpExclusionPlan::consider(admitted,scans,t.val,16);
  auto transient=RtcTransientExclusionPlan::consider(t.review->original_screening_handle(),jumps,t.val,17);
  std::vector<RtcDonorDetectorFacts> factors;
  for(std::uint32_t d=0;d<3;++d){RtcDonorDetectorFacts f;f.network=0;f.detector=d;
    f.detector_occurrence_id=t.parent->network(0).detector(d).detector_occurrence_id;factors.push_back(f);}
  auto decision=RtcEventTreatmentDecision::consider(t.review,transient,factors,"unavailable-factors","same-units",{},t.val,100);
  const auto &r=decision->records()[t.event_at(500)];
  ASSERT_EQ(r.disposition,RtcEventTreatmentClass::admitted_level_shift);EXPECT_FALSE(r.donor);
  EXPECT_TRUE(transient->excludes(0,550,0));EXPECT_TRUE(transient->excludes(0,850,0));
  for(auto s:decision->facts_handle()->find(0,0)->stable_segments)EXPECT_FALSE(s.first<600 && s.past_last>600);
}
TEST(processing_scan_native, consecutive_native_rows_across_physical_gap_remain_separate) {
  Input in;for(std::size_t i=100;i<in.times.size();++i){in.times[i]+=.25;in.counters[i]+=30;}
  auto parent=in.freeze();Eigen::VectorXd common(4);std::vector<NativeSlotAssociation> associations;
  for(int i=0;i<4;++i){common[i]=in.times[i+98];associations.push_back({198+i});}
  Eigen::MatrixXI scans(4,1);scans<<0,3,0,3;
  const auto out=project_processing_scans_to_native(parent,0,common,associations,scans,.004096,"existing-generation","existing-relation");
  ASSERT_EQ(out.scans[0].science_native.size(),2);
  EXPECT_EQ(out.scans[0].science_native[0].past_last,200);EXPECT_EQ(out.scans[0].science_native[1].first,200);
}
}

namespace {
TEST(rtc_event_treatment, same_spikes_do_not_authorize_mixing_event_generations) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto other=learn_rtc_event_assessment(t.spikes,t.assessment->population_handle(),5000);
  auto other_review=RtcEventAssessmentDecision::consider(other,t.val,5001);
  ASSERT_EQ(other->spike_handle().get(),t.assessment->spike_handle().get());
  ASSERT_NE(other.get(),t.assessment.get());
  EXPECT_THROW(RtcEventTreatmentDecision::consider(other_review,t.transient,resolved_factors(t),
    "known-prior","same-units",{},t.val,5002),std::invalid_argument);
}
}

namespace {
void isolated_speed_excursions(Trial &t) {
  auto times=Eigen::VectorXd::LinSpaced(5000,999,1098.98);
  Eigen::VectorXd ra(times.size()),dec=Eigen::VectorXd::Zero(times.size());
  double position=0;
  for(Eigen::Index i=0;i<times.size();++i){
    const double v=times[i]>=1010 && times[i]<1010.5 ? .5 :
        times[i]>=1018 && times[i]<1018.5 ? 150. : 10.;
    if(i)position+=v*(times[i]-times[i-1]);
    ra[i]=position*std::numbers::pi/(180*3600);
  }
  AstScanMotionSourceMetadata m{AstScanMotionProducerKind::real_toltec,"Science","Lissajous",1,2000,0,50,
      AstScanMotionFieldRegistry::source_ra_act_source_dec_act_j2000_radians,"isolated-low-and-high-speed-test"};
  auto source=AstScanMotionSource::admit(t.parent->scope(),t.parent->scope(),0,m,times,ra,dec);
  t.domain.motion=AstScanMotionNetworkView::admit(build_ast_scan_motion_product(source,{1,2,3,4}),
      t.parent->network(0).occurrence_axis().native_timing_handle());
  t.domain.speed_ceiling_arcsec_per_sec=235;
}
}
TEST(rtc_notch_recovery, speed_support_is_separate_from_centers_and_all_nonspeed_blockers) {
  Input in(4000);
  in.rs[1240*3]=NativeReadoutCoordinateState::measured(true,false,true,true);
  Trial t(in);isolated_speed_excursions(t);
  const auto first=t.parent->network(0).occurrence_axis().first_native_row();
  std::size_t low=0,high=0,overlap=0;
  for(auto treatment:{RtcSpeedSupportTreatment::comparison_reject_both,
      RtcSpeedSupportTreatment::comparison_low_only,RtcSpeedSupportTreatment::comparison_high_only,
      RtcSpeedSupportTreatment::original_paired_measurements}) {
    t.domain.speed_support=treatment;
    auto plan=t.plan(false,false,{.25,.5,.25});auto out=t.apply(plan);
    for(std::size_t i=0;i<in.times.size();++i){
      const auto speed=plan->speed_restrictions()[i];const auto cause=plan->input_causes()[i];
      low+=speed==RtcSpeedRestriction::below_minimum;high+=speed==RtcSpeedRestriction::above_output_sampling_limit;
      if(cause==RtcNotchRecoveryCause::producer_invalid){
        EXPECT_EQ(plan->support_causes()[i],cause);overlap+=speed!=RtcSpeedRestriction::none;
      }
      bool support=cause==RtcNotchRecoveryCause::retained;
      support |= cause==RtcNotchRecoveryCause::below_minimum_speed && (treatment==RtcSpeedSupportTreatment::comparison_low_only || treatment==RtcSpeedSupportTreatment::original_paired_measurements);
      support |= cause==RtcNotchRecoveryCause::insufficient_output_sampling && (treatment==RtcSpeedSupportTreatment::comparison_high_only || treatment==RtcSpeedSupportTreatment::original_paired_measurements);
      EXPECT_EQ(plan->support_causes()[i]==RtcNotchRecoveryCause::retained,support);
      if(speed!=RtcSpeedRestriction::none)EXPECT_FALSE(out->map_center_admitted(first+i));
      if(i<2 || i+2>=in.times.size())continue;
      bool complete=true;
      for(std::size_t k=i-2;k<=i+2;++k)complete &= plan->support_causes()[k]==RtcNotchRecoveryCause::retained;
      EXPECT_EQ(out->coordinate_stage_available(NativeReadoutCoordinate::x,first+i,true),complete);
      EXPECT_EQ(out->coordinate_stage_available(NativeReadoutCoordinate::r,first+i,true),complete);
      EXPECT_EQ(out->map_center_admitted(first+i),complete && i%2==0 && cause==RtcNotchRecoveryCause::retained);
      if(complete){
        // Exact two-stage operator, including speed-excluded intermediate centers.
        double expected=0;
        for(int j=-1;j<=1;++j){
          double intermediate=0;
          for(int k=-1;k<=1;++k)intermediate=std::fma(k==0?.5:.25,in.x(i+j+k,0),intermediate);
          expected=std::fma(j==0?.5:.25,intermediate,expected);
        }
        EXPECT_DOUBLE_EQ(out->filtered_native_pair()(i,0),expected);
      }
    }
  }
  EXPECT_GT(low,0);EXPECT_GT(high,0);EXPECT_GT(overlap,0);
  EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x,first+1240,0),in.x(1240,0));
}
TEST(rtc_pipeline, numerical_speed_support_does_not_expand_existing_spectral_use) {
  Input in(4000);Trial t(in);isolated_speed_excursions(t);
  std::vector<std::shared_ptr<const RtcPipelineResult>> results;
  for(auto treatment:{RtcSpeedSupportTreatment::comparison_reject_both,RtcSpeedSupportTreatment::original_paired_measurements}){
    t.domain.speed_support=treatment;std::vector<std::shared_ptr<const RtcNotchRecoveryPlan>> plans;
    for(int d=0;d<3;++d)plans.push_back(t.plan(false,false,{.25,.5,.25},d));
    auto complete=RtcPipelinePlan::consider(plans,t.joint->joint_handle(),500+results.size());
    const std::array parts{t.spikes->input_handle()};
    results.push_back(RtcPipelineResult::apply(complete,t.spikes->input_handle(),t.val,parts));
  }
  for(bool lowpass:{false,true}){
    auto old=results[0]->native_product(lowpass,t.val),now=results[1]->native_product(lowpass,t.val);
    std::size_t new_numeric=0;
    for(std::size_t d=0;d<3;++d){
      EXPECT_EQ(old->columns()[d].review_use_admitted,now->columns()[d].review_use_admitted);
      EXPECT_EQ(old->columns()[d].speed_restrictions,now->columns()[d].speed_restrictions);
      for(std::size_t i=0;i<in.times.size();++i)new_numeric+=(now->columns()[d].state[i]&3)==3 && (old->columns()[d].state[i]&3)!=3;
    }
    EXPECT_GT(new_numeric,0);
    const std::vector<RtcSpectralCadenceDomain> cadence{{0,"unchanged-cadence",.008192,1e-7}};
    auto a=RtcNativeSpectralEvidence::learn_conditioned(old,t.val,cadence,600),b=RtcNativeSpectralEvidence::learn_conditioned(now,t.val,cadence,601);
    ASSERT_EQ(a->spectra().size(),b->spectra().size());
    for(std::size_t i=0;i<a->spectra().size();++i)EXPECT_EQ(a->spectra()[i].psd,b->spectra()[i].psd);
  }
}

TEST(rtc_notch_recovery, speed_permission_preserves_coincident_optical_domain_restriction) {
  Input in; Trial t(in,250);
  t.domain.speed_ceiling_arcsec_per_sec=235;
  const auto plan=t.plan();
  for(std::size_t i=0;i<plan->input_causes().size();++i) {
    EXPECT_EQ(plan->input_causes()[i],RtcNotchRecoveryCause::insufficient_output_sampling);
    EXPECT_EQ(plan->speed_restrictions()[i],RtcSpeedRestriction::above_output_sampling_limit);
    EXPECT_EQ(plan->support_causes()[i],RtcNotchRecoveryCause::motion_outside_domain);
  }
  const std::array parts{t.spikes->input_handle()};
  const auto result=RtcNotchRecoveryResult::apply(plan,t.spikes->input_handle(),t.val,parts);
  EXPECT_TRUE(result->output_native_rows().empty());
}

TEST(rtc_common_mode, diagnostic_does_not_change_frozen_plan_flags_or_science_output) {
  Input in; Trial t(in);const auto plan=t.plan();const auto before=t.apply(plan);
  RtcCommonModeDomain domain;domain.network=0;domain.motion=t.domain.motion;
  domain.nominal_interval_seconds=.008192;domain.output_factor=2;domain.speed_ceiling_arcsec_per_sec=20;
  domain.population_authority="controlled-three-original-peers";
  domain.scans=RtcExistingScanBinding::admit(t.parent,"existing-controlled-scan","exact-native","known",
      RtcExistingScanSupportState::conservative_native_support_bound,{{0,{0,100,1200}}});
  for(std::uint32_t d=0;d<3;++d)domain.members.push_back({t.parent->network(0).detector(d).detector_occurrence_id,true,1.});
  auto diagnostic=RtcCommonModeEvidence::learn(t.spikes,domain,100);
  auto review=RtcCommonModeConsideration::compare(diagnostic,t.lines);
  auto after=t.apply(plan);
  EXPECT_EQ(before->causes(),after->causes());
  EXPECT_EQ(before->output_native_rows(),after->output_native_rows());
  for(auto row=100;row<1200;++row)for(auto coordinate:{NativeReadoutCoordinate::x,NativeReadoutCoordinate::r})for(bool stage:{false,true})
    EXPECT_EQ(before->coordinate_stage_available(coordinate,row,stage),after->coordinate_stage_available(coordinate,row,stage));
  const auto &a=before->filtered_native_pair(),&b=after->filtered_native_pair();
  ASSERT_EQ(a.size(),b.size());
  for(Eigen::Index i=0;i<a.size();++i)EXPECT_TRUE(a.data()[i]==b.data()[i] || (std::isnan(a.data()[i]) && std::isnan(b.data()[i])));
  EXPECT_EQ(t.val->generation().value,0);
  EXPECT_FALSE(review.rejection_authorized);
  for(std::size_t i=0;i<in.times.size();++i)for(std::size_t d=0;d<3;++d){
    EXPECT_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x,100+i,d),in.x(i,d));
    EXPECT_EQ(t.parent->network(0).value(NativeReadoutCoordinate::r,100+i,d),in.r(i,d));
    EXPECT_EQ(t.parent->network(0).state(NativeReadoutCoordinate::x,100+i,d).valid(),in.xs[i*3+d].valid());
    EXPECT_EQ(t.parent->network(0).state(NativeReadoutCoordinate::r,100+i,d).valid(),in.rs[i*3+d].valid());
  }
}

Input consequence_input(TimestreamNativeRow first) {Input in(6000);in.first_native_row=first;return in;}
struct ConsequenceFixture {
  Input input;
  Trial trial{input};
  std::shared_ptr<const RtcPipelineResult> base=complete_apply(trial,RtcPipelinePlan::consider(complete_plans(trial),trial.joint->joint_handle(),31));
  std::vector<RtcRecoveryInjection> source;
  std::vector<RtcConsequenceEvidence::Positions> positions;
  RtcConsequenceDomain domain;
  explicit ConsequenceFixture(TimestreamNativeRow first=100):input(consequence_input(first)) {
    domain.identity="controlled-source";domain.source_model="constant algebra control";
    domain.regime="native fixture units";domain.geometry_identity="declared-linear-fixture-arcsec";
    domain.units="native x;arcsec";domain.purposes={citlali::config::ReductionType::science,citlali::config::ReductionType::pointing};
    domain.injected_identity="source-probe";domain.required_unavailable={"purpose-specific acceptance limit","two-dimensional pointing fit"};
    for(const auto &plan:base->plan_handle()->detector_plans()) {
      source.push_back({plan,"source-probe",Eigen::Matrix<double,Eigen::Dynamic,2>::Ones(6000,2)});
      RtcConsequenceEvidence::Positions xy(6000,2);
      for(int i=0;i<6000;++i){xy(i,0)=.1*i;xy(i,1)=0;}positions.push_back(xy);
      domain.windows.push_back({1000,1200});domain.ringing_windows.push_back({950,1250});
    }
  }
  auto apply(const std::vector<RtcRecoveryInjection> &s) {
    const std::array parts{trial.spikes->input_handle()};
    return RtcPipelineResult::apply(base->plan_handle(),trial.spikes->input_handle(),trial.val,parts,s);
  }
  auto learn(const std::shared_ptr<const RtcPipelineResult> &value,std::shared_ptr<const RtcPipelineResult> no_line={}) {
    return RtcConsequenceEvidence::learn(base,base,value,no_line,source,positions,domain,trial.val,70);
  }
};
TEST(rtc_purpose_consequence, final_scheduled_unit_response_keeps_purpose_and_missing_requirements) {
  ConsequenceFixture f;auto injected=f.apply(f.source);auto e=f.learn(injected);
  ASSERT_EQ(e->records().size(),3);EXPECT_FALSE(e->acceptance_requirement_selected);
  EXPECT_EQ(e->domain().purposes.size(),2);EXPECT_EQ(e->domain().required_unavailable.size(),2);
  for(const auto &r:e->records()) {
    ASSERT_TRUE(r.available);EXPECT_EQ(r.rows.size(),100);EXPECT_EQ(r.expected,100);
    for(auto row:r.rows)EXPECT_EQ(row%2,0);
    EXPECT_NEAR(r.measured.projection,1,1e-12);EXPECT_NEAR(r.measured.peak_ratio,1,1e-12);
    EXPECT_NEAR(r.measured.centroid_x_arcsec,0,1e-10);EXPECT_NEAR(r.measured.waveform_error,0,1e-12);
  }
  auto spectral=conditioned_learn(f.trial,f.base,true);
  auto conditioned=RtcSpectralTransientConsideration::consider(spectral,f.trial.val,f.trial.review,f.trial.val,72);
  auto outcome=RtcTreatmentOutcomeEvidence::learn(f.trial.lines->spectral_handle(),spectral,73);
  auto reassess=RtcPipelineReassessment::consider(f.base,conditioned,74,outcome,{e});
  auto selection=retain_selection(reassess);auto decision=RtcPipelineDecision::consider(reassess,f.trial.val,selection,75);
  EXPECT_EQ(decision->disposition(),RtcPipelineDisposition::retain);
  EXPECT_EQ(decision->reassessment_handle()->consequence_handles()[0],e);EXPECT_FALSE(decision->scientifically_qualified);
  selection.intent=RtcPipelineSelectionIntent::require_scientific_qualification;
  EXPECT_EQ(RtcPipelineDecision::consider(reassess,f.trial.val,selection,76)->disposition(),RtcPipelineDisposition::unavailable);
}
TEST(rtc_purpose_consequence, line_parameter_error_does_not_cancel_like_paired_source_transfer) {
  ConsequenceFixture f;auto absent=f.source,present=f.source;
  f.domain.line_free_identity="same-sky-noise-no-line";f.domain.injected_identity="same-sky-noise-plus-line";
  for(std::size_t d=0;d<3;++d){
    absent[d].identity=f.domain.line_free_identity;present[d].identity=f.domain.injected_identity;
    for(int i=0;i<6000;++i){
      const double sky=std::exp(-.5*std::pow((i-1000.)/30,2));
      f.source[d].delta(i,0)=sky;f.source[d].delta(i,1)=.2*sky;
      const double noise=.01*std::sin(.63*i),line=.2*std::sin(.09*i+.2*d);
      absent[d].delta(i,0)=sky+noise;absent[d].delta(i,1)=.2*sky;
      present[d].delta.row(i)=absent[d].delta.row(i);present[d].delta(i,0)+=line;
    }
  }
  auto a=f.apply(absent),p=f.apply(present);auto e=f.learn(p,a);
  double largest=0;
  for(const auto &r:e->records()) {
    ASSERT_TRUE(r.available);ASSERT_TRUE(r.line_free);double dot=0,energy=0;
    for(auto row:r.rows){auto i=row-100;double s=f.source[r.detector].delta(i,0);
      dot+=s*(p->detector_results()[r.detector]->filtered_native_pair()(i,0)-a->detector_results()[r.detector]->filtered_native_pair()(i,0));energy+=s*s;}
    EXPECT_NEAR(r.measured.projection-r.line_free->projection,dot/energy,1e-12);
    largest=std::max(largest,std::abs(dot/energy));
  }
  EXPECT_GT(largest,.001);
}
TEST(rtc_purpose_consequence, incomplete_crossing_and_absent_model_remain_unavailable) {
  ConsequenceFixture f;auto y=f.apply(f.source);
  f.domain.windows[0]={100,300};f.domain.ringing_windows[0]={100,350};
  f.domain.window_unavailable={"","no principal crossing","OOF defocused fixture missing"};
  auto e=f.learn(y);for(const auto &r:e->records()){EXPECT_FALSE(r.available);EXPECT_FALSE(r.unavailable.empty());}
}
TEST(rtc_purpose_consequence, foreign_plan_VAL_geometry_and_overlay_cannot_bind) {
  ConsequenceFixture f;auto y=f.apply(f.source);
  auto original=f.source[0].plan;f.source[0].plan=f.trial.plan(false,false,{},0);
  EXPECT_THROW(f.learn(y),std::invalid_argument);f.source[0].plan=original;
  f.domain.injected_identity="other-overlay";EXPECT_THROW(f.learn(y),std::invalid_argument);f.domain.injected_identity="source-probe";
  f.positions[0](1,0)=NAN;EXPECT_THROW(f.learn(y),std::invalid_argument);f.positions[0](1,0)=.1;
  ValDeltaBuilder b{f.trial.val,{ValProducer::rtc,80}};b.propose(f.trial.val->address(0,100,0),ValFactCode{1},ValFactState{1},ValFactCause{1});auto later=ValSnapshot::commit(b.freeze());
  EXPECT_THROW(RtcConsequenceEvidence::learn(f.base,f.base,y,nullptr,f.source,f.positions,f.domain,later,81),std::invalid_argument);
  f.domain.purposes.clear();EXPECT_THROW(f.learn(y),std::invalid_argument);
}
TEST(rtc_purpose_consequence, foreign_Apply_cannot_reuse_consequences_in_reassessment) {
  ConsequenceFixture f;auto e=f.learn(f.apply(f.source));auto other=complete_apply(f.trial,f.base->plan_handle());
  auto spectral=conditioned_learn(f.trial,other,true);
  auto c=RtcSpectralTransientConsideration::consider(spectral,f.trial.val,f.trial.review,f.trial.val,72);
  EXPECT_THROW(RtcPipelineReassessment::consider(other,c,74,nullptr,{e}),std::invalid_argument);
}

TEST(rtc_purpose_consequence, declared_window_counts_schedule_relative_to_odd_native_origin) {
  ConsequenceFixture f(101);
  for(std::size_t d=0;d<3;++d){f.domain.windows[d]={1001,1202};f.domain.ringing_windows[d]={951,1252};}
  auto e=f.learn(f.apply(f.source));
  for(const auto &r:e->records()) {
    ASSERT_TRUE(r.available);EXPECT_EQ(r.expected,101);EXPECT_EQ(r.rows.size(),101);
    EXPECT_EQ(r.rows.front(),1001);EXPECT_EQ(r.rows.back(),1201);
    for(auto row:r.rows)EXPECT_EQ((row-101)%2,0);
    EXPECT_NEAR(r.measured.projection,1,1e-12);
  }
}

TEST(rtc_output_grid, exact_output_slots_keep_unavailable_positions_and_original_parent) {
  for (auto origin : {100, 101}) {
    Input in(1600); in.first_native_row=origin; Trial t(in);
    auto result=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31));
    const auto before=t.parent->network(0).value(NativeReadoutCoordinate::x,origin+700,0);
    auto g=RtcOutputGrid::prepare(result,output_align(t));
    EXPECT_EQ(g->applied_handle(),result);EXPECT_EQ(g->input_val_snapshot_handle(),t.val);
    ASSERT_EQ(g->detectors().size(),3);EXPECT_EQ(g->detectors()[0].scheduled_count,800);
    auto edge=g->occurrence(0,0);EXPECT_FALSE(edge.x_available);EXPECT_FALSE(edge.filter_footprint);
    EXPECT_EQ(edge.representative.network_occurrence.native_row(),origin);
    for(std::size_t slot=0;slot<800;++slot) {
      const auto fact=g->occurrence(0,slot);const auto row=origin+2*slot;
      EXPECT_EQ(fact.slot,slot);EXPECT_EQ(fact.representative.network_occurrence.native_row(),row);
      EXPECT_DOUBLE_EQ(fact.representative.assigned_time_unix_sec,t.parent->network(0).occurrence_axis().native_identity(row).reconstructed_time_unix_sec());
      if(fact.x_available) {
        EXPECT_DOUBLE_EQ(*g->value(0,slot,NativeReadoutCoordinate::x),result->detector_results()[0]->filtered_native_pair()(2*slot,0));
        ASSERT_TRUE(fact.filter_footprint);EXPECT_EQ(fact.filter_footprint->first,row-2);
        EXPECT_EQ(fact.filter_footprint->past_last,row+3);
      }
    }
    EXPECT_DOUBLE_EQ(before,t.parent->network(0).value(NativeReadoutCoordinate::x,origin+700,0));
    EXPECT_LT(g->owned_descriptor_bytes(),1024);
    EXPECT_THROW(g->occurrence(0,800),std::out_of_range);
    EXPECT_THROW(g->occurrence(3,0),std::out_of_range);
  }
}
TEST(rtc_output_grid, foreign_parent_snapshot_and_motion_cannot_be_relabelled) {
  Input in(1600);Trial t(in),foreign(in);
  auto result=complete_apply(t,RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31));
  EXPECT_THROW(RtcOutputGrid::prepare(result,output_align(foreign)),std::invalid_argument);
  EXPECT_THROW(RtcOutputGrid::prepare(result,output_align(t,ValSnapshot::initial(t.parent))),std::invalid_argument);
  auto other=build_ast_scan_motion_product(t.domain.motion->raw_product_handle()->source_handle(),{9,8,7,6});
  EXPECT_THROW(RtcOutputGrid::prepare(result,output_align(t,t.val,other)),std::invalid_argument);
  EXPECT_THROW(RtcOutputGrid::prepare(result,nullptr),std::invalid_argument);
}
TEST(rtc_output_grid, chunking_does_not_change_grid_and_physical_gap_has_no_footprint) {
  Input in(1600);for(std::size_t i=800;i<in.times.size();++i){in.times[i]+=.5;in.counters[i]+=4;}Trial t(in);
  auto plan=RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25}),t.joint->joint_handle(),31);
  auto one=complete_apply(t,plan);
  const std::array parts{NativePairedReadoutView::admit(t.parent,{{0,100,513}}),
      NativePairedReadoutView::admit(t.parent,{{0,513,1001}}),
      NativePairedReadoutView::admit(t.parent,{{0,1001,1700}})};
  auto many=RtcPipelineResult::apply(plan,t.spikes->input_handle(),t.val,parts);
  auto align=output_align(t);auto a=RtcOutputGrid::prepare(one,align),b=RtcOutputGrid::prepare(many,align);
  EXPECT_EQ(a->detectors()[0].scheduled_count,b->detectors()[0].scheduled_count);
  for(std::size_t slot=0;slot<800;++slot) {
    auto x=a->occurrence(0,slot),y=b->occurrence(0,slot);
    EXPECT_EQ(x.representative,y.representative);EXPECT_EQ(x.x_available,y.x_available);
    EXPECT_EQ(x.realized_cause,y.realized_cause);
    EXPECT_EQ(a->value(0,slot,NativeReadoutCoordinate::x),b->value(0,slot,NativeReadoutCoordinate::x));
    if(x.filter_footprint)EXPECT_FALSE(x.filter_footprint->first<900 && x.filter_footprint->past_last>900);
  }
  EXPECT_FALSE(a->occurrence(0,400).filter_footprint);
}
TEST(rtc_output_grid, donor_numerical_support_is_distinct_from_replacement_and_r_availability) {
  Input in;in.spike();Trial t(in,10,RtcSpikeProtection::outside_source);
  auto plan=RtcPipelinePlan::consider(complete_plans(t,{.25,.5,.25},{explicit_donor(t)}),t.joint->joint_handle(),31);
  auto g=RtcOutputGrid::prepare(complete_apply(t,plan),output_align(t));
  auto replaced=g->occurrence(0,250),neighbor=g->occurrence(0,249);
  EXPECT_TRUE(replaced.representative_replaced);EXPECT_TRUE(replaced.representative_excluded);
  EXPECT_TRUE(neighbor.x_available);EXPECT_FALSE(neighbor.r_available);
  EXPECT_TRUE(neighbor.replacement_influence);EXPECT_FALSE(neighbor.representative_excluded);
  EXPECT_TRUE(g->value(0,249,NativeReadoutCoordinate::x));EXPECT_FALSE(g->value(0,249,NativeReadoutCoordinate::r));
  EXPECT_EQ(g->applied_handle()->detector_results()[0]->donor_results().size(),1);
}
TEST(rtc_output_grid, injected_diagnostic_cannot_become_original_conditioned_output) {
  Input in(1600);Trial t(in);auto plan=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  std::vector<RtcRecoveryInjection> overlays;
  for(const auto &p:plan->detector_plans())overlays.push_back({p,"diagnostic-overlay",Eigen::Matrix<double,Eigen::Dynamic,2>::Zero(1600,2)});
  const std::array parts{t.spikes->input_handle()};
  auto diagnostic=RtcPipelineResult::apply(plan,t.spikes->input_handle(),t.val,parts,overlays);
  EXPECT_THROW(RtcOutputGrid::prepare(diagnostic,output_align(t)),std::invalid_argument);
}

TEST(rtc_output_grid, val_outputs_are_disjoint_from_native_coordinates_and_other_realizations) {
  Input in(1600);Trial t(in);auto plan=RtcPipelinePlan::consider(complete_plans(t),t.joint->joint_handle(),31);
  auto result=complete_apply(t,plan);auto align=output_align(t);
  auto a=RtcOutputGrid::prepare(result,align),b=RtcOutputGrid::prepare(result,align);
  auto x=RtcOutputGrid::val_target(a,0,100,NativeReadoutCoordinate::x);
  auto r=RtcOutputGrid::val_target(a,0,100,NativeReadoutCoordinate::r);
  auto other=RtcOutputGrid::val_target(b,0,100,NativeReadoutCoordinate::x);
  const ValProducerProductIdentity producer{ValProducer::rtc,31};
  const ValFactCode code{1};const ValFactState state{1};const ValFactCause cause{1};
  ValDeltaBuilder delta(t.val,producer);delta.propose(x,code,state,cause);
  auto next=ValSnapshot::commit(delta.freeze());
  ASSERT_TRUE(next->find({producer,x,code}));EXPECT_FALSE(next->find({producer,r,code}));
  EXPECT_FALSE(next->find({producer,other,code}));EXPECT_FALSE(next->find({producer,x.address(),code}));
  EXPECT_FALSE(t.val->find({producer,x,code}));EXPECT_TRUE(next->contains(x));
  auto native=ValNativeRealization::create(t.parent,producer,1,ValNativeProductRole::derived_residual,0);
  auto native_target=t.val->native_target(native,x.address(),NativeReadoutCoordinate::x);
  EXPECT_FALSE(next->find({producer,native_target,code}));
  auto foreign=ValSnapshot::initial(t.parent);ValDeltaBuilder bad(foreign,producer);
  EXPECT_THROW(bad.propose(x,code,state,cause),std::invalid_argument);
  auto moved=std::move(other);(void)moved;ValDeltaBuilder moved_from(t.val,producer);
  EXPECT_THROW(moved_from.propose(other,code,state,cause),std::invalid_argument);
  EXPECT_EQ(next->generation().value,1);EXPECT_EQ(a->input_val_snapshot_handle()->generation().value,0);
  EXPECT_EQ(next->memory_evidence().referenced_rtc_output_target_count,1);
  EXPECT_EQ(next->memory_evidence().referenced_native_target_count,0);
  RecordProperty("rtc_output_target_bytes",sizeof(ValRtcOutputTarget));
  RecordProperty("finding_key_bytes",sizeof(ValFindingKey));
}
