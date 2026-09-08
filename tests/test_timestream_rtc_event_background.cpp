#include <citlali/core/pipeline/timestream_rtc_event_background.h>
#include "timestream_successor_identity_test_support.h"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>

namespace {
using namespace citlali::pipeline;
namespace support = citlali::test::timestream_successor;

struct Input {
    std::vector<double> times;
    std::vector<TimestreamPacketCounter> counters;
    NativePairedReadoutMatrix x, r;
    std::vector<NativeReadoutCoordinateState> xs, rs;
    explicit Input(std::size_t rows = 1001)
        : times(rows), counters(rows), x(rows, 1), r(rows, 1),
          xs(support::valid_states(rows)), rs(xs) {
        for (std::size_t i = 0; i < rows; ++i) {
            times[i] = 1000 + 0.008192*i;
            counters[i] = 2000 + i;
            const double t = 0.008192*(static_cast<double>(i) - 500);
            const double noise = std::array{-0.1, 0.0, 0.1, 0.0, 0.0, -0.1, 0.1}[i%7];
            x(i,0) = 10 + 0.8*t + 0.1*t*t - 0.05*t*t*t + noise;
            r(i,0) = -3 + 2*noise;
        }
    }
    void step(double dx = 4, double dr = -3, std::size_t at = 500) {
        for (std::size_t i = at; i < times.size(); ++i) { x(i,0) += dx; r(i,0) += dr; }
    }
    auto freeze() const {
        std::vector<NativePairedReadoutNetwork> networks;
        networks.push_back(NativePairedReadoutNetwork::admit(
            support::occurrence_axis(0, 100, times, counters), support::detector_axis(0, 1),
            support::mapping_authority(0, "background-test"), x, r, xs, rs));
        return support::make_observation(std::move(networks), {0});
    }
};

struct Fixture {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const NativePairedReadoutView> view;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcSpikeSourceProtection> protection;
    std::shared_ptr<const RtcSpikeEvidence> spikes;
    explicit Fixture(const Input &in, RtcSpikeProtection state = RtcSpikeProtection::outside_source,
                     std::vector<RtcSpikeProtectionRegion> regions = {})
        : parent(in.freeze()), view(NativePairedReadoutView::full(parent)), val(ValSnapshot::initial(parent)),
          protection(RtcSpikeSourceProtection::admit(parent, "source-test-v1", state, std::move(regions))),
          spikes(learn_rtc_spike_candidates(view, val, protection, 81)) {}
    std::size_t candidate(TimestreamNativeRow row = 600) const {
        for (std::size_t i = 0; i < spikes->candidates().size(); ++i)
            if (spikes->candidates()[i].coordinate == NativeReadoutCoordinate::x &&
                spikes->candidates()[i].later_row == row) return i;
        throw std::runtime_error("fixture did not produce requested original x candidate");
    }
    auto learn(TimestreamNativeRow first = 594, TimestreamNativeRow end = 607,
               TimestreamNativeRow candidate_row = 600) const {
        return learn_rtc_event_background(spikes, {candidate(candidate_row), first, end}, 91);
    }
};

TEST(rtc_event_background, joint_offset_recovers_signed_changes_on_cubic_background) {
    Input in;
    in.step();
    Fixture f(in);
    const auto e = f.learn();
    for (const auto &c : e->coordinates()) {
        ASSERT_TRUE(c.available()) << static_cast<int>(c.support_cause) << ' '
            << static_cast<int>(c.pre_scale_fit.cause) << ' ' << static_cast<int>(c.cubic.cause);
        EXPECT_DOUBLE_EQ(c.cubic.scale, c.pre_scale_fit.scale);
        EXPECT_DOUBLE_EQ(c.cubic_with_offset.scale, c.pre_scale_fit.scale);
        EXPECT_LT(c.cubic_with_offset.huber_loss, c.cubic.huber_loss);
        EXPECT_EQ(c.support[0].usable, 244U);
        EXPECT_EQ(c.support[1].usable, 244U);
    }
    EXPECT_NEAR(e->coordinates()[0].cubic_with_offset.offset, 4, 0.02);
    EXPECT_NEAR(e->coordinates()[1].cubic_with_offset.offset, -3, 0.04);
    EXPECT_NEAR(e->coordinates()[1].pre_scale_fit.scale, 2*e->coordinates()[0].pre_scale_fit.scale, 0.003);
    EXPECT_EQ(e->spike_evidence_handle(), f.spikes);
    EXPECT_EQ(e->attempt(), 91U);
    EXPECT_EQ(e->request().excluded_first, 594);
    EXPECT_FALSE(e->offset_uncertainty_available());
    EXPECT_FALSE(e->exclusion_containment_established());
    std::cout << "RTC_EVENT_BACKGROUND_INJECTION x_offset=" << e->coordinates()[0].cubic_with_offset.offset
              << " r_offset=" << e->coordinates()[1].cubic_with_offset.offset
              << " pre_x_scale=" << e->coordinates()[0].pre_scale_fit.scale << '\n';
}

TEST(rtc_event_background, excluded_spike_does_not_become_offset_or_change_original_data) {
    Input in;
    in.x(500,0) += 40;
    Fixture f(in);
    const auto before = f.parent->network(0).value(NativeReadoutCoordinate::x, 600, 0);
    const auto e = f.learn();
    ASSERT_TRUE(e->coordinates()[0].available());
    EXPECT_NEAR(e->coordinates()[0].cubic_with_offset.offset, 0, 0.02);
    EXPECT_DOUBLE_EQ(f.parent->network(0).value(NativeReadoutCoordinate::x, 600, 0), before);
    EXPECT_EQ(f.val->generation().value, 0U);
    const auto req = RtcEventBackgroundDecision::consider(e, f.val, 1)->requirements(NativeReadoutCoordinate::x);
    EXPECT_FALSE(req.numerical_evidence_unavailable);
    EXPECT_TRUE(req.offset_uncertainty_and_acceptance_required);
    EXPECT_TRUE(req.extent_containment_and_recovery_required);
    EXPECT_TRUE(req.background_adequacy_and_coverage_required);
}

TEST(rtc_event_background, offset_is_invariant_to_added_cubic_drift) {
    Input a;
    a.step();
    Input b = a;
    for (std::size_t i = 0; i < b.times.size(); ++i) {
        const double t = 0.008192*(static_cast<double>(i)-500);
        b.x(i,0) += -8*t + 0.3*t*t + 0.2*t*t*t;
    }
    const auto ea = Fixture(a).learn(), eb = Fixture(b).learn();
    ASSERT_TRUE(ea->coordinates()[0].available());
    ASSERT_TRUE(eb->coordinates()[0].available());
    EXPECT_NEAR(ea->coordinates()[0].cubic_with_offset.offset, eb->coordinates()[0].cubic_with_offset.offset, 1e-7);
}

TEST(rtc_event_background, post_noise_cannot_widen_pre_scale) {
    Input a;
    a.step();
    Input b = a;
    for (std::size_t i = 507; i < b.times.size(); ++i) b.x(i,0) += 0.015*std::sin(0.19*i);
    const auto ea = Fixture(a).learn(), eb = Fixture(b).learn();
    ASSERT_TRUE(ea->coordinates()[0].available());
    ASSERT_TRUE(eb->coordinates()[0].available());
    EXPECT_DOUBLE_EQ(ea->coordinates()[0].pre_scale_fit.scale, eb->coordinates()[0].pre_scale_fit.scale);
    EXPECT_DOUBLE_EQ(eb->coordinates()[0].cubic.scale, eb->coordinates()[0].pre_scale_fit.scale);
    EXPECT_DOUBLE_EQ(eb->coordinates()[0].cubic_with_offset.scale, eb->coordinates()[0].pre_scale_fit.scale);
}

TEST(rtc_event_background, guard_is_explicit_trial_support_without_extent_claim) {
    Input in;
    in.step();
    Fixture f(in);
    const auto narrow = f.learn(), wide = f.learn(585, 616);
    ASSERT_TRUE(narrow->coordinates()[0].available());
    ASSERT_TRUE(wide->coordinates()[0].available());
    EXPECT_GT(wide->excluded_support().duration_sec(), narrow->excluded_support().duration_sec());
    EXPECT_FALSE(wide->exclusion_containment_established());
    EXPECT_NEAR(wide->coordinates()[0].cubic_with_offset.offset, 4, 0.03);
    EXPECT_THROW(f.learn(600, 607), std::invalid_argument);
    EXPECT_THROW(f.learn(594, 600), std::invalid_argument);
}

TEST(rtc_event_background, internal_partition_boundaries_do_not_change_fit) {
    Input in;
    in.step();
    Fixture f(in);
    std::array partitions{
        NativePairedReadoutView::admit(f.parent, {{0,100,600}}),
        NativePairedReadoutView::admit(f.parent, {{0,600,1101}})};
    const auto partitioned = learn_rtc_spike_candidates_partitioned(f.view, partitions, f.val, f.protection, 82);
    const auto a = f.learn();
    const auto b = learn_rtc_event_background(partitioned, a->request(), 92);
    EXPECT_EQ(a->coordinates()[0].cubic_with_offset.coefficients, b->coordinates()[0].cubic_with_offset.coefficients);
    EXPECT_DOUBLE_EQ(a->coordinates()[0].cubic_with_offset.offset, b->coordinates()[0].cubic_with_offset.offset);
    EXPECT_FALSE(b->observation_truncated()[0]);
    EXPECT_FALSE(b->gap_truncated()[0]);
}

TEST(rtc_event_background, observation_end_retains_incomplete_context_constraint) {
    Input in;
    in.step(4, -3, 150);
    Fixture f(in);
    const auto e = f.learn(244,257,250);
    ASSERT_TRUE(e->coordinates()[0].available());
    EXPECT_TRUE(e->observation_truncated()[0]);
    EXPECT_FALSE(e->gap_truncated()[0]);
    EXPECT_TRUE(RtcEventBackgroundDecision::consider(e,f.val,1)->requirements(NativeReadoutCoordinate::x).incomplete_observation_context);
}

TEST(rtc_event_background, actual_gap_is_never_crossed_or_called_a_scan_edge) {
    Input in;
    in.step();
    for (std::size_t i = 350; i < in.times.size(); ++i) in.counters[i] += 4;
    Fixture f(in);
    const auto e = f.learn();
    EXPECT_TRUE(e->gap_truncated()[0]);
    EXPECT_FALSE(e->observation_truncated()[0]);
    EXPECT_GE(e->coordinates()[0].support[0].first_used, 450);
    EXPECT_TRUE(RtcEventBackgroundDecision::consider(e,f.val,1)->requirements(NativeReadoutCoordinate::x).physical_gap_context);
    EXPECT_THROW(f.learn(449,607), std::invalid_argument);
}

TEST(rtc_event_background, invalid_payload_is_excluded_and_minimum_is_per_coordinate) {
    Input in;
    in.step();
    // Leave exactly 63 valid pre-fit r samples, without disturbing x.
    for (std::size_t i = 413; i < 494; ++i) {
        in.rs[i] = NativeReadoutCoordinateState::measured(true, false, true, false);
        in.r(i,0) = std::numeric_limits<double>::quiet_NaN();
    }
    for (std::size_t i = 250; i < 350; ++i) {
        in.rs[i] = NativeReadoutCoordinateState::measured(true, false, true, false);
        in.r(i,0) = std::numeric_limits<double>::quiet_NaN();
    }
    Fixture f(in);
    const auto e = f.learn();
    ASSERT_TRUE(e->coordinates()[0].available());
    EXPECT_EQ(e->coordinates()[1].support[0].usable, 63U);
    EXPECT_EQ(e->coordinates()[1].support_cause, RtcEventFitCause::insufficient_samples);
    EXPECT_TRUE(std::isnan(e->coordinates()[1].cubic_with_offset.offset));
}

TEST(rtc_event_background, known_additional_candidate_keeps_affected_coordinate_unavailable) {
    Input in;
    in.step();
    in.x(650,0) += 40;
    Fixture f(in);
    const auto e = f.learn();
    EXPECT_EQ(e->coordinates()[0].support_cause, RtcEventFitCause::additional_candidate);
    EXPECT_TRUE(e->coordinates()[1].available());
    EXPECT_TRUE(RtcEventBackgroundDecision::consider(e,f.val,1)->requirements(NativeReadoutCoordinate::x).numerical_evidence_unavailable);
}

TEST(rtc_event_background, exactly_64_usable_samples_meets_count_without_adequacy_claim) {
    Input in;
    in.step();
    for (std::size_t i = 250; i < 430; ++i) {
        in.rs[i] = NativeReadoutCoordinateState::measured(true, false, true, false);
        in.r(i,0) = std::numeric_limits<double>::quiet_NaN();
    }
    Fixture f(in);
    const auto e = f.learn();
    EXPECT_EQ(e->coordinates()[1].support[0].usable, 64U);
    EXPECT_EQ(e->coordinates()[1].support_cause, RtcEventFitCause::none);
    EXPECT_TRUE(RtcEventBackgroundDecision::consider(e,f.val,1)->requirements(NativeReadoutCoordinate::r).background_adequacy_and_coverage_required);
}

TEST(rtc_event_background, zero_pre_residual_scale_has_no_floor_or_white_noise_substitute) {
    Input in;
    in.step();
    // r need not have a candidate to receive coordinate-local background facts.
    in.r.setConstant(2);
    Fixture f(in);
    const auto e = f.learn();
    ASSERT_TRUE(e->coordinates()[0].available());
    EXPECT_EQ(e->coordinates()[1].pre_scale_fit.cause, RtcEventFitCause::zero_scale);
    EXPECT_TRUE(std::isnan(e->coordinates()[1].pre_scale_fit.scale));
    EXPECT_FALSE(e->coordinates()[1].available());
    const auto d = RtcEventBackgroundDecision::consider(e,f.val,1);
    EXPECT_TRUE(d->requirements(NativeReadoutCoordinate::x).candidate_block_pair_screening_exclusion_required);
    EXPECT_EQ(d->screening_decision_handle()->evidence_handle(), f.spikes);
}

TEST(rtc_event_background, source_protection_covers_exclusion_beyond_seed_edge) {
    Input in;
    in.step();
    Fixture f(in, RtcSpikeProtection::outside_source,
        {{0,0,604,605,RtcSpikeProtection::protected_source}, {0,0,605,606,RtcSpikeProtection::unavailable}});
    EXPECT_EQ(f.spikes->candidates()[f.candidate()].protection, RtcSpikeProtection::outside_source);
    const auto e = f.learn();
    const auto r = RtcEventBackgroundDecision::consider(e,f.val,1)->requirements(NativeReadoutCoordinate::x);
    EXPECT_TRUE(r.protected_optical_assessment_required);
    EXPECT_TRUE(r.source_protection_unavailable);
    EXPECT_TRUE(r.extent_containment_and_recovery_required);
}

TEST(rtc_event_background, consider_rejects_foreign_and_stale_snapshots) {
    Input in;
    in.step();
    Fixture f(in), foreign(in);
    const auto e = f.learn();
    EXPECT_THROW(RtcEventBackgroundDecision::consider(e,foreign.val,1),std::invalid_argument);
    EXPECT_THROW(RtcEventBackgroundDecision::consider(e,ValSnapshot::initial(f.parent),1),std::invalid_argument);
    EXPECT_THROW(RtcEventBackgroundDecision::consider(e,f.val,0),std::invalid_argument);
    EXPECT_THROW(learn_rtc_event_background(nullptr,{},1),std::invalid_argument);
    EXPECT_THROW(learn_rtc_event_background(f.spikes,{f.spikes->candidates().size(),594,607},1),std::invalid_argument);
    EXPECT_THROW(learn_rtc_event_background(f.spikes,e->request(),0),std::invalid_argument);
}

TEST(rtc_event_background, numerical_failures_are_typed_and_never_zero_filled) {
    Eigen::MatrixXd design = Eigen::MatrixXd::Ones(64,4);
    Eigen::VectorXd values = Eigen::VectorXd::LinSpaced(64,0,1);
    auto result = rtc_event_background_detail::fit(design,values);
    EXPECT_EQ(result.cause,RtcEventFitCause::rank_deficient);
    EXPECT_TRUE(std::isnan(result.offset));
    design(0,0) = std::numeric_limits<double>::infinity();
    EXPECT_EQ(rtc_event_background_detail::fit(design,values).cause,RtcEventFitCause::nonfinite);
}

TEST(rtc_event_background, bounded_synthetic_time_and_scratch_witness) {
    Input in(16384);
    in.step();
    Fixture f(in);
    const auto start = std::chrono::steady_clock::now();
    const auto e = f.learn();
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    ASSERT_TRUE(e->coordinates()[0].available());
    EXPECT_LE(e->peak_scratch_rows(), 488U);
    std::cout << "RTC_EVENT_BACKGROUND_MEASUREMENT rows=" << in.times.size()
              << " scratch_rows=" << e->peak_scratch_rows() << " seconds=" << seconds
              << " logical_summary_bytes=" << sizeof(RtcEventBackgroundEvidence) << '\n';
}
} // namespace
