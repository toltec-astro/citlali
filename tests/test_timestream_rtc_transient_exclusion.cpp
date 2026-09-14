#include <citlali/core/pipeline/timestream_rtc_transient_exclusion.h>
#include "timestream_rtc_reassessment_test_support.h"
#include <gtest/gtest.h>
#include <bit>
#include <chrono>
#include <iostream>

namespace {
using namespace citlali::pipeline;
using citlali::test::rtc_reassessment::Input;
using citlali::test::rtc_reassessment::Fixture;

struct Trial : Fixture {
    std::shared_ptr<const RtcJumpExclusionPlan> jumps;
    explicit Trial(const Input &in, RtcSpikeProtection protection = RtcSpikeProtection::outside_source,
                   bool bind_scans = true) : Fixture(in, protection) {
        const auto audit = RtcJumpSupportEvidence::learn(transition, 9);
        const auto refit = RtcJumpRefitEvidence::learn(RtcJumpRefitRequest::consider(audit, val, 10), 11);
        const auto remeasurement = RtcJumpReassessmentEvidence::learn(
            RtcJumpRemeasureRequest::consider(refit, val, 12), 13);
        const auto admission = RtcJumpAdmissionDecision::consider(
            RtcJumpReassessmentDecision::consider(remeasurement, val, 14), val, 15);
        std::shared_ptr<const RtcExistingScanBinding> scans;
        if (bind_scans) scans = RtcExistingScanBinding::admit(parent, "fixture-existing-scan-generation",
            "fixture-native-relation", "fixture-conservative-support",
            RtcExistingScanSupportState::conservative_native_support_bound, {{7, spikes->input_handle()->span(0)}});
        jumps = RtcJumpExclusionPlan::consider(admission, scans, val, 16);
    }
    auto plan() const {
        return RtcTransientExclusionPlan::consider(review->original_screening_handle(), jumps, val, 17);
    }
    auto apply(const std::shared_ptr<const RtcTransientExclusionPlan> &p) const {
        const std::array partitions{spikes->input_handle()};
        return RtcTransientExclusionResult::apply(p, spikes->input_handle(), val, partitions);
    }
};

TEST(rtc_transient_exclusion, one_coordinate_screening_failure_excludes_pair_without_inventing_event) {
    Input in; in.r.col(0).setConstant(-3.0); Trial t(in);
    auto p = t.plan(); auto out = t.apply(p);
    const auto c = p->causes(0, 500, 0);
    EXPECT_EQ(c.screening[0], RtcSpikeNoiseCause::none);
    EXPECT_EQ(c.screening[1], RtcSpikeNoiseCause::zero_scale);
    EXPECT_FALSE(c.accepted_jump);
    EXPECT_EQ(p->counts().screening_pair_cells, in.times.size());
    EXPECT_EQ(p->counts().union_pair_cells, in.times.size());
    EXPECT_EQ(p->counts().jump_pair_cells, 0);
    for (auto coord : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r}) {
        EXPECT_FALSE(out->value_if_retained(coord, 0, 500, 0));
        EXPECT_TRUE(out->value_if_retained(coord, 0, 500, 1));
    }
    EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::r, 500, 0), -3.0);
}

TEST(rtc_transient_exclusion, coordinate_causes_and_union_count_remain_distinct) {
    Input in; in.x.col(0).setConstant(10); in.r.col(0).setConstant(-3); Trial t(in);
    auto p = t.plan(); const auto c = p->causes(0, 500, 0);
    EXPECT_EQ(c.screening[0], RtcSpikeNoiseCause::zero_scale);
    EXPECT_EQ(c.screening[1], RtcSpikeNoiseCause::zero_scale);
    EXPECT_EQ(p->counts().union_pair_cells, in.times.size());
}

TEST(rtc_transient_exclusion, short_run_failure_uses_exact_support_and_needs_no_scan_binding) {
    Input in(100); Trial t(in, RtcSpikeProtection::unavailable, false);
    auto p = t.plan(); EXPECT_EQ(p->counts().union_pair_cells, 300);
    EXPECT_EQ(p->causes(0, 100, 0).screening[0], RtcSpikeNoiseCause::insufficient_population);
    EXPECT_TRUE(p->excludes(0, 199, 2));
    EXPECT_THROW(p->excludes(0, 200, 2), std::out_of_range);
}

TEST(rtc_transient_exclusion, failed_block_does_not_expand_to_other_blocks_or_detectors) {
    Input in(3000);
    for (std::size_t i = 0; i < 1221; ++i) in.r(i, 0) = -3;
    Trial t(in); auto p = t.plan();
    ASSERT_EQ(p->detectors().size(), 1);
    const auto b = p->detectors()[0].screening[0];
    EXPECT_EQ(b.rows.first, 100); EXPECT_EQ(b.rows.past_last, 1321);
    EXPECT_TRUE(p->excludes(0, 1320, 0)); EXPECT_FALSE(p->excludes(0, 1321, 0));
    EXPECT_FALSE(p->excludes(0, 500, 1)); EXPECT_FALSE(p->excludes(0, 2000, 0));
}

TEST(rtc_transient_exclusion, overlaps_count_once_while_preserving_both_causes) {
    Input in(3000); in.step(500, 4, 0);
    for (std::size_t i = 2442; i < in.times.size(); ++i) in.r(i, 0) = -3;
    Trial t(in); auto p = t.plan();
    ASSERT_EQ(t.jumps->excluded_pair_cells(), in.times.size());
    ASSERT_GT(p->counts().screening_pair_cells, 0);
    const auto c = p->causes(0, 2800, 0);
    EXPECT_TRUE(c.accepted_jump); EXPECT_EQ(c.screening[1], RtcSpikeNoiseCause::zero_scale);
    EXPECT_EQ(p->counts().overlap_pair_cells, p->counts().screening_pair_cells);
    EXPECT_EQ(p->counts().union_pair_cells, in.times.size());
    EXPECT_EQ(p->counts().union_pair_cells,
        p->counts().screening_pair_cells + p->counts().jump_pair_cells - p->counts().overlap_pair_cells);
}

TEST(rtc_transient_exclusion, jump_only_composition_is_identical_to_reviewed_jump_operation) {
    Input in; in.step(); Trial t(in); auto p = t.plan(); auto out = t.apply(p);
    EXPECT_EQ(p->counts().screening_pair_cells, 0);
    EXPECT_EQ(p->counts().jump_pair_cells, t.jumps->excluded_pair_cells());
    for (auto row = 100; row < 1200; ++row) for (std::uint32_t d = 0; d < 3; ++d) {
        EXPECT_EQ(p->excludes(0, row, d), t.jumps->excludes(0, row, d));
        if (!p->excludes(0, row, d)) for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r})
            EXPECT_EQ(std::bit_cast<std::uint64_t>(*out->value_if_retained(c, 0, row, d)),
                      std::bit_cast<std::uint64_t>(t.parent->network(0).value(c, row, d)));
    }
}

TEST(rtc_transient_exclusion, candidate_spikes_and_unknown_or_protected_jumps_are_not_promoted) {
    for (auto protection : {RtcSpikeProtection::outside_source, RtcSpikeProtection::protected_source,
                            RtcSpikeProtection::unavailable}) {
        Input in; in.spike(); Trial t(in, protection); auto p = t.plan();
        EXPECT_GT(t.spikes->candidates().size(), 0);
        EXPECT_EQ(p->counts().union_pair_cells, 0);
        EXPECT_EQ(p->screening_handle()->evidence_handle(), t.spikes);
        EXPECT_EQ(p->jump_plan_handle()->admission_handle()->assessment().spike_handle(), t.spikes);
    }
    for (auto protection : {RtcSpikeProtection::protected_source, RtcSpikeProtection::unavailable}) {
        Input in; in.step(); Trial t(in, protection); EXPECT_EQ(t.plan()->counts().union_pair_cells, 0);
    }
}

TEST(rtc_transient_exclusion, successful_source_samples_remain_bitwise_unchanged) {
    std::size_t comparisons = 0;
    for (double amplitude : {0.0, 3.0, 30.0, 300.0}) {
        Input in;
        for (std::size_t i = 0; i < in.times.size(); ++i) {
            const double t = (static_cast<double>(i) - 500) / 30;
            in.x(i, 0) += amplitude * std::exp(-0.5*t*t);
            in.r(i, 0) += amplitude * 0.2 * std::exp(-0.5*t*t);
        }
        Trial t(in, RtcSpikeProtection::protected_source); auto p = t.plan(); auto out = t.apply(p);
        EXPECT_EQ(p->counts().union_pair_cells, 0);
        for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r})
            for (TimestreamNativeRow row = 100; row < 1200; ++row) {
                ASSERT_TRUE(out->value_if_retained(c, 0, row, 0));
                EXPECT_EQ(std::bit_cast<std::uint64_t>(*out->value_if_retained(c, 0, row, 0)),
                          std::bit_cast<std::uint64_t>(t.parent->network(0).value(c, row, 0)));
                ++comparisons;
            }
    }
    EXPECT_EQ(comparisons, 8800);
}

TEST(rtc_transient_exclusion, screening_failure_is_not_cleared_by_source_protection) {
    for (auto protection : {RtcSpikeProtection::protected_source, RtcSpikeProtection::unavailable}) {
        Input in; in.r.col(0).setConstant(-3); Trial t(in, protection);
        EXPECT_TRUE(t.plan()->excludes(0, 500, 0));
        EXPECT_FALSE(t.plan()->causes(0, 500, 0).accepted_jump);
    }
}

TEST(rtc_transient_exclusion, exact_learn_evidence_and_both_decisions_are_required) {
    Input in; Trial t(in);
    const auto second = learn_rtc_spike_candidates(t.spikes->input_handle(), t.val,
        t.spikes->protection_handle(), 101);
    auto screening = RtcSpikeLearningDecision::consider(second, t.val, 102);
    EXPECT_THROW(RtcTransientExclusionPlan::consider(screening, t.jumps, t.val, 17), std::invalid_argument);
    EXPECT_THROW(RtcTransientExclusionPlan::consider(nullptr, t.jumps, t.val, 17), std::invalid_argument);
    EXPECT_THROW(RtcTransientExclusionPlan::consider(t.review->original_screening_handle(), nullptr, t.val, 17), std::invalid_argument);
    EXPECT_THROW(RtcTransientExclusionPlan::consider(t.review->original_screening_handle(), t.jumps, t.val, 0), std::invalid_argument);
}

TEST(rtc_transient_exclusion, stale_snapshot_and_rebound_input_are_rejected) {
    Input in; Trial t(in); auto p = t.plan(); const std::array parts{t.spikes->input_handle()};
    auto other_val = ValSnapshot::initial(t.parent);
    EXPECT_THROW(RtcTransientExclusionPlan::consider(t.review->original_screening_handle(), t.jumps, other_val, 17), StaleRtcValGeneration);
    EXPECT_THROW(RtcTransientExclusionResult::apply(p, t.spikes->input_handle(), other_val, parts), StaleRtcValGeneration);
    EXPECT_THROW(RtcTransientExclusionResult::apply(p, NativePairedReadoutView::full(t.parent), t.val, parts), std::invalid_argument);
}

TEST(rtc_transient_exclusion, partitions_do_not_change_treatment_or_repeat_learning) {
    Input in; in.r.col(0).setConstant(-3); Trial t(in); auto p = t.plan();
    std::vector<std::shared_ptr<const NativePairedReadoutView>> parts{
        NativePairedReadoutView::admit(t.parent, {{0, 100, 450}}),
        NativePairedReadoutView::admit(t.parent, {{0, 450, 1200}})};
    auto out = RtcTransientExclusionResult::apply(p, t.spikes->input_handle(), t.val, parts);
    EXPECT_EQ(out->realized_counts(), t.apply(p)->realized_counts());
    EXPECT_EQ(out->plan_handle(), p); EXPECT_EQ(p->screening_handle()->evidence_handle(), t.spikes);
    std::reverse(parts.begin(), parts.end());
    EXPECT_THROW(RtcTransientExclusionResult::apply(p, t.spikes->input_handle(), t.val, parts), std::invalid_argument);
    parts.pop_back();
    EXPECT_THROW(RtcTransientExclusionResult::apply(p, t.spikes->input_handle(), t.val, parts), std::invalid_argument);
}

TEST(rtc_transient_exclusion, invalid_payloads_are_unavailable_without_inventing_treatment) {
    Input in; in.x(300, 1) = NAN;
    in.xs[3*300 + 1] = NativeReadoutCoordinateState::measured(true, false, true, false);
    Trial t(in); auto p = t.plan(); auto out = t.apply(p);
    EXPECT_FALSE(p->excludes(0, 400, 1));
    EXPECT_FALSE(out->value_if_retained(NativeReadoutCoordinate::x, 0, 400, 1));
    EXPECT_FALSE(out->value_if_retained(NativeReadoutCoordinate::r, 0, 400, 1));
    EXPECT_THROW(p->causes(0, 99, 0), std::out_of_range);
    EXPECT_THROW(p->causes(0, 100, 3), std::out_of_range);
    EXPECT_THROW(out->value_if_retained(static_cast<NativeReadoutCoordinate>(9), 0, 100, 0), std::invalid_argument);
}

TEST(rtc_transient_exclusion, repeated_plans_are_deterministic_and_sparse) {
    Input in(3000); in.r.col(0).setConstant(-3); Trial t(in); const auto p = t.plan();
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        const auto next = t.plan(); auto result = t.apply(next);
        EXPECT_EQ(result->realized_counts(), p->counts());
        EXPECT_EQ(next->logical_owned_bytes(), p->logical_owned_bytes());
        EXPECT_EQ(result->owned_numeric_bytes, 0); EXPECT_EQ(result->owned_state_plane_bytes, 0);
    }
    EXPECT_LT(p->logical_owned_bytes(), 1024);
    std::cout << "transient_plan_apply_1000_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()
              << " logical_owned_bytes=" << p->logical_owned_bytes() << '\n';
}

TEST(rtc_transient_exclusion, physical_gap_starts_new_screening_run_without_spreading_exclusion) {
    Input in(3000);
    for (std::size_t i = 0; i < 1500; ++i) in.r(i, 0) = -3;
    for (std::size_t i = 1500; i < in.times.size(); ++i) {
        in.times[i] += 30; in.counters[i] += 10;
    }
    Trial t(in); auto p = t.plan();
    ASSERT_EQ(t.parent->network(0).occurrence_axis().contiguous_runs().size(), 2);
    EXPECT_EQ(p->counts().union_pair_cells, 1500);
    EXPECT_TRUE(p->excludes(0, 1599, 0)); EXPECT_FALSE(p->excludes(0, 1600, 0));
    EXPECT_FALSE(p->excludes(0, 3000, 0));
    EXPECT_EQ(p->screening_handle()->evidence_handle()->input_handle()->parent_handle(), t.parent);
}
} // namespace
