#include <citlali/core/pipeline/timestream_rtc_jump_exclusion.h>
#include "timestream_rtc_reassessment_test_support.h"
#include <gtest/gtest.h>
#include <bit>
#include <chrono>
#include <iostream>

namespace {
using namespace citlali::pipeline;
using citlali::test::rtc_reassessment::Input;

struct Trial {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const NativePairedReadoutView> input;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcJumpAdmissionDecision> admission;
    explicit Trial(const Input &data, RtcSpikeProtection protection = RtcSpikeProtection::outside_source,
                   std::vector<RtcSpikeProtectionRegion> regions = {}) {
        parent = data.freeze(); input = NativePairedReadoutView::full(parent); val = ValSnapshot::initial(parent);
        auto spikes = learn_rtc_spike_candidates(input, val,
            RtcSpikeSourceProtection::admit(parent, "analytical-source-authority", protection, std::move(regions)), 1);
        std::vector<RtcEventPeerEligibility> peers;
        for (std::uint32_t d = 0; d < 3; ++d)
            peers.push_back({0, d, parent->network(0).detectors()[d].detector_occurrence_id, true});
        auto assessment = learn_rtc_event_assessment(spikes, RtcEventPeerPopulation::admit(spikes, "fixture", peers), 2);
        auto review = RtcEventAssessmentDecision::consider(assessment, val, 3);
        auto amplitude = RtcJumpAmplitudeDecision::consider(review, val, 4);
        auto shorter = RtcJumpConsistencyEvidence::learn(amplitude, 5);
        auto consistency = RtcJumpConsistencyDecision::consider(shorter, val, 6);
        auto transition_request = RtcJumpTransitionRequest::consider(consistency, val, 7);
        auto transition = RtcJumpTransitionEvidence::learn(transition_request, 8);
        auto audit = RtcJumpSupportEvidence::learn(transition, 9);
        auto refit_request = RtcJumpRefitRequest::consider(audit, val, 10);
        auto refit = RtcJumpRefitEvidence::learn(refit_request, 11);
        auto request = RtcJumpRemeasureRequest::consider(refit, val, 12);
        auto evidence = RtcJumpReassessmentEvidence::learn(request, 13);
        admission = RtcJumpAdmissionDecision::consider(RtcJumpReassessmentDecision::consider(evidence, val, 14), val, 15);
    }
    auto scans(std::vector<RtcExistingScanNativeSupport> supports = {}) const {
        if (supports.empty()) supports.push_back({71, input->span(0)});
        return RtcExistingScanBinding::admit(parent, "analytical-existing-PCA-generation",
            "analytical-exact-native-scan-relation", "analytical-exact-native-cell-support", RtcExistingScanSupportState::conservative_native_support_bound, std::move(supports));
    }
    auto plan(std::shared_ptr<const RtcExistingScanBinding> binding) const {
        return RtcJumpExclusionPlan::consider(admission, std::move(binding), val, 16);
    }
    auto apply(std::shared_ptr<const RtcJumpExclusionPlan> p) const {
        const std::array partitions{input};
        return RtcJumpExclusionResult::apply(std::move(p), input, val, partitions);
    }
    std::vector<RtcJumpAdmissionGroup> admitted() const {
        std::vector<RtcJumpAdmissionGroup> result;
        for (const auto &g : admission->groups()) if (g.admitted()) result.push_back(g);
        return result;
    }
};

TEST(rtc_jump_exclusion, x_origin_selects_paired_scan_and_preserves_original_payload) {
    Input data; data.step(500, 4, 0); Trial t(data);
    auto groups = t.admitted(); ASSERT_EQ(groups.size(), 1);
    EXPECT_TRUE(groups[0].coordinates[0].admitted()); EXPECT_FALSE(groups[0].coordinates[1].admitted());
    auto p = t.plan(t.scans()); auto result = t.apply(p);
    ASSERT_EQ(p->detectors().size(), 1); EXPECT_FALSE(p->detectors()[0].whole_observation);
    EXPECT_EQ(p->detectors()[0].original_groups.size(), 1);
    EXPECT_EQ(result->realized_excluded_pair_cells(), data.times.size());
    for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r}) {
        EXPECT_FALSE(result->value_if_retained(c, 0, 600, 0));
        ASSERT_TRUE(result->value_if_retained(c, 0, 600, 1));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(*result->value_if_retained(c, 0, 600, 1)),
                  std::bit_cast<std::uint64_t>(t.parent->network(0).value(c, 600, 1)));
    }
    EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x, 600, 0), data.x(500, 0));
    EXPECT_EQ(result->owned_numeric_bytes, 0); EXPECT_EQ(result->owned_state_plane_bytes, 0);
}

TEST(rtc_jump_exclusion, r_origin_can_exclude_x_without_cross_coordinate_correction) {
    Input data; data.step(500, 0, -3); Trial t(data);
    auto groups = t.admitted(); ASSERT_EQ(groups.size(), 1);
    EXPECT_FALSE(groups[0].coordinates[0].admitted()); EXPECT_TRUE(groups[0].coordinates[1].admitted());
    auto result = t.apply(t.plan(t.scans()));
    EXPECT_FALSE(result->value_if_retained(NativeReadoutCoordinate::x, 0, 600, 0));
    EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::r, 600, 0), data.r(500, 0));
}

TEST(rtc_jump_exclusion, protected_and_unknown_source_measurements_cannot_authorize_rejection) {
    Input data; data.step();
    for (auto protection : {RtcSpikeProtection::protected_source, RtcSpikeProtection::unavailable}) {
        Trial t(data, protection); EXPECT_TRUE(t.admitted().empty());
        bool cause_present = false;
        for (const auto &g : t.admission->groups()) for (const auto &c : g.coordinates)
            cause_present |= c.cause == (protection == RtcSpikeProtection::unavailable ?
                RtcJumpAdmissionCause::source_protection_unavailable : RtcJumpAdmissionCause::protected_optical_test_unavailable);
        EXPECT_TRUE(cause_present);
        auto result = t.apply(t.plan(nullptr));
        EXPECT_EQ(result->realized_excluded_pair_cells(), 0);
        EXPECT_DOUBLE_EQ(*result->value_if_retained(NativeReadoutCoordinate::x, 0, 600, 0), data.x(500, 0));
    }
}

TEST(rtc_jump_exclusion, source_in_fit_flank_is_not_cleared_by_outside_source_seed) {
    Input data; data.step();
    Trial t(data, RtcSpikeProtection::outside_source, {{0, 0, 440, 445, RtcSpikeProtection::protected_source}});
    EXPECT_TRUE(t.admitted().empty());
    EXPECT_FALSE(t.apply(t.plan(nullptr))->excluded(0, 600, 0));
}

TEST(rtc_jump_exclusion, isolated_spike_and_no_jump_controls_do_not_become_persistent_jumps) {
    for (bool spike : {false, true}) {
        Input data; if (spike) data.spike(); Trial t(data);
        EXPECT_TRUE(t.admitted().empty());
        EXPECT_EQ(t.apply(t.plan(nullptr))->realized_excluded_pair_cells(), 0);
    }
}

TEST(rtc_jump_exclusion, unavailable_recovery_never_substitutes_for_no_return) {
    Input data; data.step();
    data.xs[3*710] = NativeReadoutCoordinateState::measured(true, false, true, true);
    data.rs[3*710] = NativeReadoutCoordinateState::measured(true, false, true, true);
    Trial t(data); EXPECT_TRUE(t.admitted().empty());
    bool recovery_missing = false;
    for (const auto &g : t.admission->groups()) for (const auto &c : g.coordinates)
        recovery_missing |= c.cause == RtcJumpAdmissionCause::recovery_unavailable;
    EXPECT_TRUE(recovery_missing);
}

Input repeated(std::size_t jumps) {
    Input data(6000);
    for (Eigen::Index row = 0; row < data.x.rows(); ++row) for (Eigen::Index d = 0; d < 3; ++d) {
        data.x(row,d) = std::array{-.1,0.,.1,0.,0.,-.1,.1}[(row+d)%7];
        data.r(row,d) = 2*data.x(row,d);
    }
    for (std::size_t j = 0; j < jumps; ++j) data.step(1000+1500*j, 4, -3);
    return data;
}
TEST(rtc_jump_exclusion, three_original_groups_count_xr_once_and_exclude_only_that_detector_observation) {
    for (std::size_t n : {2U, 3U}) {
        Trial t(repeated(n)); ASSERT_EQ(t.admitted().size(), n);
        auto p = t.plan(t.scans({{1,{0,100,2100}},{2,{0,2100,4100}},{3,{0,4100,6100}}}));
        ASSERT_EQ(p->detectors().size(), 1);
        EXPECT_EQ(p->detectors()[0].original_groups.size(), n);
        EXPECT_EQ(p->detectors()[0].whole_observation, n == 3);
        EXPECT_EQ(p->excluded_pair_cells(), n == 3 ? 6000 : 4000);
        auto result = t.apply(p);
        EXPECT_EQ(result->excluded(0, 6000, 0), n == 3);
        EXPECT_FALSE(result->excluded(0, 6000, 1));
    }
}

TEST(rtc_jump_exclusion, every_intersected_scan_is_selected_and_endpoint_contact_is_not_overlap) {
    Input data; data.step(500,4,0); Trial t(data);
    auto g = t.admitted(); ASSERT_EQ(g.size(),1);
    auto a = g[0].coordinates[0].affected; ASSERT_GT(a.past_last-a.first,1);
    auto p = t.plan(t.scans({{1,{0,100,a.first}},{2,{0,a.first,a.first+1}},
                           {3,{0,a.first+1,a.past_last}},{4,{0,a.past_last,1200}}}));
    ASSERT_EQ(p->detectors().size(),1);
    EXPECT_EQ(p->detectors()[0].scans,(std::vector<std::uint64_t>{2,3}));
    EXPECT_FALSE(p->excludes(0,a.first-1,0)); EXPECT_TRUE(p->excludes(0,a.first,0));
    EXPECT_FALSE(p->excludes(0,a.past_last,0));
    EXPECT_EQ(p->excluded_pair_cells(),a.past_last-a.first);
}

TEST(rtc_jump_exclusion, selecting_scan_includes_its_disjoint_support_and_unions_overlapping_scans) {
    Input data; data.step(500,4,0); Trial t(data);
    auto p = t.plan(t.scans({{7,{0,400,700}},{7,{0,1000,1100}},{8,{0,500,800}}}));
    EXPECT_EQ(p->excluded_pair_cells(),500);
    EXPECT_TRUE(p->excludes(0,1050,0)); EXPECT_FALSE(p->excludes(0,900,0));
}

TEST(rtc_jump_exclusion, missing_scan_relation_cannot_publish_a_partial_plan) {
    Input data; data.step(); Trial t(data);
    EXPECT_THROW(t.plan(nullptr),RtcJumpScanBindingUnavailable);
    EXPECT_THROW(t.plan(t.scans({{1,{0,100,200}}})),RtcJumpScanBindingUnavailable);
}

TEST(rtc_jump_exclusion, exact_parent_snapshot_and_nonzero_decision_identity_are_required) {
    Input data; data.step(); Trial t(data), other(data);
    EXPECT_THROW(t.plan(other.scans()),std::invalid_argument);
    EXPECT_THROW(RtcJumpAdmissionDecision::consider(t.admission->parent_handle(), other.val, 1),std::invalid_argument);
    EXPECT_THROW(RtcJumpAdmissionDecision::consider(t.admission->parent_handle(), t.val, 0),std::invalid_argument);
    EXPECT_THROW(RtcJumpExclusionPlan::consider(t.admission,t.scans(),ValSnapshot::initial(t.parent),1),std::invalid_argument);
    EXPECT_THROW(RtcJumpExclusionPlan::consider(t.admission,t.scans(),t.val,0),std::invalid_argument);
    const std::array partitions{t.input}; auto p=t.plan(t.scans());
    EXPECT_THROW(RtcJumpExclusionResult::apply(p,other.input,t.val,partitions),std::invalid_argument);
    EXPECT_THROW(RtcJumpExclusionResult::apply(p,t.input,ValSnapshot::initial(t.parent),partitions),StaleRtcValGeneration);
}

TEST(rtc_jump_exclusion, scan_binding_rejects_missing_identity_duplicates_and_out_of_parent_support) {
    Input data; Trial t(data);
    EXPECT_THROW(RtcExistingScanBinding::admit(t.parent,"","relation","uncertainty",RtcExistingScanSupportState::conservative_native_support_bound,{{1,{0,100,200}}}),std::invalid_argument);
    EXPECT_THROW(RtcExistingScanBinding::admit(t.parent,"generation","relation","",RtcExistingScanSupportState::conservative_native_support_bound,{{1,{0,100,200}}}),std::invalid_argument);
    EXPECT_THROW(RtcExistingScanBinding::admit(t.parent,"generation","relation","uncertainty",RtcExistingScanSupportState::unavailable,{{1,{0,100,200}}}),RtcJumpScanBindingUnavailable);
    EXPECT_THROW(t.scans({{1,{0,99,200}}}),std::invalid_argument);
    EXPECT_THROW(t.scans({{1,{0,100,200}},{1,{0,199,300}}}),std::invalid_argument);
}

TEST(rtc_jump_exclusion, apply_schedule_and_partition_independence) {
    Input data; data.step(); Trial t(data); auto p=t.plan(t.scans());
    const auto left=NativePairedReadoutView::admit(t.parent,{{0,100,650}});
    const auto right=NativePairedReadoutView::admit(t.parent,{{0,650,1200}});
    const std::array parts{left,right};
    auto partitioned=RtcJumpExclusionResult::apply(p,t.input,t.val,parts);
    auto whole=t.apply(p);
    EXPECT_EQ(whole->realized_excluded_pair_cells(),partitioned->realized_excluded_pair_cells());
    EXPECT_EQ(whole->plan_handle(),partitioned->plan_handle());
    const std::array incomplete{left};
    EXPECT_THROW(RtcJumpExclusionResult::apply(p,t.input,t.val,incomplete),std::invalid_argument);
    const std::array reversed{right,left};
    EXPECT_THROW(RtcJumpExclusionResult::apply(p,t.input,t.val,reversed),std::invalid_argument);
}

TEST(rtc_jump_exclusion, invalid_producer_payload_remains_separate_and_unread) {
    Input data; data.xs[3*500+1]=NativeReadoutCoordinateState::measured(true,false,true,false);
    data.x(500,1)=NAN; Trial t(data); auto result=t.apply(t.plan(nullptr));
    EXPECT_FALSE(result->excluded(0,600,1));
    EXPECT_FALSE(result->value_if_retained(NativeReadoutCoordinate::x,0,600,1));
    EXPECT_FALSE(result->value_if_retained(NativeReadoutCoordinate::r,0,600,1));
    EXPECT_THROW(result->excluded(0,99,0),std::out_of_range);
    EXPECT_THROW(result->excluded(0,600,3),std::out_of_range);
}

TEST(rtc_jump_exclusion, plan_and_apply_are_deterministic_and_sparse) {
    Trial t(repeated(3)); const auto scans=t.scans();
    const auto start=std::chrono::steady_clock::now();
    for (int n=0;n<1000;++n) {
        auto p=t.plan(scans); auto applied=t.apply(p);
        EXPECT_EQ(applied->realized_excluded_pair_cells(),6000);
        ASSERT_EQ(p->detectors().size(),1); EXPECT_EQ(p->detectors()[0].original_groups,t.plan(scans)->detectors()[0].original_groups);
        EXPECT_LT(p->logical_owned_bytes(),1024);
    }
    std::cout << "jump_plan_apply_1000_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << " admission_owned_bytes=" << t.admission->logical_owned_bytes() << '\n';
}

TEST(rtc_jump_exclusion, source_brightness_does_not_select_a_clean_scan_for_rejection) {
    // Timestream-only injection: rerun the entire Learn/Consider chain for
    // each brightness, rather than checking only a previously frozen mask.
    for (double amplitude : {0., 3., 30., 300.}) for (bool jump : {false, true}) {
        auto data = repeated(0);
        if (jump) data.step(1000,4,0);
        for (std::size_t i=0;i<data.times.size();++i) {
            const double t=(static_cast<double>(i)-3100)*.008192;
            const double optical=amplitude*std::exp(-.5*t*t/(.2*.2));
            data.x(i,0)+=optical; data.r(i,0)+=.3*optical;
        }
        Trial t(data,RtcSpikeProtection::outside_source,
                {{0,0,2100,6100,RtcSpikeProtection::protected_source}});
        ASSERT_EQ(t.admitted().size(),jump ? 1 : 0);
        auto result=t.apply(t.plan(t.scans({{1,{0,100,2100}},{2,{0,2100,6100}}})));
        EXPECT_EQ(result->realized_excluded_pair_cells(),jump ? 2000 : 0);
        for (auto c : {NativeReadoutCoordinate::x,NativeReadoutCoordinate::r})
            for (TimestreamNativeRow row=2800;row<3600;++row) {
                auto value=result->value_if_retained(c,0,row,0);ASSERT_TRUE(value);
                EXPECT_EQ(std::bit_cast<std::uint64_t>(*value),
                    std::bit_cast<std::uint64_t>(t.parent->network(0).value(c,row,0)));
            }
    }
}
} // namespace
