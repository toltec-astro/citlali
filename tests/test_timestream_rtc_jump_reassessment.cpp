#include "timestream_rtc_reassessment_test_support.h"
#include <gtest/gtest.h>
namespace {
using namespace citlali::pipeline;
using namespace citlali::test::rtc_reassessment;
struct Reassessment {
    std::shared_ptr<const RtcJumpSupportEvidence> audit;
    std::shared_ptr<const RtcJumpRefitRequest> request;
    std::shared_ptr<const RtcJumpRefitEvidence> refit;
    std::shared_ptr<const RtcJumpRemeasureRequest> measure;
    std::shared_ptr<const RtcJumpReassessmentEvidence> evidence;
    std::shared_ptr<const RtcJumpReassessmentDecision> decision;
    explicit Reassessment(const Fixture &f) {
        audit=RtcJumpSupportEvidence::learn(f.transition,9);
        request=RtcJumpRefitRequest::consider(audit,f.val,10);
        refit=RtcJumpRefitEvidence::learn(request,11);
        measure=RtcJumpRemeasureRequest::consider(refit,f.val,12);
        evidence=RtcJumpReassessmentEvidence::learn(measure,13);
        decision=RtcJumpReassessmentDecision::consider(evidence,f.val,14);
    }
};
TEST(rtc_jump_reassessment, exact_rows_exclude_neighbor_masks_and_coordinate_invalid_samples) {
    Input in;in.step();in.xs[3*300]=NativeReadoutCoordinateState::measured(true,false,true,false);in.x(300,0)=NAN;Fixture f(in);auto event=f.assessment->events()[f.event_at(500)];
    const auto &spikes=*f.spikes;
    auto rows=rtc_jump_reassessment_detail::admitted(spikes,event,0,event.background[0].support,2);
    for(std::size_t side=0;side<2;++side) {
        EXPECT_EQ(rtc_jump_reassessment_detail::size(rows[side]),event.background[0].support[side].usable);
        for(auto r:rows[side]) for(auto row=r.first;row<r.past_last;++row)
            EXPECT_FALSE(rtc_event_assessment_detail::contains(event.neighbor_exclusions,row));
    }
    EXPECT_FALSE(rtc_event_assessment_detail::contains(rows[0],400));
    const auto partner=rtc_jump_reassessment_detail::admitted(spikes,event,1,event.background[1].support,2);
    EXPECT_TRUE(rtc_event_assessment_detail::contains(partner[0],400));
    ++event.background[0].support[0].usable;
    EXPECT_THROW(rtc_jump_reassessment_detail::admitted(spikes,event,0,event.background[0].support,2),std::invalid_argument);
}
TEST(rtc_jump_reassessment, endpoint_extension_is_not_sample_overlap) {
    RtcJumpFitRows rows{{{{1,3},{8,11}},{{20,24}}}};
    EXPECT_EQ(rtc_jump_reassessment_detail::overlap(rows,{{3,8}}),0);
    EXPECT_EQ(rtc_jump_reassessment_detail::overlap(rows,{{7,9}}),1);
    EXPECT_EQ(rtc_jump_reassessment_detail::overlap(rows,{{11,20}}),0);
}
TEST(rtc_jump_reassessment, paired_mask_union_preserves_disjoint_neighbor_exclusions) {
    RtcJumpFitRows rows{{{{0,10}},{{20,30}}}};
    auto mask=rtc_event_assessment_detail::merge({{3,5},{4,8},{22,23},{24,27}});
    const auto result=rtc_jump_reassessment_detail::subtract(rows,mask);
    EXPECT_EQ(rtc_jump_reassessment_detail::size(result[0]),5);
    EXPECT_EQ(rtc_jump_reassessment_detail::size(result[1]),6);
    EXPECT_EQ(rtc_jump_reassessment_detail::overlap(result,mask),0);
}
TEST(rtc_jump_reassessment, isolated_step_inside_exclusion_requests_no_refit) {
    Input in;in.step();Fixture f(in);Reassessment r(f);
    ASSERT_FALSE(r.audit->audits().empty());
    EXPECT_EQ(r.refit->counts().requested_groups,0);
    EXPECT_EQ(r.refit->counts().fit_calls,0);
    bool measured=false;
    for(const auto &pair:r.decision->coordinates()) for(auto cause:pair)
        measured|=cause==RtcJumpReassessmentCause::unchanged_measured;
    EXPECT_TRUE(measured);
    EXPECT_FALSE(r.decision->apply_authorized);
}
TEST(rtc_jump_reassessment, refit_freezes_scale_basis_and_uses_original_values) {
    Input in;in.step();Fixture f(in);const auto &event=f.assessment->events()[f.event_at(500)];
    auto rows=rtc_jump_reassessment_detail::admitted(*f.spikes,event,0,event.background[0].support,2);
    rows=rtc_jump_reassessment_detail::subtract(rows,{{event.trial_exclusion.past_last,event.trial_exclusion.past_last+20}});
    RtcJumpRefitCounts counts;
    const auto fit=rtc_jump_reassessment_detail::refit(*f.spikes,event,0,rows,event.background[0].pre_scale_fit,true,counts);
    ASSERT_TRUE(fit.available());
    EXPECT_DOUBLE_EQ(fit.cubic.scale,event.background[0].pre_scale_fit.scale);
    EXPECT_DOUBLE_EQ(fit.cubic_with_offset.scale,event.background[0].pre_scale_fit.scale);
    EXPECT_NEAR(fit.cubic_with_offset.offset,4,.02);
    EXPECT_EQ(counts.fit_calls,2);
    EXPECT_EQ(fit.support[1].usable,event.background[0].support[1].usable-19);
    EXPECT_DOUBLE_EQ(f.parent->network(0).value(NativeReadoutCoordinate::x,600,0),in.x(500,0));
}
TEST(rtc_jump_reassessment, reduced_support_cannot_expand_context_or_lower_order) {
    Input in;in.step();Fixture f(in);const auto &event=f.assessment->events()[f.event_at(500)];
    auto rows=rtc_jump_reassessment_detail::admitted(*f.spikes,event,0,event.background[0].support,2);
    rows[1]={{event.background[0].support[1].first_used,event.background[0].support[1].first_used+63}};
    RtcJumpRefitCounts counts;const auto fit=rtc_jump_reassessment_detail::refit(*f.spikes,event,0,rows,event.background[0].pre_scale_fit,true,counts);
    EXPECT_FALSE(fit.available());EXPECT_EQ(fit.support_cause,RtcEventFitCause::insufficient_samples);EXPECT_EQ(counts.fit_calls,0);
}
TEST(rtc_jump_reassessment, missing_frozen_scale_never_estimates_a_replacement) {
    Input in;in.step();Fixture f(in);const auto &event=f.assessment->events()[f.event_at(500)];
    auto rows=rtc_jump_reassessment_detail::admitted(*f.spikes,event,0,event.background[0].support,2);
    auto scale=event.background[0].pre_scale_fit;scale.scale=NAN;
    RtcJumpRefitCounts counts;const auto fit=rtc_jump_reassessment_detail::refit(*f.spikes,event,0,rows,scale,true,counts);
    EXPECT_FALSE(fit.available());EXPECT_EQ(fit.support_cause,RtcEventFitCause::zero_scale);EXPECT_EQ(counts.fit_calls,0);
}
TEST(rtc_jump_reassessment, requires_original_identity_chain_and_nonzero_ids) {
    Input in;in.step();Fixture f(in),other(in);Reassessment r(f);
    EXPECT_THROW(RtcJumpSupportEvidence::learn(nullptr,9),std::invalid_argument);
    EXPECT_THROW(RtcJumpSupportEvidence::learn(f.transition,0),std::invalid_argument);
    EXPECT_THROW(RtcJumpRefitRequest::consider(r.audit,other.val,10),std::invalid_argument);
    EXPECT_THROW(RtcJumpRefitEvidence::learn(nullptr,11),std::invalid_argument);
    EXPECT_THROW(RtcJumpRemeasureRequest::consider(r.refit,other.val,12),std::invalid_argument);
    EXPECT_THROW(RtcJumpReassessmentEvidence::learn(nullptr,13),std::invalid_argument);
    EXPECT_THROW(RtcJumpReassessmentDecision::consider(r.evidence,other.val,14),std::invalid_argument);
    EXPECT_EQ(r.evidence->request_handle()->refit_handle()->request_handle()->audit_handle()->parent_handle().get(),f.transition.get());
}
TEST(rtc_jump_reassessment, repeat_is_deterministic_and_never_applies_flags) {
    Input in;in.step();Fixture f(in);Reassessment a(f),b(f);
    EXPECT_EQ(a.decision->coordinates(),b.decision->coordinates());
    EXPECT_EQ(a.refit->counts().fit_calls,b.refit->counts().fit_calls);
    EXPECT_EQ(RtcJumpReassessmentPolicy::maximum_additional_passes,1);
    EXPECT_FALSE(a.evidence->hard_event_accepted);EXPECT_FALSE(a.decision->apply_authorized);
}
TEST(rtc_jump_reassessment, one_pass_refits_transition_overlap_with_frozen_scales) {
    Input in;in.step();
    for(std::size_t row=500;row<in.times.size();++row) {
        in.x(row,0)+=2*std::exp(-double(row-500)/8);
        in.r(row,0)-=2*std::exp(-double(row-500)/8);
    }
    Fixture f(in);Reassessment r(f);
    ASSERT_GT(r.refit->counts().requested_groups,0);
    EXPECT_GT(r.refit->counts().fit_calls,0);
    for(std::size_t i=0;i<r.audit->audits().size();++i) {
        if(!r.request->selections()[i].requested) continue;
        const auto &s=r.audit->audits()[i];
        for(std::size_t c=0;c<2;++c) {
            const auto &fit=r.refit->coordinates()[i][c];
            EXPECT_EQ(rtc_jump_reassessment_detail::overlap(fit.primary_rows,r.request->selections()[i].paired_mask),0);
            if(fit.primary.available()) EXPECT_DOUBLE_EQ(fit.primary.cubic_with_offset.scale,f.assessment->events()[s.event].background[c].pre_scale_fit.scale);
            if(fit.shorter.available()) EXPECT_DOUBLE_EQ(fit.shorter.cubic_with_offset.scale,f.short_evidence->coordinates()[s.event][c].pre_scale_fit.scale);
        }
    }
    EXPECT_FALSE(r.decision->hard_event_accepted);
}
} // namespace
