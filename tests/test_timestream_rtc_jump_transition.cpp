#include <citlali/core/pipeline/timestream_rtc_jump_transition.h>
#include "timestream_successor_identity_test_support.h"
#include <gtest/gtest.h>

namespace {
using namespace citlali::pipeline;
namespace support = citlali::test::timestream_successor;
struct Input {
    std::vector<double> times;
    std::vector<TimestreamPacketCounter> counters;
    NativePairedReadoutMatrix x, r;
    std::vector<NativeReadoutCoordinateState> xs, rs;
    double integration_half=.004096;
    explicit Input(std::size_t n=1100, double cadence=.008192) : times(n), counters(n), x(n,3), r(n,3),
        xs(support::valid_states(3*n)), rs(xs) {
        integration_half=cadence/2;
        for (std::size_t i=0; i<times.size(); ++i) {
            times[i]=1000+cadence*i; counters[i]=2000+i;
            for (std::size_t d=0; d<3; ++d) {
                const double t=cadence*(static_cast<double>(i)-500);
                const double noise=std::array{-0.1,0.0,0.1,0.0,0.0,-0.1,0.1}[(i+d)%7];
                x(i,d)=10+.8*t+.1*t*t-.05*t*t*t+noise; r(i,d)=-3+2*noise;
            }
        }
    }
    void step(std::size_t at=500,double dx=4,double dr=-3) {
        for (std::size_t i=at;i<times.size();++i) { x(i,0)+=dx; r(i,0)+=dr; }
    }
    void spike(std::size_t at=500,double dx=40,double dr=25) { x(at,0)+=dx; r(at,0)+=dr; }
    auto freeze() const {
        auto timing=std::make_shared<const NativeNetworkAlignment>(0,100,support::time_vector(times),counters);
        std::vector<NativePairedReadoutOccurrenceBinding> occurrences;
        for (std::size_t i=0;i<times.size();++i)
            occurrences.push_back({static_cast<std::int64_t>(i+10000),static_cast<std::int64_t>(i+20000),{times[i]-integration_half,times[i]+integration_half}});
        auto axis=std::make_shared<const NativePairedReadoutOccurrenceAxis>(timing,100,std::move(occurrences));
        std::vector<NativePairedReadoutNetwork> networks;
        networks.push_back(NativePairedReadoutNetwork::admit(axis,support::detector_axis(0,3),
            support::mapping_authority(0,"jump-test"),x,r,xs,rs));
        return support::make_observation(std::move(networks),{0});
    }
};
struct Fixture {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcSpikeEvidence> spikes;
    std::shared_ptr<const RtcEventAssessmentEvidence> assessment;
    std::shared_ptr<const RtcEventAssessmentDecision> review;
    std::shared_ptr<const RtcJumpAmplitudeDecision> amplitude;
    std::shared_ptr<const RtcJumpConsistencyEvidence> short_evidence;
    std::shared_ptr<const RtcJumpConsistencyDecision> decision;
    std::shared_ptr<const RtcJumpTransitionRequest> request;
    std::shared_ptr<const RtcJumpTransitionEvidence> transition;
    explicit Fixture(const Input &in, RtcSpikeProtection protection=RtcSpikeProtection::outside_source) {
        parent=in.freeze(); val=ValSnapshot::initial(parent);
        spikes=learn_rtc_spike_candidates(NativePairedReadoutView::full(parent),val,
            RtcSpikeSourceProtection::admit(parent,"jump-test-protection",protection),1);
        std::vector<RtcEventPeerEligibility> peers;
        for (std::uint32_t d=0;d<3;++d) peers.push_back({0,d,parent->network(0).detectors()[d].detector_occurrence_id,true});
        assessment=learn_rtc_event_assessment(spikes,RtcEventPeerPopulation::admit(spikes,"jump-test-peers",std::move(peers)),2);
        review=RtcEventAssessmentDecision::consider(assessment,val,3);
        amplitude=RtcJumpAmplitudeDecision::consider(review,val,4);
        short_evidence=RtcJumpConsistencyEvidence::learn(amplitude,5);
        decision=RtcJumpConsistencyDecision::consider(short_evidence,val,6);
        request=RtcJumpTransitionRequest::consider(decision,val,7);
        transition=RtcJumpTransitionEvidence::learn(request,8);
    }
    std::size_t event_at(std::size_t input_row) const {
        for (std::size_t i=0;i<assessment->events().size();++i)
            if (spikes->candidates()[assessment->events()[i].seed].later_row==static_cast<TimestreamNativeRow>(100+input_row)) return i;
        throw std::runtime_error("expected original event seed missing");
    }
};

RtcJumpTransition measured(const Fixture &f, const RtcAssessedEvent &event, std::size_t c=0) {
    const auto &axis=f.parent->network(event.network).occurrence_axis();
    const auto row=f.spikes->candidates()[event.seed].earlier_row;
    std::vector<std::size_t> candidates;
    for(std::size_t i=0;i<f.spikes->candidates().size();++i){
        const auto &b=f.spikes->blocks()[f.spikes->candidates()[i].noise_block_index];
        if(b.network_id==event.network && b.detector_index==event.detector)candidates.push_back(i);
    }
    std::sort(candidates.begin(),candidates.end(),[&](auto a,auto b){return f.spikes->candidates()[a].earlier_row<f.spikes->candidates()[b].earlier_row;});
    for(const auto &run:axis.contiguous_runs())
        if(row>=run.first_native_row && row<run.past_last_native_row)
            return rtc_jump_transition_detail::measure(*f.spikes,event,c,run,
                rtc_jump_transition_detail::neighbor_masks(*f.spikes,event,run,candidates));
    throw std::runtime_error("fixture run missing");
}

TEST(rtc_jump_transition, inclusive_residual_band_uses_frozen_scale) {
    EXPECT_TRUE(rtc_jump_transition_detail::agrees(14,10,1));
    EXPECT_TRUE(rtc_jump_transition_detail::agrees(6,10,1));
    EXPECT_FALSE(rtc_jump_transition_detail::agrees(std::nextafter(14.,15.),10,1));
    EXPECT_FALSE(rtc_jump_transition_detail::agrees(10,10,0));
    EXPECT_FALSE(rtc_jump_transition_detail::agrees(NAN,10,1));
    EXPECT_FALSE(rtc_jump_transition_detail::agrees(10,INFINITY,1));
    EXPECT_FALSE(rtc_jump_transition_detail::agrees(10,10,std::numeric_limits<double>::max()));
}
TEST(rtc_jump_transition, positive_and_negative_steps_have_original_cell_brackets) {
    for(double sign:{-1.,1.}) {
        Input in; in.step(500,sign*4,-sign*3); Fixture f(in);const auto i=f.event_at(500);
        const auto &axis=f.parent->network(0).occurrence_axis();
        for(std::size_t c=0;c<2;++c) {
            const auto &b=f.transition->coordinates()[i][c];ASSERT_TRUE(b.available())
                << "cause=" << static_cast<int>(b.cause) << " request=" << static_cast<int>(f.request->coordinates()[i][c])
                << " consistency=" << static_cast<int>(f.decision->coordinates()[i][c].cause)
                << " recovered=" << f.decision->coordinates()[i][c].confirmed_recovery_excludes_persistent_shift;
            EXPECT_EQ(b.affected.first,599);EXPECT_EQ(b.affected.past_last,601);
            EXPECT_EQ(b.confirmations[0].rows.past_last,b.affected.first);
            EXPECT_EQ(b.confirmations[1].rows.first,b.affected.past_last);
            EXPECT_DOUBLE_EQ(b.physical_bound.begin_unix_sec,axis.occurrence(599).integration_support.begin_unix_sec);
            EXPECT_DOUBLE_EQ(b.physical_bound.end_unix_sec,axis.occurrence(600).integration_support.end_unix_sec);
            EXPECT_DOUBLE_EQ(b.frozen_residual_scale,f.assessment->events()[i].background[c].pre_scale_fit.scale);
            EXPECT_FALSE(b.exceeds_fitting_exclusion);
        }
    }
}
TEST(rtc_jump_transition, confirms_elapsed_integration_time_at_two_cadences) {
    std::array<std::int64_t,2> post_rows{};std::size_t k=0;
    for(double cadence:{.008192,.012288}) {
        Input in(1100,cadence);in.step();Fixture f(in);const auto i=f.event_at(500);
        const auto &b=f.transition->coordinates()[i][0];ASSERT_TRUE(b.available());
        const auto &post=b.confirmations[1];
        EXPECT_GE(post.physical.duration_sec(),.05);
        EXPECT_LT(post.physical.duration_sec()-cadence,.05);
        post_rows[k++]=post.rows.past_last-post.rows.first;
    }
    EXPECT_NE(post_rows[0],post_rows[1]);
}
TEST(rtc_jump_transition, gradual_settling_can_extend_beyond_trial_mask) {
    Input in;in.step(500,4,0);
    for(std::size_t row=500;row<530;++row)in.x(row,0)+=1.2*(530-row)/30.;
    Fixture f(in);const auto i=f.event_at(500);const auto &b=f.transition->coordinates()[i][0];
    ASSERT_TRUE(b.available());EXPECT_GT(b.affected.past_last,607);
    EXPECT_TRUE(b.exceeds_fitting_exclusion);
    EXPECT_FALSE(f.transition->hard_event_accepted);EXPECT_FALSE(f.transition->apply_authorized);
}
TEST(rtc_jump_transition, isolated_pulse_is_not_requested_as_persistent_transition) {
    Input in;in.spike();Fixture f(in);const auto i=f.event_at(500);
    EXPECT_EQ(f.transition->requested_coordinates(),0U);
    EXPECT_EQ(f.transition->coordinates()[i][0].cause,RtcJumpTransitionCause::not_requested);
    EXPECT_TRUE(f.decision->coordinates()[i][0].confirmed_recovery_excludes_persistent_shift);
}
TEST(rtc_jump_transition, unseeded_coordinate_does_not_borrow_step_or_scale) {
    Input in;in.step(500,4,0);Fixture f(in);const auto i=f.event_at(500);
    ASSERT_TRUE(f.transition->coordinates()[i][0].available());
    EXPECT_EQ(f.transition->coordinates()[i][1].cause,RtcJumpTransitionCause::not_requested);
    EXPECT_TRUE(std::isnan(f.transition->coordinates()[i][1].frozen_residual_scale));
}
TEST(rtc_jump_transition, unknown_protection_remains_on_exact_parent_chain) {
    Input in;in.step();Fixture f(in,RtcSpikeProtection::unavailable);const auto i=f.event_at(500);
    ASSERT_TRUE(f.transition->coordinates()[i][0].available());
    EXPECT_TRUE(f.review->event_reviews()[i].source_protection_unavailable);
    EXPECT_EQ(f.transition->request_handle()->consistency_handle().get(),f.decision.get());
    EXPECT_FALSE(f.transition->coordinates()[i][0].timing_uncertainty_quantified);
}
TEST(rtc_jump_transition, invalid_cells_between_confirmations_do_not_become_available_bound) {
    Input in;in.step(500,4,0);
    for(std::size_t row=501;row<509;++row){
        in.xs[3*row]=NativeReadoutCoordinateState::measured(true,false,true,false);in.x(row,0)=NAN;
    }
    Fixture f(in);const auto i=f.event_at(500);const auto &b=f.transition->coordinates()[i][0];
    EXPECT_EQ(b.cause,RtcJumpTransitionCause::invalid_transition_support);
    EXPECT_GT(b.invalid_rows,0U);EXPECT_FALSE(b.available());
}
TEST(rtc_jump_transition, whole_confirmation_fitting_both_references_is_ambiguous) {
    Input in;in.step();Fixture f(in);auto event=f.assessment->events()[f.event_at(500)];
    event.background[0].pre_scale_fit.scale=20;
    const auto b=measured(f,event);
    EXPECT_EQ(b.cause,RtcJumpTransitionCause::ambiguous_reference);
    EXPECT_TRUE(b.confirmations[0].also_matches_other_reference);
    EXPECT_FALSE(b.available());
}
TEST(rtc_jump_transition, missing_pre_or_post_confirmation_is_explicit) {
    Input in;in.step();Fixture f(in);auto event=f.assessment->events()[f.event_at(500)];
    event.background[0].cubic_with_offset.coefficients[0]+=100;
    EXPECT_EQ(measured(f,event).cause,RtcJumpTransitionCause::pre_confirmation_missing);
    event=f.assessment->events()[f.event_at(500)];
    event.background[0].cubic_with_offset.offset+=100;
    EXPECT_EQ(measured(f,event).cause,RtcJumpTransitionCause::post_confirmation_missing);
}
TEST(rtc_jump_transition, nonfinite_model_and_unavailable_background_remain_unavailable) {
    Input in;in.step();Fixture f(in);auto event=f.assessment->events()[f.event_at(500)];
    event.background[0].cubic_with_offset.coefficients[0]=INFINITY;
    EXPECT_EQ(measured(f,event).cause,RtcJumpTransitionCause::nonfinite);
    event=f.assessment->events()[f.event_at(500)];
    event.background[0].pre_scale_fit.cause=RtcEventFitCause::not_attempted;
    EXPECT_EQ(measured(f,event).cause,RtcJumpTransitionCause::background_unavailable);
}
TEST(rtc_jump_transition, competing_neighbor_exclusion_is_not_swallowed_by_bound) {
    Input in;in.spike(491,40,0);in.step();Fixture f(in);auto event=f.assessment->events()[f.event_at(500)];
    const auto b=measured(f,event);
    EXPECT_EQ(b.cause,RtcJumpTransitionCause::competing_exclusion);
    EXPECT_GT(b.excluded_rows,0U);EXPECT_FALSE(b.available());
}
TEST(rtc_jump_transition, later_distinct_step_cannot_be_swallowed_as_one_onset) {
    Input in;in.step(500,4,0);in.step(520,1,0);Fixture f(in);const auto i=f.event_at(500);
    const auto b=measured(f,f.assessment->events()[i]);
    EXPECT_FALSE(b.available());EXPECT_TRUE(b.multiple_candidate_edges);
    EXPECT_EQ(b.cause,RtcJumpTransitionCause::competing_exclusion);
    EXPECT_FALSE(b.physical_event_identity_resolved);
    EXPECT_FALSE(f.transition->hard_event_accepted);EXPECT_FALSE(f.transition->apply_authorized);
}
TEST(rtc_jump_transition, later_grouped_spike_keeps_stable_plateau_and_original_members) {
    for (bool other_coordinate : {false,true}) {
        Input in;in.step();in.spike(600,other_coordinate?0:40,other_coordinate?25:0);
        Fixture f(in);const auto i=f.event_at(500);const auto &event=f.assessment->events()[i];
        const auto members=event.candidates;
        ASSERT_TRUE(std::any_of(members.begin(),members.end(),[&](auto n){return f.spikes->candidates()[n].later_row>=700;}));
        const auto onset=rtc_jump_transition_detail::onset_edges(*f.spikes,event);
        EXPECT_EQ(onset.first,599);EXPECT_EQ(onset.past_last,601);
        for (std::size_t c=0;c<2;++c) {
            const auto b=measured(f,event,c);ASSERT_TRUE(b.available());
            EXPECT_EQ(b.affected.first,599);EXPECT_EQ(b.affected.past_last,601);
            EXPECT_LT(b.confirmations[1].rows.past_last,690);
            EXPECT_TRUE(b.multiple_candidate_edges);
        }
        EXPECT_EQ(event.candidates,members);
        EXPECT_FALSE(f.transition->hard_event_accepted);EXPECT_FALSE(f.transition->apply_authorized);
    }
}
TEST(rtc_jump_transition, nearby_disconnected_member_guard_still_blocks_onset_bound) {
    Input in;in.step();in.spike(505,40,0);Fixture f(in);
    const auto &event=f.assessment->events()[f.event_at(500)];
    const auto onset=rtc_jump_transition_detail::onset_edges(*f.spikes,event);
    EXPECT_EQ(onset.first,599);EXPECT_EQ(onset.past_last,601);
    const auto b=measured(f,event);
    EXPECT_FALSE(b.available());EXPECT_EQ(b.cause,RtcJumpTransitionCause::competing_exclusion);
    EXPECT_GT(b.excluded_rows,0U);
}
TEST(rtc_jump_transition, finite_transition_with_adjacent_edges_keeps_measured_bracket) {
    Input in;in.step(500,2.4,-2);in.step(501,1.6,-1);Fixture f(in);const auto i=f.event_at(500);
    for(std::size_t c=0;c<2;++c){
        const auto &b=f.transition->coordinates()[i][c];ASSERT_TRUE(b.available());
        EXPECT_TRUE(b.multiple_candidate_edges);EXPECT_FALSE(b.physical_event_identity_resolved);
        EXPECT_EQ(b.affected.first,599);EXPECT_EQ(b.affected.past_last,602);
    }
}
TEST(rtc_jump_transition, real_gap_clips_context_without_cross_gap_confirmation) {
    Input in;
    for(std::size_t row=400;row<in.times.size();++row){in.times[row]+=.05;in.counters[row]+=5;}
    in.step();Fixture f(in);const auto i=f.event_at(500);
    const auto b=measured(f,f.assessment->events()[i]);
    EXPECT_TRUE(b.acquisition_truncated);EXPECT_GE(b.examined.first,500);
    EXPECT_FALSE(b.observation_truncated);
}
TEST(rtc_jump_transition, true_observation_edge_reports_incomplete_context) {
    Input in;in.step(80);Fixture f(in);const auto i=f.event_at(80);
    const auto b=measured(f,f.assessment->events()[i]);
    EXPECT_TRUE(b.observation_truncated);EXPECT_FALSE(b.acquisition_truncated);
    EXPECT_GE(b.examined.first,100);
}
TEST(rtc_jump_transition, overlapping_integration_support_is_not_assumed_independent_time) {
    Input in;in.step();in.integration_half=.005;Fixture f(in);const auto i=f.event_at(500);
    const auto b=measured(f,f.assessment->events()[i]);
    EXPECT_EQ(b.cause,RtcJumpTransitionCause::support_geometry_unavailable);
}
TEST(rtc_jump_transition, exact_snapshot_identity_and_nonzero_attempt_are_required) {
    Input in;in.step();Fixture f(in);const auto other=ValSnapshot::initial(in.freeze());
    EXPECT_THROW(RtcJumpTransitionRequest::consider(f.decision,other,9),std::invalid_argument);
    EXPECT_THROW(RtcJumpTransitionRequest::consider(f.decision,f.val,0),std::invalid_argument);
    EXPECT_THROW(RtcJumpTransitionRequest::consider(nullptr,f.val,9),std::invalid_argument);
    EXPECT_THROW(RtcJumpTransitionEvidence::learn(f.request,0),std::invalid_argument);
    EXPECT_THROW(RtcJumpTransitionEvidence::learn(nullptr,9),std::invalid_argument);
}
TEST(rtc_jump_transition, repeat_preserves_original_pair_and_requires_no_refit) {
    Input in;in.step();Fixture f(in);const auto again=RtcJumpTransitionEvidence::learn(f.request,9);
    EXPECT_EQ(again->requested_coordinates(),f.transition->requested_coordinates());
    EXPECT_EQ(again->examined_rows(),f.transition->examined_rows());
    for(std::size_t i=0;i<again->coordinates().size();++i)for(std::size_t c=0;c<2;++c){
        const auto &a=again->coordinates()[i][c],&b=f.transition->coordinates()[i][c];
        EXPECT_EQ(a.cause,b.cause);EXPECT_EQ(a.affected.first,b.affected.first);
        EXPECT_EQ(a.affected.past_last,b.affected.past_last);EXPECT_EQ(a.physical_bound,b.physical_bound);
    }
    EXPECT_EQ(f.parent->network(0).value(NativeReadoutCoordinate::x,600,0),in.x(500,0));
    EXPECT_FALSE(again->hard_event_accepted);EXPECT_FALSE(again->apply_authorized);
}
} // namespace
