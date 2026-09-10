#include <citlali/core/pipeline/timestream_rtc_jump_consistency.h>
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
    explicit Input(std::size_t n=1100) : times(n), counters(n), x(n,3), r(n,3),
        xs(support::valid_states(3*n)), rs(xs) {
        for (std::size_t i=0; i<times.size(); ++i) {
            times[i]=1000+.008192*i; counters[i]=2000+i;
            for (std::size_t d=0; d<3; ++d) {
                const double t=.008192*(static_cast<double>(i)-500);
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
            occurrences.push_back({static_cast<std::int64_t>(i+10000),static_cast<std::int64_t>(i+20000),{times[i]-.004096,times[i]+.004096}});
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
    }
    std::size_t event_at(std::size_t input_row) const {
        for (std::size_t i=0;i<assessment->events().size();++i)
            if (spikes->candidates()[assessment->events()[i].seed].later_row==static_cast<TimestreamNativeRow>(100+input_row)) return i;
        throw std::runtime_error("expected original event seed missing");
    }
};

TEST(rtc_jump_consistency, amplitude_inclusive_five_sigma_and_both_signs) {
    for (double sign : {-1.,1.}) {
        EXPECT_EQ(rtc_jump_detail::amplitude(sign*5,1),RtcJumpAmplitudeCause::passes);
        EXPECT_EQ(rtc_jump_detail::amplitude(sign*std::nextafter(5.,0.),1),RtcJumpAmplitudeCause::below_threshold);
    }
}
TEST(rtc_jump_consistency, invalid_noise_and_overflow_are_unavailable) {
    for (double scale : std::array<double,4>{0.,-1.,NAN,INFINITY})
        EXPECT_EQ(rtc_jump_detail::amplitude(8,scale),RtcJumpAmplitudeCause::noise_unavailable);
    EXPECT_EQ(rtc_jump_detail::amplitude(INFINITY,1),RtcJumpAmplitudeCause::arithmetic_nonfinite);
    EXPECT_EQ(rtc_jump_detail::amplitude(8,std::numeric_limits<double>::max()),RtcJumpAmplitudeCause::arithmetic_nonfinite);
    EXPECT_EQ(rtc_jump_detail::amplitude(8,std::numeric_limits<double>::denorm_min()),RtcJumpAmplitudeCause::arithmetic_nonfinite);
}
TEST(rtc_jump_consistency, agreement_is_inclusive_at_two_sigma_and_sign_preserving) {
    for (double sign : {-1.,1.}) {
        EXPECT_EQ(rtc_jump_detail::consistency(sign*7,sign*5,1),RtcJumpConsistencyCause::passes);
        EXPECT_EQ(rtc_jump_detail::consistency(sign*std::nextafter(7.,8.),sign*5,1),RtcJumpConsistencyCause::offset_disagreement);
    }
    EXPECT_EQ(rtc_jump_detail::consistency(5,-5,1),RtcJumpConsistencyCause::sign_disagreement);
    EXPECT_EQ(rtc_jump_detail::consistency(5,4.9,1),RtcJumpConsistencyCause::short_below_threshold);
    EXPECT_EQ(rtc_jump_detail::consistency(4.9,5,1),RtcJumpConsistencyCause::primary_not_passed);
    EXPECT_EQ(rtc_jump_detail::consistency(5,NAN,1),RtcJumpConsistencyCause::arithmetic_nonfinite);
}
TEST(rtc_jump_consistency, cubic_with_oppositely_signed_coordinate_steps_is_stable) {
    Input in; in.step(); Fixture f(in); const auto i=f.event_at(500);
    for (std::size_t c=0;c<2;++c) {
        ASSERT_TRUE(f.amplitude->coordinates()[i][c].passes());
        ASSERT_TRUE(f.short_evidence->coordinates()[i][c].available());
        EXPECT_TRUE(f.decision->coordinates()[i][c].passes());
        EXPECT_NEAR(f.short_evidence->coordinates()[i][c].cubic_with_offset.offset,c==0?4:-3,.06);
    }
    EXPECT_EQ(f.short_evidence->counts().requested_coordinates,2U);
    EXPECT_EQ(f.short_evidence->counts().pre_fit_calls,2U);
    EXPECT_EQ(f.short_evidence->counts().joint_fit_calls,2U);
    EXPECT_FALSE(f.decision->hard_event_accepted); EXPECT_FALSE(f.decision->apply_authorized);
    EXPECT_EQ(f.parent->network(0).value(NativeReadoutCoordinate::x,600,0),in.x(500,0));
}
TEST(rtc_jump_consistency, noncubic_background_can_fail_offset_stability_despite_large_step) {
    Input in; in.step(500,4,0);
    for(std::size_t i=0;i<in.times.size();++i) {
        const double t=std::clamp(.008192*(static_cast<double>(i)-500),-2.4,2.4);
        in.x(i,0)+=.3*t*t*t*t*t;
    }
    Fixture f(in); const auto i=f.event_at(500);
    ASSERT_TRUE(f.amplitude->coordinates()[i][0].passes());
    ASSERT_TRUE(f.short_evidence->coordinates()[i][0].available());
    EXPECT_EQ(f.decision->coordinates()[i][0].cause,RtcJumpConsistencyCause::offset_disagreement);
    EXPECT_GT(f.decision->coordinates()[i][0].offset_difference_sigma,2.);
}
TEST(rtc_jump_consistency, isolated_spike_keeps_recovery_and_avoids_extra_fit) {
    Input in; in.spike(); Fixture f(in); const auto i=f.event_at(500);
    EXPECT_EQ(f.short_evidence->counts().requested_coordinates,0U);
    EXPECT_EQ(f.short_evidence->counts().pre_fit_calls,0U);
    EXPECT_EQ(f.short_evidence->counts().joint_fit_calls,0U);
    for (std::size_t c=0;c<2;++c) {
        EXPECT_TRUE(f.decision->coordinates()[i][c].confirmed_recovery_excludes_persistent_shift);
        EXPECT_EQ(f.short_evidence->coordinates()[i][c].cause,RtcJumpShortFitCause::not_requested);
    }
}
TEST(rtc_jump_consistency, unseeded_coordinate_never_borrows_other_coordinate_noise) {
    Input in; in.step(500,4,0); Fixture f(in); const auto i=f.event_at(500);
    EXPECT_TRUE(f.amplitude->coordinates()[i][0].candidate.has_value());
    EXPECT_FALSE(f.amplitude->coordinates()[i][1].candidate.has_value());
    EXPECT_EQ(f.amplitude->coordinates()[i][1].cause,RtcJumpAmplitudeCause::not_seeded);
    EXPECT_EQ(f.short_evidence->coordinates()[i][1].cause,RtcJumpShortFitCause::not_requested);
}
TEST(rtc_jump_consistency, coordinate_onset_uses_exact_first_member_noise_block) {
    Input in; in.spike(500,40,0); in.step(530,0,8); Fixture f(in);
    for (std::size_t i=0;i<f.assessment->events().size();++i) for (std::size_t c=0;c<2;++c) {
        const auto &gate=f.amplitude->coordinates()[i][c];
        std::optional<std::size_t> first;
        for (auto candidate:f.assessment->events()[i].candidates)
            if (f.spikes->candidates()[candidate].coordinate==rtc_event_assessment_detail::coord(c)) {first=candidate;break;}
        EXPECT_EQ(gate.candidate,first);
        if (first) {
            const auto block=f.spikes->candidates()[*first].noise_block_index;
            EXPECT_EQ(gate.noise_block,block);
            EXPECT_DOUBLE_EQ(gate.sigma_delta,f.spikes->blocks()[block].coordinates[c].scale);
        }
    }
}
TEST(rtc_jump_consistency, ten_second_boundary_edge_keeps_later_endpoint_block) {
    Input in(2600);
    const auto boundary=static_cast<std::size_t>(std::lower_bound(in.times.begin(),in.times.end(),in.times[0]+10)-in.times.begin());
    in.step(boundary); Fixture f(in); const auto i=f.event_at(boundary);
    const auto &gate=f.amplitude->coordinates()[i][0]; ASSERT_TRUE(gate.passes());
    const auto &seed=f.spikes->candidates()[*gate.candidate];
    const auto &block=f.spikes->blocks()[*gate.noise_block];
    EXPECT_EQ(seed.later_row,block.first); EXPECT_EQ(seed.earlier_row+1,block.first);
    EXPECT_EQ(gate.noise_block,seed.noise_block_index);
    EXPECT_DOUBLE_EQ(gate.sigma_delta,block.coordinates[0].scale);
}
TEST(rtc_jump_consistency, short_flanks_are_original_subset_with_unchanged_neighbor_masks) {
    Input in; in.spike(425,40,0); in.step(); Fixture f(in); const auto i=f.event_at(500);
    const auto &event=f.assessment->events()[i]; const auto &out=f.short_evidence->coordinates()[i][0];
    ASSERT_TRUE(out.available()); EXPECT_GT(out.neighbor_excluded[0],0U);
    const auto &axis=f.parent->network(0).occurrence_axis();
    const double begin=axis.occurrence(event.trial_exclusion.first).integration_support.begin_unix_sec;
    const double end=axis.occurrence(event.trial_exclusion.past_last-1).integration_support.end_unix_sec;
    for (std::size_t side=0;side<2;++side) {
        const auto &s=out.support[side]; const auto &p=event.background[0].support[side];
        EXPECT_GE(s.first_used,p.first_used); EXPECT_LE(s.last_used,p.last_used); EXPECT_LT(s.usable,p.usable);
        EXPECT_GE(s.begin_unix_sec,side==0?begin-1:end);
        EXPECT_LE(s.end_unix_sec,side==0?begin:end+1);
    }
    EXPECT_NEAR(out.cubic_with_offset.offset,4,.08);
}
TEST(rtc_jump_consistency, missing_one_second_at_observation_edge_is_explicit) {
    Input in; in.step(80); Fixture f(in); const auto i=f.event_at(80);
    ASSERT_TRUE(f.amplitude->coordinates()[i][0].passes());
    EXPECT_EQ(f.short_evidence->coordinates()[i][0].cause,RtcJumpShortFitCause::context_truncated);
    EXPECT_EQ(f.decision->coordinates()[i][0].cause,RtcJumpConsistencyCause::short_fit_unavailable);
}
TEST(rtc_jump_consistency, physical_gap_never_supplies_missing_short_flank) {
    Input in;
    for(std::size_t i=150;i<in.times.size();++i) { in.times[i]+=.008192*4; in.counters[i]+=4; }
    in.step(230);
    Fixture f(in); const auto i=f.event_at(230);
    ASSERT_TRUE(f.amplitude->coordinates()[i][0].passes());
    EXPECT_EQ(f.short_evidence->coordinates()[i][0].cause,RtcJumpShortFitCause::context_truncated);
}
TEST(rtc_jump_consistency, sparse_inner_flank_does_not_borrow_outer_primary_samples) {
    Input in; in.step(500,4,0);
    for(std::size_t i=390;i<480;++i) {
        in.xs[3*i]=NativeReadoutCoordinateState::measured(true,false,true,false);
        in.x(i,0)=NAN;
    }
    Fixture f(in); const auto i=f.event_at(500);
    ASSERT_TRUE(f.amplitude->coordinates()[i][0].passes());
    const auto &s=f.short_evidence->coordinates()[i][0];
    EXPECT_EQ(s.cause,RtcJumpShortFitCause::insufficient_samples);
    EXPECT_LT(s.support[0].usable,64U); EXPECT_GT(s.support[0].invalid,0U);
    EXPECT_EQ(s.pre_scale_fit.cause,RtcEventFitCause::not_attempted);
}
TEST(rtc_jump_consistency, protection_and_prior_screening_remain_on_exact_parent_chain) {
    Input in; in.step(); Fixture f(in,RtcSpikeProtection::unavailable); const auto i=f.event_at(500);
    ASSERT_TRUE(f.decision->coordinates()[i][0].passes());
    EXPECT_TRUE(f.review->event_reviews()[i].source_protection_unavailable);
    EXPECT_EQ(f.decision->evidence_handle()->amplitude_handle()->review_handle().get(),f.review.get());
    EXPECT_EQ(f.amplitude->review_handle()->original_screening_handle().get(),f.review->original_screening_handle().get());
    EXPECT_FALSE(f.decision->hard_event_accepted); EXPECT_FALSE(f.decision->apply_authorized);
}
TEST(rtc_jump_consistency, mismatched_snapshot_and_missing_identity_are_rejected) {
    Input in; in.step(); Fixture f(in); const auto other=ValSnapshot::initial(in.freeze());
    EXPECT_THROW(RtcJumpAmplitudeDecision::consider(f.review,other,1),std::invalid_argument);
    EXPECT_THROW(RtcJumpConsistencyDecision::consider(f.short_evidence,other,1),std::invalid_argument);
    EXPECT_THROW(RtcJumpAmplitudeDecision::consider(f.review,f.val,0),std::invalid_argument);
    EXPECT_THROW(RtcJumpConsistencyEvidence::learn(f.amplitude,0),std::invalid_argument);
    EXPECT_THROW(RtcJumpConsistencyDecision::consider(f.short_evidence,f.val,0),std::invalid_argument);
    EXPECT_THROW(RtcJumpAmplitudeDecision::consider(nullptr,f.val,1),std::invalid_argument);
    EXPECT_THROW(RtcJumpConsistencyEvidence::learn(nullptr,1),std::invalid_argument);
    EXPECT_THROW(RtcJumpConsistencyDecision::consider(nullptr,f.val,1),std::invalid_argument);
}
TEST(rtc_jump_consistency, fixed_short_fit_is_deterministic_and_has_no_refinement_loop) {
    Input in; in.step(); Fixture f(in); const auto again=RtcJumpConsistencyEvidence::learn(f.amplitude,7);
    EXPECT_EQ(again->counts().requested_coordinates,f.short_evidence->counts().requested_coordinates);
    for(std::size_t i=0;i<again->coordinates().size();++i) for(std::size_t c=0;c<2;++c) {
        const auto &a=again->coordinates()[i][c],&b=f.short_evidence->coordinates()[i][c];
        EXPECT_EQ(a.cause,b.cause);
        if(a.available()) {EXPECT_EQ(a.cubic_with_offset.coefficients,b.cubic_with_offset.coefficients);EXPECT_DOUBLE_EQ(a.cubic_with_offset.offset,b.cubic_with_offset.offset);}
    }
    EXPECT_LE(again->counts().pre_fit_calls,again->counts().requested_coordinates);
    EXPECT_LE(again->counts().joint_fit_calls,again->counts().pre_fit_calls);
    // Timing counters retain failed numerical work, including the entry that
    // diagnoses a zero scale; failures before any loop entry report zero.
    Eigen::MatrixXd design(64,4);
    for (Eigen::Index row=0;row<design.rows();++row) {
        const double t=static_cast<double>(row)/64.;
        design.row(row)<<1,t,t*t,t*t*t;
    }
    Eigen::VectorXd values=Eigen::VectorXd::Zero(64);
    const auto zero_scale=rtc_event_background_detail::fit(design,values);
    EXPECT_EQ(zero_scale.cause,RtcEventFitCause::zero_scale);
    EXPECT_EQ(zero_scale.iterations,1U);
    values[0]=NAN;
    const auto nonfinite=rtc_event_background_detail::fit(design,values);
    EXPECT_EQ(nonfinite.cause,RtcEventFitCause::nonfinite);
    EXPECT_EQ(nonfinite.iterations,0U);
}
} // namespace
