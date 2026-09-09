#include <citlali/core/pipeline/timestream_rtc_event_assessment.h>
#include "timestream_successor_identity_test_support.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
using namespace citlali::pipeline;
namespace support=citlali::test::timestream_successor;
struct Input {
    std::vector<double> times;
    std::vector<TimestreamPacketCounter> counters;
    NativePairedReadoutMatrix x,r;
    std::vector<NativeReadoutCoordinateState> xs,rs;
    explicit Input(std::size_t n=1100,std::size_t detectors=3)
        : times(n),counters(n),x(n,detectors),r(n,detectors),xs(support::valid_states(n*detectors)),rs(xs) {
        for(std::size_t i=0;i<n;++i) {
            times[i]=1000+.008192*i;counters[i]=2000+i;
            for(std::size_t d=0;d<detectors;++d) {
                const double t=.008192*(static_cast<double>(i)-500);
                const double noise=std::array{-0.1,0.0,0.1,0.0,0.0,-0.1,0.1}[(i+d)%7];
                x(i,d)=10+.8*t+.1*t*t-.05*t*t*t+noise;r(i,d)=-3+2*noise;
            }
        }
    }
    void spike(std::size_t at=500,std::size_t d=0,double dx=40,double dr=25) {x(at,d)+=dx;r(at,d)+=dr;}
    void step(std::size_t at=500,std::size_t d=0,double dx=4,double dr=-3) {for(std::size_t i=at;i<times.size();++i) {x(i,d)+=dx;r(i,d)+=dr;}}
    auto freeze() const {
        auto timing=std::make_shared<const NativeNetworkAlignment>(0,100,support::time_vector(times),counters);
        std::vector<NativePairedReadoutOccurrenceBinding> occurrences;
        for(std::size_t i=0;i<times.size();++i) occurrences.push_back({static_cast<std::int64_t>(i+10000),static_cast<std::int64_t>(i+20000),{times[i]-.004096,times[i]+.004096}});
        auto axis=std::make_shared<const NativePairedReadoutOccurrenceAxis>(timing,100,std::move(occurrences));
        std::vector<NativePairedReadoutNetwork> networks;
        networks.push_back(NativePairedReadoutNetwork::admit(axis,
            support::detector_axis(0,x.cols()),support::mapping_authority(0,"assessment-test"),x,r,xs,rs));
        return support::make_observation(std::move(networks),{0});
    }
};
struct Fixture {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const NativePairedReadoutView> view;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcSpikeSourceProtection> protection;
    std::shared_ptr<const RtcSpikeEvidence> spikes;
    std::shared_ptr<const RtcEventPeerPopulation> population;
    explicit Fixture(const Input &in,RtcSpikeProtection state=RtcSpikeProtection::outside_source)
        :parent(in.freeze()),view(NativePairedReadoutView::full(parent)),val(ValSnapshot::initial(parent)),
        protection(RtcSpikeSourceProtection::admit(parent,"source-test",state)),spikes(learn_rtc_spike_candidates(view,val,protection,11)) {
        std::vector<RtcEventPeerEligibility> records;
        for(std::uint32_t d=0;d<parent->network(0).detectors().size();++d) records.push_back({0,d,parent->network(0).detector(d).detector_occurrence_id,true});
        population=RtcEventPeerPopulation::admit(spikes,"test-population-exact",std::move(records));
    }
    auto learn() const {return learn_rtc_event_assessment(spikes,population,12);}
};

TEST(rtc_event_assessment, isolated_paired_spike_has_one_event_and_separate_confirmation) {
    Input in;in.spike();Fixture f(in);const auto e=f.learn();
    ASSERT_EQ(e->events().size(),1U);const auto &a=e->events()[0];
    EXPECT_EQ(a.candidates.size(),4U);
    for(const auto &r:a.recovery) {ASSERT_TRUE(r.recovered())<<static_cast<int>(r.cause);EXPECT_EQ(r.affected.first,600);EXPECT_EQ(r.affected.past_last,601);EXPECT_EQ(r.confirmation.first,601);EXPECT_GE(r.confirmation.past_last-r.confirmation.first,7);}
    EXPECT_LT(a.trial_exclusion.first,a.recovery[0].affected.first);
    EXPECT_GT(a.trial_exclusion.past_last,a.recovery[0].affected.past_last);
    const auto d=RtcEventAssessmentDecision::consider(e,f.val,13);
    EXPECT_EQ(d->event_reviews()[0].disposition,RtcEventReviewDisposition::recovered_candidate);
    EXPECT_FALSE(d->event_reviews()[0].apply_authorized);
    EXPECT_DOUBLE_EQ(f.parent->network(0).value(NativeReadoutCoordinate::x,600,0),in.x(500,0));
}
TEST(rtc_event_assessment, separate_nearby_pulses_keep_clean_intervening_background) {
    Input in;in.spike();in.spike(564);in.spike(628);Fixture f(in);const auto e=f.learn();
    ASSERT_EQ(e->events().size(),3U);
    for(const auto &a:e->events()) {
        ASSERT_TRUE(a.background[0].available());ASSERT_TRUE(a.background[1].available());
        EXPECT_TRUE(a.recovery[0].recovered());EXPECT_GT(a.excluded_neighbor_samples[0],0U);
        EXPECT_LT(a.excluded_neighbor_samples[0],50U);EXPECT_EQ(a.candidates.size(),4U);
    }
}
TEST(rtc_event_assessment, another_coordinate_excludes_the_same_neighbor_samples) {
    Input in;in.spike(500,0,40,0);in.spike(564,0,0,25);Fixture f(in);const auto e=f.learn();
    ASSERT_EQ(e->events().size(),2U);const auto &a=e->events()[0];
    EXPECT_TRUE(a.background[0].available());EXPECT_TRUE(a.background[1].available());
    EXPECT_EQ(a.excluded_neighbor_samples[0],a.excluded_neighbor_samples[1]);
    EXPECT_GT(a.excluded_neighbor_samples[0],0U);
}
TEST(rtc_event_assessment, shift_remains_persistent_with_separate_small_pre_spike) {
    Input in;in.spike(432,0,3,0);in.step();Fixture f(in);const auto e=f.learn();
    const auto it=std::find_if(e->events().begin(),e->events().end(),[&](const auto &a){return f.spikes->candidates()[a.seed].later_row==600;});
    ASSERT_NE(it,e->events().end());ASSERT_TRUE(it->background[0].available());
    EXPECT_NEAR(it->background[0].cubic_with_offset.offset,4,.04);
    EXPECT_EQ(it->recovery[0].cause,RtcEventRecoveryCause::search_limit);
    EXPECT_FALSE(it->recovery[0].recovered());
}
TEST(rtc_event_assessment, jumps_before_recovery_form_compound_assessment) {
    Input in;for(std::size_t i=500;i<530;++i) {in.x(i,0)+=30;in.r(i,0)+=20;}in.spike(515);
    Fixture f(in);const auto e=f.learn();ASSERT_FALSE(e->events().empty());
    EXPECT_GT(e->events()[0].candidates.size(),4U);
}
TEST(rtc_event_assessment, delayed_other_coordinate_step_cannot_inherit_earlier_recovery) {
    Input in;
    for(std::size_t i=500;i<600;++i) in.x(i,0)+=30;
    in.step(550,0,0,20);
    Fixture f(in);const auto e=f.learn();const auto d=RtcEventAssessmentDecision::consider(e,f.val,1);
    ASSERT_FALSE(e->events().empty());const auto &a=e->events()[0];
    EXPECT_TRUE(a.seeded[0]);EXPECT_TRUE(a.seeded[1]);
    ASSERT_TRUE(a.background[1].available());EXPECT_FALSE(a.recovery[1].recovered());
    EXPECT_EQ(d->event_reviews()[0].disposition,RtcEventReviewDisposition::persistent_or_compound_unresolved);
}
TEST(rtc_event_assessment, later_edges_interrupt_confirmation_and_all_members_precede_recovery) {
    Input in;in.spike(500,0,30,0);in.spike(505,0,0,20);in.spike(509,0,30,0);
    Fixture f(in);const auto e=f.learn();ASSERT_EQ(e->events().size(),1U);
    const auto &a=e->events()[0];EXPECT_EQ(a.candidates.size(),6U);
    for(auto i:a.candidates) {
        const auto &s=f.spikes->candidates()[i];const auto c=s.coordinate==NativeReadoutCoordinate::x?0U:1U;
        ASSERT_TRUE(a.recovery[c].recovered());EXPECT_GE(a.recovery[c].confirmation.first,s.later_row);
    }
}
TEST(rtc_event_assessment, peer_context_excludes_self_and_preserves_sample_pairing) {
    Input in;in.step(500,0);in.step(500,1);Fixture f(in);const auto e=f.learn();
    ASSERT_FALSE(e->events().empty());const auto &p=e->events()[0].peers[0];
    EXPECT_EQ(p.eligible_peers,2U);EXPECT_NE(p.strongest_peer,0U);
    EXPECT_GT(p.strongest_level_correlation,.99);EXPECT_NEAR(p.strongest_edge_delay_seconds,0,1e-9);
}
TEST(rtc_event_assessment, exact_parent_and_snapshot_are_required) {
    Input in;in.spike();Fixture f(in),other(in);
    EXPECT_THROW(learn_rtc_event_assessment(f.spikes,other.population,1),std::invalid_argument);
    EXPECT_THROW(RtcEventAssessmentDecision::consider(f.learn(),other.val,1),std::invalid_argument);
    EXPECT_THROW(learn_rtc_event_assessment(f.spikes,f.population,0),std::invalid_argument);
    EXPECT_THROW(RtcEventPeerPopulation::admit(f.spikes,"identity",{}),std::invalid_argument);
}
TEST(rtc_event_assessment, physical_gap_and_observation_end_cannot_confirm_recovery) {
    Input in;in.step(1080);Fixture f(in);const auto e=f.learn();
    ASSERT_FALSE(e->events().empty());EXPECT_TRUE(e->events()[0].observation_truncated);
    EXPECT_FALSE(e->events()[0].recovery[0].recovered());
    Input gap;gap.step();for(std::size_t i=540;i<gap.times.size();++i)gap.counters[i]+=4;
    Fixture g(gap);const auto q=g.learn();ASSERT_FALSE(q->events().empty());
    EXPECT_TRUE(q->events()[0].gap_truncated);EXPECT_FALSE(q->events()[0].recovery[0].recovered());
}
TEST(rtc_event_assessment, missing_source_authority_does_not_become_spike_acceptance) {
    Input in;in.spike();Fixture f(in,RtcSpikeProtection::unavailable);
    const auto d=RtcEventAssessmentDecision::consider(f.learn(),f.val,1);
    ASSERT_EQ(d->event_reviews().size(),1U);EXPECT_TRUE(d->event_reviews()[0].source_protection_unavailable);
    EXPECT_FALSE(d->event_reviews()[0].hard_event_accepted);
    EXPECT_EQ(d->original_screening_handle()->evidence_handle(),f.spikes);
}
TEST(rtc_event_assessment, protected_source_keeps_optical_requirement) {
    Input in;in.spike();Fixture f(in,RtcSpikeProtection::protected_source);
    const auto d=RtcEventAssessmentDecision::consider(f.learn(),f.val,1);
    ASSERT_EQ(d->event_reviews().size(),1U);EXPECT_TRUE(d->event_reviews()[0].protected_optical_assessment_required);
}
TEST(rtc_event_assessment, partitioned_learning_keeps_identical_event_support) {
    Input in;in.spike();Fixture f(in);
    std::array parts{NativePairedReadoutView::admit(f.parent,{{0,100,600}}),NativePairedReadoutView::admit(f.parent,{{0,600,1200}})};
    auto s=learn_rtc_spike_candidates_partitioned(f.view,parts,f.val,f.protection,99);
    std::vector<RtcEventPeerEligibility> records;for(std::uint32_t d=0;d<3;++d)records.push_back({0,d,f.parent->network(0).detector(d).detector_occurrence_id,true});
    const auto p=RtcEventPeerPopulation::admit(s,"test",records);
    const auto a=f.learn(),b=learn_rtc_event_assessment(s,p,1);
    ASSERT_EQ(a->events().size(),b->events().size());
    EXPECT_EQ(a->events()[0].recovery[0].affected,b->events()[0].recovery[0].affected);
    EXPECT_EQ(a->events()[0].background[0].cubic_with_offset.coefficients,b->events()[0].background[0].cubic_with_offset.coefficients);
}
TEST(rtc_event_assessment, persistent_r_pathology_raises_review_only_and_preserves_pair_screening) {
    Input in(8000);
    for(std::size_t i=0;i<in.times.size();++i) in.r(i,0)=-3+40*(in.r(i,0)+3)+(i%40==10?1000:0);
    Fixture f(in);const auto e=f.learn();const auto d=RtcEventAssessmentDecision::consider(e,f.val,1);
    ASSERT_EQ(d->health_reviews().size(),3U);const auto &h=d->health_reviews()[0];
    EXPECT_EQ(h.complete_blocks,6U);EXPECT_EQ(h.concerning_blocks[1],6U);
    EXPECT_TRUE(h.coordinate_concern[1]);EXPECT_FALSE(h.coordinate_concern[0]);
    EXPECT_FALSE(d->health_reviews()[1].concern());EXPECT_FALSE(d->health_reviews()[2].concern());
    for(const auto &r:d->event_reviews()) {EXPECT_TRUE(r.health_concern);EXPECT_FALSE(r.apply_authorized);}
    EXPECT_EQ(d->original_screening_handle()->evidence_handle(),f.spikes);
}
TEST(rtc_event_assessment, large_variability_without_frequent_candidates_is_not_health_rejection) {
    Input in(8000);for(std::size_t i=0;i<in.times.size();++i)in.r(i,0)=-3+40*(in.r(i,0)+3);
    Fixture f(in);const auto d=RtcEventAssessmentDecision::consider(f.learn(),f.val,1);
    ASSERT_EQ(d->health_reviews().size(),3U);EXPECT_FALSE(d->health_reviews()[0].concern());
}
TEST(rtc_event_assessment, five_complete_blocks_are_insufficient_for_health_concern) {
    Input in(7000);for(std::size_t i=0;i<in.times.size();++i)in.r(i,0)=-3+40*(in.r(i,0)+3)+(i%40==10?1000:0);
    Fixture f(in);const auto d=RtcEventAssessmentDecision::consider(f.learn(),f.val,1);
    ASSERT_EQ(d->health_reviews().size(),3U);EXPECT_EQ(d->health_reviews()[0].complete_blocks,5U);
    EXPECT_FALSE(d->health_reviews()[0].concern());
}
TEST(rtc_event_assessment, unavailable_peer_population_cannot_become_healthy_evidence) {
    Input in(8000,1);Fixture f(in);const auto d=RtcEventAssessmentDecision::consider(f.learn(),f.val,1);
    ASSERT_EQ(d->health_reviews().size(),1U);const auto &h=d->health_reviews()[0];
    EXPECT_EQ(h.available_blocks[0],0U);EXPECT_EQ(h.available_blocks[1],0U);EXPECT_FALSE(h.concern());
    EXPECT_FALSE(h.assessment_available[0]);EXPECT_FALSE(h.assessment_available[1]);
}
TEST(rtc_event_assessment, health_eighty_percent_boundary_is_inclusive) {
    for(const int active_blocks:{7,8}) {
        Input in(12500);
        for(std::size_t i=0;i<in.times.size();++i) if(.008192*i<10*active_blocks)in.r(i,0)=-3+40*(in.r(i,0)+3)+(i%40==10?1000:0);
        Fixture f(in);const auto d=RtcEventAssessmentDecision::consider(f.learn(),f.val,1);
        const auto &h=d->health_reviews()[0];EXPECT_EQ(h.complete_blocks,10U);
        EXPECT_EQ(h.concerning_blocks[1],static_cast<std::size_t>(active_blocks));
        EXPECT_EQ(h.coordinate_concern[1],active_blocks==8);
    }
}
TEST(rtc_event_assessment, each_original_candidate_has_its_own_peer_context) {
    Input in;in.spike();in.spike(800);Fixture f(in);const auto e=f.learn();
    EXPECT_EQ(e->candidate_peer_context().size(),f.spikes->candidates().size());
    for(const auto &p:e->candidate_peer_context()) {EXPECT_EQ(p[0].eligible_peers,2U);EXPECT_GT(p[0].strongest_shared_samples,100U);}
}
TEST(rtc_event_assessment, edge_without_departure_from_cubic_band_is_not_a_recovered_excursion) {
    Input in;
    for(std::size_t i=0;i<in.times.size();++i)in.x(i,0)+=2*std::sin(6.283185307179586*.008192*i);
    in.spike(500,0,2,0);Fixture f(in);
    const auto d=RtcEventAssessmentDecision::consider(f.learn(),f.val,1);
    ASSERT_FALSE(d->event_reviews().empty());
    EXPECT_EQ(d->event_reviews()[0].disposition,RtcEventReviewDisposition::no_resolved_excursion);
}
TEST(rtc_event_assessment, producer_invalid_cells_are_not_recovery_confirmation) {
    Input in;in.spike();
    for(std::size_t i=501;i<514;++i) {in.xs[3*i]=NativeReadoutCoordinateState::measured(true,false,true,false);in.x(i,0)=NAN;}
    Fixture f(in);const auto e=f.learn();ASSERT_FALSE(e->events().empty());
    const auto &r=e->events()[0].recovery[0];ASSERT_TRUE(r.recovered());EXPECT_GE(r.confirmation.first,614);
}
TEST(rtc_event_assessment, original_case_e_injection_retains_spike_with_spectrum_explicitly_unavailable) {
    const auto path=std::filesystem::path{__FILE__}.parent_path()/"fixtures/timestream_rtc_event_assessment/case_e_original.txt";
    std::ifstream file(path);ASSERT_TRUE(file.good());
    std::vector<std::array<double,4>> rows;std::array<double,4> row;
    while(file>>row[0]>>row[1]>>row[2]>>row[3])rows.push_back(row);
    ASSERT_GT(rows.size(),500U);Input in(rows.size(),1);
    for(std::size_t i=0;i<rows.size();++i) {in.times[i]=1000+rows[i][1]-rows[0][1];in.x(i,0)=rows[i][2];in.r(i,0)=rows[i][3];}
    Fixture baseline(in,RtcSpikeProtection::unavailable);
    const auto at=rows.size()/2;const double original=in.x(at,0);
    const double amplitude=50*baseline.spikes->blocks()[0].coordinates[0].scale;
    ASSERT_TRUE(std::isfinite(amplitude));ASSERT_GT(amplitude,0);
    in.x(at,0)+=amplitude;Fixture injected(in,RtcSpikeProtection::unavailable);
    const auto e=injected.learn();const auto d=RtcEventAssessmentDecision::consider(e,injected.val,1);
    const auto found=std::find_if(injected.spikes->candidates().begin(),injected.spikes->candidates().end(),[&](const auto &s){return s.coordinate==NativeReadoutCoordinate::x && s.later_row==static_cast<TimestreamNativeRow>(at+100);});
    ASSERT_NE(found,injected.spikes->candidates().end());EXPECT_GT(found->absolute_score,40);
    EXPECT_DOUBLE_EQ(baseline.parent->network(0).value(NativeReadoutCoordinate::x,at+100,0),original);
    for(const auto &r:d->event_reviews()) {EXPECT_TRUE(r.spectral_context_unavailable);EXPECT_TRUE(r.source_protection_unavailable);EXPECT_FALSE(r.hard_event_accepted);}
    std::cout<<"CASE_E_INJECTION native_row="<<rows[at][0]<<" amplitude="<<amplitude<<" candidate_score="<<found->absolute_score<<" assessment_events="<<e->events().size()<<'\n';
}
}
