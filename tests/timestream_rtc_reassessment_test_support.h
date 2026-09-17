#pragma once
#include <citlali/core/pipeline/timestream_rtc_jump_reassessment.h>
#include "timestream_successor_identity_test_support.h"
namespace citlali::test::rtc_reassessment {
using namespace citlali::pipeline;
namespace support = citlali::test::timestream_successor;
struct Input {
    std::vector<double> times;
    std::vector<TimestreamPacketCounter> counters;
    NativePairedReadoutMatrix x, r;
    std::vector<NativeReadoutCoordinateState> xs, rs;
    double integration_half=.004096;
    TimestreamNativeRow first_native_row=100;
    std::string paired_identity="jump-test";
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
        auto timing=std::make_shared<const NativeNetworkAlignment>(0,first_native_row,support::time_vector(times),counters);
        std::vector<NativePairedReadoutOccurrenceBinding> occurrences;
        for (std::size_t i=0;i<times.size();++i)
            occurrences.push_back({static_cast<std::int64_t>(i+10000),static_cast<std::int64_t>(i+20000),{times[i]-integration_half,times[i]+integration_half}});
        auto axis=std::make_shared<const NativePairedReadoutOccurrenceAxis>(timing,first_native_row,std::move(occurrences));
        std::vector<NativePairedReadoutNetwork> networks;
        networks.push_back(NativePairedReadoutNetwork::admit(axis,support::detector_axis(0,3),
            support::mapping_authority(0,paired_identity),x,r,xs,rs));
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

} // namespace citlali::test::rtc_reassessment
