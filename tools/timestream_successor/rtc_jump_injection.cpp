// Fixed validation catalog, not a detector policy or production route.
#include <citlali/core/pipeline/timestream_rtc_jump_reassessment.h>
#include "../../tests/timestream_successor_identity_test_support.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include "rtc_jump_loss_anchor.h"
namespace {
using namespace citlali::pipeline;
namespace support=citlali::test::timestream_successor;
struct Background {
    std::vector<double> times,x,r;
    std::vector<NativeReadoutIntegrationSupport> cells;
    std::vector<NativeReadoutCoordinateState> xs,rs;
    std::vector<std::int64_t> source_rows;
    std::string identity;
    auto freeze() const {
        std::vector<TimestreamPacketCounter> counters(times.size());
        std::vector<NativePairedReadoutOccurrenceBinding> occurrences;
        for(std::size_t i=0;i<times.size();++i) {counters[i]=i+1;occurrences.push_back({source_rows[i],source_rows[i],cells[i]});}
        auto timing=std::make_shared<const NativeNetworkAlignment>(0,0,support::time_vector(times),counters);
        auto axis=std::make_shared<const NativePairedReadoutOccurrenceAxis>(timing,0,std::move(occurrences));
        NativePairedReadoutMatrix mx(times.size(),1),mr(times.size(),1);
        for(std::size_t i=0;i<times.size();++i){mx(i,0)=x[i];mr(i,0)=r[i];}
        std::vector<NativePairedReadoutNetwork> networks;
        networks.push_back(NativePairedReadoutNetwork::admit(axis,support::detector_axis(0,1),support::mapping_authority(0,identity),std::move(mx),std::move(mr),xs,rs));
        return support::make_observation(std::move(networks),{0});
    }
};
struct Attempt {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcSpikeEvidence> spikes;
    std::shared_ptr<const RtcEventAssessmentEvidence> assessment;
    std::shared_ptr<const RtcJumpTransitionEvidence> transition;
    std::shared_ptr<const RtcJumpReassessmentDecision> final;
    explicit Attempt(const Background &b) {
        parent=b.freeze();val=ValSnapshot::initial(parent);
        spikes=learn_rtc_spike_candidates(NativePairedReadoutView::full(parent),val,
            RtcSpikeSourceProtection::admit(parent,"injection-protection-unavailable",RtcSpikeProtection::unavailable),1);
        std::vector<RtcEventPeerEligibility> peers{{0,0,parent->network(0).detectors()[0].detector_occurrence_id,true}};
        assessment=learn_rtc_event_assessment(spikes,RtcEventPeerPopulation::admit(spikes,"single-detector-injection",std::move(peers)),2);
        auto review=RtcEventAssessmentDecision::consider(assessment,val,3);
        auto amplitude=RtcJumpAmplitudeDecision::consider(review,val,4);
        auto shorter=RtcJumpConsistencyEvidence::learn(amplitude,5);
        auto consistency=RtcJumpConsistencyDecision::consider(shorter,val,6);
        transition=RtcJumpTransitionEvidence::learn(RtcJumpTransitionRequest::consider(consistency,val,7),8);
        auto audit=RtcJumpSupportEvidence::learn(transition,9);
        auto refit=RtcJumpRefitEvidence::learn(RtcJumpRefitRequest::consider(audit,val,10),11);
        auto measured=RtcJumpReassessmentEvidence::learn(RtcJumpRemeasureRequest::consider(refit,val,12),13);
        final=RtcJumpReassessmentDecision::consider(measured,val,14);
    }
};
void bound(std::ostream &out,const RtcJumpTransition &b) {
    if(!b.available()){out<<"null";return;}
    out<<'['<<std::setprecision(17)<<b.physical_bound.begin_unix_sec<<','<<b.physical_bound.end_unix_sec<<']';
}
}
#include "rtc_jump_loss_diagnostic.h"
int main(int argc,char **argv) {
    try {
        const bool diagnostic=argc==4 && std::string(argv[3])=="--loss-diagnosis";
        if(argc!=3 && !diagnostic) throw std::invalid_argument("expected background text (or synthetic), exact identity, optional --loss-diagnosis");
        Background original;original.identity=argv[2];
        if(std::string(argv[1])=="synthetic") {
            for(int i=0;i<2442;++i) {
                const double t=1000+.008192*i,u=(i-1221)*.008192;
                const double n=std::array{-.1,.0,.1,.0,.0,-.1,.1}[i%7];
                original.times.push_back(t);original.cells.push_back({t-.004096,t+.004096});original.source_rows.push_back(i);
                original.x.push_back(10+.8*u+.1*u*u-.005*u*u*u+n);original.r.push_back(-3+.4*u+2*n);
                original.xs.push_back(support::valid_states(1)[0]);original.rs.push_back(support::valid_states(1)[0]);
            }
        } else {
            std::ifstream in(argv[1]);if(!in) throw std::invalid_argument("missing fixed real background");
            std::int64_t row;double t,b,e,x,r;bool xv,rv;
            while(in>>row>>t>>b>>e>>x>>xv>>r>>rv) {
                if(!std::isfinite(t)||!std::isfinite(x)||!std::isfinite(r)||!(b<e)) throw std::invalid_argument("nonfinite injection background");
                original.times.push_back(t);original.cells.push_back({b,e});original.source_rows.push_back(row);
                original.x.push_back(x);original.r.push_back(r);
                if(!xv||!rv) throw std::invalid_argument("fixed background contains producer-invalid samples; unavailable fixture");
                original.xs.push_back(support::valid_states(1)[0]);original.rs.push_back(support::valid_states(1)[0]);
            }
            if(!in.eof()) throw std::invalid_argument("invalid background record");
        }
        if(original.times.size()<1500) throw std::invalid_argument("fixed background insufficient");
        const std::size_t center=original.times.size()/2;
        Attempt baseline(original);
        std::array<double,2> sigma{NAN,NAN};
        for(const auto &b:baseline.spikes->blocks()) if(b.first<=static_cast<TimestreamNativeRow>(center) && b.past_last>static_cast<TimestreamNativeRow>(center))
            for(std::size_t c=0;c<2;++c) if(b.coordinates[c].available()) sigma[c]=b.coordinates[c].scale;
        for(double s:sigma) if(!std::isfinite(s)||s<=0) throw std::invalid_argument("fixed background noise unavailable");
        if(diagnostic) return diagnose_injections(original,sigma,std::string(argv[1])=="synthetic");
        struct Trial {const char *name;int cells;bool jump,spike;};
        const std::array trials{Trial{"unmodified",0,false,false},Trial{"sharp_step",0,true,false},Trial{"finite_3_cells",3,true,false},Trial{"finite_12_cells",12,true,false},Trial{"sharp_plus_neighbor_spike",0,true,true},Trial{"spike_only_control",0,false,true}};
        for(const auto trial:trials) {
            auto input=original;input.identity=original.identity+":"+trial.name;
            if(trial.jump) for(std::size_t i=center;i<input.times.size();++i) {
                const double f=trial.cells ? std::min(1.,(double(i-center)+.5)/trial.cells) : i==center ?.5:1.;
                input.x[i]+=20*sigma[0]*f;input.r[i]-=20*sigma[1]*f;
            }
            if(trial.spike) {const auto i=trial.jump?center+25:center;input.x[i]+=12*sigma[0];input.r[i]+=12*sigma[1];}
            Attempt a(input);std::vector<std::size_t> candidates;
            const auto last=center+std::max(1,trial.cells);
            for(std::size_t i=0;i<a.spikes->candidates().size();++i){const auto &s=a.spikes->candidates()[i];if(s.earlier_row<static_cast<TimestreamNativeRow>(last) && s.later_row>=static_cast<TimestreamNativeRow>(center)) candidates.push_back(i);}
            std::cout<<"{\"background_identity\":"<<std::quoted(original.identity)<<",\"trial\":"<<std::quoted(trial.name)<<",\"jump_injected\":"<<(trial.jump?"true":"false")<<",\"truth_cells\":["<<center<<','<<last<<"],\"truth_time\":["<<std::setprecision(17)<<(trial.cells?input.cells[center].begin_unix_sec:input.times[center])<<','<<(trial.cells?input.cells[last-1].end_unix_sec:input.times[center])<<"],\"sigma_delta_original\":["<<sigma[0]<<','<<sigma[1]<<"],\"candidate_edges_touching_truth_cells\":"<<candidates.size()<<",\"all_candidate_edges\":"<<a.spikes->candidates().size()<<",\"groups\":[";
            bool comma=false;
            const auto &re=*a.final->evidence_handle();const auto &audit=*re.request_handle()->refit_handle()->request_handle()->audit_handle();
            for(std::size_t i=0;i<a.assessment->events().size();++i) {
                const auto &event=a.assessment->events()[i];bool associated=false;
                for(auto c:event.candidates) associated|=std::find(candidates.begin(),candidates.end(),c)!=candidates.end();
                if(!associated)continue;if(comma)std::cout<<',';comma=true;
                const auto found=std::find_if(audit.audits().begin(),audit.audits().end(),[&](const auto &s){return s.event==i;});
                std::cout<<"{\"event\":"<<i<<",\"coordinates\":[";
                for(std::size_t c=0;c<2;++c){if(c)std::cout<<',';std::cout<<"{\"before\":";bound(std::cout,a.transition->coordinates()[i][c]);
                    std::cout<<",\"before_cause\":"<<int(a.transition->coordinates()[i][c].cause)<<",\"after\":";
                    if(found==audit.audits().end()){std::cout<<"null,\"diagnostic_cause\":1";}
                    else {const auto index=std::distance(audit.audits().begin(),found);const auto cause=a.final->coordinates()[index][c];
                        if(cause==RtcJumpReassessmentCause::unchanged_measured)bound(std::cout,a.transition->coordinates()[i][c]);
                        else if(cause==RtcJumpReassessmentCause::reassessed_measured)bound(std::cout,re.coordinates()[index][c].transition);else std::cout<<"null";
                        std::cout<<",\"diagnostic_cause\":"<<int(cause);
                    }std::cout<<'}';
                }std::cout<<"]}";
            }
            std::cout<<"],\"hard_event_accepted\":false,\"apply_authorized\":false}\n";
        }
        return 0;
    } catch(const std::exception &e){std::cerr<<"injection unavailable: "<<e.what()<<'\n';return 2;}
}
