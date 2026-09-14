#include <citlali/core/pipeline/timestream_rtc_native_spectral_learn.h>
#include "timestream_rtc_reassessment_test_support.h"
#include "timestream_rtc_native_spectral_reference.h"
#include <gtest/gtest.h>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using namespace citlali::pipeline;
using citlali::test::rtc_reassessment::Input;

Input signal(std::size_t count=241, double dt=1./16) {
    Input in(count,dt);
    for (std::size_t i=0;i<count;++i) for (std::size_t d=0;d<3;++d) {
        const double t=i*dt;
        in.x(i,d)=3+.002*i+std::sin(2*std::numbers::pi*3*t)+.3*std::cos(2*std::numbers::pi*1.2*t)+d*.1;
        in.r(i,d)=-.5+.2*std::cos(2*std::numbers::pi*2.1*t)+d*.01;
    }
    if(count>90) in.x(90,0)+=7;
    return in;
}
void exclude(Input &in, std::size_t first, std::size_t last, NativeReadoutCoordinate c=NativeReadoutCoordinate::x) {
    auto &states=c==NativeReadoutCoordinate::x ? in.xs : in.rs;
    for(auto i=first;i<last;++i) states[i*3]=NativeReadoutCoordinateState::measured(true,false,true,true);
}
struct Trial {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcSpikeEvidence> spikes;
    std::shared_ptr<const RtcSpectralInputIdentity> binding;
    RtcSpectralCadenceDomain cadence;
    Trial(const Input &in, RtcSpikeProtection protection=RtcSpikeProtection::outside_source) {
        parent=in.freeze();val=ValSnapshot::initial(parent);
        spikes=learn_rtc_spike_candidates(NativePairedReadoutView::full(parent),val,
            RtcSpikeSourceProtection::admit(parent,"fixture-source",protection),1);
        auto subject=ValNativeRealization::create(parent,{ValProducer::align,1},1,ValNativeProductRole::original_input,0);
        binding=RtcSpectralInputIdentity::bind(subject,val,spikes->input_handle()->span(0),RtcSpectralInputStage::original_reference,"fixture-original-native",1);
        cadence={0,"fixture-cadence-authority",2*in.integration_half,1e-8};
    }
    auto learn() const {return RtcNativeSpectralEvidence::learn_initial(spikes,{binding},{cadence},2);}
    auto transients() const {
        std::vector<RtcEventPeerEligibility> peers;
        for(std::uint32_t d=0;d<3;++d)peers.push_back({0,d,parent->network(0).detector(d).detector_occurrence_id,true});
        return RtcEventAssessmentDecision::consider(learn_rtc_event_assessment(spikes,RtcEventPeerPopulation::admit(spikes,"fixture-peers",peers),3),val,4);
    }
};
// Isolated fault injection after the ingress invariant has already passed.
// The original matrix allocation is mutable storage; production APIs expose
// only const access. Restore the fixture's cell and never provide a production
// escape hatch merely to manufacture an otherwise impossible admitted NaN.
struct PayloadFault {
    double *cell; double saved;
    PayloadFault(const Trial &t, std::size_t row, double value) {
        cell=const_cast<double *>(t.parent->network(0).values(NativeReadoutCoordinate::x).data())+row*3;
        saved=*cell;*cell=value;
    }
    ~PayloadFault(){*cell=saved;}
};
template<class A,class B>void check_reference(const Input &in,const A &frequency,const B &psd,std::size_t windows) {
    Trial t(in);auto e=t.learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);const auto &n=e->network(0);
    ASSERT_TRUE(s.available());ASSERT_EQ(s.windows.size(),windows);ASSERT_EQ(s.psd.size(),psd.size());
    for(std::size_t i=0;i<psd.size();++i){EXPECT_NEAR(n.frequency_hz[i],frequency[i],1e-13);EXPECT_NEAR(s.psd[i],psd[i],2e-12*std::max(1.,psd[i]));}
    EXPECT_EQ(e->original_spike_handle(),t.spikes);EXPECT_EQ(n.input,t.binding);
}
TEST(rtc_native_spectral, even_grid_matches_preserved_python_estimator) {
    check_reference(signal(),spectral_reference::even_frequency,spectral_reference::even_psd,spectral_reference::even_windows);
}
TEST(rtc_native_spectral, odd_grid_preserves_inherited_last_bin_convention) {
    check_reference(signal(141,4./33),spectral_reference::odd_frequency,spectral_reference::odd_psd,spectral_reference::odd_windows);
}
TEST(rtc_native_spectral, short_stretch_padding_matches_preserved_estimator) {
    auto in=signal();exclude(in,70,80);exclude(in,112,241);
    check_reference(in,spectral_reference::padded_frequency,spectral_reference::padded_psd,spectral_reference::padded_windows);
    const auto e=Trial(in).learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);
    ASSERT_EQ(s.windows.size(),3);EXPECT_EQ(s.windows.back().rows.first,180);EXPECT_EQ(s.windows.back().rows.past_last,212);
    EXPECT_EQ(s.windows.back().padded_samples,32);
}
TEST(rtc_native_spectral, physical_runs_pool_windows_without_crossing_gap) {
    auto in=signal();for(std::size_t i=120;i<in.times.size();++i){in.counters[i]+=10;in.times[i]+=.625;}
    check_reference(in,spectral_reference::gapped_frequency,spectral_reference::gapped_psd,spectral_reference::gapped_windows);
    auto e=Trial(in).learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);ASSERT_EQ(s.runs.size(),2);
    for(const auto &w:s.windows){EXPECT_TRUE(w.rows.past_last<=220 || w.rows.first>=220);EXPECT_EQ(w.padded_samples,0);}
}
TEST(rtc_native_spectral, minimum_window_count_is_pooled_across_separate_full_length_runs) {
    auto in=signal(128);for(std::size_t i=64;i<in.times.size();++i){in.counters[i]+=3;in.times[i]+=1;}
    auto e=Trial(in).learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);
    ASSERT_TRUE(s.available());ASSERT_EQ(s.windows.size(),2);ASSERT_EQ(s.runs.size(),2);
    EXPECT_EQ(s.runs[0].past_last_window-s.runs[0].first_window,1);
    EXPECT_EQ(s.runs[1].past_last_window-s.runs[1].first_window,1);
    EXPECT_GT(s.windows[1].support_begin_unix_sec-s.windows[0].support_end_unix_sec,.9);
}
TEST(rtc_native_spectral, end_anchor_and_exact_native_integration_support_are_recorded) {
    auto in=signal();Trial t(in);auto e=t.learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);
    const std::array<TimestreamNativeRow,7> starts{100,132,164,196,228,260,277};ASSERT_EQ(s.windows.size(),starts.size());
    for(std::size_t i=0;i<starts.size();++i){const auto &w=s.windows[i];EXPECT_EQ(w.rows.first,starts[i]);EXPECT_EQ(w.rows.past_last,starts[i]+64);
        EXPECT_DOUBLE_EQ(w.support_begin_unix_sec,t.parent->network(0).occurrence_axis().occurrence(starts[i]).integration_support.begin_unix_sec);
        EXPECT_DOUBLE_EQ(w.support_end_unix_sec,t.parent->network(0).occurrence_axis().occurrence(starts[i]+63).integration_support.end_unix_sec);}
}
TEST(rtc_native_spectral, declared_invalid_nan_is_excluded_without_input_consistency_failure) {
    auto in=signal();in.x(70,0)=NAN;in.xs[70*3]=NativeReadoutCoordinateState::measured(true,false,true,false);
    Trial t(in);auto e=t.learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);
    ASSERT_TRUE(s.available());EXPECT_EQ(s.runs[0].declared_invalid_samples,1);EXPECT_EQ(s.runs[0].unexpected_nonfinite_samples,0);
    for(const auto &w:s.windows)EXPECT_TRUE(w.rows.past_last<=170 || w.rows.first>170);
    EXPECT_TRUE(std::isnan(t.parent->network(0).value(NativeReadoutCoordinate::x,170,0)));
    EXPECT_EQ(e->spectrum(0,0,NativeReadoutCoordinate::r).runs[0].declared_invalid_samples,0);
}
TEST(rtc_native_spectral, ingress_rejects_misdeclared_nonfinite_payload) {
    auto in=signal();in.x(70,0)=NAN;EXPECT_THROW(in.freeze(),std::invalid_argument);
}
TEST(rtc_native_spectral, unexpected_admitted_nan_and_infinity_invalidate_only_affected_coordinate_run) {
    for(double v:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){
        auto in=signal();Trial t(in);PayloadFault fault(t,70,v);auto e=t.learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);
        EXPECT_FALSE(s.available());EXPECT_EQ(s.cause,RtcSpectralCause::input_consistency_failure);
        EXPECT_EQ(s.runs[0].cause,RtcSpectralRunCause::input_consistency_failure);EXPECT_EQ(s.runs[0].unexpected_nonfinite_samples,1);
        EXPECT_EQ(s.runs[0].first_unexpected_nonfinite,170);EXPECT_TRUE(s.windows.empty());EXPECT_TRUE(s.psd.empty());
        EXPECT_TRUE(e->spectrum(0,0,NativeReadoutCoordinate::r).available());EXPECT_TRUE(e->spectrum(0,1,NativeReadoutCoordinate::x).available());
        EXPECT_FALSE(std::isfinite(t.parent->network(0).value(NativeReadoutCoordinate::x,170,0)));
    }
}
TEST(rtc_native_spectral, failed_run_is_named_and_never_silently_contributes_to_other_run_spectrum) {
    auto in=signal(500);for(std::size_t i=120;i<in.times.size();++i){in.counters[i]+=10;in.times[i]+=.625;}
    Trial t(in);PayloadFault fault(t,70,NAN);auto e=t.learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);
    ASSERT_TRUE(s.available());EXPECT_EQ(s.cause,RtcSpectralCause::available_with_unavailable_runs);
    EXPECT_EQ(s.runs[0].cause,RtcSpectralRunCause::input_consistency_failure);EXPECT_EQ(s.runs[0].past_last_window,0);
    for(auto w:s.windows){EXPECT_EQ(w.run_index,1);EXPECT_GE(w.rows.first,220);}
    for(auto r:s.centering_support)EXPECT_GE(r.first,220);
}
TEST(rtc_native_spectral, insufficient_windows_and_fixed_grid_are_explicit_unavailable) {
    auto one=Trial(signal(64)).learn();const auto &a=one->spectrum(0,0,NativeReadoutCoordinate::x);
    EXPECT_EQ(a.cause,RtcSpectralCause::insufficient_windows);EXPECT_EQ(a.windows.size(),1);EXPECT_TRUE(a.psd.empty());
    auto short_input=signal(80);for(std::size_t i=40;i<short_input.times.size();++i)short_input.counters[i]+=3;
    auto short_e=Trial(short_input).learn();const auto &b=short_e->spectrum(0,0,NativeReadoutCoordinate::x);
    EXPECT_EQ(b.cause,RtcSpectralCause::fixed_grid_unavailable);EXPECT_TRUE(b.psd.empty());EXPECT_EQ(b.runs.size(),2);
}
TEST(rtc_native_spectral, excluded_short_stretches_are_never_compacted_into_a_window) {
    auto in=signal();for(std::size_t i=30;i<in.times.size();i+=31)exclude(in,i,i+1);
    auto e=Trial(in).learn();const auto &s=e->spectrum(0,0,NativeReadoutCoordinate::x);
    EXPECT_EQ(s.cause,RtcSpectralCause::fixed_grid_unavailable);EXPECT_TRUE(s.windows.empty());EXPECT_GT(s.centering_support.size(),4);
}
TEST(rtc_native_spectral, astronomical_signal_is_retained_for_outside_protected_and_unknown_status) {
    auto in=signal();for(std::size_t i=0;i<in.times.size();++i)in.x(i,0)+=80*std::exp(-std::pow((static_cast<double>(i)-100)/12,2));
    auto outside=Trial(in,RtcSpikeProtection::outside_source).learn();auto protected_e=Trial(in,RtcSpikeProtection::protected_source).learn();auto unknown=Trial(in,RtcSpikeProtection::unavailable).learn();
    const auto &a=outside->spectrum(0,0,NativeReadoutCoordinate::x);ASSERT_TRUE(a.available());
    EXPECT_EQ(a.psd,protected_e->spectrum(0,0,NativeReadoutCoordinate::x).psd);EXPECT_EQ(a.psd,unknown->spectrum(0,0,NativeReadoutCoordinate::x).psd);
    for(auto w:protected_e->spectrum(0,0,NativeReadoutCoordinate::x).windows)EXPECT_EQ(w.source_counts[1],w.rows.past_last-w.rows.first);
    for(auto w:unknown->spectrum(0,0,NativeReadoutCoordinate::x).windows)EXPECT_EQ(w.source_counts[2],w.rows.past_last-w.rows.first);
}
TEST(rtc_native_spectral, failed_noise_screening_is_not_mistaken_for_producer_invalid_input) {
    Trial t(signal());EXPECT_FALSE(t.spikes->blocks()[0].coordinates[0].available());EXPECT_TRUE(t.learn()->spectrum(0,0,NativeReadoutCoordinate::x).available());
}
TEST(rtc_native_spectral, exact_initial_input_stage_support_and_val_are_required) {
    Trial t(signal());auto other=ValSnapshot::initial(t.parent);
    auto bad=RtcSpectralInputIdentity::bind(t.binding->subject_handle(),other,t.binding->support(),RtcSpectralInputStage::original_reference,"original",1);
    EXPECT_THROW(RtcNativeSpectralEvidence::learn_initial(t.spikes,{bad},{t.cadence},2),std::invalid_argument);
    auto support=t.binding->support();++support.first_native_row;
    bad=RtcSpectralInputIdentity::bind(t.binding->subject_handle(),t.val,support,RtcSpectralInputStage::original_reference,"original",1);
    EXPECT_THROW(RtcNativeSpectralEvidence::learn_initial(t.spikes,{bad},{t.cadence},2),std::invalid_argument);
    bad=RtcSpectralInputIdentity::bind(t.binding->subject_handle(),t.val,t.binding->support(),RtcSpectralInputStage::original_reference,"original",99);
    EXPECT_THROW(RtcNativeSpectralEvidence::learn_initial(t.spikes,{bad},{t.cadence},2),std::invalid_argument);
    Trial other_input(signal());EXPECT_THROW(RtcNativeSpectralEvidence::learn_initial(t.spikes,{other_input.binding},{t.cadence},2),std::invalid_argument);
    EXPECT_THROW(RtcNativeSpectralEvidence::learn_initial(t.spikes,{t.binding},{t.cadence},0),std::invalid_argument);
}
TEST(rtc_native_spectral, future_stage_identity_can_bind_later_val_without_relabeling_initial_evidence) {
    Trial t(signal());auto first=t.learn();
    ValDeltaBuilder b{t.val,{ValProducer::rtc,44}};b.propose(t.val->address(0,100,0),ValFactCode{1},ValFactState{1},ValFactCause{1});
    auto later=ValSnapshot::commit(b.freeze());
    auto derived=ValNativeRealization::create(t.parent,{ValProducer::rtc,45},2,ValNativeProductRole::derived_residual,0);
    auto identity=RtcSpectralInputIdentity::bind(derived,later,t.binding->support(),RtcSpectralInputStage::native_conditioned,"explicit-future-native-stage",2);
    EXPECT_EQ(identity->snapshot_handle(),later);EXPECT_EQ(identity->subject_handle(),derived);EXPECT_EQ(identity->snapshot_handle()->generation().value,1);
    EXPECT_EQ(first->network(0).input->snapshot_handle(),t.val);EXPECT_EQ(first->network(0).input->stage(),RtcSpectralInputStage::original_reference);
    EXPECT_THROW(RtcNativeSpectralEvidence::learn_initial(t.spikes,{identity},{t.cadence},3),std::invalid_argument);
    EXPECT_THROW(RtcSpectralInputIdentity::bind(derived,later,t.binding->support(),RtcSpectralInputStage::original_reference,"false-original",2),std::invalid_argument);
}
TEST(rtc_native_spectral, cadence_domain_is_explicit_and_invalid_cadence_stays_unavailable) {
    Trial t(signal());auto domain=t.cadence;domain.nominal_interval_seconds*=2;
    auto e=RtcNativeSpectralEvidence::learn_initial(t.spikes,{t.binding},{domain},2);
    EXPECT_EQ(e->spectrum(0,0,NativeReadoutCoordinate::x).cause,RtcSpectralCause::cadence_unavailable);
    domain=t.cadence;domain.authority.clear();EXPECT_THROW(RtcNativeSpectralEvidence::learn_initial(t.spikes,{t.binding},{domain},2),std::invalid_argument);
}
TEST(rtc_native_spectral, finite_arithmetic_overflow_cannot_produce_available_spectrum) {
    auto in=signal(242);for(Eigen::Index i=0;i<in.x.rows();++i)in.x(i,0)=std::numeric_limits<double>::max();
    auto e=Trial(in).learn();EXPECT_EQ(e->spectrum(0,0,NativeReadoutCoordinate::x).cause,RtcSpectralCause::arithmetic_nonfinite);
    EXPECT_TRUE(e->spectrum(0,0,NativeReadoutCoordinate::r).available());
}
TEST(rtc_native_spectral, controlled_transient_and_spectral_evidence_are_joint_consider_inputs_without_admission) {
    Input in(1600,1./128);in.spike();Trial t(in);auto e=t.learn();auto review=t.transients();ASSERT_FALSE(review->evidence_handle()->events().empty());
    auto joint=RtcSpectralTransientConsideration::consider(e,t.val,review,t.val,5);
    EXPECT_EQ(joint->spectral_handle(),e);EXPECT_EQ(joint->transient_handle(),review);EXPECT_TRUE(joint->event_run_contributes(0,NativeReadoutCoordinate::x));
    EXPECT_FALSE(joint->notch_admitted);EXPECT_FALSE(joint->spike_admitted);EXPECT_FALSE(joint->apply_authorized);
    EXPECT_FALSE(review->event_reviews()[0].hard_event_accepted);EXPECT_TRUE(review->event_reviews()[0].spectral_context_unavailable);
    auto other=ValSnapshot::initial(t.parent);EXPECT_THROW(RtcSpectralTransientConsideration::consider(e,other,review,t.val,5),std::invalid_argument);
    EXPECT_THROW(RtcSpectralTransientConsideration::consider(e,t.val,review,other,5),std::invalid_argument);
    Trial different(in);EXPECT_THROW(RtcSpectralTransientConsideration::consider(e,t.val,different.transients(),different.val,5),std::invalid_argument);
    const auto &s=joint->event_spectrum(0,NativeReadoutCoordinate::x);bool contains=false;
    for(auto w:s.windows)contains|=w.rows.first<=600 && 600<w.rows.past_last;EXPECT_TRUE(contains);
}
TEST(rtc_native_spectral, learn_preserves_every_original_coordinate_bit_and_publishes_review_fixture) {
    auto in=signal();Trial t(in);auto e=t.learn();std::size_t checked=0;
    for(auto c:{NativeReadoutCoordinate::x,NativeReadoutCoordinate::r})for(std::size_t i=0;i<in.times.size();++i)for(std::size_t d=0;d<3;++d){
        EXPECT_EQ(std::bit_cast<std::uint64_t>(t.parent->network(0).value(c,i+100,d)),std::bit_cast<std::uint64_t>((c==NativeReadoutCoordinate::x ? in.x : in.r)(i,d)));++checked;
    }
    EXPECT_EQ(checked,1446);
    if(const char *path=std::getenv("CITLALI_SPECTRAL_REVIEW_CSV")){
        std::ofstream f(path);ASSERT_TRUE(f);f<<std::setprecision(17)<<"frequency_hz,x_psd,r_psd\n";
        const auto &n=e->network(0);const auto &x=e->spectrum(0,0,NativeReadoutCoordinate::x);const auto &r=e->spectrum(0,0,NativeReadoutCoordinate::r);
        for(std::size_t i=0;i<n.frequency_hz.size();++i)f<<n.frequency_hz[i]<<','<<x.psd[i]<<','<<r.psd[i]<<'\n';
        std::ofstream support(std::string(path)+".windows.csv");ASSERT_TRUE(support);
        support<<std::setprecision(17)<<"first_row,past_last_row,support_begin_unix_sec,support_end_unix_sec,padded_samples\n";
        for(auto w:x.windows)support<<w.rows.first<<','<<w.rows.past_last<<','<<w.support_begin_unix_sec<<','<<w.support_end_unix_sec<<','<<w.padded_samples<<'\n';
        std::ofstream raw(std::string(path)+".raw.csv");ASSERT_TRUE(raw);raw<<std::setprecision(17)<<"time_seconds,x,r\n";
        for(std::size_t i=0;i<in.times.size();++i)raw<<in.times[i]-1000<<','<<in.x(i,0)<<','<<in.r(i,0)<<'\n';
    }
}
TEST(rtc_native_spectral, bounded_learning_timing_reports_actual_numeric_and_support_storage) {
    Trial t(signal(1600,1./128));auto start=std::chrono::steady_clock::now();std::shared_ptr<const RtcNativeSpectralEvidence> e;
    for(int i=0;i<100;++i)e=t.learn();
    std::cout<<"spectral_learn_100_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
        <<" logical_bytes="<<e->logical_owned_bytes()<<" peak_scratch_samples="<<e->peak_scratch_samples()<<'\n';
    EXPECT_GT(e->logical_owned_bytes(),0);EXPECT_LT(e->peak_scratch_samples(),10000);
}
TEST(rtc_native_spectral, scratch_bound_includes_cadence_when_all_coordinate_samples_are_invalid) {
    auto in=signal(10000);
    for(auto &state:in.xs)state=NativeReadoutCoordinateState::measured(true,false,true,true);
    for(auto &state:in.rs)state=NativeReadoutCoordinateState::measured(true,false,true,true);
    auto e=Trial(in).learn();
    for(const auto &s:e->spectra())EXPECT_FALSE(s.available());
    // Cadence retains the original time axis even when no coordinate samples
    // qualify; its median copy coexists with the interval population.
    EXPECT_GE(e->peak_scratch_samples(),2*(in.times.size()-1));
}
} // namespace
