#include <citlali/core/pipeline/timestream_rtc_line_power.h>
#include "timestream_rtc_reassessment_test_support.h"
#include <gtest/gtest.h>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>

namespace {
using namespace citlali::pipeline;
using Input=citlali::test::rtc_reassessment::Input;
constexpr auto X=NativeReadoutCoordinate::x;
constexpr auto R=NativeReadoutCoordinate::r;
struct Trial {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcSpikeEvidence> spikes;
    std::shared_ptr<const RtcNativeSpectralEvidence> spectral;
    explicit Trial(const Input &in,RtcSpikeProtection protection=RtcSpikeProtection::unavailable,double tolerance=1e-8) {
        parent=in.freeze();val=ValSnapshot::initial(parent);
        spikes=learn_rtc_spike_candidates(NativePairedReadoutView::full(parent),val,
            RtcSpikeSourceProtection::admit(parent,"line-fixture-source",protection),1);
        auto native=ValNativeRealization::create(parent,{ValProducer::align,1},1,ValNativeProductRole::original_input,0);
        auto identity=RtcSpectralInputIdentity::bind(native,val,spikes->input_handle()->span(0),RtcSpectralInputStage::original_reference,"line-fixture-native",1);
        spectral=RtcNativeSpectralEvidence::learn_initial(spikes,{identity},{{0,"line-fixture-cadence",2*in.integration_half,tolerance}},2);
    }
    auto learn(RtcLinePowerProfile p=RtcLinePowerProfile::initial_2_hz) const {return RtcLinePowerEvidence::learn(spectral,val,p,3);}
    auto joint() const {
        std::vector<RtcEventPeerEligibility> peers;
        for(std::uint32_t d=0;d<3;++d)peers.push_back({0,d,parent->network(0).detector(d).detector_occurrence_id,true});
        auto review=RtcEventAssessmentDecision::consider(learn_rtc_event_assessment(spikes,RtcEventPeerPopulation::admit(spikes,"line-fixture-peers",peers),4),val,5);
        return RtcSpectralTransientConsideration::consider(spectral,val,review,val,6);
    }
};
Input zero(std::size_t count=241,double dt=1./16) {Input in(count,dt);in.x.setZero();in.r.setZero();return in;}
void tone(Input &in,double frequency,double amplitude=2.,bool in_r=false) {
    for(std::size_t i=0;i<in.times.size();++i)(in_r?in.r:in.x)(i,0)+=amplitude*std::sin(2*std::numbers::pi*frequency*(in.times[i]-in.times[0])+.31);
}
std::vector<double> grid(std::size_t n=33,double df=.25){std::vector<double> f(n);for(std::size_t i=0;i<n;++i)f[i]=i*df;return f;}
struct Gaussian {
    std::mt19937 generator;
    explicit Gaussian(unsigned seed):generator(seed){}
    double operator()(){const double u=(generator()+.5)/4294967296.,v=(generator()+.5)/4294967296.;return std::sqrt(-2*std::log(u))*std::cos(2*std::numbers::pi*v);}
};
Input noise(std::size_t n,unsigned seed,double rho=0,double dt=1./16) {
    auto in=zero(n,dt);Gaussian g(seed);double state=g();
    for(std::size_t i=0;i<n;++i){state=rho*state+std::sqrt(1-rho*rho)*g();in.x(i,0)=state;in.r(i,0)=.5*g();}
    return in;
}
void gap_and_padding(Input &in) {
    for(std::size_t i=0;i<in.times.size();++i){
        if(i>=80){in.times[i]+=.625;in.counters[i]+=10;}
        if((i>=70&&i<80)||i>=112)for(std::size_t d=0;d<3;++d){
            in.xs[i*3+d]=NativeReadoutCoordinateState::measured(true,false,true,true);
            in.rs[i*3+d]=NativeReadoutCoordinateState::measured(true,false,true,true);
        }
    }
}
TEST(rtc_line_power, finite_single_bin_excess_has_nonzero_power) {
    auto f=grid();std::vector<double> p(f.size(),1);p[12]=100;
    auto s=rtc_line_power_detail::measure(f,p,2);ASSERT_EQ(s.regions.size(),1);
    const auto &c=s.regions[0];EXPECT_DOUBLE_EQ(c.positive_excess_power,24.75);EXPECT_DOUBLE_EQ(c.bin_span_hz,.25);
    EXPECT_DOUBLE_EQ(s.total_stored_psd_power,33.);EXPECT_DOUBLE_EQ(c.stored_psd_power_fraction,.75);
    EXPECT_DOUBLE_EQ(c.peak_contrast,100);EXPECT_FALSE(c.extent_incomplete);EXPECT_FALSE(c.background_neighborhood_truncated);
}
TEST(rtc_line_power, neighborhoods_use_actual_inclusive_grid_and_record_clipping) {
    auto f=grid();std::vector<double> p(f.size(),1);p[2]=5;
    auto s=rtc_line_power_detail::measure(f,p,2);
    EXPECT_EQ(s.neighborhoods[8].first_bin,0);EXPECT_EQ(s.neighborhoods[8].past_last_bin,17);
    EXPECT_FALSE(s.neighborhoods[8].clipped_low);EXPECT_TRUE(s.neighborhoods[7].clipped_low);
    EXPECT_EQ(s.neighborhoods[2].past_last_bin,11);ASSERT_EQ(s.regions.size(),1);
    EXPECT_FALSE(s.regions[0].extent_incomplete);EXPECT_TRUE(s.regions[0].background_neighborhood_truncated);
    p[0]=4;s=rtc_line_power_detail::measure(f,p,2);EXPECT_TRUE(s.regions[0].extent_incomplete);
    f=grid(33,.2501);s=rtc_line_power_detail::measure(f,p,2);
    EXPECT_EQ(s.neighborhoods[16].first_bin,9);EXPECT_EQ(s.neighborhoods[16].past_last_bin,24);
}
TEST(rtc_line_power, equality_splits_regions_and_connected_peaks_remain_one_descriptor) {
    auto f=grid();std::vector<double> p(f.size(),1);p[12]=5;p[13]=2;p[14]=7;
    auto s=rtc_line_power_detail::measure(f,p,2);ASSERT_EQ(s.regions.size(),1);EXPECT_EQ(s.regions[0].peak_bin,14);EXPECT_DOUBLE_EQ(s.regions[0].bin_span_hz,.75);
    p[13]=1;s=rtc_line_power_detail::measure(f,p,2);ASSERT_EQ(s.regions.size(),2);
    p[13]=2;p[14]=5;s=rtc_line_power_detail::measure(f,p,2);EXPECT_EQ(s.regions[0].peak_bin,12);
}
TEST(rtc_line_power, zero_spectrum_is_available_without_candidates_or_noise_claim) {
    Trial t(zero());auto e=t.learn();const auto &s=e->coordinate(0,0,X);
    ASSERT_TRUE(s.available());EXPECT_TRUE(s.regions.empty());EXPECT_DOUBLE_EQ(s.total_stored_psd_power,0);EXPECT_FALSE(s.persistence_measured);
}
TEST(rtc_line_power, background_and_parent_psd_preserve_negative_residuals) {
    auto in=noise(1024,341);Trial t(in);auto e=t.learn();const auto &s=e->coordinate(0,0,X);const auto &p=t.spectral->spectrum(0,0,X).psd;
    ASSERT_TRUE(s.available());bool negative=false;long double signed_power=0;
    for(std::size_t i=0;i<p.size();++i){negative|=p[i]<s.background[i];signed_power+=(p[i]-s.background[i])*s.bin_increment_hz;}
    EXPECT_TRUE(negative);auto b=e->measure_band(0,0,X,"predeclared-full-native",0,8);EXPECT_NEAR(b.signed_residual_power,static_cast<double>(signed_power),1e-14);
    EXPECT_EQ(b.first_bin,0);EXPECT_EQ(b.past_last_bin,p.size());
    EXPECT_THROW(e->measure_band(0,0,X,"",0,8),std::invalid_argument);
    EXPECT_THROW(e->measure_band(0,0,X,"outside",0,9),std::invalid_argument);
    EXPECT_THROW(e->measure_band(0,0,X,"empty",.01,.02),std::invalid_argument);
}
TEST(rtc_line_power, spectral_unavailability_remains_coordinate_local_and_retains_exact_cause_parent) {
    auto in=zero();tone(in,3);
    for(std::size_t i=0;i<in.times.size();++i){in.x(i,0)=NAN;in.xs[3*i]=NativeReadoutCoordinateState::measured(true,false,true,false);}
    Trial t(in);auto e=t.learn();EXPECT_EQ(e->coordinate(0,0,X).cause,RtcLinePowerCause::spectral_unavailable);EXPECT_TRUE(e->coordinate(0,0,R).available());
    EXPECT_EQ(e->spectral_handle(),t.spectral);EXPECT_TRUE(std::isnan(t.parent->network(0).value(X,100,0)));
    EXPECT_THROW(e->measure_band(0,0,X,"unavailable",1,4),std::invalid_argument);
    Trial too_short(zero(64));EXPECT_EQ(too_short.learn()->coordinate(0,0,X).cause,RtcLinePowerCause::spectral_unavailable);
}
TEST(rtc_line_power, exact_snapshot_attempt_and_named_profile_are_required) {
    Trial t(zero());auto other=ValSnapshot::initial(t.parent);
    EXPECT_THROW(RtcLinePowerEvidence::learn(t.spectral,other,RtcLinePowerProfile::initial_2_hz,3),std::invalid_argument);
    EXPECT_THROW(RtcLinePowerEvidence::learn(t.spectral,t.val,static_cast<RtcLinePowerProfile>(99),3),std::invalid_argument);
    EXPECT_THROW(RtcLinePowerEvidence::learn(t.spectral,t.val,RtcLinePowerProfile::initial_2_hz,0),std::invalid_argument);
    EXPECT_THROW(RtcLinePowerEvidence::learn(nullptr,t.val,RtcLinePowerProfile::initial_2_hz,3),std::invalid_argument);
    EXPECT_DOUBLE_EQ(t.learn(RtcLinePowerProfile::sensitivity_1_hz)->radius_hz(),1);
    EXPECT_DOUBLE_EQ(t.learn(RtcLinePowerProfile::sensitivity_4_hz)->radius_hz(),4);
}
TEST(rtc_line_power, source_status_is_annotation_and_originals_and_spectra_are_unchanged) {
    auto in=noise(241,123);tone(in,3);std::vector<double> reference;
    for(auto protection:{RtcSpikeProtection::outside_source,RtcSpikeProtection::protected_source,RtcSpikeProtection::unavailable}){
        Trial t(in,protection);const auto saved=t.spectral->spectrum(0,0,X).psd;auto e=t.learn();
        if(reference.empty())reference=e->coordinate(0,0,X).background;else EXPECT_EQ(reference,e->coordinate(0,0,X).background);
        EXPECT_EQ(saved,t.spectral->spectrum(0,0,X).psd);EXPECT_EQ(e->snapshot_handle(),t.val);
        for(auto c:{X,R})for(std::size_t i=0;i<in.times.size();++i)for(std::size_t d=0;d<3;++d)
            EXPECT_EQ(std::bit_cast<std::uint64_t>(t.parent->network(0).value(c,100+i,d)),std::bit_cast<std::uint64_t>((c==X?in.x:in.r)(i,d)));
    }
}
TEST(rtc_line_power, r_only_features_are_evidence_without_shared_notch_action) {
    auto in=zero();tone(in,3,4,true);Trial t(in);auto e=t.learn();auto ranked=RtcLinePowerConsideration::rank(e,t.joint(),7);
    EXPECT_TRUE(e->coordinate(0,0,X).regions.empty());EXPECT_FALSE(e->coordinate(0,0,R).regions.empty());
    EXPECT_TRUE(ranked->shared_notch_requires_direct_x);EXPECT_FALSE(ranked->notch_proposed);EXPECT_FALSE(ranked->interference_admitted);EXPECT_FALSE(ranked->apply_authorized);
    EXPECT_TRUE(ranked->ranks()[0].empty());EXPECT_FALSE(ranked->ranks()[1].empty());
    Trial other(in);
    EXPECT_THROW(RtcLinePowerConsideration::rank(e,other.joint(),7),std::invalid_argument);
}
TEST(rtc_line_power, diagnostic_ranking_is_deterministic_and_keeps_joint_transient_context) {
    Input in(1600,1./128);in.spike();tone(in,3);Trial t(in);auto e=t.learn();auto joint=t.joint();auto a=RtcLinePowerConsideration::rank(e,joint,7);auto b=RtcLinePowerConsideration::rank(e,joint,8);
    EXPECT_EQ(a->ranks(),b->ranks());EXPECT_EQ(a->joint_handle(),joint);EXPECT_EQ(a->line_handle(),e);
    ASSERT_FALSE(joint->transient_handle()->evidence_handle()->events().empty());
    for(std::size_t i=0;i<a->ranks().size();++i)for(std::size_t j=1;j<a->ranks()[i].size();++j)
        EXPECT_GE(e->coordinates()[i].regions[a->ranks()[i][j-1]].positive_excess_power,e->coordinates()[i].regions[a->ranks()[i][j]].positive_excess_power);
    EXPECT_FALSE(e->coordinate(0,0,X).persistence_measured);EXPECT_FALSE(a->apply_authorized);
}
TEST(rtc_line_power, invalid_grids_and_overflow_cannot_be_available_measurements) {
    auto f=grid();std::vector<double> p(f.size(),1);p[12]=NAN;
    EXPECT_THROW(rtc_line_power_detail::measure(f,p,2),std::invalid_argument);
    p[12]=-1;
    EXPECT_THROW(rtc_line_power_detail::measure(f,p,2),std::invalid_argument);p[12]=1;f[12]+=.01;
    EXPECT_THROW(rtc_line_power_detail::measure(f,p,2),std::invalid_argument);
    f=grid();std::fill(p.begin(),p.end(),std::numeric_limits<double>::max());auto s=rtc_line_power_detail::measure(f,p,2);EXPECT_EQ(s.cause,RtcLinePowerCause::arithmetic_unavailable);EXPECT_TRUE(s.regions.empty());
}

// Independent direct-DFT audit of the actual recorded estimator windows. This
// is test-only: it neither supplies the measured PSD nor defines a new Welch.
struct WindowAudit {double second_moment=0,stored_power=0,odd_endpoint_deficit=0;std::vector<double> psd;};
std::vector<WindowAudit> audit(const Trial &t) {
    const auto &s=t.spectral->spectrum(0,0,X);const auto &n=t.spectral->network(0);const auto N=n.fft_samples;
    std::vector<WindowAudit> result;std::vector<double> average(s.psd.size(),0);
    for(const auto &window:s.windows){
        std::vector<double> y(N,0);long double weighted_square=0;double sum_w2=0;
        for(std::size_t i=0;i<N;++i){const double w=.5-.5*std::cos(2*std::numbers::pi*i/(N-1));sum_w2+=w*w;
            if(i<static_cast<std::size_t>(window.rows.past_last-window.rows.first))y[i]=(t.parent->network(0).value(X,window.rows.first+i,0)-s.population_median-window.centered_chunk_median)*w;
            weighted_square+=static_cast<long double>(y[i])*y[i];}
        WindowAudit a;a.second_moment=static_cast<double>(weighted_square/sum_w2);a.psd.resize(N/2+1);
        for(std::size_t k=0;k<a.psd.size();++k){std::complex<long double> z{0,0};
            for(std::size_t j=0;j<N;++j){const long double phase=-2*std::numbers::pi_v<long double>*k*j/N;z+=static_cast<long double>(y[j])*std::complex<long double>{std::cos(phase),std::sin(phase)};}
            const double base=static_cast<double>(std::norm(z))/n.window_norm;
            a.psd[k]=base*((k>0&&k+1<a.psd.size())?2:1);a.stored_power+=a.psd[k]/(N*n.interval_seconds);
            average[k]+=a.psd[k]/s.windows.size();
            if(N%2&&k+1==a.psd.size())a.odd_endpoint_deficit=base/(N*n.interval_seconds);
        }
        EXPECT_NEAR(a.stored_power+a.odd_endpoint_deficit,a.second_moment,
            2e-11*std::max(std::numeric_limits<double>::min(),a.second_moment));result.push_back(std::move(a));
    }
    const double spectral_scale=std::max(std::numeric_limits<double>::min(),*std::max_element(s.psd.begin(),s.psd.end()));
    for(std::size_t k=0;k<average.size();++k)EXPECT_NEAR(average[k],s.psd[k],2e-11*spectral_scale);
    return result;
}
TEST(rtc_line_power, low_amplitude_complete_and_gap_padded_audits_preserve_power_scaling) {
    for(bool gaps:{false,true}){
        auto in=noise(241,882);tone(in,3);if(gaps)gap_and_padding(in);
        Trial full(in);auto expected=full.learn();
        in.x*=1e-6;in.r*=1e-6;Trial scaled(in);auto actual=scaled.learn();
        ASSERT_FALSE(audit(scaled).empty());
        const auto &reference=full.spectral->spectrum(0,0,X).psd;
        const auto &small=scaled.spectral->spectrum(0,0,X).psd;
        ASSERT_EQ(reference.size(),small.size());
        const double peak=*std::max_element(reference.begin(),reference.end());
        for(std::size_t k=0;k<reference.size();++k)EXPECT_NEAR(small[k]/1e-12,reference[k],2e-11*peak);
        EXPECT_NEAR(actual->coordinate(0,0,X).total_stored_psd_power/1e-12,
            expected->coordinate(0,0,X).total_stored_psd_power,2e-11*expected->coordinate(0,0,X).total_stored_psd_power);
    }
}
TEST(rtc_line_power, even_odd_and_padded_power_matches_actual_windowed_moment_with_inherited_deficit) {
    for(int kind=0;kind<3;++kind){auto in=kind==1?zero(141,4./33):noise(241,992);
        if(kind==1){tone(in,4.,2);tone(in,3.9,1);}if(kind==2)gap_and_padding(in);
        Trial t(in);auto e=t.learn();const auto &s=t.spectral->spectrum(0,0,X);ASSERT_TRUE(s.available());auto windows=audit(t);ASSERT_FALSE(windows.empty());
        double moment=0,deficit=0;bool padded=false;for(const auto &a:windows){moment+=a.second_moment/windows.size();deficit+=a.odd_endpoint_deficit/windows.size();}
        EXPECT_NEAR(e->coordinate(0,0,X).total_stored_psd_power,moment-deficit,2e-11*std::max(1.,moment));
        if(kind==1)EXPECT_GT(deficit,0.01);else EXPECT_DOUBLE_EQ(deficit,0);
        for(auto w:s.windows)padded|=w.padded_samples>0;if(kind==2)EXPECT_TRUE(padded);
        std::cout<<"POWER_ACCOUNTING case="<<kind<<" windows="<<windows.size()<<" moment="<<moment<<" inherited_deficit="<<deficit<<" stored_power="<<e->coordinate(0,0,X).total_stored_psd_power<<'\n';
    }
}

struct Recorder {
    std::ofstream cases,spectra,regions,windows,raw;
    Recorder(){if(const char *path=std::getenv("CITLALI_LINE_VALIDATION_DIR")){
        std::filesystem::create_directories(path);auto root=std::filesystem::path(path);
        cases.open(root/"cases.csv");spectra.open(root/"spectra.csv");regions.open(root/"regions.csv");windows.open(root/"windows.csv");raw.open(root/"raw.csv");
        if(!cases||!spectra||!regions||!windows||!raw)throw std::runtime_error("Cannot write required line validation artifacts");
        for(auto *f:{&cases,&spectra,&regions,&windows,&raw})*f<<std::setprecision(17);
        cases<<"family,support,seed,radius_hz,rows,windows,region_count,total_stored_power,positive_excess_sum,strongest_region_power,strongest_fraction,band_low_hz,band_high_hz,band_stored_power,band_background_power,band_signed_residual,matched_regions_power,injected_input_mean_square,injected_estimator_band_power,paired_null_raw_band,paired_null_signed_band\n";
        spectra<<"family,support,seed,radius_hz,frequency_hz,psd,background,input_stationary_psd,neighborhood_first,neighborhood_past_last,clipped_low,clipped_high\n";
        regions<<"family,support,seed,radius_hz,first_bin,past_last_bin,peak_bin,first_hz,last_hz,bin_span_hz,positive_excess,psd_fraction,peak_contrast,extent_incomplete,background_truncated\n";
        windows<<"family,window,run,first_row,past_last_row,support_begin,support_end,padded,overlap_previous_rows,contains_control_event,band_stored_power,band_signed_residual,windowed_second_moment,stored_power,odd_endpoint_deficit\n";
        raw<<"family,native_row,relative_time,x,r\n";
    }}
    void record(const std::string &family,const std::string &support,unsigned seed,const Input &in,const Trial &t,
        const std::shared_ptr<const RtcLinePowerEvidence> &e,double rho=NAN,double low=2,double high=4,
        double injection_ms=NAN,double injection_band=NAN,double null_raw=NAN,double null_signed=NAN){
        const auto &c=e->coordinate(0,0,X);const auto &s=t.spectral->spectrum(0,0,X);ASSERT_TRUE(c.available());
        auto band=e->measure_band(0,0,X,"predeclared-"+support+"-comparison-band",low,high);
        double sum=0,maximum=0,matched=0;
        for(const auto &region:c.regions){sum+=region.positive_excess_power;maximum=std::max(maximum,region.positive_excess_power);
            const double peak=t.spectral->network(0).frequency_hz[region.peak_bin];if(low<=peak&&peak<=high)matched+=region.positive_excess_power;
            if(regions.is_open())regions<<family<<','<<support<<','<<seed<<','<<e->radius_hz()<<','<<region.first_bin<<','<<region.past_last_bin<<','<<region.peak_bin<<','<<region.first_frequency_hz<<','<<region.last_frequency_hz<<','<<region.bin_span_hz<<','<<region.positive_excess_power<<','<<region.stored_psd_power_fraction<<','<<region.peak_contrast<<','<<region.extent_incomplete<<','<<region.background_neighborhood_truncated<<'\n';
        }
        EXPECT_LE(sum,c.total_stored_psd_power+1e-12*std::max(1.,c.total_stored_psd_power));
        if(cases.is_open())cases<<family<<','<<support<<','<<seed<<','<<e->radius_hz()<<','<<in.times.size()<<','<<s.windows.size()<<','<<c.regions.size()<<','<<c.total_stored_psd_power<<','<<sum<<','<<maximum<<','<<(c.total_stored_psd_power>0?maximum/c.total_stored_psd_power:0)<<','<<low<<','<<high<<','<<band.stored_psd_power<<','<<band.background_power<<','<<band.signed_residual_power<<','<<matched<<','<<injection_ms<<','<<injection_band<<','<<null_raw<<','<<null_signed<<'\n';
        const auto &n=t.spectral->network(0);
        if(spectra.is_open())for(std::size_t i=0;i<c.background.size();++i){
            const double f=n.frequency_hz[i];double expected=NAN;
            // Infinite-record stationary input reference. Centering, finite
            // windows, padding and inherited odd endpoint can change its image.
            if(std::isfinite(rho)){expected=(1-rho*rho)*n.interval_seconds/(1+rho*rho-2*rho*std::cos(2*std::numbers::pi*f*n.interval_seconds));if(i>0&&!(n.fft_samples%2==0&&i+1==c.background.size()))expected*=2;}
            const auto &h=c.neighborhoods[i];spectra<<family<<','<<support<<','<<seed<<','<<e->radius_hz()<<','<<f<<','<<s.psd[i]<<','<<c.background[i]<<','<<expected<<','<<h.first_bin<<','<<h.past_last_bin<<','<<h.clipped_low<<','<<h.clipped_high<<'\n';
        }
    }
    void witness(const std::string &family,const Input &in,const Trial &t,TimestreamNativeRow event=-1,double low=2,double high=4){
        auto a=audit(t);auto e=t.learn();const auto &s=t.spectral->spectrum(0,0,X);const auto &n=t.spectral->network(0);const auto &c=e->coordinate(0,0,X);
        auto band=e->measure_band(0,0,X,"predeclared-witness-band",low,high);
        for(std::size_t j=0;j<a.size();++j){const auto &w=s.windows[j];double power=0,residual=0;
            for(auto k=band.first_bin;k<band.past_last_bin;++k){power+=a[j].psd[k]*c.bin_increment_hz;residual+=(a[j].psd[k]-c.background[k])*c.bin_increment_hz;}
            TimestreamNativeRow overlap=0;if(j&&s.windows[j-1].run_index==w.run_index)overlap=std::max<TimestreamNativeRow>(0,s.windows[j-1].rows.past_last-w.rows.first);
            if(windows.is_open())windows<<family<<','<<j<<','<<w.run_index<<','<<w.rows.first<<','<<w.rows.past_last<<','<<w.support_begin_unix_sec<<','<<w.support_end_unix_sec<<','<<w.padded_samples<<','<<overlap<<','<<(w.rows.first<=event&&event<w.rows.past_last)<<','<<power<<','<<residual<<','<<a[j].second_moment<<','<<a[j].stored_power<<','<<a[j].odd_endpoint_deficit<<'\n';
        }
        if(raw.is_open())for(std::size_t i=0;i<in.times.size();++i)raw<<family<<','<<i+100<<','<<in.times[i]-in.times[0]<<','<<in.x(i,0)<<','<<in.r(i,0)<<'\n';
    }
};
constexpr std::array profiles{RtcLinePowerProfile::sensitivity_1_hz,RtcLinePowerProfile::initial_2_hz,RtcLinePowerProfile::sensitivity_4_hz};
TEST(rtc_line_power, bounded_null_injection_and_transient_measurement_validation) {
    Recorder output;std::size_t null_cases=0,injection_cases=0;double null_positive=0;
    for(double rho:{0.,.95})for(const std::string support:{"short","long","gap_padding"})for(unsigned seed=0;seed<32;++seed){
        auto in=noise(support=="short"?96:support=="long"?1024:241,1000+seed,rho);if(support=="gap_padding")gap_and_padding(in);
        Trial t(in);for(auto profile:profiles){auto e=t.learn(profile);output.record(rho==0?"null_white":"null_sloping",support,seed,in,t,e,rho);++null_cases;
            for(auto region:e->coordinate(0,0,X).regions)null_positive+=region.positive_excess_power;}
    }
    // No requirement for zero null candidates. Their measured distribution is
    // the result; a deterministic positive floor witnesses selection bias.
    EXPECT_EQ(null_cases,576);EXPECT_GT(null_positive,0);
    for(double rho:{0.,.95})for(const std::string shape:{"on_bin","off_bin","weak","neighboring","chirp","dense"})for(unsigned seed=0;seed<8;++seed){
        auto background=noise(1024,1000+seed,rho);auto line=zero(1024);
        if(shape=="on_bin")tone(line,3);
        if(shape=="off_bin")tone(line,3.125);
        if(shape=="weak")tone(line,3,.5);
        if(shape=="neighboring"){tone(line,2.75,1.4);tone(line,3.25,1.4);}
        if(shape=="chirp")for(std::size_t i=0;i<line.times.size();++i){double t=line.times[i]-line.times[0];line.x(i,0)=2*std::sin(2*std::numbers::pi*(2.4*t+.6*t*t/64)+.31);}
        if(shape=="dense")for(double f:{2.,2.5,3.,3.5,4.})tone(line,f,.9);
        auto combined=background;combined.x+=line.x;Trial base(background),truth(line),t(combined);double input_ms=line.x.col(0).squaredNorm()/line.x.rows();
        auto truth_band=truth.learn()->measure_band(0,0,X,"predeclared-injection-2-to-4-Hz",2,4);
        for(auto profile:profiles){auto e=t.learn(profile);auto null_band=base.learn(profile)->measure_band(0,0,X,"predeclared-injection-2-to-4-Hz",2,4);
            output.record((rho==0?"white_":"sloping_")+shape,"long",seed,combined,t,e,rho,2,4,input_ms,truth_band.stored_psd_power,null_band.stored_psd_power,null_band.signed_residual_power);++injection_cases;}
        if(rho==0&&shape=="on_bin"&&seed==0)output.witness("persistent_tone",combined,t);
    }
    EXPECT_EQ(injection_cases,288);
    for(const std::string shape:{"impulse","level_shift"}){
        // At 128 Hz, ten-second noise blocks exceed the accepted transient
        // learner's 256-difference minimum. The 16 Hz spectral-only null and
        // injection fixtures intentionally cannot establish transient evidence.
        auto in=noise(8192,1000,0,1./128);if(shape=="impulse")in.x(3200,0)+=40;else for(std::size_t i=3200;i<in.times.size();++i)in.x(i,0)+=8;
        Trial t(in);for(auto profile:profiles)output.record(shape,"long",0,in,t,t.learn(profile),0);
        output.witness(shape,in,t,3300);auto joint=t.joint();auto ranked=RtcLinePowerConsideration::rank(t.learn(),joint,7);
        EXPECT_FALSE(joint->transient_handle()->evidence_handle()->events().empty());EXPECT_FALSE(ranked->interference_admitted);EXPECT_FALSE(ranked->apply_authorized);
    }
    // Unmodified retained real Case E values in an explicitly reindexed test
    // parent. No operational cadence/source authority is inferred from this.
    const auto path=std::filesystem::path{__FILE__}.parent_path()/"fixtures/timestream_rtc_event_assessment/case_e_original.txt";
    std::ifstream file(path);ASSERT_TRUE(file);std::vector<std::array<double,4>> rows;std::array<double,4> row;
    while(file>>row[0]>>row[1]>>row[2]>>row[3])rows.push_back(row);ASSERT_EQ(rows.size(),550);
    auto in=zero(rows.size(),.008192);
    for(std::size_t i=0;i<rows.size();++i){in.times[i]=1000+rows[i][1]-rows[0][1];in.x(i,0)=rows[i][2];in.r(i,0)=rows[i][3];}
    Trial t(in,RtcSpikeProtection::unavailable,1e-4);ASSERT_TRUE(t.spectral->spectrum(0,0,X).available());
    for(auto profile:profiles)output.record("retained_case_e","550_rows",0,in,t,t.learn(profile),NAN,2,20);
    output.witness("retained_case_e",in,t,-1,2,20);
    for(std::size_t i=0;i<rows.size();++i){EXPECT_DOUBLE_EQ(t.parent->network(0).value(X,i+100,0),rows[i][2]);EXPECT_DOUBLE_EQ(t.parent->network(0).value(R,i+100,0),rows[i][3]);}
    std::cout<<"BOUNDED_VALIDATION null_profiles="<<null_cases<<" injection_profiles="<<injection_cases<<" controls=6 real_profiles=3 seeds_null=32 seeds_injection=8\n";
}
TEST(rtc_line_power, timing_reports_bounded_measurement_cost_separately_from_spectral_learning) {
    auto in=noise(1600,451);tone(in,3);Trial t(in);auto start=std::chrono::steady_clock::now();std::shared_ptr<const RtcLinePowerEvidence> e;
    for(int i=0;i<1000;++i)e=t.learn();
    std::cout<<"line_power_1000_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" logical_bytes="<<e->logical_owned_bytes()<<'\n';
    EXPECT_GT(e->logical_owned_bytes(),0);EXPECT_EQ(e->coordinates().size(),6);
}
} // namespace
