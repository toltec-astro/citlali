#include <citlali/core/pipeline/timestream_cal_wvr.h>
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <set>
#include <stdexcept>

namespace citlali::pipeline {
std::string_view cal_wvr_cause_name(CalWvrCause c) {
    switch(c) {
    case CalWvrCause::available:return "available";
    case CalWvrCause::absent:return "wvr_tau225_absent";
    case CalWvrCause::unbracketed:return "wvr_tau225_unbracketed";
    case CalWvrCause::gap_outside_source_validity:return "wvr_tau225_gap_outside_source_validity";
    case CalWvrCause::conflicting_duplicate:return "wvr_tau225_conflicting_duplicate";
    case CalWvrCause::negative:return "wvr_tau225_negative";
    case CalWvrCause::nonfinite:return "wvr_tau225_nonfinite";
    case CalWvrCause::time_mapping_unavailable:return "wvr_time_mapping_unavailable";
    }
    throw std::logic_error("unknown WVR cause");
}
std::string_view cal_opacity_quality_name(CalOpacityQuality c) {
    switch(c) {
    case CalOpacityQuality::invalid_opacity_input:return "invalid_opacity_input";
    case CalOpacityQuality::opacity_quality_unavailable:return "opacity_quality_unavailable";
    case CalOpacityQuality::outside_supported_opacity:return "outside_supported_opacity";
    case CalOpacityQuality::science_qualification_eligible:return "science_qualification_eligible";
    case CalOpacityQuality::engineering_only:return "engineering_only";
    }
    throw std::logic_error("unknown opacity quality");
}
std::shared_ptr<const CalWvrEvidence> CalWvrEvidence::learn(NativeObservationScope scope,
    std::string source,std::string mapping,std::vector<CalWvrRecord> records) {
    if(source.empty())throw std::invalid_argument("WVR source identity is required even for absent records");
    auto out=std::shared_ptr<CalWvrEvidence>(new CalWvrEvidence{scope});
    out->scope_=scope;out->source_=std::move(source);out->mapping_=std::move(mapping);
    std::set<std::string> identities;
    for(const auto &r:records) {
        if(r.identity.empty() || !identities.insert(r.identity).second || !std::isfinite(r.time_unix_sec) ||
           !std::isfinite(r.valid_first_unix_sec) || !std::isfinite(r.valid_last_unix_sec) ||
           r.valid_first_unix_sec>r.valid_last_unix_sec)
            throw std::invalid_argument("WVR record identity, time or declared validity is malformed");
    }
    std::stable_sort(records.begin(),records.end(),[](const auto &a,const auto &b){return a.time_unix_sec<b.time_unix_sec;});
    out->records_=std::move(records);
    for(std::size_t i=0;i<out->records_.size();) {
        std::size_t end=i+1;bool conflict=false;const auto &a=out->records_[i];
        while(end<out->records_.size() && out->records_[end].time_unix_sec==a.time_unix_sec) {
            const auto &b=out->records_[end++];
            auto bits=[](double v){return std::bit_cast<std::uint64_t>(v);};
            conflict|=bits(a.tau225)!=bits(b.tau225) || a.producer_valid!=b.producer_valid ||
                bits(a.valid_first_unix_sec)!=bits(b.valid_first_unix_sec) || bits(a.valid_last_unix_sec)!=bits(b.valid_last_unix_sec);
        }
        out->groups_.push_back({i,end,conflict});i=end;
    }
    return out;
}
CalWvrCause CalWvrEvidence::support_cause(const Group &lo,const Group &hi,double first,double last) const {
    if(lo.conflict || hi.conflict)return CalWvrCause::conflicting_duplicate;
    const auto &a=records_[lo.first],&b=records_[hi.first];
    if(!a.producer_valid || !b.producer_valid ||
        std::max(a.valid_first_unix_sec,b.valid_first_unix_sec)>first ||
        std::min(a.valid_last_unix_sec,b.valid_last_unix_sec)<last)
        return CalWvrCause::gap_outside_source_validity;
    if(!std::isfinite(a.tau225) || !std::isfinite(b.tau225))return CalWvrCause::nonfinite;
    if(a.tau225<0 || b.tau225<0)return CalWvrCause::negative;
    return CalWvrCause::available;
}
CalWvrSample CalWvrEvidence::at(double t) const {
    CalWvrSample out;out.mapped_time_unix_sec=t;
    if(mapping_.empty() || !std::isfinite(t) || std::fegetround()!=FE_TONEAREST){out.cause=CalWvrCause::time_mapping_unavailable;return out;}
    if(groups_.empty())return out;
    auto hi=std::lower_bound(groups_.begin(),groups_.end(),t,[&](auto g,double time){return records_[g.first].time_unix_sec<time;});
    const bool exact=hi!=groups_.end() && records_[hi->first].time_unix_sec==t;
    if(!exact && (hi==groups_.begin() || hi==groups_.end())){out.cause=CalWvrCause::unbracketed;return out;}
    const auto &lo=exact?*hi:*(hi-1);
    out.first_record=lo.first;out.last_record=hi->first;out.exact_match=exact;
    const auto &a=records_[lo.first],&b=records_[hi->first];
    out.cause=support_cause(lo,*hi,exact?t:a.time_unix_sec,exact?t:b.time_unix_sec);
    if(out.cause!=CalWvrCause::available)return out;
    out.weight=exact?0:(t-a.time_unix_sec)/(b.time_unix_sec-a.time_unix_sec);
    const double value=exact?a.tau225:a.tau225+out.weight*(b.tau225-a.tau225);
    if(!std::isfinite(value)){out.cause=CalWvrCause::nonfinite;return out;}
    out.tau225=value;out.cause=CalWvrCause::available;return out;
}
CalWvrQuality CalWvrEvidence::quality(double first,double last) const {
    CalWvrQuality q;q.first=first;q.last=last;
    if(!std::isfinite(first) || !std::isfinite(last) || last<=first || std::fegetround()!=FE_TONEAREST) {
        q.cause="classifier_numeric_failure";return q;
    }
    std::vector<double> times{first};
    for(const auto &g:groups_) {const auto t=records_[g.first].time_unix_sec;if(t>first && t<last)times.push_back(t);}
    times.push_back(last);q.breakpoint_count=times.size();
    std::vector<double> values;bool missing=false,invalid=false;
    auto observe=[&](CalWvrCause cause){
        if(cause==CalWvrCause::negative || cause==CalWvrCause::nonfinite) {
            invalid=true;q.cause=std::string(cal_wvr_cause_name(cause));
        } else if(cause!=CalWvrCause::available) {missing=true;if(!invalid)q.cause=std::string(cal_wvr_cause_name(cause));}
    };
    for(std::size_t i=0;i<times.size();++i){
        const auto sample=at(times[i]);observe(sample.cause);values.push_back(sample.tau225.value_or(0));
        // Inspect the source bracket itself: a midpoint can round onto an
        // endpoint when source times are adjacent representable binary64s.
        // Exact valid endpoint values alone never authorize an open interval.
        if(i) {
            auto hi=std::upper_bound(groups_.begin(),groups_.end(),times[i-1],
                [&](double time,auto g){return time<records_[g.first].time_unix_sec;});
            if(hi==groups_.begin() || hi==groups_.end())observe(groups_.empty()?CalWvrCause::absent:CalWvrCause::unbracketed);
            else observe(support_cause(*(hi-1),*hi,records_[(hi-1)->first].time_unix_sec,records_[hi->first].time_unix_sec));
        }
    }
    if(invalid){q.classification=CalOpacityQuality::invalid_opacity_input;return q;}
    if(missing)return q;
    q.minimum=*std::min_element(values.begin(),values.end());q.maximum=*std::max_element(values.begin(),values.end());
    q.duration=last-first;
    constexpr double threshold=0.15;
    for(std::size_t i=1;i<times.size();++i) {
        const double t0=times[i-1],t1=times[i],v0=values[i-1],v1=values[i],dt=t1-t0;
        if(!(dt>0)){q.cause="classifier_numeric_failure";return q;}
        q.area+=dt*(v0+v1)/2;
        if(v0<=threshold && v1<=threshold)continue;
        double begin=t0,end=t1;
        if(v0<=threshold)begin=v0==threshold?t0:t0+((threshold-v0)/(v1-v0))*dt;
        if(v1<=threshold)end=v1==threshold?t1:t0+((threshold-v0)/(v1-v0))*dt;
        const double duration=end-begin,peak=std::max(v0,v1);
        q.integrated_excess+=duration*(std::max(v0-threshold,0.)+std::max(v1-threshold,0.))/2;
        // Equality at a breakpoint separates the open excursion components.
        if(!q.excursions.empty() && q.excursions.back().last==begin && v0>threshold) {
            auto &e=q.excursions.back();e.last=end;e.duration=end-e.first;e.peak=std::max(e.peak,peak);
        }else q.excursions.push_back({begin,end,duration,peak});
    }
    q.mean=q.area/q.duration;
    for(const auto &e:q.excursions){q.excursion_duration+=e.duration;q.longest_excursion=std::max(q.longest_excursion,e.duration);}
    q.excursion_fraction=q.excursion_duration/q.duration;
    if(!std::isfinite(q.area) || !std::isfinite(q.duration) || !std::isfinite(q.mean) ||
       !std::isfinite(q.integrated_excess) || !std::isfinite(q.excursion_fraction) ||
       q.mean<q.minimum || q.mean>q.maximum || q.excursion_fraction<0 || q.excursion_fraction>1) {
        q.cause="classifier_numeric_failure";return q;
    }
    q.summary_available=true;q.cause="complete-source-valid-coverage";
    q.classification=q.maximum>0.25?CalOpacityQuality::outside_supported_opacity:
        q.mean<=0.15 && q.maximum<=0.175?CalOpacityQuality::science_qualification_eligible:CalOpacityQuality::engineering_only;
    return q;
}
} // namespace citlali::pipeline
