#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {
// Development RTC owner. Row-major original-domain planes; x/r dimensionless,
// AST target-centered tangent coordinates in degrees, integration times in
// seconds, independent producer validity and mapping availability. Ranges are
// zero-based half-open local rows, exactly the originating physical runs.
struct RtcPointRun { std::int64_t first=0, past_last=0;
  friend bool operator==(const RtcPointRun &,const RtcPointRun &)=default; };
struct RtcPointSelectionInput {
  std::size_t rows=0, detectors=0;
  std::span<const double> x,r,time,longitude_deg,latitude_deg;
  std::span<const std::uint8_t> valid_x,valid_r,mapping;
  std::span<const int> channels;
  std::span<const RtcPointRun> physical_runs;
  double interval_seconds=0;
  std::string parent_identity,pointing_identity;
};
struct RtcPointSelectedEvent {
  std::int64_t onset=0,first=0,past_last=0;
  std::array<double,2> contrast{},cohort_center{},cohort_scale{},adjacent_max{};
  std::size_t peers=0,coherent=0;
};
struct RtcPointRunSelection {
  std::size_t detector=0,run=0,proposals=0,eligible=0,contrast_pass=0;
  bool selected=false;
  RtcPointSelectedEvent event;
};
struct RtcPointProposedEvent {std::size_t record=0;RtcPointSelectedEvent facts;};
class RtcPointRunSelectionPlan;
// Learn proposes windows with the frozen opposite/10-MAD contrast preselection
// and measures their whole-window peer/adjacent facts. It never rejects support.
class RtcPointRunSelectionEvidence {
public:
  static constexpr const char *identity="rtc-point-physical-run-quarantine-v1";
  static constexpr double source_guard_arcsec=60.,stride_seconds=.125;
  static std::shared_ptr<const RtcPointRunSelectionEvidence> learn(const RtcPointSelectionInput &a) {
    auto need=[](bool b){if(!b)throw std::invalid_argument("RTC point selection incomplete original-domain binding");};
    need(a.rows>0 && a.detectors>0 && a.detectors<=SIZE_MAX/a.rows);
    const auto cells=a.rows*a.detectors;
    need(a.x.size()==cells && a.r.size()==cells && a.longitude_deg.size()==cells &&
      a.latitude_deg.size()==cells && a.valid_x.size()==cells && a.valid_r.size()==cells &&
      a.mapping.size()==cells && a.time.size()==a.rows && a.channels.size()==a.detectors &&
      std::isfinite(a.interval_seconds) && a.interval_seconds>0 &&
      !a.parent_identity.empty() && !a.pointing_identity.empty());
    need(std::is_sorted(a.channels.begin(),a.channels.end()) &&
      std::adjacent_find(a.channels.begin(),a.channels.end())==a.channels.end());
    std::int64_t end=0;
    for(auto run:a.physical_runs) {
      need(run.first==end && run.past_last>run.first && std::uint64_t(run.past_last)<=a.rows);
      for(auto t=run.first;t<run.past_last;++t) {
        need(std::isfinite(a.time[t]));
        if(t>run.first)need(a.time[t]>a.time[t-1] &&
          std::abs(a.time[t]-a.time[t-1]-a.interval_seconds)<=a.interval_seconds*.0001+
          4*(std::nextafter(a.time[t],INFINITY)-a.time[t]));
      }
      end=run.past_last;
    }
    need(std::uint64_t(end)==a.rows);
    auto owner=std::shared_ptr<RtcPointRunSelectionEvidence>(new RtcPointRunSelectionEvidence);auto &out=*owner;out.parent_=a.parent_identity;out.pointing_=a.pointing_identity;
    out.runs_.assign(a.physical_runs.begin(),a.physical_runs.end());
    const auto nd=a.detectors;
    for(std::size_t ri=0;ri<a.physical_runs.size();++ri) {
      const auto run=a.physical_runs[ri];const auto base=out.records_.size();
      for(std::size_t d=0;d<nd;++d)out.records_.push_back({d,ri});
      const auto at=[&](double t){return std::lower_bound(a.time.begin()+run.first,a.time.begin()+run.past_last,t)-a.time.begin();};
      const double anchor=a.time[run.first],last=a.time[run.past_last-1]+a.interval_seconds;
      for(std::size_t k=8;;++k) {
        const double onset=anchor+k*stride_seconds;
        if(onset+2>last)break;
        const auto lo=at(onset-1),hi=at(onset+2);
        const std::array<std::pair<std::int64_t,std::int64_t>,3> flank{{{at(onset-.4),at(onset-.2)},{at(onset),at(onset+.25)},{at(onset+.5),at(onset+.8)}}};
        if(hi-lo<std::ceil(2.85/a.interval_seconds) || std::any_of(flank.begin(),flank.end(),[](auto f){return f.second-f.first<20;}))continue;
        std::vector<bool> ok(nd,true);std::vector<std::array<double,2>> contrasts(nd),means(nd),ss(nd);
        for(std::size_t d=0;d<nd;++d) {
          auto &rec=out.records_[base+d];++rec.proposals;
          for(auto t=lo;t<hi;++t) {
            const auto i=t*nd+d;
            if(!a.valid_x[i] || !a.valid_r[i] || !a.mapping[i] || !std::isfinite(a.x[i]) ||
              !std::isfinite(a.r[i]) || !std::isfinite(a.longitude_deg[i]) ||
              !std::isfinite(a.latitude_deg[i]) ||
              !(3600*std::hypot(a.longitude_deg[i],a.latitude_deg[i])>source_guard_arcsec)) {ok[d]=false;break;}
          }
          if(!ok[d])continue;
          for(std::size_t c=0;c<2;++c) {
            const auto z=c?a.r:a.x;
            for(auto t=lo;t<hi;++t)means[d][c]+=z[t*nd+d];
            means[d][c]/=hi-lo;
            for(auto t=lo;t<hi;++t)ss[d][c]+=std::pow(z[t*nd+d]-means[d][c],2);
            if(!(ss[d][c]>0) || !std::isfinite(ss[d][c]))ok[d]=false;
            std::array<double,2> m;
            for(int j=0;j<2;++j){std::vector<double> v;for(auto t=flank[j].first;t<flank[j].second;++t)v.push_back(z[t*nd+d]);m[j]=median(v);}
            contrasts[d][c]=m[1]-m[0];
          }
          if(ok[d])++rec.eligible;
        }
        auto corr=[&](std::size_t d,std::size_t e,int c){
          const auto z=c?a.r:a.x;double s=0;
          for(auto t=lo;t<hi;++t)s+=(z[t*nd+d]-means[d][c])*(z[t*nd+e]-means[e][c]);
          return s/std::sqrt(ss[d][c]*ss[e][c]);
        };
        for(std::size_t d=0;d<nd;++d) {
          auto &rec=out.records_[base+d];if(!ok[d] || !(contrasts[d][0]*contrasts[d][1]<0))continue;
          std::vector<std::size_t> peers;for(std::size_t e=0;e<nd;++e)if(e!=d && ok[e])peers.push_back(e);
          if(peers.size()<100)continue;
          RtcPointSelectedEvent event;event.onset=flank[1].first;event.first=lo;event.past_last=hi;event.peers=peers.size();event.contrast=contrasts[d];bool pass=true;
          for(int c=0;c<2;++c) {
            std::vector<double> v;for(auto e:peers)v.push_back(contrasts[e][c]);
            event.cohort_center[c]=median(v);for(auto &q:v)q=std::abs(q-event.cohort_center[c]);
            event.cohort_scale[c]=1.4826*median(v);
            pass &= std::isfinite(event.cohort_scale[c]) && event.cohort_scale[c]>0 &&
              std::abs(event.contrast[c]-event.cohort_center[c])>10*event.cohort_scale[c];
          }
          if(!pass)continue;++rec.contrast_pass;
          for(auto e:peers) {const auto x=corr(d,e,0),r=corr(d,e,1);if(!std::isfinite(x)||!std::isfinite(r)){pass=false;break;}event.coherent+=std::abs(x)>=.5 && std::abs(r)>=.5;}
          event.adjacent_max={NAN,NAN};
          if(pass && d>0 && d+1<nd && a.channels[d-1]==a.channels[d]-1 && a.channels[d+1]==a.channels[d]+1 && ok[d-1] && ok[d+1])
            for(int c=0;c<2;++c)event.adjacent_max[c]=std::max(std::abs(corr(d,d-1,c)),std::abs(corr(d,d+1,c)));
          // Undefined peer arithmetic stays unavailable, never evidence of isolation.
          if(!pass)event.peers=0;
          out.proposals_.push_back({base+d,event});
        }
      }
    }
    return owner;
  }
  const auto &records()const{return records_;}
  const auto &runs()const{return runs_;}
  const auto &parent_identity()const{return parent_;}
  const auto &pointing_identity()const{return pointing_;}
private:
  friend class RtcPointRunSelectionPlan;
  RtcPointRunSelectionEvidence()=default;
  static double median(std::vector<double> &v) {
    const auto m=v.begin()+v.size()/2;std::nth_element(v.begin(),m,v.end());
    return v.size()%2?*m:std::midpoint(*std::max_element(v.begin(),m),*m);
  }
  std::string parent_,pointing_;
  std::vector<RtcPointRun> runs_;
  std::vector<RtcPointRunSelection> records_;
  std::vector<RtcPointProposedEvent> proposals_;
};
// Consider names the one isolation/adjacency policy and freezes a separate plan.
// Existing native RTC support owners apply its exact physical-run decisions.
class RtcPointRunSelectionPlan {
public:
  static constexpr const char *identity=RtcPointRunSelectionEvidence::identity;
  static constexpr double source_guard_arcsec=RtcPointRunSelectionEvidence::source_guard_arcsec;
  static RtcPointRunSelectionPlan consider(std::shared_ptr<const RtcPointRunSelectionEvidence> e) {
    if(!e)throw std::invalid_argument("RTC point selection requires learned original evidence");
    RtcPointRunSelectionPlan p;p.evidence_=std::move(e);p.records_=p.evidence_->records_;
    for(const auto &candidate:p.evidence_->proposals_) {
      auto &r=p.records_[candidate.record];const auto &f=candidate.facts;
      if(r.selected || f.peers<100 || 100*f.coherent>=f.peers)continue;
      if(std::all_of(f.adjacent_max.begin(),f.adjacent_max.end(),[](double c){return std::isfinite(c)&&c<.3;})) {r.selected=true;r.event=f;}
    }
    return p;
  }
  const auto &evidence_handle()const{return evidence_;}
  const auto &records()const{return records_;}
  const auto &runs()const{return evidence_->runs_;}
  const auto &parent_identity()const{return evidence_->parent_;}
  const auto &pointing_identity()const{return evidence_->pointing_;}
  std::vector<RtcPointRun> rejected_runs(std::size_t d)const {
    std::vector<RtcPointRun> r;for(const auto &v:records_)if(v.detector==d && v.selected)r.push_back(runs()[v.run]);return r;
  }
private:
  RtcPointRunSelectionPlan()=default;
  std::shared_ptr<const RtcPointRunSelectionEvidence> evidence_;
  std::vector<RtcPointRunSelection> records_;
};
}
