#include <citlali/core/pipeline/rtc_point_run_selection.h>
#include "../src/citlali/cli/point_rtc_probe_scope.h"
#include <cassert>
#include <iostream>
#include <random>
using namespace citlali::pipeline;
struct Data {
 std::size_t n=1600,d=120;double dt=.008192;
 std::vector<double>x,r,t,lon,lat;std::vector<std::uint8_t>vx,vr,m;std::vector<int>channels;
 std::vector<RtcPointRun> runs;
 Data():x(n*d),r(n*d),t(n),lon(n*d,.03),lat(n*d,0),vx(n*d,1),vr(vx),m(vx),channels(d),runs{{0,800},{800,1600}} {
  std::mt19937 gen(9881);std::normal_distribution<double>noise;
  std::iota(channels.begin(),channels.end(),0);
  for(std::size_t i=0;i<n;++i){t[i]=dt*i+(i>=800?2:0);for(std::size_t j=0;j<d;++j){x[i*d+j]=noise(gen);r[i*d+j]=noise(gen);}}
  for(std::size_t i=0;i<800;++i)if(t[i]>=2 && t[i]<2.25){x[i*d+50]+=100;r[i*d+50]-=100;}
 }
 auto input()const {return RtcPointSelectionInput{n,d,x,r,t,lon,lat,vx,vr,m,channels,runs,dt,"original-parent","AST-original"};}
 auto plan()const {return RtcPointRunSelectionPlan::consider(RtcPointRunSelectionEvidence::learn(input()));}
};
int main(){
 Data a;const auto raw=a.x;auto p=a.plan();assert((p.rejected_runs(50)==std::vector<RtcPointRun>{{0,800}}));assert(a.x==raw);
 assert(p.rejected_runs(1).empty());
 Data source=a;for(std::size_t i=0;i<800;++i)source.lon[i*a.d+50]=0;assert(source.plan().rejected_runs(50).empty());
 Data invalid=a;for(std::size_t i=0;i<800;++i)invalid.vr[i*a.d+50]=0;assert(invalid.plan().rejected_runs(50).empty());assert(invalid.vx==a.vx);
 Data shared=a;for(std::size_t i=0;i<800;++i)for(int j: {60,61}){shared.x[i*a.d+j]=a.x[i*a.d+50];shared.r[i*a.d+j]=a.r[i*a.d+50];}assert(shared.plan().rejected_runs(50).empty());
 Data adjacent=a;for(std::size_t i=0;i<800;++i){adjacent.x[i*a.d+49]=a.x[i*a.d+50];adjacent.r[i*a.d+49]=a.r[i*a.d+50];}assert(adjacent.plan().rejected_runs(50).empty());
 Data injection=a;for(std::size_t i=0;i<800;++i)if(a.t[i]>=2&&a.t[i]<2.25)injection.x[i*a.d+50]-=100;assert(injection.plan().rejected_runs(50).empty());assert(injection.r==a.r && injection.vr==a.vr);
 auto q=a.input();q.pointing_identity="";bool threw=false;try{RtcPointRunSelectionPlan::consider(RtcPointRunSelectionEvidence::learn(q));}catch(const std::invalid_argument&){threw=true;}assert(threw);
 q=a.input();std::vector<RtcPointRun> false_run{{0,1600}};q.physical_runs=false_run;threw=false;try{RtcPointRunSelectionPlan::consider(RtcPointRunSelectionEvidence::learn(q));}catch(const std::invalid_argument&){threw=true;}assert(threw);
 auto gate=citlali::cli::detail::point_rtc_probe_route_allowed;
 for(int nw=1;nw<=4;++nw)assert(gate(true,true,true,152420,0,2,nw,0,"owner-RTC-point-transfer-2026-09-28"));
 assert(!gate(true,true,true,152430,0,2,2,0,"owner-RTC-point-transfer-2026-09-28"));assert(!gate(true,true,true,152420,0,2,11,2,"owner-RTC-point-transfer-2026-09-28"));
 std::cout<<"PASS selector: isolated/shared/source/adjacent/paired/gap/active-injection/route\n";
}
