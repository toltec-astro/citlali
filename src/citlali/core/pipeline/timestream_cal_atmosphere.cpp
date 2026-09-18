#include <citlali/core/pipeline/timestream_cal_atmosphere.h>
#include <citlali/core/utils/sha256.h>
#include "citlali_cal_authority.h"
#include <algorithm>
#include <cfenv>
#include <charconv>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {
namespace {
double number(const std::string &s) {
    double v; auto [p, ec] = std::from_chars(s.data(), s.data()+s.size(), v);
    if(ec!=std::errc{} || p!=s.data()+s.size() || !std::isfinite(v))
        throw std::runtime_error("invalid frozen CAL node");
    return v;
}
double endpoint(double h0, double h1, double d0, double d1) {
    auto d=((2*h0+h1)*d0-h0*d1)/(h0+h1);
    if(std::signbit(d)!=std::signbit(d0)) return 0;
    if(std::signbit(d0)!=std::signbit(d1) && std::abs(d)>3*std::abs(d0)) return 3*d0;
    return d;
}
}

std::shared_ptr<const CalAtmosphereSurface> CalAtmosphereSurface::frozen(int alpha) {
    return std::shared_ptr<const CalAtmosphereSurface>(new CalAtmosphereSurface(alpha));
}
CalAtmosphereSurface::CalAtmosphereSurface(int alpha): alpha_{alpha} {
    if(std::fegetround()!=FE_TONEAREST)
        throw std::invalid_argument("CAL atmosphere requires binary64 round-to-nearest");
    if(alpha!=-1 && alpha!=0 && alpha!=2 && alpha!=4)
        throw std::invalid_argument("CAL reference alpha must be one of -1,0,2,4");
    if(citlali::utils::sha256(std::string(cal_frozen_nodes))!=nodes_sha256)
        throw std::runtime_error("CAL frozen node bytes changed");
    std::istringstream stream{std::string(cal_frozen_nodes)};std::string line;
    std::getline(stream,line);
    while(std::getline(stream,line)) {
        if(line.empty()) continue;
        std::istringstream row(line);std::vector<std::string> f;std::string cell;
        while(std::getline(row,cell,','))f.push_back(cell);
        if(f.size()!=10)throw std::runtime_error("CAL frozen node shape differs");
        int array=f[5]=="a1100"?0:f[5]=="a1400"?1:f[5]=="a2000"?2:-1;
        if(array<0)throw std::runtime_error("CAL frozen array unknown");
        nodes_.push_back({array,static_cast<int>(number(f[6])),number(f[2]),number(f[3]),number(f[7]),number(f[8])});
    }
    if(nodes_.size()!=1368)throw std::runtime_error("CAL frozen node inventory differs");
    for(int a=0;a<3;++a) {
        auto &curves=curves_[a];
        for(const auto &n:nodes_) if(n.array==a && n.alpha==alpha_) {
            auto it=std::find_if(curves.begin(),curves.end(),[&](const auto &c){return c.tau==n.tau225;});
            if(it==curves.end()){curves.push_back({n.tau225,{},{},{}});it=curves.end()-1;}
            it->elevation.push_back(n.elevation_deg);it->ordinate.push_back(n.los_tau);
        }
        std::sort(curves.begin(),curves.end(),[](const auto &x,const auto &y){return x.tau<y.tau;});
        if(curves.size()!=6)throw std::runtime_error("CAL anchor inventory differs");
        for(auto &c:curves) {
            // The supplemental anchor rows have an intentionally non-spatial
            // storage order. Sort exact (coordinate, ordinate) pairs, retaining
            // the original frozen inventory above; never regenerate a node.
            std::vector<std::pair<double,double>> ordered;
            for(std::size_t i=0;i<c.elevation.size();++i)ordered.emplace_back(c.elevation[i],c.ordinate[i]);
            std::sort(ordered.begin(),ordered.end());
            for(std::size_t i=0;i<ordered.size();++i){c.elevation[i]=ordered[i].first;c.ordinate[i]=ordered[i].second;}
            const auto n=c.elevation.size();std::vector<double> h(n-1),d(n-1);c.slope.resize(n);
            for(std::size_t i=0;i+1<n;++i){h[i]=c.elevation[i+1]-c.elevation[i];
                if(h[i]<=0)throw std::runtime_error("unordered CAL elevations");
                d[i]=(c.ordinate[i+1]-c.ordinate[i])/h[i];}
            c.slope[0]=endpoint(h[0],h[1],d[0],d[1]);
            c.slope[n-1]=endpoint(h[n-2],h[n-3],d[n-2],d[n-3]);
            for(std::size_t i=1;i+1<n;++i) {
                if(d[i-1]==0 || d[i]==0 || std::signbit(d[i-1])!=std::signbit(d[i]))c.slope[i]=0;
                else {double w1=2*h[i]+h[i-1],w2=h[i]+2*h[i-1];c.slope[i]=(w1+w2)/(w1/d[i-1]+w2/d[i]);}
            }
        }
    }
}
std::optional<double> CalAtmosphereSurface::correction(int array,double tau,double el) const {
    if(std::fegetround()!=FE_TONEAREST || array<0 || array>2 || !std::isfinite(tau) || !std::isfinite(el) ||
        tau<0 || tau>0.25 || el<25 || el>80) return {};
    if(tau==0)return 1.;
    auto eval=[&](const Curve &c){
        auto p=std::lower_bound(c.elevation.begin(),c.elevation.end(),el);
        if(p!=c.elevation.end() && *p==el)return c.ordinate[p-c.elevation.begin()];
        const auto i=static_cast<std::size_t>(p-c.elevation.begin()-1);
        const double h=c.elevation[i+1]-c.elevation[i],t=(el-c.elevation[i])/h;
        return (2*t*t*t-3*t*t+1)*c.ordinate[i]+(t*t*t-2*t*t+t)*h*c.slope[i]+
            (-2*t*t*t+3*t*t)*c.ordinate[i+1]+(t*t*t-t*t)*h*c.slope[i+1];
    };
    const auto &c=curves_[array];auto hi=std::lower_bound(c.begin(),c.end(),tau,[](const auto &v,double t){return v.tau<t;});
    double lambda;
    if(hi!=c.end() && hi->tau==tau)lambda=eval(*hi);
    else {if(hi==c.end())return {};double low_tau=0,low=0;
        if(hi!=c.begin()){low_tau=(hi-1)->tau;low=eval(*(hi-1));}
        double u=(tau-low_tau)/(hi->tau-low_tau);lambda=(1-u)*low+u*eval(*hi);}
    const double result=std::exp(lambda);
    if(!std::isfinite(result) || result<=1)return {};
    return result;
}
} // namespace citlali::pipeline
