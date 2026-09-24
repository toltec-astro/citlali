#include <citlali/core/mapmaking/jinc_discrete.h>
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>
namespace citlali::mapmaking {
namespace {
void need(bool x,const char *m){if(!x)throw std::invalid_argument(m);}
bool regular(double x){return std::isfinite(x)&&(x==0||std::fpclassify(x)!=FP_SUBNORMAL);}
double up(double x){return std::nextafter(x,std::numeric_limits<double>::infinity());}
double down(double x){return std::nextafter(x,-std::numeric_limits<double>::infinity());}
double add(double a,double b){return up(a+b);}
double mul(double a,double b){return up(a*b);}
double divup(double a,double b){return up(a/b);}
struct Work {
    JincDiscreteAccumulation r;
    std::vector<double> ln,dn,dc,dq,dt;
    std::vector<std::uint8_t> bad;
    explicit Work(std::size_t n):r{std::vector<double>(n),std::vector<double>(n),std::vector<double>(n),std::vector<double>(n),std::vector<double>(n),std::vector<std::uint64_t>(n),{},{}},ln(n),dn(n),dc(n),dq(n),dt(n),bad(n){}
};
}
JincCertifiedAccumulation JincDiscreteKernel::accumulate_certified(
    std::span<const JincDiscreteSample> samples,std::span<const double> ek,std::size_t workers,double peak)const {
    need(workers>0&&workers<=64&&std::isfinite(peak)&&peak>0,"explicit JINC certificate resources/peak");
    need(ek.size()==cache_.size(),"JINC cache/error shape differs");
    for(double e:ek)need(std::isfinite(e)&&e>=0,"invalid JINC cache bound");
    need(std::numeric_limits<double>::is_iec559&&std::numeric_limits<double>::digits==53&&std::fegetround()==FE_TONEAREST,"JINC certificate requires IEEE binary64 nearest");
    volatile double tiny=std::numeric_limits<double>::min();need(tiny*.5>0,"JINC certificate requires gradual underflow");
    const auto n=spec_.rows*spec_.columns;
    need(static_cast<long double>(n)*workers*100<=2.L*1024*1024*1024,"JINC certified worker memory guard");
    auto block=[&](std::size_t first,std::size_t last){Work w(n);
        for(std::size_t i=first;i<last;++i){const auto &s=samples[i];if(!s.admitted)continue;
            need(std::isfinite(s.x)&&std::isfinite(s.y)&&std::isfinite(s.value)&&std::isfinite(s.coefficient)&&s.coefficient>0&&std::isfinite(s.frequency_hz)&&s.frequency_hz>0,"invalid admitted JINC occurrence");
            if(s.x<.5||s.y<.5||s.x>=spec_.columns+.5||s.y>=spec_.rows+.5)continue;
            const double fx=std::floor(s.x),fy=std::floor(s.y),cx=fx+(s.x-fx>=.5),cy=fy+(s.y-fy>=.5);
            const auto qx=std::size_t(std::floor(spec_.phases*(s.x-cx+.5))),qy=std::size_t(std::floor(spec_.phases*(s.y-cy+.5)));
            need(qx<spec_.phases&&qy<spec_.phases,"JINC discrete phase domain");
            for(int dy=-int(half_);dy<=int(half_);++dy)for(int dx=-int(half_);dx<=int(half_);++dx){
                const auto x=std::int64_t(cx)+dx,y=std::int64_t(cy)+dy;
                if(x<1||x>std::int64_t(spec_.columns)||y<1||y>std::int64_t(spec_.rows))continue;
                const std::size_t p=(y-1)*spec_.columns+x-1,index=((qy*spec_.phases+qx)*side_+dy+half_)*side_+dx+half_;
                const double k=cache_[index],e=ek[index],a=s.coefficient*k,nt=a*s.value,qt=a*k,k2=k*k,tt=k2/s.frequency_hz;
                const double oe=s.coefficient*e,ne=oe*std::abs(s.value),ks=2*std::abs(k)+e,qe=oe*ks,te0=e*ks,te=te0/s.frequency_hz;
                bool bad=!(regular(s.value)&&regular(s.coefficient)&&regular(s.frequency_hz));
                for(double v:{k,e,a,nt,qt,k2,tt,oe,ne,ks,qe,te0,te})bad=bad||!regular(v);
                // Explicit underflow-to-zero detection for every nontrivial product/division.
                bad=bad||(a==0&&k!=0)||(nt==0&&a!=0&&s.value!=0)||(qt==0&&a!=0&&k!=0)||(k2==0&&k!=0)||(tt==0&&k2!=0)||
                    (oe==0&&e!=0)||(ne==0&&oe!=0&&s.value!=0)||(qe==0&&oe!=0&&ks!=0)||(te0==0&&e!=0&&ks!=0)||(te==0&&te0!=0);
                auto sum=[&](double &v,double t){v+=t;bad=bad||!regular(v);};
                sum(w.r.numerator[p],nt);sum(w.r.normalization[p],a);sum(w.r.quadratic[p],qt);sum(w.r.coefficient_squared_time[p],tt);
                sum(w.r.absolute_normalization[p],std::abs(a));sum(w.ln[p],std::abs(nt));sum(w.dn[p],ne);sum(w.dc[p],oe);sum(w.dq[p],qe);sum(w.dt[p],te);
                ++w.r.members[p];w.bad[p]|=bad;
            }
        }return w;
    };
    Work all(n);
    if(workers==1)all=block(0,samples.size());
    else{
        std::vector<std::future<Work>> jobs;
        for(std::size_t w=0;w<workers;++w)jobs.push_back(std::async(std::launch::async,block,samples.size()*w/workers,samples.size()*(w+1)/workers));
        for(auto &job:jobs){auto w=job.get();for(std::size_t p=0;p<n;++p){
            auto merge=[&](std::vector<double>&a,const std::vector<double>&b){a[p]+=b[p];all.bad[p]|=!regular(a[p]);};
            merge(all.r.numerator,w.r.numerator);merge(all.r.normalization,w.r.normalization);merge(all.r.quadratic,w.r.quadratic);merge(all.r.coefficient_squared_time,w.r.coefficient_squared_time);
            merge(all.r.absolute_normalization,w.r.absolute_normalization);merge(all.ln,w.ln);merge(all.dn,w.dn);merge(all.dc,w.dc);merge(all.dq,w.dq);merge(all.dt,w.dt);
            all.r.members[p]+=w.r.members[p];all.bad[p]|=w.bad[p];
        }}
    }
    JincCertifiedAccumulation out;out.sums=std::move(all.r);out.causes=std::move(all.bad);auto &r=out.sums;
    for(auto &v:out.error)v.assign(n,std::numeric_limits<double>::infinity());out.tolerance_lower.assign(n,0);
    r.algebraic_value.assign(n,std::numeric_limits<double>::quiet_NaN());r.algebraic_support.assign(n,0);
    constexpr double u=0x1p-53;const double tol=down(1e-6);
    for(std::size_t p=0;p<n;++p){
        if(!r.members[p]){out.causes[p]|=16;continue;}
        if(out.causes[p])continue;
        need(r.members[p]<(1ULL<<52),"JINC certificate count range");
        const double ku=mul(double(r.members[p]+workers+6),u),g=divup(ku,down(1-ku)),margin=down(1-g);
        if(!(margin>0)){out.causes[p]|=1;continue;}
        auto err=[&](double l,double d){return divup(add(mul(g,l),d),margin);};
        const double en=err(all.ln[p],all.dn[p]),ec=err(r.absolute_normalization[p],all.dc[p]),eq=err(r.quadratic[p],all.dq[p]),et=err(r.coefficient_squared_time[p],all.dt[p]);
        out.error[0][p]=en;out.error[1][p]=ec;out.error[2][p]=eq;out.error[3][p]=et;
        const double qlo=down(r.quadratic[p]-eq),tlo=down(r.coefficient_squared_time[p]-et);
        if(!(qlo>0&&tlo>0&&eq<=down(tol*qlo)&&et<=down(tol*tlo)))out.causes[p]|=4;
        if(!(std::abs(r.normalization[p])>ec)){out.causes[p]|=2;continue;}
        const double m=r.numerator[p]/r.normalization[p];r.algebraic_value[p]=m;
        if(!(regular(m)&&(m!=0||r.numerator[p]==0))){out.causes[p]|=1;continue;}
        r.algebraic_support[p]=std::isfinite(m)&&r.quadratic[p]>0;
        const double er=mul(divup(u,down(1-u)),std::abs(r.numerator[p]));
        const double bound=divup(add(add(en,mul(std::abs(m),ec)),er),down(std::abs(r.normalization[p])-ec));
        const double allowed=down(down(tol*peak)+down(tol*std::abs(m)));
        out.error[4][p]=bound;out.tolerance_lower[p]=allowed;if(!(bound<=allowed))out.causes[p]|=8;
    }return out;
}
} // namespace citlali::mapmaking
