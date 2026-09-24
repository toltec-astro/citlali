#include <citlali/core/mapmaking/jinc_discrete.h>
#include <boost/math/special_functions/bessel.hpp>
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <numbers>
#include <stdexcept>
namespace citlali::mapmaking {
namespace {
void require(bool ok,const char *why){if(!ok)throw std::invalid_argument(why);}
bool positive(double x){return std::isfinite(x)&&x>0;}
double peak_jinc(double x){return x==0?1:2*boost::math::cyl_bessel_j(1,x)/x;}
JincDiscreteAccumulation empty(std::size_t n){
    return {std::vector<double>(n),std::vector<double>(n),std::vector<double>(n),
            std::vector<double>(n),std::vector<double>(n),std::vector<std::uint64_t>(n),{}, {}};
}
void finite(const JincDiscreteAccumulation &r){
    for(const auto *p:{&r.numerator,&r.normalization,&r.quadratic,&r.coefficient_squared_time,&r.absolute_normalization})
        for(auto x:*p)require(std::isfinite(x),"nonfinite JINC accumulator");
}
}
double JincDiscreteKernel::analytic(double r,double a,double b,double c,double rmax){
    require(std::isfinite(r)&&r>=0&&positive(a)&&positive(b)&&positive(c)&&positive(rmax),"invalid explicit JINC parameters");
    // Numerical evaluation of the defining first root, not rounded 3.831706.
    const double root=boost::math::cyl_bessel_j_zero(1.,1);
    if(r==rmax)return 0.; // Exact contract-defined second-factor analytic zero.
    const double out=peak_jinc(2*std::numbers::pi*r/a)*std::exp(-std::pow(2*r/b,c))*peak_jinc(root*r/rmax);
    require(std::isfinite(out),"nonfinite JINC coefficient");return out;
}
JincDiscreteKernel::JincDiscreteKernel(JincDiscreteSpec s):spec_(s){
    require(s.rows>0&&s.columns>0&&s.rows<=std::numeric_limits<std::size_t>::max()/s.columns&&s.phases>0,"invalid JINC grid");
    for(auto x:s.pixel_matrix_rad)require(std::isfinite(x),"nonfinite JINC WCS");
    const auto m=s.pixel_matrix_rad;const double u=m[0]*m[0]+m[2]*m[2],v=m[1]*m[1]+m[3]*m[3],d=m[0]*m[1]+m[2]*m[3];
    // This is a floating representation check of M^T M=Delta^2 I, not a
    // cancellation/support rule or a numerical-adequacy certificate.
    const double tol=64*std::numeric_limits<double>::epsilon()*std::max(u,v);
    require(positive(u)&&positive(v)&&std::abs(u-v)<=tol&&std::abs(d)<=tol,"JINC requires square orthogonal affine WCS");
    require(positive(s.scale_rad)&&positive(s.a)&&positive(s.b)&&positive(s.c)&&positive(s.rmax),"missing explicit JINC kernel value");
    const double h=std::ceil(s.scale_rad*s.rmax/std::sqrt(u));
    require(std::isfinite(h)&&h>=1&&h<=10000&&s.phases<=4096,"JINC cache resource bound");half_=static_cast<std::size_t>(h);side_=2*half_+1;
    const long double count=static_cast<long double>(side_)*side_*s.phases*s.phases;
    require(count<=33554432,"JINC cache exceeds 256 MiB resource guard");cache_.resize(static_cast<std::size_t>(count));
    for(std::size_t qy=0;qy<s.phases;++qy)for(std::size_t qx=0;qx<s.phases;++qx){
        const double px=-.5+(qx+.5)/s.phases,py=-.5+(qy+.5)/s.phases;
        for(int dy=-int(half_);dy<=int(half_);++dy)for(int dx=-int(half_);dx<=int(half_);++dx){
            const double x=dx-px,y=dy-py,r=std::hypot(m[0]*x+m[1]*y,m[2]*x+m[3]*y)/s.scale_rad;
            cache_[((qy*s.phases+qx)*side_+dy+half_)*side_+dx+half_]=analytic(r,s.a,s.b,s.c,s.rmax);
        }
    }
}
double JincDiscreteKernel::coefficient(std::size_t qx,std::size_t qy,int dx,int dy)const{
    require(qx<spec_.phases&&qy<spec_.phases&&std::abs(dx)<=int(half_)&&std::abs(dy)<=int(half_),"JINC cache index");
    return cache_[((qy*spec_.phases+qx)*side_+dy+half_)*side_+dx+half_];
}
JincDiscreteAccumulation JincDiscreteKernel::accumulate(std::span<const JincDiscreteSample> samples,std::size_t workers)const{
    require(workers>0&&workers<=64,"explicit bounded JINC workers required");const auto n=spec_.rows*spec_.columns;
    // Worker-local full-map planes, not a dense deposition operator.
    require(static_cast<long double>(n)*workers*6*sizeof(double)<=2.L*1024*1024*1024,"JINC scratch exceeds 2 GiB resource guard");
    auto block=[&](std::size_t first,std::size_t last){auto r=empty(n);
        for(std::size_t i=first;i<last;++i){const auto &s=samples[i];if(!s.admitted)continue;
            require(std::isfinite(s.x)&&std::isfinite(s.y)&&std::isfinite(s.value)&&positive(s.coefficient)&&positive(s.frequency_hz),"invalid admitted JINC occurrence");
            if(s.x<.5||s.y<.5||s.x>=spec_.columns+.5||s.y>=spec_.rows+.5)continue;
            const double fx=std::floor(s.x),fy=std::floor(s.y);
            const double cx=fx+(s.x-fx>=.5),cy=fy+(s.y-fy>=.5);
            if(cx<1||cx>spec_.columns||cy<1||cy>spec_.rows)continue;
            const auto qx=static_cast<std::size_t>(std::floor(spec_.phases*(s.x-cx+.5))),qy=static_cast<std::size_t>(std::floor(spec_.phases*(s.y-cy+.5)));
            require(qx<spec_.phases&&qy<spec_.phases,"JINC phase out of half-open range");
            for(int dy=-int(half_);dy<=int(half_);++dy)for(int dx=-int(half_);dx<=int(half_);++dx){
                const auto x=static_cast<std::int64_t>(cx)+dx,y=static_cast<std::int64_t>(cy)+dy;
                if(x<1||y<1||x>static_cast<std::int64_t>(spec_.columns)||y>static_cast<std::int64_t>(spec_.rows))continue;
                const std::size_t p=(y-1)*spec_.columns+x-1;const double k=coefficient(qx,qy,dx,dy),w=s.coefficient*k;
                r.numerator[p]+=w*s.value;r.normalization[p]+=w;r.quadratic[p]+=s.coefficient*k*k;
                r.coefficient_squared_time[p]+=k*k/s.frequency_hz;r.absolute_normalization[p]+=std::abs(w);++r.members[p];
            }
        }finite(r);return r;
    };
    auto result=empty(n);
    if(workers==1)result=block(0,samples.size());
    else{
        std::vector<std::future<JincDiscreteAccumulation>> jobs;
        for(std::size_t w=0;w<workers;++w)jobs.push_back(std::async(std::launch::async,block,samples.size()*w/workers,samples.size()*(w+1)/workers));
        for(auto &job:jobs){auto r=job.get();for(std::size_t p=0;p<n;++p){
            result.numerator[p]+=r.numerator[p];result.normalization[p]+=r.normalization[p];result.quadratic[p]+=r.quadratic[p];
            result.coefficient_squared_time[p]+=r.coefficient_squared_time[p];result.absolute_normalization[p]+=r.absolute_normalization[p];result.members[p]+=r.members[p];
        }}finite(result);
    }
    result.algebraic_value.assign(n,std::numeric_limits<double>::quiet_NaN());result.algebraic_support.assign(n,0);
    for(std::size_t p=0;p<n;++p)if(result.quadratic[p]>0&&result.normalization[p]!=0){
        const double value=result.numerator[p]/result.normalization[p];if(std::isfinite(value)){result.algebraic_value[p]=value;result.algebraic_support[p]=1;}
    }
    return result;
}
} // namespace citlali::mapmaking
