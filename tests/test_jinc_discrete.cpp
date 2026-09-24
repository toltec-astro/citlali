#include <citlali/core/mapmaking/jinc_discrete.h>
#include <gtest/gtest.h>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <limits>
#include <numbers>
using namespace citlali::mapmaking;
namespace {
JincDiscreteSpec spec(std::size_t phases=4){return {11,13,phases,{1,0,0,1},1.3,1.1,2.4,2.,1.7};}
// Independent direct pixel/sample summation in long double: no cache, no owner
// phase/crop/finalization helper. This oracle tests the selected discrete rule,
// not closeness to an unquantized profile or approval of observation parameters.
long double kernel(long double r,const JincDiscreteSpec &s){
    const auto j=[](long double x){return x==0?1.L:2*boost::math::cyl_bessel_j(1.L,x)/x;};
    const auto root=boost::math::cyl_bessel_j_zero(1.L,1);
    if(r==s.rmax)return 0;
    return j(2*std::numbers::pi_v<long double>*r/s.a)*std::exp(-std::pow(2*r/s.b,s.c))*j(root*r/s.rmax);
}
void oracle(const JincDiscreteKernel &k,const std::vector<JincDiscreteSample>&samples){
    const auto &s=k.spec();auto r=k.accumulate(samples,1);const auto m=s.pixel_matrix_rad;
    const int half=int(std::ceil(s.scale_rad*s.rmax/std::hypot(m[0],m[2])));
    for(std::size_t y=1;y<=s.rows;++y)for(std::size_t x=1;x<=s.columns;++x){
        long double n=0,c=0,q=0,t=0,d=0;std::uint64_t members=0;
        for(const auto &z:samples){if(!z.admitted)continue;const auto cx=std::floor(z.x+.5),cy=std::floor(z.y+.5);
            if(cx<1||cx>s.columns||cy<1||cy>s.rows||std::abs(double(x)-cx)>half||std::abs(double(y)-cy)>half)continue;
            const auto qx=std::floor(s.phases*(z.x-cx+.5)),qy=std::floor(s.phases*(z.y-cy+.5));
            const long double dx=x-cx+.5-(qx+.5)/s.phases,dy=y-cy+.5-(qy+.5)/s.phases;
            const auto v=kernel(std::hypot(m[0]*dx+m[1]*dy,m[2]*dx+m[3]*dy)/s.scale_rad,s);const auto w=z.coefficient*v;
            n+=w*z.value;c+=w;q+=z.coefficient*v*v;t+=v*v/z.frequency_hz;d+=std::abs(w);++members;
        }
        const auto p=(y-1)*s.columns+x-1;
        for(auto [actual,expected]:{std::pair{r.numerator[p],n},{r.normalization[p],c},{r.quadratic[p],q},{r.coefficient_squared_time[p],t},{r.absolute_normalization[p],d}})
            EXPECT_NEAR(actual,double(expected),2e-12*std::max(1.L,std::abs(expected)));
        EXPECT_EQ(r.members[p],members);
    }
}
}
TEST(JincDiscrete,ExplicitParametersAndSquareWcs){
    auto s=spec();EXPECT_NO_THROW(JincDiscreteKernel{s});s.scale_rad=0;EXPECT_THROW(JincDiscreteKernel{s},std::invalid_argument);
    s=spec();s.pixel_matrix_rad={1,0,0,2};EXPECT_THROW(JincDiscreteKernel{s},std::invalid_argument);
    s=spec();s.pixel_matrix_rad={1,.2,0,1};EXPECT_THROW(JincDiscreteKernel{s},std::invalid_argument);
    s=spec();s.phases=0;EXPECT_THROW(JincDiscreteKernel{s},std::invalid_argument);
    s=spec();s.pixel_matrix_rad={0,-1,1,0};EXPECT_NO_THROW(JincDiscreteKernel{s});
    s.pixel_matrix_rad={-1,0,0,1};EXPECT_NO_THROW(JincDiscreteKernel{s});
}
TEST(JincDiscrete,IndependentSignedOracleCornersAndMissingSamples){
    const double nan=std::numeric_limits<double>::quiet_NaN();
    std::vector<JincDiscreteSample> v={{7,6,5,1,10,true},{1.49,2.51,5,2,20,true},{12.8,10.8,5,.5,11,true},{.49,4,5,1,10,true},{nan,nan,nan,nan,nan,false}};
    for(auto phases:{1U,4U}){JincDiscreteKernel k(spec(phases));EXPECT_EQ(k.half_width(),3);oracle(k,v);auto r=k.accumulate(v,1);
        bool negative=false;for(std::size_t p=0;p<r.numerator.size();++p)if(r.algebraic_support[p]){EXPECT_NEAR(r.algebraic_value[p],5.,1e-10);negative|=r.normalization[p]<0;}
        EXPECT_TRUE(negative);EXPECT_NE(k.coefficient(0,0,3,3),0.); // outside rmax, inside square
    }
}
TEST(JincDiscrete,CenterTiesPhaseTiesAndAnalyticZero){
    auto s=spec(4);JincDiscreteKernel k(s);
    const auto center=k.coefficient(2,2,0,0);EXPECT_NEAR(center,JincDiscreteKernel::analytic(std::sqrt(2.)/8/s.scale_rad,s.a,s.b,s.c,s.rmax),1e-14);EXPECT_NE(center,1.);
    EXPECT_DOUBLE_EQ(JincDiscreteKernel(spec(1)).coefficient(0,0,0,0),1.);
    EXPECT_DOUBLE_EQ(JincDiscreteKernel::analytic(s.rmax,s.a,s.b,s.c,s.rmax),0.);
    auto zero_spec=spec(1);zero_spec.scale_rad=1;zero_spec.rmax=2;JincDiscreteKernel zero(zero_spec);
    std::vector<JincDiscreteSample> one={{6,6,1,1,10,true}};const auto zeros=zero.accumulate(one,1);
    const auto zp=5*zero_spec.columns+7;EXPECT_EQ(zeros.members[zp],1U);EXPECT_DOUBLE_EQ(zeros.normalization[zp],0.);EXPECT_EQ(zeros.algebraic_support[zp],0);
    std::vector<JincDiscreteSample> tie={{.5,3.5,1,1,10,true},{13.5,4,1,1,10,true},{3,4,1,1,10,true}};oracle(k,tie);
    auto r=k.accumulate(std::span(tie).first(1),1);EXPECT_GT(r.members[3*s.columns],0U);
    EXPECT_THROW(k.coefficient(4,0,0,0),std::invalid_argument);
}
TEST(JincDiscrete,RotatedReflectedMetricAndCoefficientTimeMeaning){
    std::vector<JincDiscreteSample> v={{6.3,5.1,4,2,10,true},{7.2,6.2,-3,.7,20,true}};
    for(auto m:{std::array<double,4>{-1,0,0,1},std::array<double,4>{.6,-.8,.8,.6}}){auto s=spec();s.pixel_matrix_rad=m;oracle(JincDiscreteKernel(s),v);}
    JincDiscreteKernel k(spec());auto a=k.accumulate(v,1);for(auto &x:v)x.coefficient*=3;auto b=k.accumulate(v,1);
    for(std::size_t p=0;p<a.numerator.size();++p){EXPECT_NEAR(b.normalization[p],3*a.normalization[p],1e-13);EXPECT_NEAR(b.quadratic[p],3*a.quadratic[p],1e-13);EXPECT_DOUBLE_EQ(b.coefficient_squared_time[p],a.coefficient_squared_time[p]);}
}
TEST(JincDiscrete,LinearRejoinAndParallelEquivalence){
    JincDiscreteKernel k(spec());std::vector<JincDiscreteSample> total,clean,sky;
    for(int i=0;i<180;++i){JincDiscreteSample z{1+(i%11)+.13,1+(i%9)+.23,std::sin(i),1+(i%3)*.2,10,true};clean.push_back(z);z.value=std::cos(i);sky.push_back(z);z.value+=clean.back().value;total.push_back(z);}
    auto t=k.accumulate(total,1),p=k.accumulate(total,3),z=k.accumulate(clean,1),v=k.accumulate(sky,1);oracle(k,total);
    for(std::size_t i=0;i<t.numerator.size();++i){EXPECT_NEAR(t.numerator[i],z.numerator[i]+v.numerator[i],1e-12);EXPECT_NEAR(p.numerator[i],t.numerator[i],1e-12);EXPECT_NEAR(p.normalization[i],t.normalization[i],1e-12);EXPECT_NEAR(p.quadratic[i],t.quadratic[i],1e-12);EXPECT_NEAR(p.coefficient_squared_time[i],t.coefficient_squared_time[i],1e-12);EXPECT_EQ(p.members[i],t.members[i]);}
}
TEST(JincDiscrete,CancellationIsAlgebraicOnlyAndInvalidAdmissionFails){
    JincDiscreteKernel k(spec(1));const double negative=k.coefficient(0,0,1,0);ASSERT_LT(negative,0.);
    std::vector<JincDiscreteSample> v={{6,6,1,1,10,true},{7,6,2,-1/negative,10,true}};
    auto r=k.accumulate(v,1);const auto p=5*k.spec().columns+5;EXPECT_DOUBLE_EQ(r.normalization[p],0.);
    if(r.normalization[p]==0){EXPECT_EQ(r.algebraic_support[p],0);EXPECT_TRUE(std::isnan(r.algebraic_value[p]));}
    v[0].coefficient=0;EXPECT_THROW(k.accumulate(v,1),std::invalid_argument);v[0].admitted=false;EXPECT_NO_THROW(k.accumulate(v,1));
    EXPECT_THROW(k.accumulate(v,0),std::invalid_argument);
}
