#include <citlali/core/pipeline/ptc_inverse_noise.h>
#include <citlali/core/pipeline/ptc_empirical_map_handoff.h>
#include <citlali/core/pipeline/val_empirical_map_handoff.h>
#include <citlali/core/pipeline/fruit_cycle.h>
#include <citlali/core/mapmaking/forward_map.h>
#include <Eigen/SVD>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace citlali::pipeline;
namespace {
PtcMatrix fixture(int n=240,int d=8) {
    PtcMatrix x(n,d);
    for(int t=0;t<n;++t)for(int j=0;j<d;++j)
        x(t,j)=50+5*j+(1+.08*j)*8*std::sin(.07*t)+(.2+.1*j)*
            (std::sin((.29+.03*j)*t+.51*j)+.4*std::cos((.43+.01*j)*t));
    return x;
}
PtcSolverRequest request(int rank=2) {
    PtcSolverRequest r;r.rank=rank;r.relative_objective_tolerance=1e-8;return r;
}
auto run(const PtcMatrix &x) {
    return inverse_noise_ptc(x,PtcMask::Ones(x.rows(),x.cols()),PtcMask::Ones(x.rows(),x.cols()),request(),"fixture");
}
double independent_reference_variance(const PtcMatrix &x,int detector) {
    const auto means=x.colwise().mean().eval();std::vector<double> refs(x.rows());
    for(int t=0;t<x.rows();++t) {
        std::vector<double> row;
        for(int d=0;d<x.cols();++d)if(d!=detector)row.push_back(x(t,d)-means[d]);
        std::sort(row.begin(),row.end());
        refs[t]=row.size()%2?row[row.size()/2]:(row[row.size()/2-1]+row[row.size()/2])/2;
    }
    long double mx=0,my=0,xx=0,xy=0,sse=0;
    for(int t=0;t<x.rows();++t){mx+=refs[t];my+=x(t,detector);}mx/=x.rows();my/=x.rows();
    for(int t=0;t<x.rows();++t){xx+=(refs[t]-mx)*(refs[t]-mx);xy+=(refs[t]-mx)*(x(t,detector)-my);}
    const long double slope=xy/xx;
    for(int t=0;t<x.rows();++t){const long double e=x(t,detector)-my-slope*(refs[t]-mx);sse+=e*e;}
    return static_cast<double>(sse/(x.rows()-2));
}
std::shared_ptr<const PtcMapPublication> publication(std::string name="publication",bool exclude=false) {
    PtcMapPublicationIdentity id{name,"application","cal","rtc","processing","observation","segment","support",
        0,0,1,{"detector-a","detector-b"},{10,11,12}};
    PtcMatrix x(3,2);x<<2,8,4,12,6,14;auto causes=PtcMask::Zero(3,2).eval();if(exclude)causes(1,1)=1;
    return PtcMapPublication::bind_verified_segment(std::move(id),x,causes,true);
}
PtcDetectorNoiseEvidence evidence(double reference,double remaining) {
    PtcDetectorNoiseEvidence e;e.eligible_count=3;e.source_free_count=3;e.reference_count=3;e.remaining_count=3;
    e.reference_variance=reference;e.remaining_variance=remaining;e.cleaning_coefficient=1/reference;
    e.map_coefficient=1/std::max(reference,remaining);e.reference_reason=e.remaining_reason="available";return e;
}
std::vector<PtcEmpiricalDetectorRecord> records() {
    return {{"detector-a",evidence(2,1)},{"detector-b",evidence(4,16)}};
}
auto coefficient(const std::shared_ptr<const PtcMapPublication> &q) {
    return PtcEmpiricalCoefficients::publish(q,"coefficient","exact-noise-plan",records(),"support");
}
citlali::mapmaking::MapCoordinateParent ast(const std::shared_ptr<const PtcMapPublication> &q) {
    return {q->identity().publication,q->identity().cal_parent,q->identity().rtc_grid,q->identity().detectors,q->identity().slots};
}
}

TEST(PtcInverseNoise, ReferenceIsIndependentLeaveOutMedianRegressionWithFullResidualPower) {
    for(int detectors:{7,8}) {
        auto x=fixture(240,detectors);x(40,0)+=25; // Tail is retained, never clipped by a robust scale.
        const auto result=run(x);ASSERT_TRUE(result.fit.converged)<<result.fit.stopping_reason;
        for(int d=0;d<detectors;++d) {
            ASSERT_TRUE(result.detectors[d].reference_variance);
            EXPECT_NEAR(*result.detectors[d].reference_variance,independent_reference_variance(x,d),1e-11);
            EXPECT_EQ(result.detectors[d].reference_count,x.rows());
        }
        const auto quiet=run(fixture(240,detectors));
        EXPECT_GT(*result.detectors[0].reference_variance,*quiet.detectors[0].reference_variance);
    }
}

TEST(PtcInverseNoise, CompleteWeightedCleaningMatchesIndependentWhitenedSvd) {
    const auto x=fixture();const auto result=run(x);ASSERT_TRUE(result.fit.converged)<<result.fit.stopping_reason;
    ASSERT_EQ(result.fit_columns.size(),x.cols());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(result.prepared.centered,Eigen::ComputeThinV);
    const Eigen::MatrixXd basis=svd.matrixV().leftCols(2);
    PtcMatrix expected=result.prepared.centered-result.prepared.centered*basis*basis.transpose();
    for(int d=0;d<x.cols();++d)expected.col(d)*=std::sqrt(*result.detectors[d].reference_variance);
    EXPECT_LT((result.cleaned.values-expected).norm(),1e-9);
    EXPECT_EQ(result.cleaned.retained,x.size());EXPECT_TRUE((result.cleaned.causes==0).all());
    for(int d=0;d<x.cols();++d) {
        const auto &e=result.detectors[d];ASSERT_TRUE(e.remaining_variance);ASSERT_TRUE(e.map_coefficient);
        const double mean=result.cleaned.values.col(d).mean();
        const double variance=(result.cleaned.values.col(d).array()-mean).square().sum()/(x.rows()-1);
        EXPECT_NEAR(*e.remaining_variance,variance,1e-13);
        EXPECT_DOUBLE_EQ(*e.map_coefficient,1/std::max(*e.reference_variance,*e.remaining_variance));
        EXPECT_LE(*e.map_coefficient,*e.cleaning_coefficient);
    }
}

TEST(PtcInverseNoise, EqualNoiseReducesToExistingUniformCleaning) {
    PtcMatrix x(64,4);
    for(int t=0;t<64;++t) {
        const double a=(t&1)?8:-8,n=(t&2)?1:-1,m=(t&4)?1:-1;
        x.row(t)<<a+n,a-n,a+m,a-m;
    }
    const auto mask=PtcMask::Ones(x.rows(),x.cols()).eval();const auto r=request(1);
    const auto weighted=inverse_noise_ptc(x,mask,mask,r);ASSERT_TRUE(weighted.fit.converged);
    for(const auto &e:weighted.detectors)EXPECT_NEAR(*e.reference_variance,*weighted.detectors[0].reference_variance,1e-13);
    const auto prepared=PtcPrepared::prepare(x,mask);const auto uniform=ptc_apply(prepared,ptc_learn(prepared,r));
    EXPECT_LT((weighted.cleaned.values-uniform.values).norm(),1e-10);
}

TEST(PtcInverseNoise, CalibratedUnitScalingTransformsVariancesCoefficientsAndOutputs) {
    const auto x=fixture();const auto a=run(x),b=run(3*x);
    ASSERT_TRUE(a.fit.converged);ASSERT_TRUE(b.fit.converged);
    EXPECT_LT((b.cleaned.values-3*a.cleaned.values).norm(),1e-9);
    for(int d=0;d<x.cols();++d) {
        EXPECT_NEAR(*b.detectors[d].reference_variance,9**a.detectors[d].reference_variance,1e-10);
        EXPECT_NEAR(*b.detectors[d].remaining_variance,9**a.detectors[d].remaining_variance,1e-10);
        EXPECT_NEAR(*b.detectors[d].map_coefficient,*a.detectors[d].map_coefficient/9,1e-10);
    }
}

TEST(PtcInverseNoise, FrozenExclusionsAndProtectedTrainingDoNotBecomeObservedZerosOrMapVetoes) {
    auto x=fixture();auto mask=PtcMask::Ones(x.rows(),x.cols()).eval();mask.col(7).setZero();mask.block(30,1,10,2).setZero();
    auto source_free=PtcMask::Ones(x.rows(),x.cols()).eval();source_free.block(100,0,20,x.cols()).setZero();
    const auto r=request(1);const auto before=inverse_noise_ptc(x,mask,source_free,r);
    for(int t=0;t<x.rows();++t)for(int d=0;d<x.cols();++d)if(!mask(t,d))x(t,d)=NAN;
    const auto after=inverse_noise_ptc(x,mask,source_free,r);ASSERT_TRUE(after.fit.converged)<<after.fit.stopping_reason;
    EXPECT_DOUBLE_EQ((before.cleaned.values-after.cleaned.values).norm(),0);
    EXPECT_EQ(after.fit_columns.size(),7);EXPECT_FALSE(after.detectors[7].reference_variance);
    EXPECT_EQ(after.detectors[7].reference_reason,"no-eligible-input");
    EXPECT_EQ(after.cleaned.causes(105,0),0); // The source region is still applied/mapped.
    x.block(100,0,20,7).array()+=100;
    const auto protected_change=inverse_noise_ptc(x,mask,source_free,r);
    for(int d=0;d<7;++d)EXPECT_EQ(after.detectors[d].reference_variance,protected_change.detectors[d].reference_variance);
    EXPECT_EQ(protected_change.effective_eligible(105,0),1);
    mask(50,7)=1;EXPECT_THROW(inverse_noise_ptc(x,mask,source_free,r),std::invalid_argument);
}

TEST(PtcInverseNoise, FailedSolverCannotPublishMapWeightsOrSilentlyChangeMethod) {
    const auto x=fixture();auto mask=PtcMask::Ones(x.rows(),x.cols()).eval();mask.block(30,1,10,2).setZero();
    auto r=request();r.iteration_limit=1;
    const auto failed=inverse_noise_ptc(x,mask,mask,r);
    EXPECT_FALSE(failed.fit.converged);EXPECT_EQ(failed.fit.stopping_reason,"iteration-limit-not-converged");
    EXPECT_EQ(failed.fit.request.rank,r.rank);EXPECT_EQ(failed.fit.request.method,r.method);
    EXPECT_EQ(failed.cleaned.retained,0);
    for(const auto &e:failed.detectors){ASSERT_TRUE(e.reference_variance);EXPECT_FALSE(e.map_coefficient);}
}

TEST(PtcInverseNoise, UnsupportedVarianceDropsDetectorBeforeBasisAndNeverUsesUnityFallback) {
    auto x=fixture();auto mask=PtcMask::Ones(x.rows(),x.cols()).eval();auto free=mask;
    free.col(0).setZero();free(0,0)=free(1,0)=1;
    const auto r=inverse_noise_ptc(x,mask,free,request());ASSERT_TRUE(r.fit.converged);
    EXPECT_EQ(r.fit_columns.size(),7);EXPECT_FALSE(r.detectors[0].reference_variance);
    EXPECT_FALSE(r.detectors[0].map_coefficient);EXPECT_TRUE((r.effective_eligible.col(0)==0).all());
    EXPECT_EQ(r.detectors[0].reference_reason,"insufficient-source-free-reference-pairs");
    EXPECT_TRUE((r.cleaned.causes.col(0)!=0).all());
    x.setConstant(4);const auto constant=run(x);EXPECT_FALSE(constant.fit.converged);
    EXPECT_EQ(constant.cleaned.retained,0);EXPECT_TRUE(constant.fit_columns.empty());
    for(const auto &e:constant.detectors)EXPECT_EQ(e.reference_reason,"constant-or-nonfinite-reference");
    PtcMatrix identical(64,4);for(int t=0;t<64;++t)identical.row(t).setConstant((t&1)?1:-1);
    const auto zero=run(identical);EXPECT_EQ(zero.cleaned.retained,0);
    for(const auto &e:zero.detectors)EXPECT_EQ(e.reference_reason,"nonpositive-or-nonfinite-reference-variance");
}

TEST(PtcInverseNoise, HighNoiseDetectorHasLessInfluenceOnSharedMode) {
    PtcMatrix x=fixture(400,8);
    for(int t=0;t<x.rows();++t)x(t,7)+=40*std::sin(.83*t);
    const auto mask=PtcMask::Ones(x.rows(),x.cols()).eval();const auto r=request(1);
    const auto weighted=inverse_noise_ptc(x,mask,mask,r);ASSERT_TRUE(weighted.fit.converged);
    const auto prepared=PtcPrepared::prepare(x,mask);const auto uniform=ptc_apply(prepared,ptc_learn(prepared,r));
    EXPECT_LT(*weighted.detectors[7].cleaning_coefficient,*weighted.detectors[0].cleaning_coefficient/100);
    EXPECT_LT(weighted.cleaned.values.leftCols(7).squaredNorm(),uniform.values.leftCols(7).squaredNorm()/10);
}

TEST(PtcInverseNoise, FruitLearnsFromExactResidualAndRestoresOnlyOwnedOperand) {
    const auto residual_values=fixture();auto mask=PtcMask::Ones(residual_values.rows(),residual_values.cols()).eval();
    PtcMatrix operand=PtcMatrix::Zero(mask.rows(),mask.cols());operand.block(80,0,30,mask.cols()).setConstant(50);
    PtcMatrix original=residual_values+operand;auto source_free=mask;source_free.block(80,0,30,mask.cols()).setZero();
    const auto residual=FruitResidual::subtract("exact-CAL-current-model",original,mask,operand,mask);
    const auto r=inverse_noise_fruit(residual,source_free,request(),"cycle0");ASSERT_TRUE(r.noise.fit.converged);
    const auto independent=inverse_noise_ptc(residual->values(),mask,source_free,request(),"cycle0");
    EXPECT_EQ(r.residual.get(),residual.get());EXPECT_LT((r.total-r.noise.cleaned.values-operand).norm(),1e-11);
    EXPECT_EQ(r.noise.detectors[0].reference_variance,independent.detectors[0].reference_variance);
    EXPECT_EQ(r.noise.detectors[0].remaining_variance,independent.detectors[0].remaining_variance);
    auto unsupported=mask;unsupported(100,0)=0;
    const auto absent=FruitResidual::subtract("unsupported-model",original,mask,operand,unsupported);
    const auto excluded=inverse_noise_fruit(absent,source_free,request());EXPECT_NE(excluded.noise.cleaned.causes(100,0),0);
    EXPECT_EQ(excluded.total(100,0),0);EXPECT_THROW(inverse_noise_fruit({},source_free,request()),std::invalid_argument);
}

TEST(PtcEmpiricalMap, CoefficientsBindExactPublicationDetectorSupportAndConservativeRule) {
    const auto q=publication("publication",true);const auto g=coefficient(q);
    const auto admission=ValEmpiricalMapAdmission::bind(q,g,ast(q));
    const auto a=admission.at(0,0,0),b=admission.at(0,1,0);
    ASSERT_TRUE(a.pass());ASSERT_TRUE(b.pass());EXPECT_EQ(g->value(a.coefficient_qc,0,0),.5);
    EXPECT_EQ(g->value(b.coefficient_qc,0,1),.0625);
    EXPECT_FALSE(admission.at(1,1,0).pass());EXPECT_FALSE(admission.at(0,0,1).pass());
    EXPECT_FALSE(g->value(a.coefficient_qc,0,1));
    const auto foreign=publication();EXPECT_THROW(ValEmpiricalMapAdmission::bind(foreign,g,ast(foreign)),std::invalid_argument);
    EXPECT_FALSE(ValEmpiricalHandoffEvaluator::evaluate(*g,foreign,0,0,true).pass());
    auto reordered=records();std::swap(reordered[0],reordered[1]);
    EXPECT_THROW(PtcEmpiricalCoefficients::publish(q,"id","noise",reordered,"support"),std::invalid_argument);
    EXPECT_THROW(PtcEmpiricalCoefficients::publish(q,"id","noise",records(),"foreign-support"),std::invalid_argument);
    auto stale=records();stale[0].noise.map_coefficient=7;
    EXPECT_THROW(PtcEmpiricalCoefficients::publish(q,"id","noise",stale,"support"),std::invalid_argument);
    const auto before=ast(q);auto wrong=before;wrong.slots[0]=9;
    EXPECT_THROW(ValEmpiricalMapAdmission::bind(q,g,wrong),std::invalid_argument);
}

TEST(PtcEmpiricalMap, MissingRemainingVarianceHasNoCoefficientButFiniteZeroIsValid) {
    const auto q=publication();auto data=records();data[0].noise=evidence(2,0);
    data[1].noise.remaining_count=1;data[1].noise.remaining_variance.reset();data[1].noise.map_coefficient.reset();
    data[1].noise.remaining_reason="insufficient-source-free-cleaned-samples";
    const auto g=PtcEmpiricalCoefficients::publish(q,"id","noise",data,"support");
    const auto admission=ValEmpiricalMapAdmission::bind(q,g,ast(q));
    ASSERT_TRUE(admission.at(0,0,0).pass());EXPECT_EQ(g->value(admission.at(0,0,0).coefficient_qc,0,0),.5);
    EXPECT_TRUE(q->member(0,1));EXPECT_FALSE(admission.at(0,1,0).pass());
    EXPECT_FALSE(g->value(admission.at(0,1,0).coefficient_qc,0,1));
}

TEST(PtcEmpiricalMap, WeightedMapUsesCoefficientsWithoutPrecisionOrSourceMaskReinterpretation) {
    const auto q=publication();const auto g=coefficient(q);const auto admission=ValEmpiricalMapAdmission::bind(q,g,ast(q));
    std::vector<citlali::mapmaking::ForwardMapMember> members;
    for(std::size_t t=0;t<3;++t)for(std::size_t d=0;d<2;++d) {
        const auto decision=admission.at(t,d,0);ASSERT_TRUE(decision.pass());
        members.push_back({t*2+d,0,*g->value(decision.coefficient_qc,t,d)});
    }
    const auto op=citlali::mapmaking::ForwardMapOperator::resolve({1,1,"grid"},"domain",6,std::move(members),{0,{}});
    const auto mapped=op.apply("domain",std::span<const double>{q->values().data(),6});
    EXPECT_DOUBLE_EQ(op.normalization()[0],3*(.5+.0625));
    EXPECT_DOUBLE_EQ(mapped.signal[0],(.5*(2+4+6)+.0625*(8+12+14))/(3*(.5+.0625)));
    EXPECT_NE(PtcEmpiricalCoefficients::family,PtcUniformCoefficients::family);
    EXPECT_EQ(PtcEmpiricalCoefficients::interpretation,"empirical-analysis-coefficient-not-map-precision");
}
