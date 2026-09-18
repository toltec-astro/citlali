#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>
#include <cmath>
using namespace citlali::pipeline;
namespace {
PtcMatrix fixture(int n=120,int d=12) {
    PtcMatrix x(n,d);
    for(int t=0;t<n;++t)for(int j=0;j<d;++j)
        x(t,j)=10+j+std::sin(.07*t)*(1+.1*j)+std::cos(.13*t)*std::sin(.9*j)+.05*std::sin(.31*t+.62*j*j);
    return x;
}
PtcSolverRequest request(){PtcSolverRequest r;r.rank=2;r.relative_objective_tolerance=1e-7;return r;}
}
TEST(PtcNumerics, CompleteDataMatchesSvdAndDoesNotRestoreMean) {
    const auto x=fixture();auto p=PtcPrepared::prepare(x,PtcMask::Ones(x.rows(),x.cols()));auto fit=ptc_learn(p,request());
    ASSERT_TRUE(fit.converged)<<fit.stopping_reason;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(p.centered,Eigen::ComputeThinV);
    const Eigen::MatrixXd b=svd.matrixV().leftCols(2);
    auto a=ptc_apply(p,fit);
    EXPECT_LT((a.values-(p.centered-p.centered*b*b.transpose())).norm(),1e-9);
    EXPECT_LT(a.values.colwise().mean().norm(),1e-12);EXPECT_EQ(a.retained,x.size());
    EXPECT_EQ(a.factorizations,1);EXPECT_EQ(a.factor_reuses,x.rows()-1);
}
TEST(PtcNumerics, IncompleteRowsAndGoodPartsOfFlaggedDetectorsContribute) {
    auto x=fixture();auto mask=PtcMask::Ones(x.rows(),x.cols()).eval();
    for(int t=0;t<x.rows();++t)mask(t,t%12)=0;
    auto p=PtcPrepared::prepare(x,mask);EXPECT_EQ(p.complete_times,0);EXPECT_EQ(p.eligible_count,120*11);
    auto f=ptc_learn(p,request());ASSERT_TRUE(f.converged)<<f.stopping_reason;
    auto modified=x;modified(4,3)+=100.;auto q=PtcPrepared::prepare(modified,mask);auto g=ptc_learn(q,request());
    EXPECT_NE(p.mean[3],q.mean[3]);EXPECT_GT((f.basis*f.basis.transpose()-g.basis*g.basis.transpose()).norm(),.01);
    EXPECT_EQ(ptc_apply(p,f).retained,120*11);
}
TEST(PtcNumerics, ExcludedDonorValuesIncludingNonfiniteNeverEnterArithmetic) {
    auto x=fixture();auto mask=PtcMask::Ones(x.rows(),x.cols()).eval();mask.block(20,2,15,2).setZero();
    auto p=PtcPrepared::prepare(x,mask);auto r=request();auto f=ptc_learn(p,r);auto a=ptc_apply(p,f);
    for(int t=20;t<35;++t){x(t,2)=NAN;x(t,3)=t%2?INFINITY:-INFINITY;}
    auto q=PtcPrepared::prepare(x,mask);auto g=ptc_learn(q,r);auto b=ptc_apply(q,g);
    EXPECT_EQ((p.mean-q.mean).norm(),0);EXPECT_EQ((f.basis-g.basis).norm(),0);EXPECT_EQ((a.values-b.values).norm(),0);
    EXPECT_EQ(a.retained,b.retained);EXPECT_EQ(a.causes(25,2),1);
    mask(25,2)=1;EXPECT_THROW(PtcPrepared::prepare(x,mask),std::invalid_argument);
}
TEST(PtcNumerics, NonconvergenceAndDeficiencyDoNotSwitchMethodsOrReduceRank) {
    auto x=fixture();auto m=PtcMask::Ones(x.rows(),x.cols()).eval();m.block(0,0,25,4).setZero();
    auto p=PtcPrepared::prepare(x,m);auto r=request();r.iteration_limit=1;auto f=ptc_learn(p,r);
    EXPECT_FALSE(f.converged);EXPECT_EQ(f.stopping_reason,"iteration-limit-not-converged");EXPECT_EQ(f.request.method,PtcMethod::observed_als);
    EXPECT_EQ(ptc_apply(p,f).retained,0);
    r=request();r.rank=13;EXPECT_FALSE(ptc_learn(p,r).converged);
    m.col(0).setZero();auto q=PtcPrepared::prepare(x,m);r=request();auto failed=ptc_learn(q,r);
    EXPECT_FALSE(failed.converged);EXPECT_EQ(failed.stopping_reason,"insufficient-pairwise-initialization-overlap");
}
TEST(PtcNumerics, FrozenMaskedResponseIsSameLocalLinearOperator) {
    auto x=fixture();auto m=PtcMask::Ones(x.rows(),x.cols()).eval();m.block(0,0,2,11).setZero();
    auto p=PtcPrepared::prepare(x,m);auto r=request();r.method=PtcMethod::pairwise_covariance;auto f=ptc_learn(p,r);ASSERT_TRUE(f.converged);
    PtcMatrix h=fixture()/30.;auto response=ptc_response(p,f,h);auto baseline=ptc_apply(p,f);
    auto perturbed=p;perturbed.centered+=h;auto injected=ptc_apply(perturbed,f);
    EXPECT_EQ(response.causes(0,11),4);EXPECT_EQ(response.retained,baseline.retained);
    EXPECT_LT((injected.values-baseline.values-response.values).norm(),1e-11);
    h(0,0)=NAN;EXPECT_EQ(ptc_response(p,f,h).causes(0,11),4);
    h(3,0)=NAN;EXPECT_EQ(ptc_response(p,f,h).causes(3,1),8);
}
TEST(PtcNumerics, PairwiseUsesDetectorMeansAndOverlapMinusOne) {
    auto x=fixture();auto mask=PtcMask::Ones(x.rows(),x.cols()).eval();mask.block(10,2,17,1).setZero();
    auto p=PtcPrepared::prepare(x,mask);auto r=request();r.method=PtcMethod::pairwise_covariance;auto f=ptc_learn(p,r);
    ASSERT_TRUE(f.converged);Eigen::MatrixXd c(x.cols(),x.cols());
    for(int d=0;d<x.cols();++d)for(int e=0;e<x.cols();++e) {
        double sum=0;int count=0;
        for(int t=0;t<x.rows();++t)if(mask(t,d)&&mask(t,e)){sum+=(x(t,d)-p.mean[d])*(x(t,e)-p.mean[e]);++count;}
        c(d,e)=sum/(count-1);
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(c);const Eigen::MatrixXd b=eig.eigenvectors().rightCols(2);
    EXPECT_LT((b*b.transpose()-f.basis*f.basis.transpose()).norm(),1e-12);
}
TEST(PtcNumerics, UnsupportedLargeRankAndOneTimeRemainBoundedFailures) {
    auto x=fixture(1);auto p=PtcPrepared::prepare(x,PtcMask::Ones(x.rows(),x.cols()));auto r=request();r.rank=100000000;
    auto f=ptc_learn(p,r);EXPECT_FALSE(f.converged);auto a=ptc_apply(p,f);EXPECT_EQ(a.retained,0);EXPECT_EQ(a.causes(0,0),2);
}

TEST(PtcNumerics, UnresolvedCutoffDegeneracyDoesNotChooseArbitrarySubspace) {
    PtcMatrix x(6,3);x<<1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1;
    auto p=PtcPrepared::prepare(x,PtcMask::Ones(6,3));auto r=request();r.rank=2;
    for(auto method:{PtcMethod::observed_als,PtcMethod::pairwise_covariance}) {
        r.method=method;auto f=ptc_learn(p,r);EXPECT_FALSE(f.converged);
        EXPECT_EQ(f.stopping_reason,"unresolved-eigenspace-cutoff-degeneracy");EXPECT_EQ(ptc_apply(p,f).retained,0);
    }
    r.rank=3;EXPECT_TRUE(ptc_learn(p,r).converged);
}
