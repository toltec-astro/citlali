#include <citlali/core/pipeline/fruit_cycle.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
using namespace citlali::pipeline;
using namespace citlali::mapmaking;
TEST(FruitCycle, PixelProjectionSupportAndExactBrightness) {
    std::vector<double> v(1681,0);std::vector<std::uint8_t> valid(1681,1);
    v[840]=17.25;valid[839]=0;v[839]=NAN;
    FruitPointModel m("fixture",{41,41,"grid"},21,21,-2./3600,2./3600,v,valid);
    EXPECT_DOUBLE_EQ(m.at(0,0).value,17.25);
    EXPECT_EQ(m.at(2./3600,0).support,FruitModelSupport::unavailable);
    EXPECT_EQ(m.at(60./3600,0).support,FruitModelSupport::outside);
    EXPECT_EQ(m.at(NAN,0).support,FruitModelSupport::unavailable);
    EXPECT_EQ(m.at(-2./3600,0).support,FruitModelSupport::excluded);
    EXPECT_EQ(m.at(0.999/3600,0).value,17.25);
}
TEST(FruitCycle, BypassAccountingMissingnessAndStaleState) {
    PtcMatrix y(4,3);y<<1,2,3,4,5,6,7,8,9,10,11,12;
    PtcMatrix m=PtcMatrix::Constant(4,3,0.25);PtcMask e=PtcMask::Ones(4,3),s=e;s(1,1)=0;
    auto r=FruitResidual::subtract("r",y,e,m,s);
    EXPECT_EQ(r->eligible()(1,1),0);EXPECT_EQ(r->values()(1,1),0);
    for(int i=0;i<12;++i)if(r->eligible().data()[i])EXPECT_DOUBLE_EQ(r->values().data()[i]+r->operand().data()[i],y.data()[i]);
    EXPECT_DOUBLE_EQ(y(0,0),1);PtcSolverRequest req;req.rank=1;
    const auto learned=FruitLearned::learn(r,req);
    auto other=FruitResidual::subtract("r",y,e,m,s);
    EXPECT_THROW(learned.apply(other),std::invalid_argument);
    EXPECT_NO_THROW(learned.apply(r));
    EXPECT_THROW(FruitResidual::subtract("",y,e,m,s),std::invalid_argument);
}
TEST(FruitCycle, NumericalKnownSourceProjectionCleaningRejoinMap) {
    // Three detectors traverse a three-pixel source against a rank-one temporal
    // common mode. The model is sampled through the actual containing-pixel
    // owner; PTC relearns from the current residual. No bookkeeping mock.
    std::vector<double> v(1681,0);std::vector<std::uint8_t> valid(1681,1);
    v[839]=2;v[840]=8;v[841]=2;
    FruitPointModel model("source",{41,41,"grid"},21,21,-2./3600,2./3600,v,valid);
    PtcMatrix y(60,3),m(60,3);PtcMask eligible=PtcMask::Ones(60,3);
    std::vector<ForwardMapMember> members;
    for(int t=0;t<60;++t)for(int d=0;d<3;++d){int p=(t+d)%3;
        m(t,d)=model.at((p-1)*(-2./3600),0).value;
        y(t,d)=m(t,d)+(d+1)*std::sin(t*.31)+double(d+1)*10;
        members.push_back({std::size_t(3*t+d),std::size_t(p),1});}
    auto r=FruitResidual::subtract("synthetic",y,eligible,m,eligible);PtcSolverRequest req;req.rank=1;
    auto learned=FruitLearned::learn(r,req);ASSERT_TRUE(learned.fit().converged);
    auto result=learned.apply(r);EXPECT_LT(result.cleaned.values.norm(),1e-10);
    auto op=ForwardMapOperator::resolve({1,3,"sky"},"domain",180,members,{0,{}});
    const auto out=op.apply("domain",{result.total.data(),180});
    ASSERT_EQ(out.signal.size(),3);EXPECT_NEAR(out.signal[0],2,1e-10);EXPECT_NEAR(out.signal[1],8,1e-10);EXPECT_NEAR(out.signal[2],2,1e-10);
    auto zero=FruitResidual::subtract("zero",y,eligible,PtcMatrix::Zero(60,3),eligible);
    auto direct=PtcPrepared::prepare(y,eligible);auto fit=ptc_learn(direct,req);auto reference=ptc_apply(direct,fit);
    auto replay=FruitLearned::learn(zero,req).apply(zero);
    EXPECT_EQ((replay.cleaned.causes!=reference.causes).count(),0);
    EXPECT_DOUBLE_EQ((replay.cleaned.values-reference.values).norm(),0);
    // Source removal changes the learned action, not just basis signs.
    EXPECT_GT((fit.basis*fit.basis.transpose()-learned.fit().basis*learned.fit().basis.transpose()).norm(),1e-4);
}
TEST(FruitCycle, SignedInjectionPreservesParentAndInvalidEntries) {
    PtcMatrix y(2,3);y<<4,7,NAN,1,-2,8;
    PtcMatrix i=PtcMatrix::Constant(2,3,-.25);PtcMask e=PtcMask::Ones(2,3),s=e;e(0,2)=0;s(0,2)=0;
    auto result=fruit_injected_parent(y,e,i,s);
    EXPECT_TRUE(std::isnan(result(0,2)));EXPECT_DOUBLE_EQ(y(0,0),4);
    for(int k=0;k<6;++k)if(e.data()[k])EXPECT_DOUBLE_EQ(result.data()[k]-y.data()[k],-.25);
    s(0,0)=0;EXPECT_THROW(fruit_injected_parent(y,e,i,s),std::invalid_argument);
    s(0,0)=1;i(0,0)=NAN;EXPECT_THROW(fruit_injected_parent(y,e,i,s),std::invalid_argument);
    i.setZero();result=fruit_injected_parent(y,e,i,s);EXPECT_DOUBLE_EQ(result(1,2),y(1,2));
}
TEST(FruitCycle, ReferenceSequenceFreshMeansMaskedApplyAndRankFailure) {
    PtcMatrix y(5,3);y<<10,21,30,12,25,37,14,20,35,18,22,33,11,29,31;
    PtcMask e=PtcMask::Ones(5,3);e(0,1)=e(0,2)=0;e(1,0)=0;
    PtcMatrix operand=PtcMatrix::Constant(5,3,.75);operand(0,1)=operand(0,2)=operand(1,0)=0;
    auto r=FruitResidual::subtract("masked-refseq",y,e,operand,PtcMask::Ones(5,3));
    Eigen::MatrixXd b(3,1);b<<1,0,0;PtcSolverRequest req;req.rank=1;
    auto x=fruit_apply_reference_basis(r,req,b,"saved-ref-stage-0");
    EXPECT_EQ(x.cleaned.failed_times,1);EXPECT_EQ(x.cleaned.causes(1,1),4);
    EXPECT_EQ(x.cleaned.causes(1,0),1);EXPECT_DOUBLE_EQ(x.cleaned.values(1,1),0);
    for(int d=0;d<3;++d){long double sum=0;int count=0;for(int t=0;t<5;++t)if(e(t,d)){sum+=y(t,d)-operand(t,d);++count;}
        EXPECT_DOUBLE_EQ(x.prepared.mean[d],double(sum/count));}
    for(int t=0;t<5;++t)for(int d=0;d<3;++d)if(x.cleaned.causes(t,d)==0){
        const double expected=d==0?0:y(t,d)-operand(t,d)-x.prepared.mean[d];
        EXPECT_DOUBLE_EQ(x.cleaned.values(t,d),expected);EXPECT_DOUBLE_EQ(x.total(t,d),expected+operand(t,d));}
    // Distinct application/MAP masks: deficient row excluded; valid row2 also
    // excluded by MAP alone. PTC means must still include row2's observed data.
    std::vector<ForwardMapMember> members;std::vector<double> values;
    for(int t=0;t<5;++t)for(int d=0;d<3;++d)if(x.cleaned.causes(t,d)==0){auto i=values.size();values.push_back(x.total(t,d));if(t!=2)members.push_back({i,0,1});}
    auto op=ForwardMapOperator::resolve({1,1,"sky"},"domain",values.size(),members,{0,{}});
    double sum=0;for(auto m:members)sum+=values[m.input_row];
    EXPECT_DOUBLE_EQ(op.apply("domain",values).signal[0],sum/members.size());
    EXPECT_THROW(fruit_apply_reference_basis(r,req,b,""),std::invalid_argument);
}
TEST(FruitCycle, GuardedFrozenInterventionSignedZeroMaskedAndTamper) {
    PtcMatrix y(2,3);y<<4,7,NAN,1,-2,8;
    PtcMask e=PtcMask::Ones(2,3),s=e;e(0,2)=0;s(0,2)=0;
    for(double amplitude:{0.,.25,-.25}) {
        PtcMatrix i=PtcMatrix::Constant(2,3,amplitude);i(0,2)=0;
        const auto saved=fruit_injected_parent(y,e,i,s);
        const auto verified=fruit_verified_injected_parent(y,e,i,s,i,saved);
        EXPECT_EQ(std::memcmp(verified.data(),saved.data(),std::size_t(saved.size())*sizeof(double)),0);
        EXPECT_EQ(std::memcmp(&verified(0,2),&y(0,2),sizeof(double)),0);
        EXPECT_DOUBLE_EQ(verified(0,0)-y(0,0),amplitude);
        auto wrong=saved;wrong(0,0)+=.125;
        EXPECT_THROW(fruit_verified_injected_parent(y,e,i,s,i,wrong),std::invalid_argument);
        auto wrong_i=i;wrong_i(0,0)+=.125;
        EXPECT_THROW(fruit_verified_injected_parent(y,e,i,s,wrong_i,saved),std::invalid_argument);
        wrong=saved;wrong(0,2)=0;
        EXPECT_THROW(fruit_verified_injected_parent(y,e,i,s,i,wrong),std::invalid_argument);
    }
}
TEST(FruitCycle, ReferenceSequenceStageOrderingAndAutonomousReplacement) {
    PtcMatrix y(4,3);y<<11,24,32,14,20,35,19,27,30,12,25,39;
    const PtcMask mask=PtcMask::Ones(4,3);PtcSolverRequest req;req.rank=1;
    auto execute=[&](const PtcMatrix &parent){
        PtcMatrix candidate=PtcMatrix::Zero(4,3);std::vector<PtcMatrix> outputs;
        for(int k=0;k<3;++k){Eigen::MatrixXd b=Eigen::MatrixXd::Zero(3,1);b(k,0)=1;
            const auto applied=candidate;auto residual=FruitResidual::subtract("stage-"+std::to_string(k),parent,mask,applied,mask);
            auto result=fruit_apply_reference_basis(residual,req,b,"REF-stage-"+std::to_string(k));
            // Independent coordinate-projector oracle with freshly computed means.
            PtcMatrix expected=parent-applied;const Eigen::RowVectorXd mean=expected.colwise().mean();expected.rowwise()-=mean;expected.col(k).setZero();expected+=applied;
            EXPECT_LT((result.total-expected).norm(),1e-13);
            candidate=result.total.cwiseMax(0); // complete replacement, no accumulation
            outputs.push_back(result.total);
        }return outputs;
    };
    const auto r=execute(y),zero=execute(y+PtcMatrix::Zero(4,3));
    for(int k=0;k<3;++k)EXPECT_DOUBLE_EQ((r[k]-zero[k]).norm(),0);
    EXPECT_GT((r[0]-r[1]).norm(),1);EXPECT_GT((r[1]-r[2]).norm(),1);
}
