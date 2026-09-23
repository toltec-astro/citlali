#include <citlali/core/pipeline/fruit_cycle.h>
#include <gtest/gtest.h>
#include <cmath>
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
