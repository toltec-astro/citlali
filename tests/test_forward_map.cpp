#include <citlali/core/mapmaking/forward_map.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <limits>

using namespace citlali::mapmaking;
namespace {
const ForwardMapGrid grid{2,3,"fixture-exact-AST-grid"};
const std::string domain="fixture-PTC-publication-ordered-occurrences";
auto op(std::vector<ForwardMapMember> members, double cut=0) {
    return ForwardMapOperator::resolve(grid,domain,6,std::move(members),{cut,{}});
}
}

TEST(ForwardMap, HalfOpenCentersEdgesAndOrientation) {
    EXPECT_EQ(containing_pixel(grid,0.5,0.5),0);
    EXPECT_EQ(containing_pixel(grid,1,1),0);
    EXPECT_EQ(containing_pixel(grid,1.5,1),1);
    EXPECT_EQ(containing_pixel(grid,3,2),5);
    EXPECT_EQ(containing_pixel(grid,std::nextafter(3.5,0.),1),2);
    EXPECT_FALSE(containing_pixel(grid,3.5,1));
    EXPECT_FALSE(containing_pixel(grid,1,2.5));
    EXPECT_FALSE(containing_pixel(grid,std::nextafter(0.5,0.),1));
    EXPECT_FALSE(containing_pixel(grid,NAN,1));
    EXPECT_FALSE(containing_pixel(grid,1,INFINITY));
}
TEST(ForwardMap, IndependentUnequalCoefficientCalculation) {
    // N=2*(-3)+1*6=0, Q=3: a measured zero, whereas pixel 1 is absent.
    const auto a=op({{0,0,2},{1,0,1},{3,4,0.5}});
    const std::array<double,6> z{-3,6,NAN,-8,INFINITY,NAN};
    const auto r=a.apply(domain,z);
    EXPECT_EQ(a.output_rows(),(std::vector<std::size_t>{0,4}));
    EXPECT_EQ(r.numerator,(std::vector<double>{0,0,0,0,-4,0}));
    EXPECT_EQ(a.normalization(),(std::vector<double>{3,0,0,0,0.5,0}));
    EXPECT_EQ(r.signal,(std::vector<double>{0,-8}));
    EXPECT_EQ(a.contributing_hits(),(std::vector<std::uint64_t>{2,0,0,0,1,0}));
}
TEST(ForwardMap, ConstantAndDeltaSameLinearCompanionOperator) {
    const auto a=op({{0,0,1},{1,0,3},{3,4,2}});
    const std::array<double,6> constant{7,7,NAN,7,NAN,NAN};
    EXPECT_EQ(a.apply(domain,constant).signal,(std::vector<double>{7,7}));
    const std::array<double,6> delta{0,4,NAN,0,NAN,NAN};
    EXPECT_EQ(a.apply(domain,delta).signal,(std::vector<double>{3,0}));
    const std::array<double,6> sum{7,11,NAN,7,NAN,NAN};
    EXPECT_EQ(a.apply(domain,sum).signal,(std::vector<double>{10,7}));
}
TEST(ForwardMap, ExplicitSupportSeparatesNormalizationAndScience) {
    const auto a=op({{0,0,1},{1,1,5},{2,2,10}},0.5);
    EXPECT_EQ(a.reference_normalization(),10);
    EXPECT_EQ(a.reference_index(),2);
    EXPECT_EQ(a.normalization_support(),(std::vector<std::uint8_t>{1,1,1,0,0,0}));
    EXPECT_EQ(a.science_support(),(std::vector<std::uint8_t>{0,1,1,0,0,0}));
    EXPECT_EQ(a.output_rows(),(std::vector<std::size_t>{1,2}));
}
TEST(ForwardMap, SmallPopulationSelectorAndEmptySupport) {
    // Independently tabulated zero-based order statistic for N=1..16,
    // including the first two departures from simply taking the maximum.
    const std::array<std::size_t,16> expected{0,1,2,3,4,5,6,7,7,8,9,10,11,12,13,14};
    for(std::size_t n=1;n<=expected.size();++n) {
        std::vector<ForwardMapMember> members;
        for(std::size_t i=0;i<n;++i)members.push_back({i,i,double(i+1)});
        auto a=ForwardMapOperator::resolve({1,n,"grid"},domain,n,members,{0,{}});
        EXPECT_EQ(a.reference_index(),expected[n-1]);
    }
    auto a=op({});
    EXPECT_EQ(a.reference_normalization(),0); EXPECT_FALSE(a.reference_index());
    EXPECT_TRUE(a.output_rows().empty());
    const std::array<double,6> absent{NAN,NAN,NAN,NAN,NAN,NAN};
    EXPECT_TRUE(a.apply(domain,absent).signal.empty());
    auto cut=ForwardMapOperator::resolve(grid,domain,6,{{0,0,1}},{2,"fixture-owner-same-cut-scope"});
    EXPECT_TRUE(cut.output_rows().empty()); // numerical no-row result, never a map product
}
TEST(ForwardMap, BadPlanAndPayloadNeverMutateResolvedOperator) {
    for(double c:std::array<double,4>{-1.,NAN,INFINITY,2.}) EXPECT_THROW(op({},c),std::invalid_argument);
    for(double c:std::array<double,4>{0.,-1.,NAN,INFINITY}) EXPECT_THROW(op({{0,0,c}}),std::invalid_argument);
    EXPECT_THROW(op({{0,0,1},{0,1,1}}),std::invalid_argument);
    EXPECT_THROW(op({{6,0,1}}),std::invalid_argument);
    EXPECT_THROW(op({{0,6,1}}),std::invalid_argument);
    auto a=op({{0,0,1}});
    const auto original=a.normalization();
    const std::array<double,6> bad{NAN,0,0,0,0,0};
    EXPECT_THROW(a.apply(domain,bad),std::invalid_argument);
    EXPECT_THROW(a.apply("same-shape-foreign-parent",bad),std::invalid_argument);
    EXPECT_THROW(a.apply(domain,std::span<const double>{bad.data(),5}),std::invalid_argument);
    EXPECT_EQ(a.normalization(),original);
}
TEST(ForwardMap, OverflowIsFailureNotSanitizedZero) {
    const double big=std::numeric_limits<double>::max();
    EXPECT_THROW(op({{0,0,big},{1,0,big}}),std::overflow_error);
    auto a=op({{0,0,2}});
    const std::array<double,6> values{big,0,0,0,0,0};
    EXPECT_THROW(a.apply(domain,values),std::overflow_error);
    EXPECT_EQ(a.normalization()[0],2);
    const double tiny=std::numeric_limits<double>::denorm_min();
    auto b=op({{0,0,tiny}});
    const std::array<double,6> half{0.5,0,0,0,0,0};
    EXPECT_THROW(b.apply(domain,half),std::overflow_error);
    EXPECT_THROW(op({{0,0,tiny}},0.5),std::overflow_error);
}
TEST(ForwardMap, NonzeroSubnormalRoundingCannotChangeConstantOrSupport) {
    const double tiny=std::numeric_limits<double>::denorm_min();
    const auto a=op({{0,0,tiny}});
    // These used to return 1, 2, 2 (and their negatives) instead of failing.
    for (double z:std::array<double,6>{0.75,1.5,2.5,-0.75,-1.5,-2.5}) {
        const std::array<double,6> values{z,0,0,0,0,0};
        EXPECT_THROW(a.apply(domain,values),std::overflow_error);
    }
    for (double z:std::array<double,5>{0,1,2,-1,-2}) {
        const std::array<double,6> values{z,0,0,0,0,0};
        EXPECT_EQ(a.apply(domain,values).signal,(std::vector<double>{z}));
    }
    // Reference*cut rounds to a nonzero subnormal, or threshold/10 does.
    EXPECT_THROW(op({{0,0,100*tiny}},0.751),std::overflow_error);
    EXPECT_THROW(op({{0,0,15*tiny}},1),std::overflow_error);
    EXPECT_NO_THROW(op({{0,0,100*tiny}},0.5));
    // Also catch an inexact result rounded up to the minimum normal value.
    const double normal=std::numeric_limits<double>::min();
    const auto edge=op({{0,0,normal}});
    const std::array<double,6> almost_one{std::nextafter(1.,0.),0,0,0,0,0};
    EXPECT_THROW(edge.apply(domain,almost_one),std::overflow_error);
}
TEST(ForwardMap, NormalizationRejectsInexactSubnormalButPreservesExact) {
    const double tiny=std::numeric_limits<double>::denorm_min();
    // Products and sums are exact: numerator=3*tiny, denominator=2.
    const auto a=op({{0,0,1},{1,0,1}});
    const std::array<double,6> inexact{tiny,2*tiny,0,0,0,0};
    EXPECT_THROW(a.apply(domain,inexact),std::overflow_error);
    const std::array<double,6> exact{tiny,tiny,0,0,0,0};
    EXPECT_EQ(a.apply(domain,exact).signal,(std::vector<double>{tiny}));
    const std::array<double,6> zero{tiny,-tiny,0,0,0,0};
    EXPECT_EQ(a.apply(domain,zero).signal,(std::vector<double>{0}));
}
TEST(ForwardMap, OrderBlockAssemblyAndConcurrentReadOnlyApplication) {
    std::vector<ForwardMapMember> members{{0,0,1},{1,0,1},{2,0,1},{3,2,1}};
    auto a=op(members);std::reverse(members.begin(),members.end());auto b=op(members);
    const std::array<double,6> values{1e16,1,-1e16,3,NAN,NAN};
    EXPECT_EQ(a.apply(domain,values).signal,b.apply(domain,values).signal);
    auto first=std::async(std::launch::async,[&]{return a.apply(domain,values);});
    auto second=std::async(std::launch::async,[&]{return a.apply(domain,values);});
    EXPECT_EQ(first.get().signal,second.get().signal);
}
TEST(ForwardMap, OriginalFootprintsUnionAndDoNotFollowDescendantPixels) {
    const std::vector<ForwardMapOriginal> originals{{4,0.25},{1,0.5},{5,0},{std::nullopt,1}};
    const std::vector<ForwardMapAncestor> ancestry{{0,true,false},{0,true,true},{0,true,true},
        {1,true,false},{2,true,true},{3,true,true}};
    const auto r=accumulate_original_footprints(grid,"ALIGN-exact-original-domain",originals,ancestry);
    EXPECT_EQ(r.upstream_eligible,(std::vector<double>{0,0.5,0,0,0.25,0}));
    EXPECT_EQ(r.retained,(std::vector<double>{0,0,0,0,0.25,0}));
    EXPECT_THROW(accumulate_original_footprints(grid,"",originals,ancestry),std::invalid_argument);
    const std::vector<ForwardMapAncestor> foreign{{4,true,true}}, invalid{{0,false,true}};
    EXPECT_THROW(accumulate_original_footprints(grid,"originals",originals,foreign),std::invalid_argument);
    EXPECT_THROW(accumulate_original_footprints(grid,"originals",originals,invalid),std::invalid_argument);
}

#include <citlali/core/pipeline/ptc_map_handoff.h>
using namespace citlali::pipeline;
namespace {
auto published_segment(bool empty=false) {
    PtcMapPublicationIdentity id{"q","application","CAL","RTC","processing","observation","segment","support",0,0,1,{"detector-A","detector-B"},{3,5}};
    PtcMatrix values(2,2);values<<1,2,3,NAN;
    PtcMask causes(2,2);causes<<0,0,0,1;if(empty)causes.setConstant(1);
    return PtcMapPublication::bind_verified_segment(id,values,causes,true);
}
}
TEST(PtcMapHandoff, ExplicitUniformSelectionExactSupportAndRequest) {
    const auto q=published_segment();
    const auto g=PtcUniformCoefficients::publish(q,{"owner-selection",std::string(PtcUniformCoefficients::family)},"g","support");
    ASSERT_TRUE(g);EXPECT_EQ(q->retained(),3);
    auto admitted=ValUniformHandoffEvaluator::evaluate(*g,q,0,0,true);EXPECT_TRUE(admitted.pass());EXPECT_EQ(g->value(admitted,0,0),1);
    EXPECT_FALSE(g->value(admitted,0,1)); // a decision cannot move between occurrences
    EXPECT_FALSE(ValUniformHandoffEvaluator::evaluate(*g,q,0,0,false).realized);
    EXPECT_FALSE(ValUniformHandoffEvaluator::evaluate(*g,q,1,1,true).applicable);
    EXPECT_FALSE(ValUniformHandoffEvaluator::evaluate(*g,published_segment(),0,0,true).structurally_bound);
    EXPECT_FALSE(PtcUniformCoefficients::publish(published_segment(true),{"selection",std::string(PtcUniformCoefficients::family)},"empty","support"));
    for(const auto &selection:std::vector<UniformSelection>{{"wrong","other-family"},{"missing",std::nullopt},{"conflicting",std::string(PtcUniformCoefficients::family),true}}) {
        const auto bad=PtcUniformCoefficients::publish(q,selection,"g-other","support");
        auto decision=ValUniformHandoffEvaluator::evaluate(*bad,q,0,0,true);EXPECT_TRUE(decision.realized);EXPECT_FALSE(decision.pass());
        decision.restrictions.fill(UniformFact::true_value);EXPECT_FALSE(bad->value(decision,0,0));
    }
    const auto bad=PtcUniformCoefficients::publish(q,{"selection",std::string(PtcUniformCoefficients::family)},"bad","wrong-support");
    EXPECT_FALSE(ValUniformHandoffEvaluator::evaluate(*bad,q,0,0,true).pass());
}
TEST(PtcMapHandoff, FourValuedFidelityAndStructuralFailures) {
    using F=UniformFact;
    EXPECT_EQ(uniform_all(std::array{F::conflict,F::unknown,F::false_value}),F::false_value);
    EXPECT_EQ(uniform_all(std::array{F::conflict,F::unknown}),F::conflict);
    EXPECT_EQ(uniform_all(std::array{F::true_value,F::unknown}),F::unknown);
    const auto q=published_segment();auto id=q->identity();
    id.slots={3,3};EXPECT_THROW(PtcMapPublication::bind_verified_segment(id,q->values(),q->causes(),true),std::invalid_argument);
    id=q->identity();id.detectors={"same","same"};EXPECT_THROW(PtcMapPublication::bind_verified_segment(id,q->values(),q->causes(),true),std::invalid_argument);
    id=q->identity();id.rank=0;EXPECT_THROW(PtcMapPublication::bind_verified_segment(id,q->values(),q->causes(),true),std::invalid_argument);
    EXPECT_THROW(PtcMapPublication::bind_verified_segment(q->identity(),q->values(),q->causes(),false),std::invalid_argument);
}

#include <citlali/core/mapmaking/forward_map_admission.h>
#include <citlali/core/pipeline/val_map_handoff.h>
TEST(ForwardMapAdmission, ExactParentsAndSeparateProducerAndMapDecisions) {
    const auto q=published_segment();const auto g=PtcUniformCoefficients::publish(q,{"selection",std::string(PtcUniformCoefficients::family)},"g","support");
    MapCoordinateParent ast{q->identity().publication,q->identity().cal_parent,q->identity().rtc_grid,q->identity().detectors,q->identity().slots};
    const auto use=ValMapAdmission::bind(q,g,ast);
    EXPECT_TRUE(use.at(0,0,cal_available).pass());
    auto rejected=use.at(0,0,cal_direct_replacement_or_exclusion);EXPECT_TRUE(rejected.coefficient_qc.pass());EXPECT_FALSE(rejected.pass());
    rejected.eligible=true;EXPECT_FALSE(rejected.pass()); // public axes cannot forge evaluator seal
    EXPECT_FALSE(use.at(1,1,cal_available).pass());EXPECT_FALSE(use.at(0,0,cal_available,false).realized);
    ast.slots={3,6};EXPECT_THROW(ValMapAdmission::bind(q,g,ast),std::invalid_argument);
    ast.slots=q->identity().slots;ast.rtc_grid="foreign";EXPECT_THROW(ValMapAdmission::bind(q,g,ast),std::invalid_argument);
}
TEST(ForwardMap, AstCenteredTanGridPreservesOrientationAndMargin) {
    const auto w=AstRtcTanGrid::centered(1.,0.5,2.,4./3600,2./3600);
    EXPECT_EQ(w.columns,7);EXPECT_EQ(w.rows,5);EXPECT_EQ(w.crpix1,4);EXPECT_EQ(w.crpix2,3);
    auto [x,y]=w.continuous_pixel(2./3600,2./3600);EXPECT_DOUBLE_EQ(x,3);EXPECT_DOUBLE_EQ(y,4);
    EXPECT_THROW(AstRtcTanGrid::centered(1,0.5,0,0,0),std::invalid_argument);
}

TEST(ForwardMap, SourceOffsetAltAzUsesCorrectedElevationAndTelescopeCorrections) {
    NativeTelescopeData d;
    auto v=[](double a,double b){Eigen::VectorXd x(2);x<<a,b;return x;};
    d["TelTime"]=v(10,12);d["TelAzAct"]=v(2*pi+0.42,2*pi+0.44);
    d["SourceAz"]=v(.4,.42);d["TelElAct"]=v(.62,.66);d["SourceEl"]=v(.6,.64);
    d["TelAzCor"]=v(.001,.003);d["TelElCor"]=v(.01,.01);
    const RawTelescopeTrajectory tr(d);
    NativePointingOffsetModel offsets({{"az",v(2,4)},{"alt",v(-3,-1)}},v(10,12));
    const auto times=v(10,11);
    auto s=evaluate_ast_rtc_schedule(tr,offsets,0,0,times,AstRtcMapFrame::delta_source_altaz);
    const auto xy=s.detector_tangent_deg(5,7);
    for(int i=0;i<2;++i){const double el=.61+.02*i;
        const double az=.02*std::cos(el)-(.001+.001*i)+(std::cos(el)*5-std::sin(el)*7+2+i)*pi/180/3600;
        const double alt=.01+(std::cos(el)*7+std::sin(el)*5-3+i)*pi/180/3600;
        EXPECT_NEAR(xy(i,0),az*180/pi,1e-12);EXPECT_NEAR(xy(i,1),alt*180/pi,1e-12);
        EXPECT_NEAR(s.elevation_deg[i],el*180/pi,1e-13);
    }
    EXPECT_TRUE(s.detector_tangent_deg(NAN,0).array().isNaN().all());
    EXPECT_EQ(tr.telescope_data().at("TelAzAct")[0],2*pi+.42);
    auto bad=d;bad.erase("SourceAz");
    EXPECT_THROW(evaluate_ast_rtc_schedule(RawTelescopeTrajectory(bad),offsets,0,0,times,AstRtcMapFrame::delta_source_altaz),std::invalid_argument);
    EXPECT_THROW(evaluate_ast_rtc_schedule(tr,offsets,0,0,v(9,11),AstRtcMapFrame::delta_source_altaz),std::invalid_argument);
    const auto grid=AstRtcTanGrid::source_offset(1,2./3600,3./3600);
    EXPECT_EQ(grid.crval1_deg,0);EXPECT_EQ(grid.crval2_deg,0);EXPECT_EQ(grid.cdelt1_deg,-1./3600);
}
TEST(ForwardMap, SourceOffsetZeroAndDetectorRotationSign) {
    auto v=[](double a){return Eigen::VectorXd::Constant(2,a).eval();};
    NativeTelescopeData d{{"TelTime",v(1)},{"TelAzAct",v(.7)},{"SourceAz",v(.7)},
      {"TelElAct",v(pi/2)},{"SourceEl",v(pi/2)},{"TelAzCor",v(0)},{"TelElCor",v(0)}};
    d["TelTime"][1]=2;
    NativePointingOffsetModel off({{"az",v(0)},{"alt",v(0)}},d["TelTime"]);
    const auto s=evaluate_ast_rtc_schedule(RawTelescopeTrajectory(d),off,0,0,d["TelTime"],AstRtcMapFrame::delta_source_altaz);
    EXPECT_TRUE(s.detector_tangent_deg(0,0).isZero());
    const auto xy=s.detector_tangent_deg(4,3);
    EXPECT_NEAR(xy(0,0)*3600,-3,1e-14);EXPECT_NEAR(xy(0,1)*3600,4,1e-14);
    EXPECT_TRUE(s.outside_source_radius(0,0,35).isZero());
    EXPECT_TRUE((s.outside_source_radius(36,0,35)==1).all());
    EXPECT_TRUE((s.outside_source_radius(43,0,42.5)==1).all());
    EXPECT_TRUE((s.outside_source_radius(51,0,50)==1).all());
    EXPECT_TRUE(s.outside_source_radius(NAN,0,35).isZero());
    EXPECT_THROW(s.outside_source_radius(0,0,0),std::invalid_argument);
    EXPECT_THROW(s.outside_source_radius(0,0,NAN),std::invalid_argument);
    auto wrong_frame=s;wrong_frame.frame=AstRtcMapFrame::fk5_tan;
    EXPECT_THROW(wrong_frame.outside_source_radius(0,0,35),std::invalid_argument);
}
