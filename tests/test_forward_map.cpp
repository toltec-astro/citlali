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
