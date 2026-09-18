#include <citlali/core/pipeline/timestream_cal_atmosphere.h>
#include <citlali/core/pipeline/timestream_cal_wvr.h>
#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <limits>

using namespace citlali::pipeline;
namespace {
auto wvr(std::vector<CalWvrRecord> r,std::string mapping="ALIGN-controlled-Unix") {
    return CalWvrEvidence::learn(NativeObservationScope{1,0,0},"controlled-TEL",std::move(mapping),std::move(r));
}
CalWvrRecord record(std::string id,double time,double tau,bool valid=true,double first=0,double last=100) {
    return {std::move(id),time,tau,valid,first,last};
}
}

TEST(cal_atmosphere, every_frozen_node_and_analytic_zero_in_all_reference_spectra) {
    for(int alpha:{-1,0,2,4}) {
        auto surface=CalAtmosphereSurface::frozen(alpha);ASSERT_EQ(surface->nodes().size(),1368);
        for(const auto &node:surface->nodes())if(node.alpha==alpha && node.elevation_deg>=25) {
            auto value=surface->correction(node.array,node.tau225,node.elevation_deg);ASSERT_TRUE(value);
            EXPECT_NEAR(*value,node.correction,2e-14*node.correction);
        }
        for(int a=0;a<3;++a)EXPECT_DOUBLE_EQ(*surface->correction(a,0,40),1);
    }
}
TEST(cal_atmosphere, independent_scipy_pchip_and_optical_depth_interpolation_oracle) {
    // scipy.interpolate.PchipInterpolator on the exact approved node table,
    // then numpy.interp on lambda, not on extinction correction.
    const double oracle[3][3]={{1.0691546386630877,1.2702366825722653,1.4643631650256363},
        {1.0458980372606719,1.1738896705468875,1.2917370269949187},
        {1.0265060174795118,1.085087291088988,1.131428345701599}};
    auto s=CalAtmosphereSurface::frozen();
    for(int a=0;a<3;++a)for(int i=0;i<3;++i)
        EXPECT_NEAR(*s->correction(a,std::array{.03,.12,.225}[i],std::array{47.,53.,68.}[i]),oracle[a][i],2e-14);
}
TEST(cal_atmosphere, boundaries_fail_closed_without_alpha_or_support_extrapolation) {
    auto s=CalAtmosphereSurface::frozen();EXPECT_THROW(CalAtmosphereSurface::frozen(1),std::invalid_argument);
    EXPECT_TRUE(s->correction(0,.25,25));EXPECT_TRUE(s->correction(2,.25,80));
    for(double tau:{-.001,std::nextafter(.25,1.),double(NAN),double(INFINITY)})EXPECT_FALSE(s->correction(0,tau,40));
    for(double el:{std::nextafter(25.,0.),std::nextafter(80.,100.),double(NAN)})EXPECT_FALSE(s->correction(0,.1,el));
    EXPECT_FALSE(s->correction(3,.1,40));
}
TEST(cal_wvr, exact_records_and_separately_rounded_linear_interpolation_preserve_lineage) {
    auto e=wvr({record("a",0,.1),record("b",10,.2)});
    auto exact=e->at(10);ASSERT_TRUE(exact.tau225);EXPECT_EQ(std::bit_cast<std::uint64_t>(*exact.tau225),std::bit_cast<std::uint64_t>(.2));
    EXPECT_TRUE(exact.exact_match);EXPECT_EQ(exact.first_record,1);EXPECT_EQ(exact.last_record,1);
    auto mid=e->at(2.5);ASSERT_TRUE(mid.tau225);EXPECT_DOUBLE_EQ(*mid.tau225,.125);EXPECT_DOUBLE_EQ(mid.weight,.25);
    EXPECT_EQ(mid.first_record,0);EXPECT_EQ(mid.last_record,1);EXPECT_FALSE(mid.exact_match);
    EXPECT_EQ(e->at(-1).cause,CalWvrCause::unbracketed);EXPECT_EQ(e->at(11).cause,CalWvrCause::unbracketed);
}
TEST(cal_wvr, declared_gaps_invalid_records_and_duplicate_conflicts_are_not_skipped) {
    auto gap=wvr({record("a",0,.1,true,0,4),record("b",10,.2,true,6,10)});
    EXPECT_TRUE(gap->at(0).tau225);EXPECT_TRUE(gap->at(10).tau225);
    EXPECT_EQ(gap->at(5).cause,CalWvrCause::gap_outside_source_validity);
    EXPECT_FALSE(gap->quality(0,10).summary_available);
    auto invalid=wvr({record("a",0,.1),record("b",5,.15,false),record("c",10,.2)});
    EXPECT_EQ(invalid->at(2).cause,CalWvrCause::gap_outside_source_validity);
    EXPECT_EQ(invalid->at(7).cause,CalWvrCause::gap_outside_source_validity);
    auto duplicates=wvr({record("a",0,.1),record("b",0,.1),record("c",10,.2)});
    EXPECT_DOUBLE_EQ(*duplicates->at(5).tau225,.15);EXPECT_EQ(duplicates->records().size(),3);
    auto conflict=wvr({record("a",0,.1),record("b",0,.12),record("c",10,.2)});
    EXPECT_EQ(conflict->at(0).cause,CalWvrCause::conflicting_duplicate);
    EXPECT_EQ(conflict->at(5).cause,CalWvrCause::conflicting_duplicate);EXPECT_TRUE(conflict->at(10).tau225);
}
TEST(cal_wvr, absent_nonfinite_negative_mapping_and_long_valid_gap_have_distinct_dispositions) {
    EXPECT_EQ(wvr({})->at(1).cause,CalWvrCause::absent);
    EXPECT_EQ(wvr({record("a",0,NAN)})->at(0).cause,CalWvrCause::nonfinite);
    EXPECT_EQ(wvr({record("a",0,-.1)})->at(0).cause,CalWvrCause::negative);
    EXPECT_EQ(wvr({record("a",0,.1)},"")->at(0).cause,CalWvrCause::time_mapping_unavailable);
    EXPECT_TRUE(wvr({record("a",0,.1,true,0,1e6),record("b",1e6,.2,true,0,1e6)})->at(5e5).tau225);
}
TEST(cal_wvr, chronological_quality_mean_analytic_crossings_and_excursion_accounting) {
    auto e=wvr({record("a",0,.1),record("b",10,.2),record("c",20,.1)});
    auto q=e->quality(0,20);ASSERT_TRUE(q.summary_available);EXPECT_NEAR(q.mean,.15,1e-16);
    EXPECT_DOUBLE_EQ(q.minimum,.1);EXPECT_DOUBLE_EQ(q.maximum,.2);ASSERT_EQ(q.excursions.size(),1);
    EXPECT_NEAR(q.excursions[0].first,5,1e-14);EXPECT_NEAR(q.excursions[0].last,15,1e-14);
    EXPECT_NEAR(q.excursion_duration,10,1e-14);EXPECT_NEAR(q.integrated_excess,.25,1e-14);
    EXPECT_EQ(q.classification,CalOpacityQuality::engineering_only);
    auto split=wvr({record("a",0,.2),record("b",10,.15),record("c",20,.2)})->quality(0,20);
    ASSERT_TRUE(split.summary_available);EXPECT_EQ(split.excursions.size(),2);
}
TEST(cal_wvr, opacity_classes_have_inclusive_boundaries_and_do_not_grant_sample_support) {
    auto constant=[](double tau){return wvr({record("a",0,tau),record("b",10,tau)})->quality(0,10);};
    EXPECT_EQ(constant(.15).classification,CalOpacityQuality::science_qualification_eligible);
    EXPECT_EQ(constant(std::nextafter(.15,1.)).classification,CalOpacityQuality::engineering_only);
    EXPECT_EQ(constant(.25).classification,CalOpacityQuality::engineering_only);
    EXPECT_EQ(constant(std::nextafter(.25,1.)).classification,CalOpacityQuality::outside_supported_opacity);
    EXPECT_EQ(constant(-.1).classification,CalOpacityQuality::invalid_opacity_input);
    EXPECT_EQ(constant(NAN).classification,CalOpacityQuality::invalid_opacity_input);
    auto peak=wvr({record("a",0,.1),record("b",5,.175),record("c",10,.1)});
    EXPECT_EQ(peak->quality(0,10).classification,CalOpacityQuality::science_qualification_eligible);
    EXPECT_EQ(peak->at(11).cause,CalWvrCause::unbracketed);
    auto invalid=wvr({record("bad",0,-.1),record("good",10,.1)});
    EXPECT_EQ(invalid->quality(0,11).classification,CalOpacityQuality::invalid_opacity_input);
    EXPECT_TRUE(invalid->at(10).tau225);
}
