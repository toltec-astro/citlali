#include "../src/citlali/cli/native_rtc_run_report.h"
#include <gtest/gtest.h>

namespace {
using namespace citlali::cli;
YAML::Node effective_config() {
    return YAML::Load(R"yaml(
timestream:
  processed_time_chunk:
    clean:
      enabled: true
      grouping: [nw]
      mask_radius_arcsec: 0
      standard_pca:
        enabled: true
        stddev_limit: 0
        n_eig_to_cut: {a1100: [3], a1400: [7], a2000: [10]}
)yaml");
}

TEST(NativePtcSetup, ResolvesOnlyVerifiedArrayAndPreservesRequest) {
    auto yaml=effective_config();
    const auto before=YAML::Dump(yaml);
    NativePtcConfiguration configured(yaml,{});
    yaml["timestream"]["processed_time_chunk"]["clean"]["standard_pca"]["n_eig_to_cut"]["a1400"][0]=99;
    const auto plan=configured.resolve(YAML::Load("{}"),1);
    EXPECT_EQ(plan.configured_rank,7);
    EXPECT_EQ(plan.requested_rank,0);
    EXPECT_EQ(plan.solver.rank,7);
    EXPECT_EQ(plan.solver.iteration_limit,100);
    EXPECT_EQ(plan.solver.relative_rank_tolerance,1e-10);
    EXPECT_EQ(plan.solver.relative_objective_tolerance,1e-5);
    auto original=YAML::Load(before);
    const NativePtcConfiguration overridden(original,{"pairwise-covariance",4});
    const auto explicit_plan=overridden.resolve(YAML::Load("{}"),1);
    EXPECT_EQ(explicit_plan.configured_rank,7);
    EXPECT_EQ(explicit_plan.requested_rank,4);
    EXPECT_EQ(explicit_plan.solver.rank,4);
    EXPECT_EQ(explicit_plan.solver.method,citlali::pipeline::PtcMethod::pairwise_covariance);
    EXPECT_EQ(YAML::Dump(original),before);
}

TEST(NativePtcSetup, RejectsUnsupportedCleaningBeforeArrayResolution) {
    for(const auto *key:{"adaptive_selector","corr_grouping","marchenko_pastur","null_model"}) {
        auto yaml=effective_config();
        yaml["timestream"]["processed_time_chunk"]["clean"][key]["enabled"]=true;
        EXPECT_THROW((NativePtcConfiguration{yaml,{}}),std::invalid_argument);
    }
    for(const auto *key:{"mask_radius_arcsec","tau"}) {
        auto yaml=effective_config();
        yaml["timestream"]["processed_time_chunk"]["clean"][key]=1.;
        EXPECT_THROW((NativePtcConfiguration{yaml,{}}),std::invalid_argument);
    }
    auto yaml=effective_config();
    yaml["timestream"]["processed_time_chunk"]["clean"]["grouping"]=std::vector<std::string>{"array"};
    EXPECT_THROW((NativePtcConfiguration{yaml,{}}),std::invalid_argument);
    yaml=effective_config();
    yaml["timestream"]["processed_time_chunk"]["clean"]["standard_pca"]["stddev_limit"]=1.;
    EXPECT_THROW((NativePtcConfiguration{yaml,{}}),std::invalid_argument);
    EXPECT_THROW((NativePtcConfiguration{effective_config(),{"unknown",0}}),std::invalid_argument);
    EXPECT_THROW((NativePtcConfiguration{effective_config(),{"observed-als",-1}}),std::invalid_argument);
}

TEST(NativePtcSetup, ExplicitOverrideStillRequiresSinglePositiveConfiguredRank) {
    for(const auto &ranks:std::vector<std::vector<int>>{{},{0},{-1},{2,3}}) {
        auto yaml=effective_config();
        yaml["timestream"]["processed_time_chunk"]["clean"]["standard_pca"]["n_eig_to_cut"]["a1400"]=ranks;
        const NativePtcConfiguration configured(yaml,{"observed-als",5});
        EXPECT_THROW(configured.resolve(YAML::Load("{}"),1),std::invalid_argument);
        EXPECT_EQ(configured.resolve(YAML::Load("{}"),0).solver.rank,5);
    }
    const NativePtcConfiguration configured(effective_config(),{});
    EXPECT_THROW(configured.resolve(YAML::Load("{}"),-1),std::invalid_argument);
    EXPECT_THROW(configured.resolve(YAML::Load("{}"),3),std::invalid_argument);
}

TEST(NativePtcSetup, PreservesSpecialProfileScopeAndIterationLimits) {
    auto transfer=YAML::Load("{observation: 152420, network: 1, point_transfer_runtime_profile: point-transfer-runtime-2026-09-29-v1}");
    const NativePtcConfiguration configured(effective_config(),{"observed-als",10});
    EXPECT_EQ(configured.resolve(transfer,0).solver.iteration_limit,500);
    EXPECT_THROW(configured.resolve(transfer,1),std::invalid_argument);
    transfer["network"]=5;
    EXPECT_THROW(configured.resolve(transfer,0),std::invalid_argument);
    auto oof=YAML::Load("{observation: 152385, subobservation: 0, scan: 1, mode: OOF, network: 1, development_reference: oof-cold-m0-v1}");
    const NativePtcConfiguration cold(effective_config(),{});
    EXPECT_EQ(cold.resolve(oof,0).solver.iteration_limit,500);
    EXPECT_THROW(cold.resolve(oof,2),std::invalid_argument);
    oof["development_reference"]="oof-a2000-m0-v1";oof["network"]=11;
    const NativePtcConfiguration longwave(effective_config(),{"observed-als",3});
    EXPECT_EQ(longwave.resolve(oof,2).solver.iteration_limit,500);
    const NativePtcConfiguration masked(effective_config(),{"masked-covariance-count-v1",15});
    auto scope=YAML::Load("{observation: 152430, network: 12}");
    EXPECT_EQ(masked.resolve(scope,2).solver.rank,15);
    scope["network"]=10;
    EXPECT_THROW(masked.resolve(scope,2),std::invalid_argument);
}

TEST(NativePtcSetup, SetupReportSeparatesConfiguredOverrideAndExecution) {
    const auto cfg=YAML::Load("{observation: 152389, network: 7}");
    NativeRtcOutputLayout::NetworkPaths paths;
    paths.root="/already/claimed/network7";
    const auto setup=NativePtcConfiguration(effective_config(),{"observed-als",4}).resolve(cfg,1);
    const auto record=native_setup_record(cfg,DevelopmentTerminal::ptc,paths,&setup);
    EXPECT_EQ(record["PTC"]["configured_rank"].as<int>(),7);
    EXPECT_EQ(record["PTC"]["requested_rank_override"].as<int>(),4);
    EXPECT_EQ(record["PTC"]["effective_rank"].as<int>(),4);
    const auto summary=native_setup_summary(record);
    EXPECT_NE(summary.find("rank_source=explicit-request-override"),std::string::npos);
    EXPECT_NE(summary.find("execution not yet observed"),std::string::npos);
    const auto inherited=NativePtcConfiguration(effective_config(),{}).resolve(cfg,1);
    const auto inherited_record=native_setup_record(cfg,DevelopmentTerminal::ptc,paths,&inherited);
    EXPECT_TRUE(inherited_record["PTC"]["requested_rank_override"].IsNull());
    EXPECT_NE(native_setup_summary(inherited_record).find("rank_source=bound-effective-config"),std::string::npos);
}

TEST(NativePtcSetup, ResultReportShowsPartialAvailabilityAndDoesNotInventDownstreamWork) {
    auto receipt=YAML::Load(R"yaml(
status: UNAVAILABLE-PTC-fit-or-output;upstream-preserved
donor_continuity:
  terminal: {scheduled_slots: 20, x_available: 15, directly_excluded: 3, replacement_influenced: 2}
CAL: {available: 12}
PTC: {rank: 7, available: 8, failed_fits: 1}
)yaml");
    const auto summary=native_result_summary(7,receipt);
    EXPECT_NE(summary.find("status=UNAVAILABLE"),std::string::npos);
    EXPECT_NE(summary.find("x_available=15/20"),std::string::npos);
    EXPECT_NE(summary.find("CAL available=12"),std::string::npos);
    EXPECT_NE(summary.find("PTC effective_rank=7 available=8 failed_fits=1"),std::string::npos);
    EXPECT_NE(summary.find("scientific qualification unresolved"),std::string::npos);
    receipt.remove("CAL");receipt.remove("PTC");
    const auto rtc_only=native_result_summary(7,receipt);
    EXPECT_EQ(rtc_only.find("CAL available="),std::string::npos);
    EXPECT_EQ(rtc_only.find("PTC effective_rank="),std::string::npos);
}
} // namespace
