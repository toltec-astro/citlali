#include <kids/toltec/toltec.h>

#include <citlali/core/pipeline/raw_timestream_provenance.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>
#include <citlali/core/timestream/rtc/rtcproc.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <set>
#include <string>

namespace {

using citlali::pipeline::RawTimestreamExecutionPlan;
using timestream::RTCProc;

RTCProc::RTCPhaseIndependentSnapshot complete_snapshot() {
    RTCProc::RTCPhaseIndependentSnapshot snapshot;
    snapshot.observation_scope = "project:152389";

    RTCProc::RTCStageProvenance stage;
    stage.stage_identity = "rtc-stage:sha256:stage0";
    stage.parent_identity = "rtc-assigned-grid:sha256:parent0";
    stage.process_identity = "rtc-process:sha256:process0";
    stage.observation_scope = snapshot.observation_scope;
    stage.process_label = "standard";
    stage.stage_view = "inner";
    stage.assigned_grid_identity =
        "rtc-realized-assigned-grid:sha256:grid0";
    stage.representative_assigned_time_hex = "0x1p+0";
    stage.assigned_time_values_digest =
        "rtc-assigned-time-values:sha256:times0";
    stage.scan_id = 0;
    stage.input_sample_count = 9;
    stage.output_sample_count = 5;
    stage.detector_count = 2;
    stage.inner_sample_count = 9;
    stage.native_sample_rate_hz = 488.0;
    stage.effective_sample_rate_hz = 244.0;
    stage.downsample_factor = 2;
    stage.complete_response_available = true;
    stage.signal_stage_bits = 18;
    stage.response_stage_bits = 18;
    stage.influenced_sample_count = 2;
    stage.influence_intervals.push_back(
        {0, 1, 2,
         timestream::RTCInfluenceCause::replacement_or_synthesis});
    stage.fir_coefficients_hex = {"0x1p-2", "0x1p-1", "0x1p-2"};
    stage.notch_a_coefficients_hex = {"0x1p+0", "-0x1p+0"};
    stage.notch_b_coefficients_hex = {"0x1p+0", "-0x1p+0"};
    stage.iir_highpass_alpha_hex = "0x1.f4p-1";
    stage.iir_highpass_order = 1;
    stage.notch_zero_phase = true;
    snapshot.stages.push_back(stage);

    auto product_stage = stage;
    product_stage.stage_identity =
        "rtc-output-stage:sha256:output-stage0";
    product_stage.parent_identity = stage.stage_identity;
    product_stage.stage_view = "rtc_inner_full";
    snapshot.stages.push_back(product_stage);

    RTCProc::RTCProductProvenance product;
    product.product_identity = "rtc-product:sha256:product0";
    product.stage_identity = product_stage.stage_identity;
    product.parent_identity = stage.stage_identity;
    product.process_identity = stage.process_identity;
    product.completion_identity = "rtc-completion:sha256:completion0";
    product.assigned_grid_identity = stage.assigned_grid_identity;
    product.product_kind = "rtc_inner_full";
    product.filepath = "toltec0_timestream.nc";
    product.scan_id = 0;
    product.output_row = 0;
    product.complete = true;
    snapshot.products.push_back(product);
    return snapshot;
}

RawTimestreamExecutionPlan initialized_plan() {
    RawTimestreamExecutionPlan plan;
    plan.reset_from_request({});
    plan.begin_observation();
    return plan;
}

TEST(RtcPhaseIndependentProvenance,
     RequestedEffectiveResolvedAndRealizedRoundTripExactly) {
    auto plan = initialized_plan();
    const auto snapshot = complete_snapshot();
    citlali::pipeline::synchronize_raw_rtc_realized_state(
        plan, snapshot, 1, 1);
    citlali::pipeline::complete_raw_timestream_observation(plan, 1, 1);

    const YAML::Node root =
        citlali::pipeline::raw_timestream_provenance_node(plan);
    EXPECT_EQ(root["schema_version"].as<std::string>(),
              "citlali-raw-timestream-provenance-v2");
    EXPECT_EQ(root["requested"]["rtc_contract"]
                  ["assigned_grid_authority"].as<std::string>(),
              "ALIGN-ASSIGNED-TIME-COMPAT-001");
    EXPECT_EQ(root["effective"]["config"]["rtc_contract"]
                  ["physical_event_semantics"].as<std::string>(),
              "unavailable");
    EXPECT_EQ(root["observation"]["value"]["rtc_contract"]
                  ["assigned_time_semantics"].as<std::string>(),
              "compatibility_state_only");
    ASSERT_TRUE(root["realized"]["rtc"]["bundle_complete"].as<bool>());
    EXPECT_FALSE(root["realized"]["rtc"]["bundle_identity"]
                     .as<std::string>().empty());

    const auto stage = root["realized"]["rtc"]["stages"][0];
    EXPECT_EQ(stage["stage_identity"].as<std::string>(),
              snapshot.stages[0].stage_identity);
    EXPECT_EQ(stage["physical_event_semantics"].as<std::string>(),
              "unavailable");
    EXPECT_EQ(stage["source_mask"]["timing_sensitive_accuracy"]
                  .as<std::string>(),
              "unavailable");
    EXPECT_EQ(stage["representative_assigned_time"]["rule"]
                  .as<std::string>(),
              "phase_zero_first_cell_compatibility_value");
    EXPECT_EQ(stage["assigned_time_values_digest"].as<std::string>(),
              snapshot.stages[0].assigned_time_values_digest);
    EXPECT_EQ(stage["operator_coefficients"]["fir_hex"][1]
                  .as<std::string>(),
              "0x1p-1");
    EXPECT_EQ(stage["operator_coefficients"]["notch_section_layout"]
                  .as<std::string>(),
              "ordered_sos_a_then_b_each_section_size_three");
    EXPECT_EQ(stage["influence"]["intervals"][0]["cause_bits"]
                  .as<std::uint32_t>(),
              static_cast<std::uint32_t>(
                  timestream::RTCInfluenceCause::
                      replacement_or_synthesis));

    const auto product = root["realized"]["rtc"]["products"][0];
    EXPECT_EQ(product["stage_identity"].as<std::string>(),
              snapshot.stages[1].stage_identity);
    EXPECT_EQ(product["completion_identity"].as<std::string>(),
              snapshot.products[0].completion_identity);
}

TEST(RtcPhaseIndependentProvenance,
     IncompleteDuplicateAndStaleBundlesFailClosed) {
    {
        auto plan = initialized_plan();
        auto missing = complete_snapshot();
        missing.stages.clear();
        EXPECT_THROW(
            citlali::pipeline::synchronize_raw_rtc_realized_state(
                plan, missing, 1, 1),
            std::logic_error);
    }
    {
        auto plan = initialized_plan();
        auto duplicate = complete_snapshot();
        duplicate.products.push_back(duplicate.products.front());
        EXPECT_THROW(
            citlali::pipeline::synchronize_raw_rtc_realized_state(
                plan, duplicate, 1, 2),
            std::logic_error);
    }
    {
        auto plan = initialized_plan();
        auto stale = complete_snapshot();
        stale.products.front().stage_identity =
            "rtc-stage:sha256:stale";
        EXPECT_THROW(
            citlali::pipeline::synchronize_raw_rtc_realized_state(
                plan, stale, 1, 1),
            std::logic_error);
    }
}

TEST(RtcPhaseIndependentProvenance,
     ProductIdentityAttributesAndYamlReopenPreserveLinks) {
    const std::filesystem::path directory =
        std::filesystem::path{::testing::TempDir()} /
        "citlali_rtc_phase_independent_provenance";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const auto snapshot = complete_snapshot();
    const auto nc_path = directory / "rtc_identity.nc";
    {
        netCDF::NcFile file(nc_path.string(), netCDF::NcFile::replace);
        RTCProc::write_phase_independent_product_identity(
            file, snapshot.products.front());
    }
    {
        netCDF::NcFile file(nc_path.string(), netCDF::NcFile::read);
        std::string product_identity;
        std::string event_semantics;
        std::string stage_identity;
        file.getAtt("rtc_product_identity").getValues(product_identity);
        file.getAtt("rtc_physical_event_semantics")
            .getValues(event_semantics);
        file.getAtt("rtc_stage_identity_row_0")
            .getValues(stage_identity);
        EXPECT_EQ(product_identity,
                  snapshot.products.front().product_identity);
        EXPECT_EQ(event_semantics, "unavailable");
        EXPECT_EQ(stage_identity,
                  snapshot.products.front().stage_identity);
    }

    auto plan = initialized_plan();
    citlali::pipeline::synchronize_raw_rtc_realized_state(
        plan, snapshot, 1, 1);
    citlali::pipeline::complete_raw_timestream_observation(plan, 1, 1);
    citlali::pipeline::write_raw_timestream_provenance_file(
        directory, plan);
    const auto reopened = YAML::LoadFile(
        citlali::pipeline::raw_timestream_provenance_path(directory)
            .string());
    EXPECT_EQ(reopened["realized"]["rtc"]["products"][0]
                  ["product_identity"].as<std::string>(),
              snapshot.products.front().product_identity);
    EXPECT_EQ(reopened["realized"]["rtc"]["stages"][0]
                  ["parent_identity"].as<std::string>(),
              snapshot.stages.front().parent_identity);

    std::filesystem::remove_all(directory);
}

TEST(RtcPhaseIndependentProvenance,
     ProductAndCompletionIdentitiesDoNotCollideAcrossKindsOrRows) {
    RTCProc proc;
    auto snapshot = complete_snapshot();
    proc.rtc_observation_scope = snapshot.observation_scope;
    proc.publish_phase_independent_stage(snapshot.stages.front());

    const auto inner = proc.make_phase_independent_product(
        0, "/tmp/rtc.nc", "rtc_inner_full", 0, false, false);
    const auto diagnostic = proc.make_phase_independent_product(
        0, "/tmp/rtcdiag.nc", "rtc_diagnostic", 0, false, false);
    const auto next_row = proc.make_phase_independent_product(
        0, "/tmp/rtc.nc", "rtc_inner_full", 1, false, false);
    const auto mini = proc.make_phase_independent_product(
        0, "/tmp/rtc-mini.nc", "rtc_inner_mini", 0, true, false);
    EXPECT_NE(inner.product_identity, diagnostic.product_identity);
    EXPECT_NE(inner.stage_identity, diagnostic.stage_identity);
    EXPECT_NE(inner.stage_identity, mini.stage_identity);
    EXPECT_EQ(inner.parent_identity,
              snapshot.stages.front().stage_identity);
    EXPECT_EQ(inner.product_identity, next_row.product_identity);
    EXPECT_NE(inner.completion_identity, next_row.completion_identity);
}

}  // namespace
