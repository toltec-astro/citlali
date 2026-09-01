#include <citlali/core/pipeline/timestream_identity_route_context.h>

#include "timestream_successor_identity_test_support.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;
namespace support = citlali::test::timestream_successor;

constexpr double arcsec_to_radians =
    std::numbers::pi_v<double> / (180.0 * 3600.0);

Eigen::VectorXd vector(const std::vector<double> &values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    for (std::size_t index = 0; index < values.size(); ++index) {
        result(static_cast<Eigen::Index>(index)) = values[index];
    }
    return result;
}

pipeline::AstScanMotionSourceMetadata ast_metadata() {
    return {pipeline::AstScanMotionProducerKind::real_toltec,
            "Science",
            "Lissajous",
            1,
            2000.0,
            0,
            50.0,
            pipeline::AstScanMotionFieldRegistry::
                source_ra_act_source_dec_act_j2000_radians,
            "tel_toltec:identity-route-context"};
}

std::shared_ptr<const pipeline::AstScanMotionProduct> ast_product(
    const pipeline::NativeObservationScope &scope,
    double start_time = 999.80) {
    constexpr std::size_t count = 31;
    std::vector<double> times(count);
    std::vector<double> ra(count);
    std::vector<double> dec(count, 0.0);
    for (std::size_t index = 0; index < count; ++index) {
        const double elapsed = 0.020 * static_cast<double>(index);
        times[index] = start_time + elapsed;
        ra[index] = 10.0 * elapsed * arcsec_to_radians;
    }
    auto source = pipeline::AstScanMotionSource::admit(
        scope, scope, 100, ast_metadata(), vector(times), vector(ra),
        vector(dec));
    return pipeline::build_ast_scan_motion_product(
        std::move(source), {11, 12, 13, 14});
}

std::shared_ptr<const pipeline::AstScanMotionNetworkViews> ast_views(
    const std::shared_ptr<const pipeline::NativePairedReadoutObservation>
        &native,
    double source_start_time = 999.80) {
    std::vector<std::shared_ptr<const pipeline::NativeNetworkAlignment>>
        timings;
    for (const auto network_id : native->participant_network_ids()) {
        timings.push_back(native->network(network_id)
                              .occurrence_axis_handle()
                              ->native_timing_handle());
    }
    return pipeline::AstScanMotionNetworkViews::admit(
        native->scope(), ast_product(native->scope(), source_start_time),
        std::move(timings));
}

struct RouteFixture {
    std::shared_ptr<const pipeline::NativePairedReadoutObservation> native;
    std::shared_ptr<const pipeline::AstScanMotionNetworkViews> ast;
    std::shared_ptr<const pipeline::IdentityRouteAlignContext> align;
};

RouteFixture route_fixture(double paired_time_offset = 0.0,
                           double ast_source_start_time = 999.80) {
    auto nw0_x = support::valid_states(4);
    auto nw0_r = support::valid_states(4);
    nw0_r[2] = pipeline::NativeReadoutCoordinateState::measured(
        true, false, true, true);
    auto nw7_x = support::valid_states(3);
    auto nw7_r = support::valid_states(3);
    nw7_x[1] = pipeline::NativeReadoutCoordinateState::measured(
        true, false, true, true);

    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(support::make_network(
        0, 10,
        {1000.0000 + paired_time_offset,
         1000.0100 + paired_time_offset,
         1000.0200 + paired_time_offset,
         1000.0300 + paired_time_offset},
        {100, 101, 102, 103}, 1, 1.0, 101.0,
        std::move(nw0_x), std::move(nw0_r)));
    networks.push_back(support::make_network(
        7, 70,
        {1000.0025 + paired_time_offset,
         1000.0125 + paired_time_offset,
         1000.0325 + paired_time_offset},
        {700, 701, 703}, 1, 11.0, 111.0,
        std::move(nw7_x), std::move(nw7_r)));
    auto native =
        support::make_observation(std::move(networks), {0, 7});
    auto ast = ast_views(native, ast_source_start_time);
    auto align = pipeline::IdentityRouteAlignContext::admit(native, ast);
    return {std::move(native), std::move(ast), std::move(align)};
}

pipeline::RtcOnlyRouteRequest rtc_request(
    const RouteFixture &fixture,
    std::uint64_t run = 13) {
    const auto logical_spans =
        pipeline::full_native_occurrence_spans(*fixture.native);
    const auto cardinality = fixture.native->cardinality();
    return {{run},
            fixture.native,
            logical_spans,
            {{{0, 10, 12}, {7, 70, 71}},
             {{0, 12, 14}, {7, 71, 73}}},
            {11,
             {run},
             fixture.native,
             cardinality.native_occurrence_count,
             cardinality.detector_occurrence_count,
             true}};
}

std::shared_ptr<const pipeline::NativePairedReadoutObservation>
non_midpoint_native() {
    auto timing =
        std::make_shared<const pipeline::NativeNetworkAlignment>(
            0, 10, support::time_vector({1000.0, 1000.01}),
            std::vector<pipeline::TimestreamPacketCounter>{100, 101});
    std::vector<pipeline::NativePairedReadoutOccurrenceBinding>
        occurrences{{100, 100, {1000.0, 1000.008}},
                    {101, 101, {1000.01, 1000.018}}};
    auto axis = std::make_shared<const
                                 pipeline::NativePairedReadoutOccurrenceAxis>(
        timing, 10, std::move(occurrences));
    const auto cells = 2U;
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(pipeline::NativePairedReadoutNetwork::admit(
        std::move(axis), support::detector_axis(0, 1),
        support::mapping_authority(0, "non-midpoint"),
        support::matrix(2, 1, 1.0), support::matrix(2, 1, 101.0),
        support::valid_states(cells), support::valid_states(cells)));
    return support::make_observation(std::move(networks), {0});
}

TEST(identity_route_context,
     assembles_exact_ast_rtc_context_and_truthful_map_facing_unavailability) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;
    const auto outcome = pipeline::run_identity_route_context(
        {fixture.align, rtc_request(fixture)}, publication);

    ASSERT_TRUE(outcome.map_facing_context_complete());
    ASSERT_NE(outcome.map_facing_bundle, nullptr);
    EXPECT_EQ(outcome.state,
              pipeline::IdentityRouteContextState::
                  map_facing_context_complete);
    EXPECT_EQ(outcome.failure_cause,
              pipeline::IdentityRouteContextFailureCause::none);
    EXPECT_TRUE(outcome.failure_detail.empty());

    const auto &bundle = *outcome.map_facing_bundle;
    const auto &output = *bundle.rtc_context_handle();
    const auto &input = *output.input_context_handle();
    EXPECT_EQ(input.align_context_handle(), fixture.align);
    EXPECT_EQ(input.ast_views_handle(), fixture.ast);
    EXPECT_EQ(input.signal_handle()->parent_handle(), fixture.native);
    EXPECT_EQ(input.ast_dependency(),
              pipeline::IdentityRtcAstDependency::not_applicable);
    EXPECT_EQ(output.rtc_terminal_handle(), publication.snapshot());
    EXPECT_EQ(output.signal_handle()->native_parent_handle(),
              fixture.native);
    EXPECT_EQ(output.signal_handle()->input_handle(),
              input.signal_handle());
    EXPECT_EQ(output.ast_views_handle(), fixture.ast);
    EXPECT_EQ(output.sampling_relation(),
              pipeline::IdentityRtcSamplingRelation::
                  native_factor_one_phase_zero);
    EXPECT_EQ(output.signal_unit_state(),
              pipeline::IdentityRouteSignalUnitState::raw_paired_xr);

    const auto assignment = output.occurrence_assignment(0, 10);
    EXPECT_EQ(assignment.network_occurrence,
              output.signal_handle()->representative_native_identity(0, 10));
    EXPECT_EQ(assignment.parent_readout_occurrence_key, 10010);
    EXPECT_EQ(assignment.paired_xr_occurrence_key, 20010);
    EXPECT_EQ(assignment.integration_support,
              output.signal_handle()->integration_support(0, 10));
    EXPECT_DOUBLE_EQ(assignment.assigned_time_unix_sec, 1000.0);
    EXPECT_EQ(fixture.align->occurrence_time_policy(),
              pipeline::IdentityRouteOccurrenceTimePolicy::
                  integration_interval_midpoint);
    EXPECT_TRUE(output.ast_motion_record(0, 10).available());
    ASSERT_TRUE(output.ast_motion_support(0, 10));
    EXPECT_EQ(output.ast_motion_support(0, 10)->network_occurrence,
              assignment.network_occurrence);

    EXPECT_EQ(std::bit_cast<std::uint64_t>(
                  output.signal_handle()->value(
                      pipeline::NativeReadoutCoordinate::x, 0, 10, 0)),
              std::bit_cast<std::uint64_t>(1.0));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(
                  output.signal_handle()->value(
                      pipeline::NativeReadoutCoordinate::r, 0, 10, 0)),
              std::bit_cast<std::uint64_t>(101.0));
    EXPECT_EQ(output.signal_handle()->memory_evidence().owned_numeric_bytes,
              0U);
    EXPECT_EQ(fixture.align->memory_evidence().logical_owned_bytes(), 0U);
    EXPECT_EQ(input.memory_evidence().owned_numeric_bytes, 0U);
    EXPECT_EQ(output.memory_evidence().logical_owned_bytes(), 0U);

    const auto &cal = bundle.calibration_state();
    EXPECT_EQ(cal.rtc_context_handle(), bundle.rtc_context_handle());
    EXPECT_EQ(cal.product_state(),
              pipeline::IdentityCalibrationProductState::
                  unavailable_component_not_admitted);
    EXPECT_EQ(cal.unit_state(),
              pipeline::IdentityCalibrationUnitState::
                  unavailable_no_calibration_product);
    EXPECT_EQ(cal.response_state(),
              pipeline::IdentityCalibrationResponseState::
                  unavailable_no_calibration_product);
    EXPECT_EQ(cal.uncertainty_state(),
              pipeline::IdentityCalibrationUncertaintyState::
                  unavailable_no_calibration_product);
    const auto &cal_val =
        bundle.calibration_for_ptc_val_disposition();
    EXPECT_EQ(cal_val.rtc_context_handle(), bundle.rtc_context_handle());
    EXPECT_EQ(cal_val.state(),
              pipeline::IdentityCalibrationForPtcAdmissionState::
                  not_evaluated_product_unavailable);

    const auto &ptc = bundle.ptc_state();
    EXPECT_EQ(ptc.rtc_context_handle(), bundle.rtc_context_handle());
    EXPECT_EQ(ptc.product_state(),
              pipeline::IdentityPtcProductState::
                  unavailable_component_not_admitted);
    EXPECT_EQ(ptc.conditioning_state(),
              pipeline::IdentityPtcConditioningState::
                  unavailable_no_ptc_product);
    EXPECT_EQ(ptc.response_state(),
              pipeline::IdentityPtcResponseState::
                  unavailable_no_ptc_product);
    EXPECT_EQ(ptc.uncertainty_state(),
              pipeline::IdentityPtcUncertaintyState::
                  unavailable_no_ptc_product);
    const auto &ptc_val = bundle.ptc_for_map_val_disposition();
    EXPECT_EQ(ptc_val.rtc_context_handle(), bundle.rtc_context_handle());
    EXPECT_EQ(ptc_val.state(),
              pipeline::IdentityPtcForMapAdmissionState::
                  not_evaluated_product_unavailable);
    EXPECT_EQ(bundle.map_admission_state(),
              pipeline::IdentityMapAdmissionState::
                  unavailable_calibration_and_ptc_products);
    EXPECT_FALSE(bundle.map_action_performed());
}

TEST(identity_route_context,
     ast_motion_unavailability_does_not_remove_ast_or_block_identity_rtc) {
    const auto fixture = route_fixture(0.0, 2000.0);
    EXPECT_FALSE(fixture.ast->network(0).record(10).available());

    pipeline::RtcOnlyProductSlot publication;
    const auto outcome = pipeline::run_identity_route_context(
        {fixture.align, rtc_request(fixture)}, publication);

    ASSERT_TRUE(outcome.map_facing_context_complete());
    const auto &output =
        *outcome.map_facing_bundle->rtc_context_handle();
    EXPECT_EQ(output.input_context_handle()->ast_dependency(),
              pipeline::IdentityRtcAstDependency::not_applicable);
    EXPECT_EQ(output.ast_views_handle(), fixture.ast);
    EXPECT_FALSE(output.ast_motion_record(0, 10).available());
    EXPECT_FALSE(output.ast_motion_support(0, 10));
    EXPECT_NE(publication.snapshot(), nullptr);
}

TEST(identity_route_context,
     align_admission_rejects_missing_participants_or_copied_timing_handles) {
    const auto fixture = route_fixture();
    EXPECT_THROW(pipeline::IdentityRouteAlignContext::admit(
                     fixture.native, nullptr),
                 std::invalid_argument);

    const auto nw0_timing = fixture.native->network(0)
                                .occurrence_axis_handle()
                                ->native_timing_handle();
    const auto nw0_only = pipeline::AstScanMotionNetworkViews::admit(
        fixture.native->scope(), ast_product(fixture.native->scope()),
        {nw0_timing});
    EXPECT_THROW(pipeline::IdentityRouteAlignContext::admit(
                     fixture.native, nw0_only),
                 std::invalid_argument);

    std::vector<std::shared_ptr<const pipeline::NativeNetworkAlignment>>
        copied_timings;
    for (const auto network_id : fixture.native->participant_network_ids()) {
        const auto original = fixture.native->network(network_id)
                                  .occurrence_axis_handle()
                                  ->native_timing_handle();
        copied_timings.push_back(
            std::make_shared<const pipeline::NativeNetworkAlignment>(
                original->network_id(), original->first_native_row(),
                original->reconstructed_times_unix_sec(),
                original->packet_counters()));
    }
    const auto copied_views = pipeline::AstScanMotionNetworkViews::admit(
        fixture.native->scope(), ast_product(fixture.native->scope()),
        std::move(copied_timings));
    EXPECT_THROW(pipeline::IdentityRouteAlignContext::admit(
                     fixture.native, copied_views),
                 std::invalid_argument);
}

TEST(identity_route_context,
     approved_midpoint_policy_rejects_a_native_time_at_interval_start) {
    const auto native = non_midpoint_native();
    const auto views = ast_views(native);
    EXPECT_THROW(pipeline::IdentityRouteAlignContext::admit(native, views),
                 std::invalid_argument);
}

TEST(identity_route_context,
     exact_align_to_rtc_input_binding_fails_before_rtc_publication) {
    const auto fixture = route_fixture();
    const auto foreign = route_fixture(10.0, 1009.80);
    pipeline::RtcOnlyProductSlot publication;
    const auto outcome = pipeline::run_identity_route_context(
        {fixture.align, rtc_request(foreign)}, publication);

    EXPECT_FALSE(outcome.map_facing_context_complete());
    EXPECT_EQ(outcome.state,
              pipeline::IdentityRouteContextState::input_context_failed);
    EXPECT_EQ(outcome.failure_cause,
              pipeline::IdentityRouteContextFailureCause::
                  align_input_binding_mismatch);
    EXPECT_EQ(publication.snapshot(), nullptr);
    EXPECT_EQ(outcome.rtc_terminal.diagnostics.native_admission_entry_count,
              0U);
}

TEST(identity_route_context,
     rtc_failure_remains_typed_and_does_not_manufacture_a_map_bundle) {
    const auto fixture = route_fixture();
    auto request = rtc_request(fixture);
    request.engineering_partitions.pop_back();
    pipeline::RtcOnlyProductSlot publication;
    const auto outcome = pipeline::run_identity_route_context(
        {fixture.align, std::move(request)}, publication);

    EXPECT_FALSE(outcome.map_facing_context_complete());
    EXPECT_EQ(outcome.state,
              pipeline::IdentityRouteContextState::rtc_failed);
    EXPECT_EQ(outcome.failure_cause,
              pipeline::IdentityRouteContextFailureCause::
                  rtc_route_rejected);
    EXPECT_EQ(outcome.rtc_terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::incomplete_logical_support);
    EXPECT_EQ(outcome.map_facing_bundle, nullptr);
    EXPECT_EQ(publication.snapshot(), nullptr);
}

TEST(identity_route_context,
     implementation_has_no_common_grid_engine_or_map_action_dependency) {
    namespace fs = std::filesystem;
    const auto repository = fs::path{__FILE__}.parent_path().parent_path();
    const std::vector<fs::path> paths{
        repository /
            "include/citlali/core/pipeline/timestream_identity_route_context.h",
        repository /
            "src/citlali/core/pipeline/timestream_identity_route_context.cpp"};
    const std::vector<std::string> forbidden{
        "CommonAnalysisGrid", "common_analysis_grid", "Engine",
        "mapmaking/map", "yaml-cpp"};

    for (const auto &path : paths) {
        std::ifstream stream(path);
        ASSERT_TRUE(stream) << path;
        std::ostringstream content;
        content << stream.rdbuf();
        for (const auto &token : forbidden) {
            EXPECT_EQ(content.str().find(token), std::string::npos)
                << path << " contains excluded dependency " << token;
        }
    }
}

}  // namespace
