#include <citlali/core/pipeline/rtc_only_route.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;

Eigen::VectorXd vector(std::initializer_list<double> values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index index = 0;
    for (const auto value : values) result(index++) = value;
    return result;
}

struct RtcOnlyFixture {
    std::shared_ptr<const pipeline::PairedReadout> native;
    std::shared_ptr<const pipeline::NativeNetworkAlignment> timing;
};

RtcOnlyFixture route_fixture() {
    constexpr pipeline::TimestreamNetworkId network_id = 0;
    constexpr pipeline::TimestreamNativeRow first_native_row = 10;
    const pipeline::NativeObservationScope scope{152390, 0, 4};
    auto timing =
        std::make_shared<const pipeline::NativeNetworkAlignment>(
            network_id, first_native_row,
            vector({100.0, 101.0, 103.0}),
            std::vector<pipeline::TimestreamPacketCounter>{20, 21, 23});
    auto axis =
        std::make_shared<const pipeline::PairedReadoutOccurrenceAxis>(
            timing, first_native_row,
            std::vector<pipeline::NativeOccurrenceInterval>{
                {99.8, 100.2}, {100.8, 101.2}, {102.8, 103.2}});
    auto mapping =
        std::make_shared<const pipeline::NativeReadoutMappingIdentity>(
            pipeline::NativeReadoutMappingIdentity{
                "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1",
                "producer:route", "tune:route", "mapping:route",
                "iq-to-xr:route", "raw-x:route", "raw-r:route"});
    std::vector<pipeline::PairedReadoutDetectorIdentity> detectors{
        {500, 0, network_id, 1000, 0}};
    pipeline::PairedReadoutMatrix x(3, 1);
    pipeline::PairedReadoutMatrix r(3, 1);
    x << 1.25, -2.5, 3.75;
    r << 101.25, -102.5, 103.75;
    const auto valid = pipeline::ReadoutMemberState::measured(
        true, true, true, true);
    std::vector<pipeline::ReadoutMemberState> x_states(3, valid);
    std::vector<pipeline::ReadoutMemberState> r_states(3, valid);
    r_states[1] = pipeline::ReadoutMemberState::measured(
        true, false, true, true);
    std::vector<pipeline::PairedReadoutNetwork> networks;
    networks.push_back(pipeline::PairedReadoutNetwork::admit(
        std::move(axis), std::move(detectors), std::move(mapping),
        std::move(x), std::move(r), std::move(x_states),
        std::move(r_states)));
    auto native = pipeline::PairedReadout::admit(
        scope, {network_id}, std::move(networks));

    return {std::move(native), std::move(timing)};
}

pipeline::RtcOnlyRouteRequest route_request(
    const RtcOnlyFixture &fixture, std::uint64_t run = 1) {
    const auto logical_spans =
        pipeline::full_native_occurrence_spans(*fixture.native);
    return {{run}, fixture.native, logical_spans, {logical_spans}};
}

TEST(rtc_only_route,
     executes_explicit_lca_and_atomically_publishes_inspectable_memory_product) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;
    const auto outcome = pipeline::run_identity_rtc_only(
        route_request(fixture, 42), publication);

    ASSERT_TRUE(outcome.complete());
    ASSERT_NE(outcome.published_product, nullptr);
    EXPECT_EQ(outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::none);
    EXPECT_TRUE(outcome.terminal.failure_detail.empty());
    EXPECT_EQ(publication.snapshot(), outcome.published_product);
    const auto &bundle = *outcome.published_product;
    EXPECT_EQ(bundle.terminal_result().identity.run, 42U);
    EXPECT_EQ(bundle.evidence_handle()->identity().attempt, 42U);
    EXPECT_EQ(bundle.plan_handle()->identity().resolution, 42U);
    EXPECT_EQ(bundle.realization().completion,
              pipeline::RtcCompletionState::complete);
    EXPECT_EQ(bundle.timestream_handle()->output_native_occurrence_count(), 3U);
    EXPECT_EQ(bundle.timestream_handle()->output_cell_count(), 3U);

    const auto &diagnostics = outcome.terminal.diagnostics;
    EXPECT_EQ(diagnostics.network_count, 1U);
    EXPECT_EQ(diagnostics.engineering_partition_count, 1U);
    EXPECT_EQ(diagnostics.detector_count, 1U);
    EXPECT_EQ(diagnostics.native_occurrence_count, 3U);
    EXPECT_EQ(diagnostics.detector_occurrence_count, 3U);
    EXPECT_EQ(diagnostics.evidence_event_count, 1U);
    EXPECT_EQ(diagnostics.direct_x_event_count, 0U);
    EXPECT_EQ(diagnostics.direct_r_event_count, 1U);
    EXPECT_EQ(diagnostics.x_and_r_event_count, 0U);
    EXPECT_EQ(diagnostics.pair_ineligible_cell_count, 1U);
    EXPECT_EQ(diagnostics.x_numerically_valid_cell_count, 3U);
    EXPECT_EQ(diagnostics.r_numerically_valid_cell_count, 2U);
    EXPECT_GT(diagnostics.derived_evidence_bytes, 0U);
    EXPECT_EQ(diagnostics.derived_plan_bytes, 0U);
    EXPECT_EQ(diagnostics.rtc_owned_numeric_bytes, 0U);
    EXPECT_EQ(diagnostics.native_admission_entry_count, 1U);
    EXPECT_EQ(diagnostics.learn_entry_count, 1U);
    EXPECT_EQ(diagnostics.consider_entry_count, 1U);
    EXPECT_EQ(diagnostics.apply_entry_count, 1U);
    EXPECT_EQ(diagnostics.publication_entry_count, 1U);

    for (const auto member :
         {pipeline::ReadoutMember::x, pipeline::ReadoutMember::r}) {
        const auto published = bundle.timestream_handle()->value(
            member, 0, 12, 0);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(published),
                  std::bit_cast<std::uint64_t>(
                      fixture.native->network(0).value(member, 12, 0)));
    }
    EXPECT_EQ(bundle.timestream_handle()->pair_decision(0, 11, 0),
              pipeline::RtcPairDecision::ineligible);
    EXPECT_TRUE(bundle.timestream_handle()
                    ->member_numerically_valid(
                        pipeline::ReadoutMember::x, 0, 11, 0));
}

TEST(rtc_only_route,
     input_failure_is_truthful_and_leaves_an_empty_publication_slot) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;
    auto request = route_request(fixture);
    request.logical_spans.front().past_last_native_row = 14;
    const auto outcome =
        pipeline::run_identity_rtc_only(request, publication);

    EXPECT_FALSE(outcome.complete());
    EXPECT_EQ(outcome.terminal.state,
              pipeline::RtcOnlyTerminalState::input_admission_failed);
    EXPECT_EQ(outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::input_contract_rejected);
    EXPECT_EQ(outcome.terminal.failure_detail,
              "native RTC view span is incomplete or outside parent support");
    EXPECT_EQ(outcome.published_product, nullptr);
    EXPECT_EQ(publication.snapshot(), nullptr);
}

TEST(rtc_only_route,
     multi_chunk_execution_finalizes_one_complete_logical_product) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot single_publication;
    pipeline::RtcOnlyProductSlot partitioned_publication;
    const auto single = pipeline::run_identity_rtc_only(
        route_request(fixture, 1), single_publication);
    auto partitioned_request = route_request(fixture, 2);
    partitioned_request.engineering_partitions = {
        {{0, 10, 11}}, {{0, 11, 13}}};
    const auto partitioned = pipeline::run_identity_rtc_only(
        partitioned_request, partitioned_publication);

    ASSERT_TRUE(single.complete());
    ASSERT_TRUE(partitioned.complete());
    EXPECT_EQ(partitioned_publication.snapshot(),
              partitioned.published_product);
    EXPECT_EQ(partitioned.terminal.diagnostics.engineering_partition_count,
              2U);
    EXPECT_EQ(partitioned.terminal.diagnostics.native_admission_entry_count,
              1U);
    EXPECT_EQ(partitioned.terminal.diagnostics.learn_entry_count, 1U);
    EXPECT_EQ(partitioned.terminal.diagnostics.consider_entry_count, 1U);
    EXPECT_EQ(partitioned.terminal.diagnostics.apply_entry_count, 1U);
    EXPECT_EQ(partitioned.terminal.diagnostics.publication_entry_count, 1U);
    EXPECT_EQ(partitioned.published_product->timestream_handle()
                  ->output_native_occurrence_count(),
              3U);

    const auto &single_product =
        *single.published_product->timestream_handle();
    const auto &partitioned_product =
        *partitioned.published_product->timestream_handle();
    for (pipeline::TimestreamNativeRow row = 10; row < 13; ++row) {
        EXPECT_EQ(partitioned_product.representative_native_identity(
                      0, row),
                  single_product.representative_native_identity(0, row));
        EXPECT_EQ(partitioned_product.representative_interval(0, row),
                  single_product.representative_interval(0, row));
        EXPECT_EQ(partitioned_product.identity(0, row, 0),
                  single_product.identity(0, row, 0));
        EXPECT_EQ(partitioned_product.pair_decision(0, row, 0),
                  single_product.pair_decision(0, row, 0));
        for (const auto member :
             {pipeline::ReadoutMember::x, pipeline::ReadoutMember::r}) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(
                          partitioned_product.value(member, 0, row, 0)),
                      std::bit_cast<std::uint64_t>(
                          single_product.value(member, 0, row, 0)));
            EXPECT_EQ(partitioned_product.member_local_causes(
                          member, 0, row, 0),
                      single_product.member_local_causes(
                          member, 0, row, 0));
        }
        const auto *partitioned_cause =
            partitioned_product.pair_causal_evidence(0, row, 0);
        const auto *single_cause =
            single_product.pair_causal_evidence(0, row, 0);
        ASSERT_EQ(partitioned_cause == nullptr, single_cause == nullptr);
        if (partitioned_cause) {
            EXPECT_EQ(partitioned_cause->origin, single_cause->origin);
        }
    }
}

TEST(rtc_only_route,
     missing_engineering_chunk_fails_without_partial_publication) {
    const auto fixture = route_fixture();
    auto request = route_request(fixture);
    request.engineering_partitions = {{{0, 10, 12}}};
    pipeline::RtcOnlyProductSlot publication;

    const auto outcome =
        pipeline::run_identity_rtc_only(request, publication);

    EXPECT_FALSE(outcome.complete());
    EXPECT_EQ(outcome.terminal.state,
              pipeline::RtcOnlyTerminalState::input_admission_failed);
    EXPECT_EQ(outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::incomplete_logical_support);
    EXPECT_EQ(outcome.terminal.failure_detail,
              "engineering partitions do not exactly cover declared logical support");
    EXPECT_EQ(outcome.published_product, nullptr);
    EXPECT_EQ(publication.snapshot(), nullptr);
}

TEST(rtc_only_route,
     failed_second_publication_does_not_replace_prior_complete_product) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;
    const auto first = pipeline::run_identity_rtc_only(
        route_request(fixture, 1), publication);
    ASSERT_TRUE(first.complete());
    const auto committed = publication.snapshot();

    const auto second = pipeline::run_identity_rtc_only(
        route_request(fixture, 2), publication);
    EXPECT_FALSE(second.complete());
    EXPECT_EQ(second.terminal.state,
              pipeline::RtcOnlyTerminalState::publication_failed);
    EXPECT_EQ(second.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::publication_slot_occupied);
    EXPECT_EQ(second.terminal.failure_detail,
              "RTC-only product slot already contains a completion");
    EXPECT_EQ(second.published_product, nullptr);
    EXPECT_EQ(publication.snapshot(), committed);
    EXPECT_EQ(publication.snapshot()->terminal_result().identity.run, 1U);
}

TEST(rtc_only_route, invalid_run_identity_has_an_exact_terminal_cause) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;

    const auto outcome = pipeline::run_identity_rtc_only(
        route_request(fixture, 0), publication);

    EXPECT_FALSE(outcome.complete());
    EXPECT_EQ(outcome.terminal.state,
              pipeline::RtcOnlyTerminalState::input_admission_failed);
    EXPECT_EQ(outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::invalid_run_identity);
    EXPECT_EQ(outcome.terminal.failure_detail,
              "RTC-only route requires a nonzero run identity");
    EXPECT_EQ(publication.snapshot(), nullptr);
}

TEST(rtc_only_route,
     identity_result_is_independent_of_external_common_slot_assignments) {
    const auto fixture = route_fixture();
    std::vector<pipeline::NativeSlotAssociation> first_associations(4);
    first_associations[0].native_row = 10;
    first_associations[1].native_row = 11;
    first_associations[2].absence_reason =
        pipeline::CoincidenceAbsenceReason::no_candidate;
    first_associations[3].native_row = 12;
    std::map<pipeline::TimestreamNetworkId,
             std::vector<pipeline::NativeSlotAssociation>> first_by_network;
    first_by_network.emplace(0, std::move(first_associations));
    const pipeline::NativeAlignmentPlan first_alignment{
        fixture.native->scope(), {*fixture.timing},
        vector({100.0, 101.0, 102.0, 103.0}),
        std::move(first_by_network)};

    std::vector<pipeline::NativeSlotAssociation> second_associations(3);
    second_associations[0].native_row = 12;
    second_associations[1].native_row = 10;
    second_associations[2].native_row = 11;
    std::map<pipeline::TimestreamNetworkId,
             std::vector<pipeline::NativeSlotAssociation>> second_by_network;
    second_by_network.emplace(0, std::move(second_associations));
    const pipeline::NativeAlignmentPlan second_alignment{
        fixture.native->scope(), {*fixture.timing},
        vector({99.5, 100.5, 103.5}), std::move(second_by_network)};
    ASSERT_NE(first_alignment.slot_count(), second_alignment.slot_count());
    ASSERT_NE(first_alignment.association(0, 0).native_row,
              second_alignment.association(0, 0).native_row);

    pipeline::RtcOnlyProductSlot first_publication;
    pipeline::RtcOnlyProductSlot second_publication;
    const auto first = pipeline::run_identity_rtc_only(
        route_request(fixture, 1), first_publication);
    const auto second = pipeline::run_identity_rtc_only(
        route_request(fixture, 2), second_publication);

    ASSERT_TRUE(first.complete());
    ASSERT_TRUE(second.complete());
    for (pipeline::TimestreamNativeRow row = 10; row < 13; ++row) {
        EXPECT_EQ(first.published_product->timestream_handle()
                      ->identity(0, row, 0),
                  second.published_product->timestream_handle()
                      ->identity(0, row, 0));
        EXPECT_DOUBLE_EQ(first.published_product->timestream_handle()
                             ->output_time_unix_sec(0, row),
                         second.published_product->timestream_handle()
                             ->output_time_unix_sec(0, row));
        EXPECT_DOUBLE_EQ(first.published_product->timestream_handle()
                             ->value(pipeline::ReadoutMember::x, 0, row, 0),
                         second.published_product->timestream_handle()
                             ->value(pipeline::ReadoutMember::x, 0, row, 0));
        EXPECT_EQ(first.published_product->timestream_handle()
                      ->pair_decision(0, row, 0),
                  second.published_product->timestream_handle()
                      ->pair_decision(0, row, 0));
    }
}

TEST(rtc_only_route,
     route_and_identity_headers_exclude_common_grid_and_later_stage_entries) {
    namespace fs = std::filesystem;
    const auto repository = fs::path{__FILE__}.parent_path().parent_path();
    const std::vector<fs::path> headers{
        repository / "include/citlali/core/pipeline/identity_rtc.h",
        repository / "include/citlali/core/pipeline/rtc_only_route.h"};
    const std::vector<std::string> forbidden{
        "AlignedPairedReadout", "NativeAlignmentPlan", "common_slot",
        "alignment_absence", "timestream_native_pointing.h",
        "telescope_pointing_operations.h", "calib.h", "ptc/", "mapmaking/"};

    for (const auto &header : headers) {
        std::ifstream stream(header);
        ASSERT_TRUE(stream) << header;
        std::ostringstream content;
        content << stream.rdbuf();
        for (const auto &token : forbidden) {
            EXPECT_EQ(content.str().find(token), std::string::npos)
                << header << " contains forbidden route dependency " << token;
        }
    }
}

}  // namespace
