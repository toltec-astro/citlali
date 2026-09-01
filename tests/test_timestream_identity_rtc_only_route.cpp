#include <citlali/core/pipeline/timestream_identity_rtc_only_route.h>

#include "timestream_successor_identity_test_support.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;
namespace support = citlali::test::timestream_successor;

struct RtcOnlyFixture {
    std::shared_ptr<const pipeline::NativePairedReadoutObservation> native;
};

RtcOnlyFixture route_fixture(double time_offset = 0.0) {
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
        {1000.0000 + time_offset, 1000.0100 + time_offset,
         1000.0200 + time_offset, 1000.0300 + time_offset},
        {100, 101, 102, 103}, 1, 1.0, 101.0,
        std::move(nw0_x), std::move(nw0_r)));
    networks.push_back(support::make_network(
        7, 70,
        {1000.0025 + time_offset, 1000.0125 + time_offset,
         1000.0325 + time_offset},
        {700, 701, 703}, 1, 11.0, 111.0,
        std::move(nw7_x), std::move(nw7_r)));
    return {support::make_observation(std::move(networks), {0, 7})};
}

pipeline::RtcOnlyRouteRequest route_request(
    const RtcOnlyFixture &fixture,
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

TEST(identity_rtc_only_route,
     publishes_one_complete_inspectable_native_axis_product) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;
    const auto outcome = pipeline::run_identity_rtc_only(
        route_request(fixture), publication);

    ASSERT_TRUE(outcome.complete());
    ASSERT_NE(outcome.published_product, nullptr);
    EXPECT_EQ(publication.snapshot(), outcome.published_product);
    const auto &bundle = *outcome.published_product;
    EXPECT_EQ(bundle.terminal_result().identity.run, 13U);
    EXPECT_EQ(bundle.terminal_result().state,
              pipeline::RtcOnlyTerminalState::complete);
    EXPECT_EQ(bundle.terminal_result().failure_cause,
              pipeline::RtcOnlyFailureCause::none);
    EXPECT_TRUE(bundle.terminal_result().failure_detail.empty());
    EXPECT_EQ(bundle.finalization().run_identity,
              bundle.terminal_result().identity);
    EXPECT_EQ(bundle.finalization().input_handle,
              bundle.timestream_handle()->native_parent_handle());
    EXPECT_EQ(bundle.plan_handle()->identity(),
              bundle.realization().plan_identity);
    EXPECT_EQ(bundle.evidence_handle(),
              bundle.plan_handle()->evidence_handle());

    const auto &product = *bundle.timestream_handle();
    EXPECT_EQ(product.native_parent_handle(), fixture.native);
    EXPECT_EQ(product.network_spans().size(), 2U);
    EXPECT_EQ(product.output_native_occurrence_count(), 7U);
    EXPECT_EQ(product.output_cell_count(), 7U);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(
                  product.output_time_unix_sec(0, 10)),
              std::bit_cast<std::uint64_t>(1000.0000));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(
                  product.output_time_unix_sec(7, 70)),
              std::bit_cast<std::uint64_t>(1000.0025));
    EXPECT_NE(product.output_time_unix_sec(0, 10),
              product.output_time_unix_sec(7, 70));
    EXPECT_EQ(product.identity(0, 10, 0).detector_occurrence_id,
              "detector-occurrence:0:0");
    EXPECT_EQ(product.identity(7, 70, 0).detector_occurrence_id,
              "detector-occurrence:7:0");
    EXPECT_EQ(product.memory_evidence().owned_numeric_bytes, 0U);

    const auto &diagnostics = outcome.terminal.diagnostics;
    EXPECT_EQ(diagnostics.network_count, 2U);
    EXPECT_EQ(diagnostics.engineering_partition_count, 2U);
    EXPECT_EQ(diagnostics.detector_count, 2U);
    EXPECT_EQ(diagnostics.native_occurrence_count, 7U);
    EXPECT_EQ(diagnostics.detector_occurrence_count, 7U);
    EXPECT_EQ(diagnostics.evidence_event_count, 2U);
    EXPECT_EQ(diagnostics.direct_x_event_count, 1U);
    EXPECT_EQ(diagnostics.direct_r_event_count, 1U);
    EXPECT_EQ(diagnostics.x_and_r_event_count, 0U);
    EXPECT_EQ(diagnostics.pair_ineligible_cell_count, 2U);
    EXPECT_EQ(diagnostics.x_payload_available_cell_count, 7U);
    EXPECT_EQ(diagnostics.r_payload_available_cell_count, 7U);
    EXPECT_EQ(diagnostics.x_numerically_valid_cell_count, 6U);
    EXPECT_EQ(diagnostics.r_numerically_valid_cell_count, 6U);
    EXPECT_EQ(diagnostics.derived_plan_bytes, 0U);
    EXPECT_EQ(diagnostics.rtc_owned_numeric_bytes, 0U);
    EXPECT_EQ(diagnostics.native_admission_entry_count, 1U);
    EXPECT_EQ(diagnostics.learn_entry_count, 1U);
    EXPECT_EQ(diagnostics.consider_entry_count, 1U);
    EXPECT_EQ(diagnostics.apply_entry_count, 1U);
    EXPECT_EQ(diagnostics.finalization_entry_count, 1U);
    EXPECT_EQ(diagnostics.publication_entry_count, 1U);
    EXPECT_STREQ(pipeline::rtc_only_terminal_state_name(
                     outcome.terminal.state),
                 "complete");
    EXPECT_STREQ(pipeline::rtc_only_failure_cause_name(
                     outcome.terminal.failure_cause),
                 "none");
}

TEST(identity_rtc_only_route,
     partitioning_preserves_values_identities_causes_and_native_gaps) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot split_publication;
    const auto split = pipeline::run_identity_rtc_only(
        route_request(fixture, 1), split_publication);
    auto single_request = route_request(fixture, 2);
    single_request.engineering_partitions =
        {single_request.logical_spans};
    pipeline::RtcOnlyProductSlot single_publication;
    const auto single = pipeline::run_identity_rtc_only(
        single_request, single_publication);

    ASSERT_TRUE(split.complete());
    ASSERT_TRUE(single.complete());
    const auto &split_product = *split.published_product->timestream_handle();
    const auto &single_product =
        *single.published_product->timestream_handle();
    EXPECT_EQ(split_product.realized_operator(),
              single_product.realized_operator());
    for (const auto network_id : {0, 7}) {
        const auto span = split_product.input_handle()->span(network_id);
        for (auto row = span.first_native_row;
             row < span.past_last_native_row; ++row) {
            EXPECT_EQ(split_product.identity(network_id, row, 0),
                      single_product.identity(network_id, row, 0));
            EXPECT_EQ(split_product.integration_support(network_id, row),
                      single_product.integration_support(network_id, row));
            EXPECT_EQ(split_product.pair_decision(network_id, row, 0),
                      single_product.pair_decision(network_id, row, 0));
            for (const auto coordinate :
                 {pipeline::NativeReadoutCoordinate::x,
                  pipeline::NativeReadoutCoordinate::r}) {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(
                              split_product.value(
                                  coordinate, network_id, row, 0)),
                          std::bit_cast<std::uint64_t>(
                              single_product.value(
                                  coordinate, network_id, row, 0)));
                EXPECT_EQ(split_product.member_local_causes(
                              coordinate, network_id, row, 0),
                          single_product.member_local_causes(
                              coordinate, network_id, row, 0));
            }
        }
    }
    EXPECT_EQ(split_product.network_spans()[0].occurrence_count(), 4U);
    EXPECT_EQ(split_product.network_spans()[1].occurrence_count(), 3U);
    EXPECT_DOUBLE_EQ(split_product.output_time_unix_sec(7, 72), 1000.0325);
    EXPECT_THROW(split_product.output_time_unix_sec(7, 73),
                 std::out_of_range);
}

TEST(identity_rtc_only_route,
     incomplete_partition_fails_before_scientific_lifecycle_or_publication) {
    const auto fixture = route_fixture();
    auto request = route_request(fixture);
    request.engineering_partitions.pop_back();
    pipeline::RtcOnlyProductSlot publication;

    const auto outcome =
        pipeline::run_identity_rtc_only(request, publication);

    EXPECT_FALSE(outcome.complete());
    EXPECT_EQ(outcome.terminal.state,
              pipeline::RtcOnlyTerminalState::input_admission_failed);
    EXPECT_EQ(outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::incomplete_logical_support);
    EXPECT_EQ(outcome.terminal.diagnostics.learn_entry_count, 0U);
    EXPECT_EQ(outcome.terminal.diagnostics.consider_entry_count, 0U);
    EXPECT_EQ(publication.snapshot(), nullptr);
}

TEST(identity_rtc_only_route,
     finalization_binds_exact_run_input_and_complete_logical_content) {
    const auto fixture = route_fixture();
    const auto foreign = route_fixture(10.0);
    auto foreign_request = route_request(foreign);
    foreign_request.finalization = route_request(fixture).finalization;
    pipeline::RtcOnlyProductSlot foreign_publication;
    const auto foreign_outcome = pipeline::run_identity_rtc_only(
        foreign_request, foreign_publication);
    EXPECT_FALSE(foreign_outcome.complete());
    EXPECT_EQ(foreign_outcome.terminal.state,
              pipeline::RtcOnlyTerminalState::finalization_failed);
    EXPECT_EQ(foreign_outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::finalization_identity_mismatch);
    EXPECT_EQ(foreign_publication.snapshot(), nullptr);

    auto stale_run = route_request(fixture);
    stale_run.finalization.run_identity.run = 999;
    pipeline::RtcOnlyProductSlot stale_run_publication;
    const auto stale_run_outcome = pipeline::run_identity_rtc_only(
        stale_run, stale_run_publication);
    EXPECT_EQ(stale_run_outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::finalization_identity_mismatch);
    EXPECT_EQ(stale_run_publication.snapshot(), nullptr);

    auto incomplete = route_request(fixture);
    --incomplete.finalization.completed_native_occurrence_count;
    pipeline::RtcOnlyProductSlot incomplete_publication;
    const auto incomplete_outcome = pipeline::run_identity_rtc_only(
        incomplete, incomplete_publication);
    EXPECT_EQ(incomplete_outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::
                  required_logical_content_incomplete);
    EXPECT_EQ(incomplete_publication.snapshot(), nullptr);

    auto unfinished = route_request(fixture);
    unfinished.finalization.observation_facts_finalized = false;
    pipeline::RtcOnlyProductSlot unfinished_publication;
    const auto unfinished_outcome = pipeline::run_identity_rtc_only(
        unfinished, unfinished_publication);
    EXPECT_EQ(unfinished_outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::observation_facts_incomplete);
    EXPECT_EQ(unfinished_publication.snapshot(), nullptr);
}

TEST(identity_rtc_only_route,
     publication_is_once_only_no_replace_and_preserves_prior_completion) {
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
    EXPECT_EQ(publication.snapshot(), committed);
    EXPECT_EQ(publication.snapshot()->terminal_result().identity.run, 1U);
}

TEST(identity_rtc_only_route,
     invalid_run_identity_has_truthful_terminal_cause_and_empty_slot) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;
    const auto outcome = pipeline::run_identity_rtc_only(
        route_request(fixture, 0), publication);

    EXPECT_FALSE(outcome.complete());
    EXPECT_EQ(outcome.terminal.state,
              pipeline::RtcOnlyTerminalState::input_admission_failed);
    EXPECT_EQ(outcome.terminal.failure_cause,
              pipeline::RtcOnlyFailureCause::invalid_run_identity);
    EXPECT_EQ(publication.snapshot(), nullptr);
}

}  // namespace
