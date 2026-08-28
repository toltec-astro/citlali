#include <citlali/core/pipeline/rtc_only_route.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <map>
#include <memory>
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
    std::shared_ptr<const pipeline::NativeAlignmentPlan> alignment;
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

    std::vector<pipeline::NativeSlotAssociation> associations(4);
    associations[0].native_row = 10;
    associations[1].native_row = 11;
    associations[2].absence_reason =
        pipeline::CoincidenceAbsenceReason::no_candidate;
    associations[3].native_row = 12;
    std::map<pipeline::TimestreamNetworkId,
             std::vector<pipeline::NativeSlotAssociation>> by_network;
    by_network.emplace(network_id, std::move(associations));
    auto alignment =
        std::make_shared<const pipeline::NativeAlignmentPlan>(
            scope,
            std::vector<pipeline::NativeNetworkAlignment>{*timing},
            vector({100.0, 101.0, 102.0, 103.0}),
            std::move(by_network));
    return {std::move(native), std::move(alignment)};
}

pipeline::RtcOnlyRouteRequest route_request(
    const RtcOnlyFixture &fixture, std::uint64_t run = 1) {
    return {{run}, fixture.native, fixture.alignment, 0, 4};
}

TEST(rtc_only_route,
     executes_explicit_lca_and_atomically_publishes_inspectable_memory_product) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;
    const auto outcome = pipeline::run_identity_rtc_only(
        route_request(fixture, 42), publication);

    ASSERT_TRUE(outcome.complete());
    ASSERT_NE(outcome.published_product, nullptr);
    EXPECT_EQ(publication.snapshot(), outcome.published_product);
    const auto &bundle = *outcome.published_product;
    EXPECT_EQ(bundle.terminal_result().identity.run, 42U);
    EXPECT_EQ(bundle.evidence_handle()->identity().attempt, 42U);
    EXPECT_EQ(bundle.plan_handle()->identity().resolution, 42U);
    EXPECT_EQ(bundle.realization().completion,
              pipeline::RtcCompletionState::complete);
    EXPECT_EQ(bundle.timestream_handle()->output_slot_count(), 4U);
    EXPECT_EQ(bundle.timestream_handle()->output_cell_count(), 4U);

    const auto &diagnostics = outcome.terminal.diagnostics;
    EXPECT_EQ(diagnostics.network_count, 1U);
    EXPECT_EQ(diagnostics.detector_count, 1U);
    EXPECT_EQ(diagnostics.aligned_cell_count, 4U);
    EXPECT_EQ(diagnostics.mapped_cell_count, 3U);
    EXPECT_EQ(diagnostics.evidence_event_count, 2U);
    EXPECT_EQ(diagnostics.direct_x_event_count, 0U);
    EXPECT_EQ(diagnostics.direct_r_event_count, 1U);
    EXPECT_EQ(diagnostics.x_and_r_event_count, 0U);
    EXPECT_EQ(diagnostics.alignment_absence_event_count, 1U);
    EXPECT_EQ(diagnostics.pair_ineligible_cell_count, 2U);
    EXPECT_EQ(diagnostics.x_numerically_valid_cell_count, 3U);
    EXPECT_EQ(diagnostics.r_numerically_valid_cell_count, 2U);
    EXPECT_GT(diagnostics.derived_evidence_bytes, 0U);
    EXPECT_GT(diagnostics.derived_plan_bytes, 0U);
    EXPECT_EQ(diagnostics.rtc_owned_numeric_bytes, 0U);

    for (const auto member :
         {pipeline::ReadoutMember::x, pipeline::ReadoutMember::r}) {
        const auto published = bundle.timestream_handle()->value(
            member, 0, 3, 0);
        ASSERT_TRUE(published);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(*published),
                  std::bit_cast<std::uint64_t>(
                      fixture.native->network(0).value(member, 12, 0)));
    }
    EXPECT_EQ(bundle.timestream_handle()->pair_decision(0, 1, 0),
              pipeline::RtcPairDecision::ineligible);
    EXPECT_TRUE(bundle.timestream_handle()
                    ->member_numerically_valid(
                        pipeline::ReadoutMember::x, 0, 1, 0));
}

TEST(rtc_only_route,
     input_failure_is_truthful_and_leaves_an_empty_publication_slot) {
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;
    auto request = route_request(fixture);
    request.past_last_common_slot = 5;
    const auto outcome =
        pipeline::run_identity_rtc_only(request, publication);

    EXPECT_FALSE(outcome.complete());
    EXPECT_EQ(outcome.terminal.state,
              pipeline::RtcOnlyTerminalState::input_admission_failed);
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
    EXPECT_EQ(second.published_product, nullptr);
    EXPECT_EQ(publication.snapshot(), committed);
    EXPECT_EQ(publication.snapshot()->terminal_result().identity.run, 1U);
}

TEST(rtc_only_route,
     absent_or_unavailable_ast_state_is_not_consumed_and_cannot_change_result) {
    struct AstStateProbe {
        bool available = false;
        std::size_t interpolation_calls = 0;
    } ast;
    const auto fixture = route_fixture();
    pipeline::RtcOnlyProductSlot publication;

    const auto outcome = pipeline::run_identity_rtc_only(
        route_request(fixture), publication);

    EXPECT_TRUE(outcome.complete());
    EXPECT_FALSE(ast.available);
    EXPECT_EQ(ast.interpolation_calls, 0U);
    EXPECT_DOUBLE_EQ(
        outcome.published_product->timestream_handle()
            ->output_time_unix_sec(3),
        103.0);
    EXPECT_EQ(outcome.published_product->timestream_handle()
                  ->representative_native_identity(0, 3)
                  ->native_row(),
              12);
}

}  // namespace
