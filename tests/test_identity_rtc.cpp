#include <citlali/core/pipeline/identity_rtc.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
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

struct IdentityRtcFixture {
    std::shared_ptr<const pipeline::PairedReadout> native;
    std::shared_ptr<const pipeline::NativePairedReadoutView> view;
};

IdentityRtcFixture identity_fixture(
                                    pipeline::TimestreamNativeRow first_row = 10,
                                    pipeline::TimestreamNativeRow past_row = 14,
                                    std::int64_t first_output_uid = 500,
                                    std::int64_t second_output_uid = 501) {
    constexpr pipeline::TimestreamNetworkId network_id = 0;
    constexpr pipeline::TimestreamNativeRow first_native_row = 10;
    const pipeline::NativeObservationScope scope{152390, 0, 4};
    auto timing =
        std::make_shared<const pipeline::NativeNetworkAlignment>(
            network_id, first_native_row,
            vector({100.0, 101.0, 102.0, 104.0}),
            std::vector<pipeline::TimestreamPacketCounter>{20, 21, 22, 24});
    auto occurrence_axis =
        std::make_shared<const pipeline::PairedReadoutOccurrenceAxis>(
            timing, first_native_row,
            std::vector<pipeline::NativeOccurrenceInterval>{
                {99.8, 100.2}, {100.8, 101.2},
                {101.8, 102.2}, {103.8, 104.2}});
    auto mapping =
        std::make_shared<const pipeline::NativeReadoutMappingIdentity>(
            pipeline::NativeReadoutMappingIdentity{
                "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1",
                "producer:0", "tune:0", "mapping:0", "iq-to-xr:0",
                "raw-x:0", "raw-r:0"});
    std::vector<pipeline::PairedReadoutDetectorIdentity> detectors{
        {first_output_uid, 0, network_id, 1000, 0},
        {second_output_uid, 0, network_id, 1000, 1}};

    pipeline::PairedReadoutMatrix x(4, 2);
    pipeline::PairedReadoutMatrix r(4, 2);
    x << 1.25, -2.5,
         3.75, -4.125,
         5.5, -6.625,
         7.875, -8.0;
    r << 101.25, -102.5,
         103.75, -104.125,
         105.5, -106.625,
         107.875, -108.0;
    const auto valid = pipeline::ReadoutMemberState::measured(
        true, true, true, true);
    std::vector<pipeline::ReadoutMemberState> x_states(8, valid);
    std::vector<pipeline::ReadoutMemberState> r_states(8, valid);
    x_states[2] = pipeline::ReadoutMemberState::measured(
        true, false, true, true);  // native row 11, detector 500
    r_states[4] = pipeline::ReadoutMemberState::measured(
        true, false, true, true);  // native row 12, detector 500
    x_states[6] = pipeline::ReadoutMemberState::measured(
        true, false, true, true);  // native row 13, detector 500
    r_states[6] = pipeline::ReadoutMemberState::measured(
        true, false, true, true);  // native row 13, detector 500

    std::vector<pipeline::PairedReadoutNetwork> networks;
    networks.push_back(pipeline::PairedReadoutNetwork::admit(
        std::move(occurrence_axis), std::move(detectors),
        std::move(mapping), std::move(x), std::move(r),
        std::move(x_states), std::move(r_states)));
    auto native = pipeline::PairedReadout::admit(
        scope, {network_id}, std::move(networks));
    auto view = pipeline::NativePairedReadoutView::admit(
        native, {{network_id, first_row, past_row}});
    return {std::move(native), std::move(view)};
}

std::shared_ptr<const pipeline::RtcPlan> identity_plan(
    const std::shared_ptr<const pipeline::NativePairedReadoutView> &input,
    std::uint64_t attempt = 1,
    std::uint64_t resolution = 1) {
    return pipeline::consider_identity_rtc(
        pipeline::learn_identity_rtc(input, attempt), resolution);
}

TEST(identity_rtc,
     native_view_references_parent_axes_without_common_analysis_grid_projection) {
    const auto fixture = identity_fixture();
    const auto &native_network = fixture.native->network(0);

    EXPECT_EQ(fixture.view->parent_handle(), fixture.native);
    EXPECT_EQ(fixture.view->native_occurrence_count(), 4U);
    EXPECT_EQ(fixture.view->detector_occurrence_count(), 8U);
    EXPECT_EQ(fixture.view->span(0),
              pipeline::NativeOccurrenceSpan({0, 10, 14}));
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>(fixture.view->network(0).value(
            pipeline::ReadoutMember::x, 12, 1)),
        std::bit_cast<std::uint64_t>(native_network.value(
            pipeline::ReadoutMember::x, 12, 1)));
}

TEST(identity_rtc,
     learn_retains_x_r_and_both_origins_with_local_causes) {
    const auto fixture = identity_fixture();
    const auto evidence = pipeline::learn_identity_rtc(fixture.view, 7);

    EXPECT_EQ(evidence->identity().attempt, 7U);
    EXPECT_EQ(evidence->input_handle(), fixture.view);
    EXPECT_EQ(evidence->summary().examined_cell_count, 8U);
    EXPECT_EQ(evidence->summary().accepted_event_count, 3U);
    EXPECT_EQ(evidence->summary().direct_x_event_count, 2U);
    EXPECT_EQ(evidence->summary().direct_r_event_count, 2U);
    EXPECT_EQ(evidence->summary().x_and_r_event_count, 1U);

    const auto *x_event = evidence->find(0, 11, 0);
    const auto *r_event = evidence->find(0, 12, 0);
    const auto *both_event = evidence->find(0, 13, 0);
    ASSERT_NE(x_event, nullptr);
    ASSERT_NE(r_event, nullptr);
    ASSERT_NE(both_event, nullptr);
    EXPECT_TRUE(x_event->direct_x());
    EXPECT_FALSE(x_event->direct_r());
    EXPECT_FALSE(r_event->direct_x());
    EXPECT_TRUE(r_event->direct_r());
    EXPECT_EQ(both_event->origin, pipeline::RtcEvidenceOrigin::x_and_r);
    EXPECT_TRUE(pipeline::has_cause(
        evidence->member_local_causes(*x_event),
        pipeline::PairedReadoutCause::x_original_invalid));
    EXPECT_FALSE(pipeline::has_cause(
        evidence->member_local_causes(*x_event),
        pipeline::PairedReadoutCause::r_original_invalid));
    EXPECT_TRUE(pipeline::has_cause(
        evidence->member_local_causes(*r_event),
        pipeline::PairedReadoutCause::r_original_invalid));
    EXPECT_EQ(evidence->scientific_identity(*r_event),
              pipeline::RtcNativeCellIdentity({0, 12, 500}));
    EXPECT_LE(sizeof(pipeline::RtcEvidenceEvent), 16U);
    EXPECT_EQ(evidence->memory_evidence().derived_event_bytes,
              evidence->events().size() *
                  sizeof(pipeline::RtcEvidenceEvent));
}

TEST(identity_rtc,
     lookup_and_pair_consequence_are_independent_of_output_uid_order) {
    const auto fixture = identity_fixture(10, 14, 501, 500);
    const auto evidence = pipeline::learn_identity_rtc(fixture.view, 4);
    const auto result = pipeline::apply_identity_rtc(
        pipeline::consider_identity_rtc(evidence, 5), fixture.view);

    EXPECT_EQ(result.product->identity(0, 10, 0).detector_uid, 501);
    EXPECT_EQ(result.product->identity(0, 10, 1).detector_uid, 500);
    ASSERT_EQ(evidence->events().size(), 3U);
    for (const auto &event : evidence->events()) {
        const auto detector = static_cast<Eigen::Index>(
            event.cell.detector_index);
        const auto native_row = evidence->native_row(event);
        EXPECT_EQ(result.product->pair_decision(
                      event.cell.network_id, native_row,
                      detector),
                  pipeline::RtcPairDecision::ineligible);
        const auto *cause = result.product->pair_causal_evidence(
            event.cell.network_id, native_row, detector);
        ASSERT_NE(cause, nullptr);
        EXPECT_EQ(evidence->scientific_identity(*cause),
                  result.product->identity(
                      event.cell.network_id, native_row, detector));
    }
    EXPECT_EQ(result.realization.pair_ineligible_cell_count,
              evidence->events().size());
}

TEST(identity_rtc,
     consider_applies_bidirectional_pair_consequence_without_erasing_causes) {
    const auto fixture = identity_fixture();
    const auto evidence = pipeline::learn_identity_rtc(fixture.view, 3);
    const auto plan = pipeline::consider_identity_rtc(evidence, 9);
    const auto result = pipeline::apply_identity_rtc(plan, fixture.view);

    EXPECT_EQ(plan->identity().evidence, evidence->identity());
    EXPECT_EQ(plan->identity().resolution, 9U);
    EXPECT_EQ(plan->pair_policy(),
              pipeline::RtcPairPolicy::conservative_pair_wide);

    // Direct r evidence makes the corresponding x occurrence ineligible,
    // while x remains numerically valid and its local cause remains empty.
    EXPECT_TRUE(result.product->member_numerically_valid(
        pipeline::ReadoutMember::x, 0, 12, 0));
    EXPECT_EQ(result.product->member_local_causes(
                  pipeline::ReadoutMember::x, 0, 12, 0),
              pipeline::ReadoutMemberCause::none);
    EXPECT_EQ(result.product->pair_decision(0, 12, 0),
              pipeline::RtcPairDecision::ineligible);
    const auto *from_r = result.product->pair_causal_evidence(0, 12, 0);
    ASSERT_NE(from_r, nullptr);
    EXPECT_TRUE(from_r->direct_r());
    EXPECT_FALSE(from_r->direct_x());

    // Direct x evidence has the symmetric pair-wide consequence for r.
    EXPECT_TRUE(result.product->member_numerically_valid(
        pipeline::ReadoutMember::r, 0, 11, 0));
    EXPECT_EQ(result.product->member_local_causes(
                  pipeline::ReadoutMember::r, 0, 11, 0),
              pipeline::ReadoutMemberCause::none);
    EXPECT_EQ(result.product->pair_decision(0, 11, 0),
              pipeline::RtcPairDecision::ineligible);
    const auto *from_x = result.product->pair_causal_evidence(0, 11, 0);
    ASSERT_NE(from_x, nullptr);
    EXPECT_TRUE(from_x->direct_x());
    EXPECT_FALSE(from_x->direct_r());

    EXPECT_EQ(result.product->pair_decision(0, 10, 0),
              pipeline::RtcPairDecision::eligible);
}

TEST(identity_rtc,
     apply_is_exact_paired_identity_and_owns_no_duplicate_timestream_plane) {
    const auto fixture = identity_fixture();
    const auto plan = identity_plan(fixture.view);
    const auto result = pipeline::apply_identity_rtc(plan, fixture.view);

    EXPECT_EQ(result.product->input_handle(), fixture.view);
    EXPECT_EQ(result.product->native_parent_handle(), fixture.native);
    EXPECT_EQ(result.product->plan_handle(), plan);
    EXPECT_EQ(result.product->output_native_occurrence_count(), 4U);
    EXPECT_EQ(result.product->output_cell_count(), 8U);
    EXPECT_EQ(result.product->realized_operator().sampling_factor, 1U);
    EXPECT_EQ(result.product->realized_operator().sampling_phase, 0U);
    EXPECT_DOUBLE_EQ(result.product->realized_operator().x_from_x, 1.0);
    EXPECT_DOUBLE_EQ(result.product->realized_operator().x_from_r, 0.0);
    EXPECT_DOUBLE_EQ(result.product->realized_operator().r_from_x, 0.0);
    EXPECT_DOUBLE_EQ(result.product->realized_operator().r_from_r, 1.0);

    for (const auto member :
         {pipeline::ReadoutMember::x, pipeline::ReadoutMember::r}) {
        for (pipeline::TimestreamNativeRow row = 10; row < 14; ++row) {
            for (Eigen::Index detector = 0; detector < 2; ++detector) {
                const auto input_value = fixture.native->network(0).value(
                    member, row, detector);
                const auto output_value = result.product->value(
                    member, 0, row, detector);
                EXPECT_EQ(std::bit_cast<std::uint64_t>(output_value),
                          std::bit_cast<std::uint64_t>(input_value));
            }
        }
    }
    EXPECT_EQ(result.product->representative_native_identity(0, 13),
              fixture.native->network(0).occurrence_axis_handle()
                  ->identity(13));
    EXPECT_EQ(result.product->representative_interval(0, 13),
              pipeline::NativeOccurrenceInterval({103.8, 104.2}));
    EXPECT_DOUBLE_EQ(result.product->output_time_unix_sec(0, 13), 104.0);

    const auto memory = result.product->memory_evidence();
    EXPECT_EQ(memory.owned_numeric_bytes, 0U);
    EXPECT_EQ(memory.owned_state_plane_bytes, 0U);
    EXPECT_EQ(memory.logical_owned_bytes(), 0U);
    EXPECT_EQ(memory.referenced_parent_count, 1U);
    EXPECT_EQ(result.realization.output_native_occurrence_count, 4U);
    EXPECT_EQ(result.realization.output_cell_count, 8U);
    EXPECT_EQ(result.realization.pair_ineligible_cell_count, 3U);
    EXPECT_EQ(result.realization.x_numerically_valid_cell_count, 6U);
    EXPECT_EQ(result.realization.r_numerically_valid_cell_count, 6U);
    EXPECT_EQ(result.realization.realized_sampling_factor, 1U);
    EXPECT_LT(sizeof(pipeline::RtcRealization), 128U);
    EXPECT_EQ(plan->evidence_handle()->memory_evidence()
                  .referenced_parent_count,
              1U);
    EXPECT_EQ(plan->memory_evidence().derived_plan_bytes, 0U);
    EXPECT_EQ(plan->memory_evidence().logical_owned_bytes(), 0U);
    EXPECT_EQ(plan->memory_evidence().referenced_evidence_count, 1U);
}

TEST(identity_rtc, apply_rejects_a_plan_bound_to_another_input_instance) {
    const auto first = identity_fixture();
    const auto second = identity_fixture();
    const auto first_plan = identity_plan(first.view);

    EXPECT_THROW(
        pipeline::apply_identity_rtc(first_plan, second.view),
        std::invalid_argument);
}

TEST(identity_rtc,
     pair_decision_queries_fail_closed_outside_the_plan_bound_domain) {
    const auto fixture = identity_fixture();
    const auto plan = identity_plan(fixture.view);
    const auto result = pipeline::apply_identity_rtc(plan, fixture.view);

    EXPECT_THROW(plan->decision(1, 10, 0), std::out_of_range);
    EXPECT_THROW(plan->decision(0, 9, 0), std::out_of_range);
    EXPECT_THROW(plan->decision(0, 14, 0), std::out_of_range);
    EXPECT_THROW(plan->decision(0, 10, -1), std::out_of_range);
    EXPECT_THROW(plan->decision(0, 10, 2), std::out_of_range);
    EXPECT_THROW(plan->causal_evidence(0, 10, 2), std::out_of_range);

    EXPECT_THROW(result.product->pair_decision(0, 10, -1),
                 std::out_of_range);
    EXPECT_THROW(result.product->pair_decision(0, 10, 2),
                 std::out_of_range);
    EXPECT_THROW(result.product->pair_causal_evidence(0, 10, 2),
                 std::out_of_range);
}

TEST(identity_rtc,
     scientific_results_are_invariant_to_engineering_chunk_partition) {
    const auto full = identity_fixture();
    const auto first_chunk = pipeline::NativePairedReadoutView::admit(
        full.native, {{0, 10, 12}});
    const auto second_chunk = pipeline::NativePairedReadoutView::admit(
        full.native, {{0, 12, 14}});

    const auto full_evidence = pipeline::learn_identity_rtc(full.view, 1);
    const auto first_evidence =
        pipeline::learn_identity_rtc(first_chunk, 2);
    const auto second_evidence =
        pipeline::learn_identity_rtc(second_chunk, 3);
    EXPECT_EQ(first_evidence->events().size() +
                  second_evidence->events().size(),
              full_evidence->events().size());
    ASSERT_NE(full_evidence->find(0, 12, 0), nullptr);
    ASSERT_NE(second_evidence->find(0, 12, 0), nullptr);
    EXPECT_NE(full_evidence->find(0, 12, 0)
                  ->cell.native_occurrence_offset,
              second_evidence->find(0, 12, 0)
                  ->cell.native_occurrence_offset);

    const auto full_result = pipeline::apply_identity_rtc(
        pipeline::consider_identity_rtc(full_evidence, 1), full.view);
    const auto first_result = pipeline::apply_identity_rtc(
        pipeline::consider_identity_rtc(first_evidence, 2), first_chunk);
    const auto second_result = pipeline::apply_identity_rtc(
        pipeline::consider_identity_rtc(second_evidence, 3), second_chunk);
    EXPECT_NE(full_result.product.get(), first_result.product.get());
    EXPECT_NE(full_result.product->plan_handle().get(),
              first_result.product->plan_handle().get());

    for (pipeline::TimestreamNativeRow row = 10; row < 14; ++row) {
        const auto &partitioned = row < 12 ? first_result : second_result;
        for (Eigen::Index detector = 0; detector < 2; ++detector) {
            EXPECT_EQ(partitioned.product->identity(0, row, detector),
                      full_result.product->identity(0, row, detector));
            EXPECT_EQ(partitioned.product->representative_native_identity(
                          0, row),
                      full_result.product->representative_native_identity(
                          0, row));
            EXPECT_EQ(partitioned.product->representative_interval(0, row),
                      full_result.product->representative_interval(0, row));
            EXPECT_DOUBLE_EQ(partitioned.product->output_time_unix_sec(0, row),
                             full_result.product->output_time_unix_sec(0, row));
            EXPECT_EQ(partitioned.product->value(
                          pipeline::ReadoutMember::x, 0, row, detector),
                      full_result.product->value(
                          pipeline::ReadoutMember::x, 0, row, detector));
            EXPECT_EQ(partitioned.product->value(
                          pipeline::ReadoutMember::r, 0, row, detector),
                      full_result.product->value(
                          pipeline::ReadoutMember::r, 0, row, detector));
            EXPECT_EQ(partitioned.product->pair_decision(
                          0, row, detector),
                      full_result.product->pair_decision(
                          0, row, detector));
            const auto *partitioned_cause =
                partitioned.product->pair_causal_evidence(0, row, detector);
            const auto *full_cause =
                full_result.product->pair_causal_evidence(0, row, detector);
            ASSERT_EQ(partitioned_cause == nullptr, full_cause == nullptr);
            if (partitioned_cause) {
                EXPECT_EQ(partitioned_cause->origin, full_cause->origin);
                EXPECT_EQ(partitioned.product->plan_handle()
                              ->evidence_handle()->scientific_identity(
                                  *partitioned_cause),
                          full_result.product->plan_handle()
                              ->evidence_handle()->scientific_identity(
                                  *full_cause));
            }
        }
    }
}

TEST(identity_rtc, succeeds_from_native_timing_without_pointing_state) {
    const auto fixture = identity_fixture();
    const auto result = pipeline::apply_identity_rtc(
        identity_plan(fixture.view), fixture.view);

    ASSERT_NE(result.product, nullptr);
    EXPECT_EQ(result.realization.completion,
              pipeline::RtcCompletionState::complete);
    EXPECT_DOUBLE_EQ(result.product->output_time_unix_sec(0, 10), 100.0);
    EXPECT_EQ(result.product->representative_native_identity(0, 10)
                  .native_row(),
              10);
}

}  // namespace
