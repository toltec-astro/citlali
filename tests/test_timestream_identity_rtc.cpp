#include <citlali/core/pipeline/timestream_identity_rtc.h>

#include "timestream_successor_identity_test_support.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;
namespace support = citlali::test::timestream_successor;

struct IdentityRtcFixture {
    std::shared_ptr<const pipeline::NativePairedReadoutObservation> native;
    std::shared_ptr<const pipeline::NativePairedReadoutView> logical;
    std::shared_ptr<const pipeline::ValSnapshot> val;
};

IdentityRtcFixture identity_fixture() {
    auto x_states = support::valid_states(8);
    auto r_states = support::valid_states(8);
    x_states[2] = pipeline::NativeReadoutCoordinateState::measured(
        true, false, true, true);
    r_states[4] = pipeline::NativeReadoutCoordinateState::measured(
        true, false, true, true);
    x_states[6] = pipeline::NativeReadoutCoordinateState::measured(
        true, false, true, true);
    r_states[6] = pipeline::NativeReadoutCoordinateState::measured(
        true, false, true, true);

    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(support::make_network(
        0, 10, {100.0, 101.0, 102.0, 104.0}, {20, 21, 22, 24},
        2, 1.25, 101.25, std::move(x_states), std::move(r_states)));
    auto native = support::make_observation(std::move(networks), {0});
    auto logical = pipeline::NativePairedReadoutView::full(native);
    auto val = pipeline::ValSnapshot::initial(native);
    return {std::move(native), std::move(logical), std::move(val)};
}

std::shared_ptr<const pipeline::RtcPlan> identity_plan(
    const std::shared_ptr<const pipeline::NativePairedReadoutView> &input,
    const std::shared_ptr<const pipeline::ValSnapshot> &val,
    std::uint64_t attempt = 1,
    std::uint64_t consideration = 1) {
    return pipeline::consider_identity_rtc(
        pipeline::learn_identity_rtc(input, val, attempt), val,
        consideration);
}

TEST(identity_rtc,
     native_view_references_canonical_parent_without_common_grid) {
    const auto fixture = identity_fixture();
    EXPECT_EQ(fixture.logical->parent_handle(), fixture.native);
    EXPECT_EQ(fixture.logical->native_occurrence_count(), 4U);
    EXPECT_EQ(fixture.logical->detector_occurrence_count(), 8U);
    EXPECT_EQ(fixture.logical->span(0),
              pipeline::NativeOccurrenceSpan({0, 10, 14}));
    EXPECT_EQ(fixture.logical->network(0).occurrence_axis_handle(),
              fixture.native->network(0).occurrence_axis_handle());
    EXPECT_EQ(fixture.logical->network(0).mapping_authority_handle(),
              fixture.native->network(0).mapping_authority_handle());
}

TEST(identity_rtc,
     learn_preserves_coordinate_local_and_pair_causes_as_sparse_evidence) {
    const auto fixture = identity_fixture();
    const auto evidence = pipeline::learn_identity_rtc(
        fixture.logical, fixture.val, 7);

    EXPECT_EQ(evidence->identity().attempt, 7U);
    EXPECT_EQ(evidence->input_handle(), fixture.logical);
    EXPECT_EQ(evidence->val_snapshot_handle(), fixture.val);
    EXPECT_EQ(evidence->val_generation(), pipeline::ValGeneration{0});
    EXPECT_TRUE(evidence->proposed_val_findings().empty());
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
        evidence->pair_local_causes(*x_event),
        pipeline::NativePairedReadoutCause::x_producer_invalid));
    EXPECT_FALSE(pipeline::has_cause(
        evidence->pair_local_causes(*x_event),
        pipeline::NativePairedReadoutCause::r_producer_invalid));
    EXPECT_TRUE(pipeline::has_cause(
        evidence->pair_local_causes(*r_event),
        pipeline::NativePairedReadoutCause::r_producer_invalid));

    const auto identity = evidence->scientific_identity(*r_event);
    EXPECT_EQ(identity.network_id, 0);
    EXPECT_EQ(identity.native_row, 12);
    EXPECT_EQ(identity.storage_column, 0);
    EXPECT_EQ(identity.detector_occurrence_id,
              "detector-occurrence:0:0");
    EXPECT_EQ(identity.parent_readout_occurrence_key, 10012);
    EXPECT_EQ(identity.paired_xr_occurrence_key, 20012);
    EXPECT_EQ(identity.mapping_record_id, "mapping-record:0");
    EXPECT_LE(sizeof(pipeline::RtcEvidenceEvent), 16U);
    EXPECT_EQ(evidence->memory_evidence().derived_event_bytes,
              evidence->events().size() *
                  sizeof(pipeline::RtcEvidenceEvent));
}

TEST(identity_rtc,
     consider_produces_one_immutable_factor_one_pair_policy) {
    const auto fixture = identity_fixture();
    const auto evidence = pipeline::learn_identity_rtc(
        fixture.logical, fixture.val, 3);
    const auto plan = pipeline::consider_identity_rtc(
        evidence, fixture.val, 9);

    EXPECT_EQ(plan->identity().evidence, evidence->identity());
    EXPECT_EQ(plan->identity().consideration, 9U);
    EXPECT_EQ(plan->pair_policy(),
              pipeline::RtcPairPolicy::conservative_pair_wide);
    EXPECT_EQ(plan->operator_spec().sampling_factor, 1U);
    EXPECT_EQ(plan->operator_spec().sampling_phase, 0U);
    EXPECT_EQ(plan->val_snapshot_handle(), fixture.val);
    EXPECT_EQ(plan->required_val_generation(), pipeline::ValGeneration{0});
    EXPECT_EQ(plan->val_change_policy(),
              pipeline::IdentityRtcValChangePolicy::
                  exact_generation_requires_relearn);
    EXPECT_DOUBLE_EQ(plan->operator_spec().x_from_x, 1.0);
    EXPECT_DOUBLE_EQ(plan->operator_spec().x_from_r, 0.0);
    EXPECT_DOUBLE_EQ(plan->operator_spec().r_from_x, 0.0);
    EXPECT_DOUBLE_EQ(plan->operator_spec().r_from_r, 1.0);

    EXPECT_EQ(plan->decision(0, 10, 0),
              pipeline::RtcPairDecision::eligible);
    EXPECT_EQ(plan->decision(0, 11, 0),
              pipeline::RtcPairDecision::ineligible);
    EXPECT_EQ(plan->decision(0, 12, 0),
              pipeline::RtcPairDecision::ineligible);
    EXPECT_EQ(plan->decision(0, 13, 0),
              pipeline::RtcPairDecision::ineligible);
}

TEST(identity_rtc,
     apply_is_bitwise_identity_and_owns_no_duplicate_payload_or_state_plane) {
    const auto fixture = identity_fixture();
    const auto plan = identity_plan(fixture.logical, fixture.val);
    const auto result = pipeline::apply_identity_rtc(
        plan, fixture.logical, fixture.val);

    EXPECT_EQ(result.product->input_handle(), fixture.logical);
    EXPECT_EQ(result.product->native_parent_handle(), fixture.native);
    EXPECT_EQ(result.product->plan_handle(), plan);
    EXPECT_EQ(result.product->val_snapshot_handle(), fixture.val);
    EXPECT_EQ(result.product->val_generation(), pipeline::ValGeneration{0});
    EXPECT_TRUE(result.product->proposed_val_findings().empty());
    EXPECT_EQ(result.product->output_native_occurrence_count(), 4U);
    EXPECT_EQ(result.product->output_cell_count(), 8U);
    EXPECT_EQ(result.product->realized_operator(),
              pipeline::RtcIdentityOperator{});

    for (const auto coordinate :
         {pipeline::NativeReadoutCoordinate::x,
          pipeline::NativeReadoutCoordinate::r}) {
        for (pipeline::TimestreamNativeRow row = 10; row < 14; ++row) {
            for (Eigen::Index detector = 0; detector < 2; ++detector) {
                const auto input_value = fixture.native->network(0).value(
                    coordinate, row, detector);
                const auto output_value = result.product->value(
                    coordinate, 0, row, detector);
                EXPECT_EQ(std::bit_cast<std::uint64_t>(output_value),
                          std::bit_cast<std::uint64_t>(input_value));
                EXPECT_EQ(result.product->member_state(
                              coordinate, 0, row, detector).valid(),
                          fixture.native->network(0)
                              .state(coordinate, row, detector).valid());
                EXPECT_EQ(result.product->member_local_causes(
                              coordinate, 0, row, detector),
                          fixture.native->network(0)
                              .state(coordinate, row, detector).causes());
            }
        }
    }

    EXPECT_EQ(result.product->representative_native_identity(0, 13),
              fixture.native->network(0)
                  .occurrence_axis().native_identity(13));
    EXPECT_EQ(result.product->integration_support(0, 13),
              fixture.native->network(0)
                  .occurrence_axis().occurrence(13).integration_support);
    EXPECT_EQ(result.product->occurrence_binding(0, 13)
                  .paired_xr_occurrence_key,
              20013);
    EXPECT_EQ(result.product->detector_binding(0, 1).tone_or_channel_id,
              "tone:0:1");
    EXPECT_EQ(result.product->mapping_authority(0).mapping_revision_id,
              "mapping-revision:0");

    const auto memory = result.product->memory_evidence();
    EXPECT_EQ(memory.owned_numeric_bytes, 0U);
    EXPECT_EQ(memory.owned_state_plane_bytes, 0U);
    EXPECT_EQ(memory.logical_owned_bytes(), 0U);
    EXPECT_EQ(memory.referenced_parent_count, 1U);
    EXPECT_EQ(result.realization.output_native_occurrence_count, 4U);
    EXPECT_EQ(result.realization.output_cell_count, 8U);
    EXPECT_EQ(result.realization.pair_ineligible_cell_count, 3U);
    EXPECT_EQ(result.realization.x_payload_available_cell_count, 8U);
    EXPECT_EQ(result.realization.r_payload_available_cell_count, 8U);
    EXPECT_EQ(result.realization.x_numerically_valid_cell_count, 6U);
    EXPECT_EQ(result.realization.r_numerically_valid_cell_count, 6U);
    EXPECT_EQ(result.realization.realized_sampling_factor, 1U);
    EXPECT_EQ(result.realization.val_generation,
              pipeline::ValGeneration{0});
    EXPECT_EQ(plan->memory_evidence().logical_owned_bytes(), 0U);
}

TEST(identity_rtc,
     exact_partition_schedule_is_scientifically_invariant_and_fail_closed) {
    const auto fixture = identity_fixture();
    const std::vector<std::shared_ptr<const pipeline::NativePairedReadoutView>>
        partitions{
            pipeline::NativePairedReadoutView::admit(
                fixture.native, {{0, 10, 12}}),
            pipeline::NativePairedReadoutView::admit(
                fixture.native, {{0, 12, 14}})};
    const auto evidence = pipeline::learn_identity_rtc_partitioned(
        fixture.logical, partitions, fixture.val, 11);
    const auto plan = pipeline::consider_identity_rtc(
        evidence, fixture.val, 12);
    const auto partitioned = pipeline::apply_identity_rtc_partitioned(
        plan, fixture.logical, partitions, fixture.val);
    const auto single = pipeline::apply_identity_rtc(
        identity_plan(fixture.logical, fixture.val, 21, 22),
        fixture.logical, fixture.val);

    EXPECT_EQ(partitioned.realization.output_cell_count,
              single.realization.output_cell_count);
    EXPECT_EQ(partitioned.realization.pair_ineligible_cell_count,
              single.realization.pair_ineligible_cell_count);
    for (pipeline::TimestreamNativeRow row = 10; row < 14; ++row) {
        for (Eigen::Index detector = 0; detector < 2; ++detector) {
            EXPECT_EQ(partitioned.product->identity(0, row, detector),
                      single.product->identity(0, row, detector));
            EXPECT_EQ(partitioned.product->pair_decision(0, row, detector),
                      single.product->pair_decision(0, row, detector));
            for (const auto coordinate :
                 {pipeline::NativeReadoutCoordinate::x,
                  pipeline::NativeReadoutCoordinate::r}) {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(
                              partitioned.product->value(
                                  coordinate, 0, row, detector)),
                          std::bit_cast<std::uint64_t>(
                              single.product->value(
                                  coordinate, 0, row, detector)));
            }
        }
    }

    const std::vector<std::shared_ptr<const pipeline::NativePairedReadoutView>>
        incomplete{partitions.front()};
    EXPECT_THROW(
        pipeline::learn_identity_rtc_partitioned(
            fixture.logical, incomplete, fixture.val, 31),
        pipeline::IncompleteNativePartitionSchedule);
}

TEST(identity_rtc, exact_plan_and_input_instance_binding_is_enforced) {
    const auto first = identity_fixture();
    const auto second = identity_fixture();
    const auto first_plan = identity_plan(first.logical, first.val);

    EXPECT_THROW(
        pipeline::apply_identity_rtc(
            first_plan, second.logical, second.val),
        std::invalid_argument);
    EXPECT_THROW(first_plan->decision(1, 10, 0), std::out_of_range);
    EXPECT_THROW(first_plan->decision(0, 9, 0), std::out_of_range);
    EXPECT_THROW(first_plan->decision(0, 14, 0), std::out_of_range);
    EXPECT_THROW(first_plan->decision(0, 10, -1), std::out_of_range);
    EXPECT_THROW(first_plan->decision(0, 10, 2), std::out_of_range);
}

TEST(identity_rtc,
     synthetic_spike_fact_generation_invalidates_stale_evidence_and_plan) {
    const auto fixture = identity_fixture();
    const auto evidence_v0 = pipeline::learn_identity_rtc(
        fixture.logical, fixture.val, 41);
    const auto plan_v0 = pipeline::consider_identity_rtc(
        evidence_v0, fixture.val, 42);

    const pipeline::ValProducerProductIdentity rtc_product{
        pipeline::ValProducer::rtc, 41};
    const auto affected_cell = fixture.val->address(0, 12, 0);
    const pipeline::ValFindingKey spike_key{
        rtc_product, affected_cell, pipeline::ValFactCode{1}};
    pipeline::ValDeltaBuilder builder{fixture.val, rtc_product};
    builder.propose(affected_cell, pipeline::ValFactCode{1},
                    pipeline::ValFactState{1},
                    pipeline::ValFactCause{1});

    // A phase that holds V0 cannot observe an uncommitted proposal.
    EXPECT_EQ(fixture.val->find(spike_key), nullptr);
    auto delta = builder.freeze();
    EXPECT_EQ(fixture.val->find(spike_key), nullptr);
    const auto val_v1 = pipeline::ValSnapshot::commit(std::move(delta));
    ASSERT_NE(val_v1->find(spike_key), nullptr);
    EXPECT_EQ(val_v1->generation(), pipeline::ValGeneration{1});

    EXPECT_THROW(
        pipeline::consider_identity_rtc(evidence_v0, val_v1, 43),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::apply_identity_rtc(plan_v0, fixture.logical, val_v1),
        pipeline::StaleRtcValGeneration);

    const auto evidence_v1 = pipeline::learn_identity_rtc(
        fixture.logical, val_v1, 44);
    const auto plan_v1 = pipeline::consider_identity_rtc(
        evidence_v1, val_v1, 45);
    const auto applied_v1 = pipeline::apply_identity_rtc(
        plan_v1, fixture.logical, val_v1);
    EXPECT_EQ(applied_v1.product->val_snapshot_handle(), val_v1);
    EXPECT_EQ(applied_v1.realization.val_generation,
              pipeline::ValGeneration{1});
}

TEST(identity_rtc, completes_from_native_timing_without_pointing_state) {
    const auto fixture = identity_fixture();
    const auto result = pipeline::apply_identity_rtc(
        identity_plan(fixture.logical, fixture.val), fixture.logical,
        fixture.val);

    ASSERT_NE(result.product, nullptr);
    EXPECT_EQ(result.realization.completion,
              pipeline::RtcCompletionState::complete);
    EXPECT_DOUBLE_EQ(result.product->output_time_unix_sec(0, 10), 100.0);
    EXPECT_EQ(result.product->representative_native_identity(0, 10)
                  .native_row(),
              10);
}

}  // namespace
