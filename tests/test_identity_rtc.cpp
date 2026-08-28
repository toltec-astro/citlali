#include <citlali/core/pipeline/identity_rtc.h>

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
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
    std::shared_ptr<const pipeline::NativeAlignmentPlan> alignment;
    std::shared_ptr<const pipeline::AlignedPairedReadout> aligned;
};

IdentityRtcFixture identity_fixture(std::size_t first_slot = 0,
                                    std::size_t past_last_slot = 5) {
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
        {500, 0, network_id, 1000, 0},
        {501, 0, network_id, 1000, 1}};

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

    std::vector<pipeline::NativeSlotAssociation> associations(5);
    associations[0].native_row = 10;
    associations[1].native_row = 11;
    associations[2].native_row = 12;
    associations[3].absence_reason =
        pipeline::CoincidenceAbsenceReason::no_candidate;
    associations[4].native_row = 13;
    std::map<pipeline::TimestreamNetworkId,
             std::vector<pipeline::NativeSlotAssociation>> by_network;
    by_network.emplace(network_id, std::move(associations));
    auto alignment =
        std::make_shared<const pipeline::NativeAlignmentPlan>(
            scope,
            std::vector<pipeline::NativeNetworkAlignment>{*timing},
            vector({100.0, 101.0, 102.0, 103.0, 104.0}),
            std::move(by_network));
    auto aligned = pipeline::AlignedPairedReadout::admit(
        native, alignment, first_slot, past_last_slot);
    return {std::move(native), std::move(alignment), std::move(aligned)};
}

std::shared_ptr<const pipeline::RtcPlan> identity_plan(
    const std::shared_ptr<const pipeline::AlignedPairedReadout> &input,
    std::uint64_t attempt = 1,
    std::uint64_t resolution = 1) {
    return pipeline::consider_identity_rtc(
        pipeline::learn_identity_rtc(input, attempt), resolution);
}

TEST(identity_rtc,
     align_relation_references_native_pair_and_preserves_align_owned_mapping) {
    const auto fixture = identity_fixture();
    const auto &native_network = fixture.native->network(0);

    EXPECT_EQ(fixture.aligned->native_parent_handle(), fixture.native);
    EXPECT_EQ(fixture.aligned->alignment_handle(), fixture.alignment);
    EXPECT_EQ(fixture.aligned->mapped_cell_count(), 8U);
    EXPECT_EQ(fixture.aligned->aligned_cell_count(), 10U);
    EXPECT_EQ(fixture.aligned->identity(0, 2, 0).native_row, 12);
    EXPECT_FALSE(fixture.aligned->identity(0, 3, 0)
                     .has_native_occurrence());
    EXPECT_EQ(fixture.aligned->absence_reason(0, 3),
              pipeline::CoincidenceAbsenceReason::no_candidate);
    ASSERT_TRUE(fixture.aligned->value(
        pipeline::ReadoutMember::x, 0, 2, 1));
    EXPECT_EQ(
        std::bit_cast<std::uint64_t>(*fixture.aligned->value(
            pipeline::ReadoutMember::x, 0, 2, 1)),
        std::bit_cast<std::uint64_t>(native_network.value(
            pipeline::ReadoutMember::x, 12, 1)));
}

TEST(identity_rtc,
     learn_retains_x_r_both_and_alignment_origins_with_local_causes) {
    const auto fixture = identity_fixture();
    const auto evidence = pipeline::learn_identity_rtc(fixture.aligned, 7);

    EXPECT_EQ(evidence->identity().attempt, 7U);
    EXPECT_EQ(evidence->input_handle(), fixture.aligned);
    EXPECT_EQ(evidence->summary().examined_cell_count, 10U);
    EXPECT_EQ(evidence->summary().mapped_cell_count, 8U);
    EXPECT_EQ(evidence->summary().accepted_event_count, 5U);
    EXPECT_EQ(evidence->summary().direct_x_event_count, 2U);
    EXPECT_EQ(evidence->summary().direct_r_event_count, 2U);
    EXPECT_EQ(evidence->summary().x_and_r_event_count, 1U);
    EXPECT_EQ(evidence->summary().alignment_absence_event_count, 2U);

    const auto *x_event = evidence->find(
        fixture.aligned->identity(0, 1, 0));
    const auto *r_event = evidence->find(
        fixture.aligned->identity(0, 2, 0));
    const auto *both_event = evidence->find(
        fixture.aligned->identity(0, 4, 0));
    const auto *absence = evidence->find(
        fixture.aligned->identity(0, 3, 1));
    ASSERT_NE(x_event, nullptr);
    ASSERT_NE(r_event, nullptr);
    ASSERT_NE(both_event, nullptr);
    ASSERT_NE(absence, nullptr);
    EXPECT_TRUE(x_event->direct_x());
    EXPECT_FALSE(x_event->direct_r());
    EXPECT_FALSE(r_event->direct_x());
    EXPECT_TRUE(r_event->direct_r());
    EXPECT_EQ(both_event->origin, pipeline::RtcEvidenceOrigin::x_and_r);
    EXPECT_TRUE(pipeline::has_cause(
        x_event->member_local_causes,
        pipeline::PairedReadoutCause::x_original_invalid));
    EXPECT_FALSE(pipeline::has_cause(
        x_event->member_local_causes,
        pipeline::PairedReadoutCause::r_original_invalid));
    EXPECT_TRUE(pipeline::has_cause(
        r_event->member_local_causes,
        pipeline::PairedReadoutCause::r_original_invalid));
    EXPECT_TRUE(absence->joint_alignment());
    EXPECT_EQ(absence->alignment_absence,
              pipeline::CoincidenceAbsenceReason::no_candidate);
}

TEST(identity_rtc,
     consider_applies_bidirectional_pair_consequence_without_erasing_causes) {
    const auto fixture = identity_fixture();
    const auto evidence = pipeline::learn_identity_rtc(fixture.aligned, 3);
    const auto plan = pipeline::consider_identity_rtc(evidence, 9);
    const auto result = pipeline::apply_identity_rtc(plan, fixture.aligned);

    EXPECT_EQ(plan->identity().evidence, evidence->identity());
    EXPECT_EQ(plan->identity().resolution, 9U);
    EXPECT_EQ(plan->actions().size(), evidence->events().size());

    // Direct r evidence makes the corresponding x occurrence ineligible,
    // while x remains numerically valid and its local cause remains empty.
    EXPECT_TRUE(result.product->member_numerically_valid(
        pipeline::ReadoutMember::x, 0, 2, 0));
    EXPECT_EQ(result.product->member_local_causes(
                  pipeline::ReadoutMember::x, 0, 2, 0),
              pipeline::ReadoutMemberCause::none);
    EXPECT_EQ(result.product->pair_decision(0, 2, 0),
              pipeline::RtcPairDecision::ineligible);
    const auto *from_r = result.product->pair_causal_evidence(0, 2, 0);
    ASSERT_NE(from_r, nullptr);
    EXPECT_TRUE(from_r->direct_r());
    EXPECT_FALSE(from_r->direct_x());

    // Direct x evidence has the symmetric pair-wide consequence for r.
    EXPECT_TRUE(result.product->member_numerically_valid(
        pipeline::ReadoutMember::r, 0, 1, 0));
    EXPECT_EQ(result.product->member_local_causes(
                  pipeline::ReadoutMember::r, 0, 1, 0),
              pipeline::ReadoutMemberCause::none);
    EXPECT_EQ(result.product->pair_decision(0, 1, 0),
              pipeline::RtcPairDecision::ineligible);
    const auto *from_x = result.product->pair_causal_evidence(0, 1, 0);
    ASSERT_NE(from_x, nullptr);
    EXPECT_TRUE(from_x->direct_x());
    EXPECT_FALSE(from_x->direct_r());

    EXPECT_EQ(result.product->pair_decision(0, 0, 0),
              pipeline::RtcPairDecision::eligible);
}

TEST(identity_rtc,
     apply_is_exact_paired_identity_and_owns_no_duplicate_timestream_plane) {
    const auto fixture = identity_fixture();
    const auto plan = identity_plan(fixture.aligned);
    const auto result = pipeline::apply_identity_rtc(plan, fixture.aligned);

    EXPECT_EQ(result.product->input_handle(), fixture.aligned);
    EXPECT_EQ(result.product->plan_handle(), plan);
    EXPECT_EQ(result.product->output_slot_count(), 5U);
    EXPECT_EQ(result.product->output_cell_count(), 10U);
    EXPECT_EQ(result.product->realized_operator().sampling_factor, 1U);
    EXPECT_EQ(result.product->realized_operator().sampling_phase, 0U);
    EXPECT_DOUBLE_EQ(result.product->realized_operator().x_from_x, 1.0);
    EXPECT_DOUBLE_EQ(result.product->realized_operator().x_from_r, 0.0);
    EXPECT_DOUBLE_EQ(result.product->realized_operator().r_from_x, 0.0);
    EXPECT_DOUBLE_EQ(result.product->realized_operator().r_from_r, 1.0);

    for (const auto member :
         {pipeline::ReadoutMember::x, pipeline::ReadoutMember::r}) {
        for (std::size_t slot : {0U, 1U, 2U, 4U}) {
            for (Eigen::Index detector = 0; detector < 2; ++detector) {
                const auto input_value = fixture.aligned->value(
                    member, 0, slot, detector);
                const auto output_value = result.product->value(
                    member, 0, slot, detector);
                ASSERT_TRUE(input_value);
                ASSERT_TRUE(output_value);
                EXPECT_EQ(std::bit_cast<std::uint64_t>(*output_value),
                          std::bit_cast<std::uint64_t>(*input_value));
            }
        }
    }
    EXPECT_FALSE(result.product->value(
        pipeline::ReadoutMember::x, 0, 3, 0));
    EXPECT_EQ(result.product->representative_native_identity(0, 4),
              fixture.aligned->representative_native_identity(0, 4));
    EXPECT_EQ(result.product->representative_interval(0, 4),
              pipeline::NativeOccurrenceInterval({103.8, 104.2}));
    EXPECT_DOUBLE_EQ(result.product->output_time_unix_sec(4), 104.0);

    const auto memory = result.product->memory_evidence();
    EXPECT_EQ(memory.owned_numeric_bytes, 0U);
    EXPECT_EQ(memory.owned_state_plane_bytes, 0U);
    EXPECT_EQ(memory.logical_owned_bytes(), 0U);
    EXPECT_EQ(memory.referenced_parent_count, 1U);
    EXPECT_EQ(result.realization.output_cell_count, 10U);
    EXPECT_EQ(result.realization.pair_ineligible_cell_count, 5U);
    EXPECT_EQ(result.realization.x_numerically_valid_cell_count, 6U);
    EXPECT_EQ(result.realization.r_numerically_valid_cell_count, 6U);
    EXPECT_EQ(result.realization.realized_sampling_factor, 1U);
    EXPECT_LT(sizeof(pipeline::RtcRealization), 128U);
    EXPECT_EQ(plan->evidence_handle()->memory_evidence()
                  .referenced_parent_count,
              1U);
    EXPECT_EQ(plan->memory_evidence().derived_action_bytes,
              plan->actions().size() * sizeof(pipeline::RtcPairAction));
    EXPECT_EQ(plan->memory_evidence().referenced_evidence_count, 1U);
}

TEST(identity_rtc, apply_rejects_a_plan_bound_to_another_input_instance) {
    const auto first = identity_fixture();
    const auto second = identity_fixture();
    const auto first_plan = identity_plan(first.aligned);

    EXPECT_THROW(
        pipeline::apply_identity_rtc(first_plan, second.aligned),
        std::invalid_argument);
}

TEST(identity_rtc,
     scientific_results_are_invariant_to_engineering_chunk_partition) {
    const auto full = identity_fixture(0, 5);
    const auto first_chunk = pipeline::AlignedPairedReadout::admit(
        full.native, full.alignment, 0, 3);
    const auto second_chunk = pipeline::AlignedPairedReadout::admit(
        full.native, full.alignment, 3, 5);

    const auto full_evidence = pipeline::learn_identity_rtc(full.aligned, 1);
    const auto first_evidence =
        pipeline::learn_identity_rtc(first_chunk, 2);
    const auto second_evidence =
        pipeline::learn_identity_rtc(second_chunk, 3);
    std::vector<pipeline::RtcEvidenceEvent> partitioned_events;
    partitioned_events.insert(partitioned_events.end(),
                              first_evidence->events().begin(),
                              first_evidence->events().end());
    partitioned_events.insert(partitioned_events.end(),
                              second_evidence->events().begin(),
                              second_evidence->events().end());
    ASSERT_EQ(partitioned_events.size(), full_evidence->events().size());
    for (std::size_t index = 0; index < partitioned_events.size(); ++index) {
        EXPECT_EQ(partitioned_events[index], full_evidence->events()[index]);
    }

    const auto full_result = pipeline::apply_identity_rtc(
        pipeline::consider_identity_rtc(full_evidence, 1), full.aligned);
    const auto first_result = pipeline::apply_identity_rtc(
        pipeline::consider_identity_rtc(first_evidence, 2), first_chunk);
    const auto second_result = pipeline::apply_identity_rtc(
        pipeline::consider_identity_rtc(second_evidence, 3), second_chunk);
    EXPECT_NE(full_result.product.get(), first_result.product.get());
    EXPECT_NE(full_result.product->plan_handle().get(),
              first_result.product->plan_handle().get());

    for (std::size_t slot = 0; slot < 5; ++slot) {
        const auto &partitioned = slot < 3 ? first_result : second_result;
        for (Eigen::Index detector = 0; detector < 2; ++detector) {
            EXPECT_EQ(partitioned.product->identity(0, slot, detector),
                      full_result.product->identity(0, slot, detector));
            EXPECT_EQ(partitioned.product->value(
                          pipeline::ReadoutMember::x, 0, slot, detector),
                      full_result.product->value(
                          pipeline::ReadoutMember::x, 0, slot, detector));
            EXPECT_EQ(partitioned.product->value(
                          pipeline::ReadoutMember::r, 0, slot, detector),
                      full_result.product->value(
                          pipeline::ReadoutMember::r, 0, slot, detector));
            EXPECT_EQ(partitioned.product->pair_decision(
                          0, slot, detector),
                      full_result.product->pair_decision(
                          0, slot, detector));
        }
    }
}

TEST(identity_rtc, succeeds_without_constructing_or_evaluating_ast) {
    const auto fixture = identity_fixture();
    const auto result = pipeline::apply_identity_rtc(
        identity_plan(fixture.aligned), fixture.aligned);

    ASSERT_NE(result.product, nullptr);
    EXPECT_EQ(result.realization.completion,
              pipeline::RtcCompletionState::complete);
    EXPECT_DOUBLE_EQ(result.product->output_time_unix_sec(0), 100.0);
    EXPECT_EQ(result.product->representative_native_identity(0, 0)
                  ->native_row(),
              10);
}

}  // namespace
