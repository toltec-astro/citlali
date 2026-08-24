#include <citlali/core/pipeline/native_observation_carriers.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;

Eigen::VectorXd values(std::initializer_list<double> input) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(input.size()));
    Eigen::Index index = 0;
    for (const double value : input) result(index++) = value;
    return result;
}

pipeline::NativeNetworkAlignment network(
    pipeline::TimestreamNetworkId network_id,
    pipeline::TimestreamNativeRow first_row,
    std::initializer_list<double> times,
    std::vector<pipeline::TimestreamPacketCounter> counters) {
    return pipeline::NativeNetworkAlignment{
        network_id, first_row, values(times), std::move(counters)};
}

std::shared_ptr<const pipeline::NativeAlignmentPlan> alignment_plan(
    pipeline::NativeObservationScope scope,
    std::vector<pipeline::NativeNetworkAlignment> networks,
    Eigen::VectorXd common_times,
    std::map<pipeline::TimestreamNetworkId,
             std::vector<pipeline::NativeSlotAssociation>> associations) {
    return std::make_shared<const pipeline::NativeAlignmentPlan>(
        scope, std::move(networks), std::move(common_times),
        std::move(associations));
}

std::shared_ptr<const pipeline::RawTelescopeTrajectory> raw_telescope() {
    pipeline::NativeTelescopeData data;
    data["TelTime"] = values({0.0, 5.0});
    data["TelUTC"] = values({0.0, 5.0});
    data["TelAzAct"] = values({10.0, 20.0});
    data["TelElAct"] = values({30.0, 40.0});
    data["Hold"] = values({0.0, 0.0});
    return std::make_shared<const pipeline::RawTelescopeTrajectory>(
        std::move(data));
}

pipeline::NativePointingOffsetModel offset_model() {
    pipeline::NativePointingOffsetsArcsec offsets;
    offsets[citlali::config::pointing_axis_az()] = values({0.0, 10.0});
    offsets[citlali::config::pointing_axis_alt()] =
        values({100.0, 200.0});
    return pipeline::NativePointingOffsetModel{
        std::move(offsets), values({0.0, 5.0})};
}

std::shared_ptr<const pipeline::NativeAlignmentPlan>
two_network_alignment(pipeline::NativeObservationScope scope,
                      bool reverse_input = false) {
    std::vector<pipeline::NativeNetworkAlignment> networks;
    networks.push_back(network(0, 10, {1.0, 3.0}, {100, 101}));
    networks.push_back(network(7, 70, {1.0, 3.0}, {700, 701}));
    if (reverse_input) std::reverse(networks.begin(), networks.end());
    return alignment_plan(
        scope, std::move(networks), values({1.0, 3.0}),
        {{0, pipeline::make_direct_native_slot_associations(10, 2)},
         {7, pipeline::make_direct_native_slot_associations(70, 2)}});
}

TEST(sci_align_native_carriers,
     reconstructs_delivered_times_and_preserves_subcadence_drop) {
    Eigen::MatrixXd timestamps = Eigen::MatrixXd::Zero(3, 6);
    timestamps.col(0).setConstant(1700000000.0);
    timestamps.col(1) = values({1.0, 2.0, 4.0});
    timestamps.col(3) = values({10.0, 11.0, 13.0});

    const auto aligned = pipeline::make_native_network_alignment(
        0, 40, timestamps, 1.0e8, 0.0);
    ASSERT_EQ(aligned.row_count(), 3);
    EXPECT_DOUBLE_EQ(
        aligned.identity(40).reconstructed_time_unix_sec(),
        1700000000.0);
    EXPECT_DOUBLE_EQ(
        aligned.identity(41).reconstructed_time_unix_sec(),
        1700000001.0);
    EXPECT_DOUBLE_EQ(
        aligned.identity(42).reconstructed_time_unix_sec(),
        1700000003.0);

    Eigen::VectorXi legacy_mask(4);
    legacy_mask << 1, 1, 0, 1;
    const auto associations =
        pipeline::make_gap_native_slot_associations(
            aligned,
            values({1700000000.0, 1700000001.0,
                    1700000002.0, 1700000003.0}),
            legacy_mask, 1.0);
    ASSERT_EQ(associations.size(), 4U);
    EXPECT_EQ(associations[0].native_row, 40);
    EXPECT_EQ(associations[1].native_row, 41);
    EXPECT_FALSE(associations[2].mapped());
    EXPECT_EQ(associations[2].absence_reason,
              pipeline::CoincidenceAbsenceReason::no_candidate);
    EXPECT_EQ(associations[3].native_row, 42);

    const auto runs = pipeline::partition_native_contiguous_runs(
        aligned, 40, 43);
    ASSERT_EQ(runs.size(), 2U);
    EXPECT_EQ(runs[0].first_native_row, 40);
    EXPECT_EQ(runs[0].past_last_native_row, 42);
    EXPECT_EQ(runs[1].first_native_row, 42);
    EXPECT_EQ(runs[1].past_last_native_row, 43);
    ASSERT_TRUE(runs[0].boundary_after.counter_discontinuity);
    EXPECT_EQ(runs[0].boundary_after.counter_discontinuity->before_counter,
              11);
    EXPECT_EQ(runs[0].boundary_after.counter_discontinuity->after_counter,
              13);
}

TEST(sci_align_native_carriers,
     pins_round_half_away_from_zero_and_inclusive_half_dt_edge) {
    const auto aligned = network(0, 5, {1.0}, {10});
    Eigen::VectorXi legacy_mask(2);
    legacy_mask << 0, 1;
    const auto associations =
        pipeline::make_gap_native_slot_associations(
            aligned, values({0.0, 2.0}), legacy_mask, 2.0);
    ASSERT_EQ(associations.size(), 2U);
    EXPECT_FALSE(associations[0].mapped());
    EXPECT_EQ(associations[1].native_row, 5);
    EXPECT_DOUBLE_EQ(std::abs(
        aligned.identity(5).reconstructed_time_unix_sec() - 2.0), 1.0);

    Eigen::VectorXi collision_mask(2);
    collision_mask << 1, 0;
    const auto collision = network(0, 0, {0.8, 0.9}, {1, 2});
    EXPECT_THROW(
        pipeline::make_gap_native_slot_associations(
            collision, values({0.0, 2.0}), collision_mask, 2.0),
        std::logic_error);

    Eigen::VectorXi wrong_mask(2);
    wrong_mask << 0, 0;
    EXPECT_THROW(
        pipeline::make_gap_native_slot_associations(
            network(0, 0, {0.0}, {1}), values({0.0, 1.0}),
            wrong_mask, 1.0),
        std::logic_error);
}

TEST(sci_align_native_carriers,
     partitions_repeat_decrease_jump_rollover_and_scan_boundaries) {
    const auto max =
        std::numeric_limits<pipeline::TimestreamPacketCounter>::max();
    const auto min =
        std::numeric_limits<pipeline::TimestreamPacketCounter>::min();
    const auto aligned = network(
        3, 100, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        {10, 11, 11, 9, 12, max, min});
    const auto runs = pipeline::partition_native_contiguous_runs(
        aligned, 101, 106);
    ASSERT_EQ(runs.size(), 5U);
    EXPECT_TRUE(runs.front().boundary_before.scan_boundary);
    EXPECT_FALSE(runs.front().boundary_before.stream_boundary);
    EXPECT_TRUE(runs.back().boundary_after.scan_boundary);
    EXPECT_FALSE(runs.back().boundary_after.stream_boundary);

    for (const auto &run : runs) {
        for (auto row = run.first_native_row + 1;
             row < run.past_last_native_row; ++row) {
            EXPECT_FALSE(aligned.discontinuity_between(row - 1, row));
        }
    }
    const auto rollover = aligned.discontinuity_between(105, 106);
    ASSERT_TRUE(rollover);
    EXPECT_EQ(rollover->before_counter, max);
    EXPECT_EQ(rollover->after_counter, min);
    EXPECT_FALSE(pipeline::packet_counters_are_contiguous(max, min));
}

TEST(sci_align_native_carriers,
     canonicalizes_network_input_order_without_changing_identity) {
    const pipeline::NativeObservationScope scope{148670, 0, 2};
    const auto first = two_network_alignment(scope, false);
    const auto reversed = two_network_alignment(scope, true);
    EXPECT_EQ(first->participant_network_ids(),
              (std::vector<pipeline::TimestreamNetworkId>{0, 7}));
    EXPECT_EQ(first->participant_network_ids(),
              reversed->participant_network_ids());
    for (const auto network_id : first->participant_network_ids()) {
        const auto &lhs = first->network(network_id);
        const auto &rhs = reversed->network(network_id);
        EXPECT_TRUE((lhs.reconstructed_times_unix_sec().array() ==
                     rhs.reconstructed_times_unix_sec().array()).all());
        for (std::size_t slot = 0; slot < first->slot_count(); ++slot) {
            EXPECT_EQ(first->association(network_id, slot),
                      reversed->association(network_id, slot));
        }
    }

    EXPECT_THROW(
        alignment_plan(
            scope,
            {network(0, 0, {1.0}, {1}),
             network(0, 1, {2.0}, {2})},
            values({1.0}),
            {{0, pipeline::make_direct_native_slot_associations(0, 1)}}),
        std::invalid_argument);
}

TEST(sci_align_native_carriers,
     evaluates_telescope_and_detector_offsets_at_exact_native_times) {
    const pipeline::NativeObservationScope scope{148670, 0, 2};
    const auto alignment = two_network_alignment(scope, true);
    const auto pointing = pipeline::make_native_pointing_plan(
        alignment, raw_telescope(), offset_model());
    EXPECT_TRUE(pointing->bound_to(alignment));
    EXPECT_EQ(pointing->participant_network_ids(),
              (std::vector<pipeline::TimestreamNetworkId>{0, 7}));

    for (const auto network_id : pointing->participant_network_ids()) {
        const auto &network_pointing = pointing->network(network_id);
        const auto first_row = network_pointing.first_native_row();
        EXPECT_DOUBLE_EQ(
            network_pointing.telescope_series("TelTime")(0), 1.0);
        EXPECT_DOUBLE_EQ(
            network_pointing.identity(first_row)
                .reconstructed_time_unix_sec(),
            1.0);
        EXPECT_DOUBLE_EQ(
            network_pointing.telescope_series("TelAzAct")(0), 12.0);
        EXPECT_DOUBLE_EQ(
            network_pointing.telescope_series("TelElAct")(1), 36.0);
        EXPECT_DOUBLE_EQ(
            network_pointing.pointing_offset_arcsec(
                citlali::config::pointing_axis_az())(0),
            2.0);
        EXPECT_DOUBLE_EQ(
            network_pointing.pointing_offset_arcsec(
                citlali::config::pointing_axis_alt())(1),
            160.0);
    }
    EXPECT_TRUE((pointing->network(0)
                     .telescope_series("TelAzAct")
                     .array() ==
                 pointing->network(7)
                     .telescope_series("TelAzAct")
                     .array())
                    .all());
}

TEST(sci_align_native_carriers,
     constant_offsets_cover_native_times_outside_legacy_common_support) {
    pipeline::NativePointingOffsetsArcsec offsets;
    offsets[citlali::config::pointing_axis_az()] = values({0.0});
    offsets[citlali::config::pointing_axis_alt()] = values({-2.5});
    const pipeline::NativePointingOffsetModel constant{
        std::move(offsets), values({1.0, 3.0})};

    const auto evaluated = constant.evaluate_at(values({0.5, 3.5}));
    EXPECT_TRUE((evaluated.at(citlali::config::pointing_axis_az()).array() ==
                 0.0).all());
    EXPECT_TRUE((evaluated.at(citlali::config::pointing_axis_alt()).array() ==
                 -2.5).all());

    EXPECT_THROW(
        offset_model().evaluate_at(values({-0.5, 3.5})),
        std::out_of_range);
}

TEST(sci_align_native_carriers,
     observation_publication_rejects_absent_stale_and_foreign_atomically) {
    const pipeline::NativeObservationScope scope{148670, 0, 2};
    const auto alignment = two_network_alignment(scope);
    const auto pointing = pipeline::make_native_pointing_plan(
        alignment, raw_telescope(), offset_model());
    pipeline::NativeObservationCarrierSlot slot{scope};
    slot.publish(alignment, pointing);
    const auto accepted = slot.handle();
    ASSERT_TRUE(accepted);

    EXPECT_THROW(slot.publish(nullptr, pointing), std::invalid_argument);
    EXPECT_EQ(slot.handle().get(), accepted.get());

    const auto stale_alignment = two_network_alignment(scope, true);
    EXPECT_THROW(slot.publish(stale_alignment, pointing),
                 std::invalid_argument);
    EXPECT_EQ(slot.handle().get(), accepted.get());

    const pipeline::NativeObservationScope foreign_scope{148671, 0, 2};
    const auto foreign_alignment = two_network_alignment(foreign_scope);
    const auto foreign_pointing = pipeline::make_native_pointing_plan(
        foreign_alignment, raw_telescope(), offset_model());
    EXPECT_THROW(slot.publish(foreign_alignment, foreign_pointing),
                 std::invalid_argument);
    EXPECT_EQ(slot.handle().get(), accepted.get());

    auto nonfinite_data = raw_telescope()->telescope_data();
    nonfinite_data["TelAzAct"](0) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        (void)pipeline::RawTelescopeTrajectory{std::move(nonfinite_data)},
        std::invalid_argument);
    EXPECT_EQ(slot.handle().get(), accepted.get());
}

TEST(sci_align_native_carriers,
     rejects_bad_counter_and_timestamp_candidates_before_publication) {
    Eigen::MatrixXd timestamps = Eigen::MatrixXd::Zero(2, 6);
    timestamps.col(0).setConstant(1700000000.0);
    timestamps.col(1) = values({1.0, 2.0});
    timestamps.col(3) = values({1.0, 2.5});
    EXPECT_THROW(
        pipeline::make_native_network_alignment(
            0, 0, timestamps, 1.0e8, 0.0),
        std::invalid_argument);

    timestamps(1, 3) = 1.0e30;
    EXPECT_THROW(
        pipeline::make_native_network_alignment(
            0, 0, timestamps, 1.0e8, 0.0),
        std::invalid_argument);
    timestamps(1, 3) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        pipeline::make_native_network_alignment(
            0, 0, timestamps, 1.0e8, 0.0),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::make_native_network_alignment(
            0, 0, Eigen::MatrixXd::Zero(0, 6), 1.0e8, 0.0),
        std::invalid_argument);
    EXPECT_THROW(
        network(0, 0, {1.0, 1.0}, {1, 2}),
        std::invalid_argument);

    auto bad_reference = values({1.0, 2.0});
    bad_reference(1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        pipeline::make_gap_native_slot_associations(
            network(0, 0, {1.0, 2.0}, {1, 2}), bad_reference,
            Eigen::VectorXi::Ones(2), 1.0),
        std::invalid_argument);
}

}  // namespace
