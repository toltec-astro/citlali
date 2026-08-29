#include <citlali/core/pipeline/paired_readout.h>
#include <citlali/core/pipeline/paired_readout_kids_adapter.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;

struct FakeChannel {
    pipeline::PairedReadoutMatrix data;
};

struct FakeSolvedData {
    FakeChannel xs;
    FakeChannel rs;
};

struct FakeSolverResult {
    FakeSolvedData data_out;
};

Eigen::VectorXd vector(std::initializer_list<double> values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index index = 0;
    for (const auto value : values) result(index++) = value;
    return result;
}

std::shared_ptr<const pipeline::PairedReadoutOccurrenceAxis> axis(
    pipeline::TimestreamNetworkId network_id,
    pipeline::TimestreamNativeRow first_row,
    std::initializer_list<double> times,
    std::vector<pipeline::TimestreamPacketCounter> counters,
    double duration_sec = 0.4) {
    auto timing = std::make_shared<const pipeline::NativeNetworkAlignment>(
        network_id, first_row, vector(times), std::move(counters));
    std::vector<pipeline::NativeOccurrenceInterval> intervals;
    intervals.reserve(times.size());
    for (const auto time : times) {
        intervals.push_back(
            {time - duration_sec / 2.0, time + duration_sec / 2.0});
    }
    return std::make_shared<const pipeline::PairedReadoutOccurrenceAxis>(
        std::move(timing), first_row, std::move(intervals));
}

std::shared_ptr<const pipeline::NativeReadoutMappingIdentity> mapping(
    std::string suffix = "0") {
    return std::make_shared<const pipeline::NativeReadoutMappingIdentity>(
        pipeline::NativeReadoutMappingIdentity{
            "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1",
            "producer:" + suffix, "tune:" + suffix, "mapping:" + suffix,
            "iq-to-xr:" + suffix, "raw-x:" + suffix, "raw-r:" + suffix});
}

std::vector<pipeline::PairedReadoutDetectorIdentity> detectors(
    pipeline::TimestreamNetworkId network_id, std::int64_t uid_base,
    Eigen::Index count) {
    std::vector<pipeline::PairedReadoutDetectorIdentity> result;
    for (Eigen::Index column = 0; column < count; ++column) {
        result.push_back({uid_base + column, network_id < 7 ? 0 : 1,
                          network_id, 1000 + network_id, column});
    }
    return result;
}

std::vector<pipeline::ReadoutMemberState> states(
    std::size_t count, bool original_valid = true) {
    return std::vector<pipeline::ReadoutMemberState>(
        count, pipeline::ReadoutMemberState::measured(
                   true, original_valid, true, true));
}

FakeSolverResult solver_result(Eigen::Index rows, Eigen::Index columns,
                               double x_base, double r_base) {
    FakeSolverResult result;
    result.data_out.xs.data.resize(rows, columns);
    result.data_out.rs.data.resize(rows, columns);
    for (Eigen::Index row = 0; row < rows; ++row) {
        for (Eigen::Index column = 0; column < columns; ++column) {
            result.data_out.xs.data(row, column) =
                x_base + 10.0 * row + column;
            result.data_out.rs.data(row, column) =
                r_base + 10.0 * row + column;
        }
    }
    return result;
}

pipeline::PairedReadoutNetwork make_network(
    pipeline::TimestreamNetworkId network_id,
    pipeline::TimestreamNativeRow first_row,
    std::initializer_list<double> times,
    std::vector<pipeline::TimestreamPacketCounter> counters,
    Eigen::Index detector_count, std::int64_t uid_base) {
    auto occurrence_axis = axis(network_id, first_row, times,
                                std::move(counters));
    auto solved = solver_result(static_cast<Eigen::Index>(times.size()),
                                detector_count, 1.0, 101.0);
    const auto cell_count = static_cast<std::size_t>(
        solved.data_out.xs.data.size());
    return pipeline::take_paired_kids_solver_result(
        pipeline::PairedReadoutNetworkIngress{
            std::move(occurrence_axis),
            detectors(network_id, uid_base, detector_count),
            mapping(std::to_string(network_id)), states(cell_count),
            states(cell_count)},
        std::move(solved));
}

TEST(paired_readout,
     atomically_moves_x_and_r_from_one_solver_result_without_repacking) {
    auto solved = solver_result(3, 2, 1.0, 101.0);
    const auto *x_storage = solved.data_out.xs.data.data();
    const auto *r_storage = solved.data_out.rs.data.data();
    auto network = pipeline::take_paired_kids_solver_result(
        pipeline::PairedReadoutNetworkIngress{
            axis(0, 40, {10.0, 11.0, 12.0}, {100, 101, 102}),
            detectors(0, 500, 2), mapping(), states(6), states(6)},
        std::move(solved));

    EXPECT_EQ(network.contiguous_values(pipeline::ReadoutMember::x).data(),
              x_storage);
    EXPECT_EQ(network.contiguous_values(pipeline::ReadoutMember::r).data(),
              r_storage);
    EXPECT_EQ(network.value(pipeline::ReadoutMember::x, 42, 1), 22.0);
    EXPECT_EQ(network.value(pipeline::ReadoutMember::r, 42, 1), 122.0);
    EXPECT_EQ(network.detector(1).output_uid, 501);
    EXPECT_EQ(network.occurrence_axis_handle()->identity(42).native_row(), 42);
    EXPECT_EQ(network.mapping_identity_handle()->x_raw_unit_id, "raw-x:0");
    EXPECT_EQ(network.mapping_identity_handle()->r_raw_unit_id, "raw-r:0");
}

TEST(paired_readout, accepts_zero_as_a_canonical_nonnegative_output_uid) {
    auto solved = solver_result(2, 2, 1.0, 101.0);
    auto network = pipeline::take_paired_kids_solver_result(
        pipeline::PairedReadoutNetworkIngress{
            axis(0, 40, {10.0, 11.0}, {100, 101}),
            detectors(0, 0, 2), mapping(), states(4), states(4)},
        std::move(solved));

    EXPECT_EQ(network.detector(0).output_uid, 0);
    EXPECT_EQ(network.detector(1).output_uid, 1);
}

TEST(paired_readout,
     preserves_independent_member_validity_and_derives_pair_wide_causes) {
    auto solved = solver_result(2, 1, 1.0, 101.0);
    solved.data_out.rs.data(1, 0) =
        std::numeric_limits<double>::quiet_NaN();
    auto x_states = states(2);
    auto r_states = states(2);
    r_states[1] = pipeline::ReadoutMemberState::measured(
        true, false, true, false);
    auto network = pipeline::take_paired_kids_solver_result(
        pipeline::PairedReadoutNetworkIngress{
            axis(7, 70, {20.0, 21.0}, {700, 701}), detectors(7, 900, 1),
            mapping("7"), std::move(x_states), std::move(r_states)},
        std::move(solved));

    EXPECT_TRUE(network.state(pipeline::ReadoutMember::x, 71, 0).valid());
    EXPECT_FALSE(network.state(pipeline::ReadoutMember::r, 71, 0).valid());
    EXPECT_TRUE(network.pair_available(71, 0));
    EXPECT_FALSE(network.pair_valid(71, 0));
    EXPECT_EQ(network.state(pipeline::ReadoutMember::x, 71, 0).causes(),
              pipeline::ReadoutMemberCause::none);
    const auto r_causes =
        network.state(pipeline::ReadoutMember::r, 71, 0).causes();
    EXPECT_TRUE(pipeline::has_cause(
        r_causes, pipeline::ReadoutMemberCause::producer_invalid));
    EXPECT_TRUE(pipeline::has_cause(
        r_causes, pipeline::ReadoutMemberCause::nonfinite_payload));
    const auto pair_causes = network.pair_causes(71, 0);
    EXPECT_TRUE(pipeline::has_cause(
        pair_causes, pipeline::PairedReadoutCause::r_original_invalid));
    EXPECT_TRUE(pipeline::has_cause(
        pair_causes, pipeline::PairedReadoutCause::r_nonfinite));
    EXPECT_FALSE(pipeline::has_cause(
        pair_causes, pipeline::PairedReadoutCause::x_original_invalid));
    EXPECT_EQ(network.state(pipeline::ReadoutMember::x, 71, 0).origin(),
              pipeline::ReadoutMemberOrigin::measured);
    EXPECT_EQ(network.state(pipeline::ReadoutMember::r, 71, 0).origin(),
              pipeline::ReadoutMemberOrigin::measured);
}

TEST(paired_readout,
     preserves_native_per_network_axes_without_common_analysis_grid_projection) {
    const pipeline::NativeObservationScope scope{152390, 0, 4};
    std::vector<pipeline::PairedReadoutNetwork> networks;
    networks.push_back(
        make_network(7, 700, {100.15, 101.35}, {10, 12}, 1, 900));
    networks.push_back(make_network(
        0, 40, {100.0, 100.8, 101.6}, {1, 2, 3}, 2, 500));
    auto product = pipeline::PairedReadout::admit(
        scope, {0, 7}, std::move(networks));

    ASSERT_EQ(product->network_count(), 2U);
    EXPECT_EQ(product->network(0).occurrence_count(), 3);
    EXPECT_EQ(product->network(7).occurrence_count(), 2);
    EXPECT_DOUBLE_EQ(
        product->network(0)
            .occurrence_axis_handle()
            ->identity(41)
            .reconstructed_time_unix_sec(),
        100.8);
    EXPECT_DOUBLE_EQ(
        product->network(7)
            .occurrence_axis_handle()
            ->identity(701)
            .reconstructed_time_unix_sec(),
        101.35);
    EXPECT_EQ(product->cardinality().network_count, 2U);
    EXPECT_EQ(product->cardinality().detector_count, 3U);
    EXPECT_EQ(product->cardinality().native_occurrence_count, 5U);
    EXPECT_EQ(product->cardinality().detector_occurrence_count, 8U);
}

TEST(paired_readout,
     retains_primitive_interval_original_validity_and_support_facts) {
    auto solved = solver_result(2, 1, 1.0, 101.0);
    auto x_states = states(2);
    x_states[1] = pipeline::ReadoutMemberState::measured(
        true, false, true, true);
    auto r_states = states(2);
    r_states[0] = pipeline::ReadoutMemberState::measured(
        true, true, false, true);
    auto network = pipeline::take_paired_kids_solver_result(
        pipeline::PairedReadoutNetworkIngress{
            axis(0, 10, {50.0, 51.0}, {20, 21}, 0.25),
            detectors(0, 100, 1), mapping(), std::move(x_states),
            std::move(r_states)},
        std::move(solved));

    EXPECT_DOUBLE_EQ(
        network.occurrence_axis_handle()->interval(10).duration_sec(),
        0.25);
    EXPECT_FALSE(
        network.state(pipeline::ReadoutMember::x, 11, 0).original_valid());
    EXPECT_TRUE(network.state(pipeline::ReadoutMember::x, 11, 0)
                    .in_acquisition_support());
    EXPECT_TRUE(
        network.state(pipeline::ReadoutMember::r, 10, 0).original_valid());
    EXPECT_FALSE(network.state(pipeline::ReadoutMember::r, 10, 0)
                     .in_acquisition_support());
}

TEST(paired_readout,
     fails_closed_for_missing_partner_shape_identity_and_participant_errors) {
    auto missing_r = solver_result(2, 1, 1.0, 101.0);
    missing_r.data_out.rs.data.resize(0, 0);
    EXPECT_THROW(
        pipeline::take_paired_kids_solver_result(
            pipeline::PairedReadoutNetworkIngress{
                axis(0, 10, {1.0, 2.0}, {1, 2}), detectors(0, 100, 1),
                mapping(), states(2), states(2)},
            std::move(missing_r)),
        std::invalid_argument);

    auto wrong_shape = solver_result(2, 1, 1.0, 101.0);
    wrong_shape.data_out.rs.data.resize(1, 2);
    EXPECT_THROW(
        pipeline::take_paired_kids_solver_result(
            pipeline::PairedReadoutNetworkIngress{
                axis(0, 10, {1.0, 2.0}, {1, 2}), detectors(0, 100, 1),
                mapping(), states(2), states(2)},
            std::move(wrong_shape)),
        std::invalid_argument);

    auto reordered_detectors = detectors(0, 100, 2);
    std::swap(reordered_detectors[0], reordered_detectors[1]);
    auto reordered = solver_result(2, 2, 1.0, 101.0);
    EXPECT_THROW(
        pipeline::take_paired_kids_solver_result(
            pipeline::PairedReadoutNetworkIngress{
                axis(0, 10, {1.0, 2.0}, {1, 2}),
                std::move(reordered_detectors), mapping(), states(4),
                states(4)},
            std::move(reordered)),
        std::invalid_argument);

    auto ambiguous_mapping = mapping();
    auto mutable_mapping = std::make_shared<pipeline::NativeReadoutMappingIdentity>(
        *ambiguous_mapping);
    mutable_mapping->mapping_revision.clear();
    auto solved = solver_result(2, 1, 1.0, 101.0);
    EXPECT_THROW(
        pipeline::take_paired_kids_solver_result(
            pipeline::PairedReadoutNetworkIngress{
                axis(0, 10, {1.0, 2.0}, {1, 2}), detectors(0, 100, 1),
                std::move(mutable_mapping), states(2), states(2)},
            std::move(solved)),
        std::invalid_argument);

    const pipeline::NativeObservationScope scope{152390, 0, 4};
    std::vector<pipeline::PairedReadoutNetwork> missing_network;
    missing_network.push_back(
        make_network(0, 40, {1.0}, {1}, 1, 500));
    EXPECT_THROW(
        pipeline::PairedReadout::admit(
            scope, {0, 7}, std::move(missing_network)),
        std::invalid_argument);
}

TEST(paired_readout,
     reports_basic_cardinality_and_logical_owned_memory_without_lineage) {
    const pipeline::NativeObservationScope scope{152390, 0, 4};
    std::vector<pipeline::PairedReadoutNetwork> networks;
    networks.push_back(make_network(
        0, 40, {1.0, 2.0, 3.0}, {1, 2, 3}, 2, 500));
    auto product = pipeline::PairedReadout::admit(
        scope, {0}, std::move(networks));
    const auto cardinality = product->cardinality();
    const auto memory = product->memory_evidence();

    EXPECT_EQ(cardinality.detector_occurrence_count, 6U);
    EXPECT_EQ(memory.numeric_payload_bytes, 12U * sizeof(double));
    EXPECT_EQ(memory.member_state_bytes,
              12U * sizeof(pipeline::ReadoutMemberState));
    EXPECT_EQ(memory.occurrence_interval_bytes,
              3U * sizeof(pipeline::NativeOccurrenceInterval));
    EXPECT_EQ(memory.detector_axis_bytes,
              2U * sizeof(pipeline::PairedReadoutDetectorIdentity));
    EXPECT_GT(memory.identity_text_bytes, 0U);
    EXPECT_EQ(memory.referenced_native_axis_count, 1U);
    EXPECT_GT(memory.logical_owned_bytes(), memory.numeric_payload_bytes);
}

}  // namespace
