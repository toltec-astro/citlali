#include <citlali/core/pipeline/timestream_native_paired_readout.h>

#include <gtest/gtest.h>

#include <algorithm>
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

Eigen::VectorXd vector(std::initializer_list<double> values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index index = 0;
    for (const auto value : values) result(index++) = value;
    return result;
}

pipeline::NativeReadoutCoordinateAuthority coordinate_authority(
    const std::string &name) {
    return pipeline::NativeReadoutCoordinateAuthority{
        name + ":meaning", name + ":unit-or-scale", name + ":sign",
        name + ":reference", name + ":normalization", name + ":metric",
        name + ":validity-domain", name + ":uncertainty-state"};
}

std::shared_ptr<const pipeline::NativeReadoutMappingAuthority> mapping(
    pipeline::TimestreamNetworkId network_id,
    const std::string &suffix = "0") {
    auto result = std::make_shared<pipeline::NativeReadoutMappingAuthority>();
    result->network_id = network_id;
    result->producer_id = "producer-family:" + suffix;
    result->producer_instance_id = "producer-instance:" + suffix;
    result->producer_interface_id =
        std::string{pipeline::native_paired_xr_producer_interface_id};
    result->mapping_record_id = "mapping-record:" + suffix;
    result->mapping_revision_id = "mapping-revision:" + suffix;
    result->tune_id = "tune:" + suffix;
    result->readout_interface_id = "readout-interface:" + suffix;
    result->input_coordinate_record_id = "native-iq:" + suffix;
    result->transform_id = "iq-to-xr:" + suffix;
    result->transform_representation_id = "jacobian-record:" + suffix;
    result->applicability_domain_id = "applicability:" + suffix;
    result->event_time_epoch_meaning_id =
        "integration-centroid-unix:" + suffix;
    result->native_time_unit_id = "s:" + suffix;
    result->native_cadence_record_id = "native-cadence:" + suffix;
    result->native_time_validity_state_id =
        "native-time-validity:" + suffix;
    result->timing_uncertainty_state_id =
        "timing-uncertainty-record:" + suffix;
    result->parent_readout_record_id = "native-iq-occurrences:" + suffix;
    result->paired_xr_record_id = "native-xr-occurrences:" + suffix;
    result->runtime_binding_rule_id = "runtime-binding:" + suffix;
    result->compatibility_rule_id = "compatibility:" + suffix;
    result->failure_semantics_id = "fail-closed:" + suffix;
    result->x = coordinate_authority("x:" + suffix);
    result->r = coordinate_authority("r:" + suffix);
    return result;
}

std::shared_ptr<const pipeline::NativePairedReadoutOccurrenceAxis> axis(
    pipeline::TimestreamNetworkId network_id,
    pipeline::TimestreamNativeRow first_row,
    std::initializer_list<double> times,
    std::vector<pipeline::TimestreamPacketCounter> counters,
    double duration_sec = 0.4) {
    auto timing =
        std::make_shared<const pipeline::NativeNetworkAlignment>(
            network_id, first_row, vector(times), std::move(counters));
    std::vector<pipeline::NativePairedReadoutOccurrenceBinding> occurrences;
    occurrences.reserve(times.size());
    std::int64_t index = 0;
    for (const auto time : times) {
        occurrences.push_back(
            {10000 + first_row + index, 20000 + first_row + index,
             {time - duration_sec / 2.0, time + duration_sec / 2.0}});
        ++index;
    }
    return std::make_shared<const
                            pipeline::NativePairedReadoutOccurrenceAxis>(
        std::move(timing), first_row, std::move(occurrences));
}

std::vector<pipeline::NativeReadoutDetectorBinding> detectors(
    pipeline::TimestreamNetworkId network_id,
    const std::string &association,
    Eigen::Index count) {
    std::vector<pipeline::NativeReadoutDetectorBinding> result;
    result.reserve(static_cast<std::size_t>(count));
    for (Eigen::Index column = 0; column < count; ++column) {
        result.push_back(
            {network_id, column,
             "detector:" + std::to_string(network_id) + ":" +
                 std::to_string(column),
             association,
             "tone:" + std::to_string(network_id) + ":" +
                 std::to_string(column)});
    }
    return result;
}

pipeline::NativePairedReadoutMatrix values(Eigen::Index rows,
                                            Eigen::Index columns,
                                            double base) {
    pipeline::NativePairedReadoutMatrix result(rows, columns);
    for (Eigen::Index row = 0; row < rows; ++row) {
        for (Eigen::Index column = 0; column < columns; ++column) {
            result(row, column) = base + 10.0 * row + column;
        }
    }
    return result;
}

std::vector<pipeline::NativeReadoutCoordinateState> states(
    std::size_t count) {
    return std::vector<pipeline::NativeReadoutCoordinateState>(
        count, pipeline::NativeReadoutCoordinateState::measured(
                   true, true, true, true));
}

pipeline::NativePairedReadoutNetwork make_network(
    pipeline::TimestreamNetworkId network_id,
    pipeline::TimestreamNativeRow first_row,
    std::initializer_list<double> times,
    std::vector<pipeline::TimestreamPacketCounter> counters,
    Eigen::Index detector_count) {
    const auto row_count = static_cast<Eigen::Index>(times.size());
    const auto cell_count = static_cast<std::size_t>(
        row_count * detector_count);
    const auto suffix = std::to_string(network_id);
    return pipeline::NativePairedReadoutNetwork::admit(
        axis(network_id, first_row, times, std::move(counters)),
        detectors(network_id, "association:" + suffix, detector_count),
        mapping(network_id, suffix), values(row_count, detector_count, 1.0),
        values(row_count, detector_count, 101.0), states(cell_count),
        states(cell_count));
}

TEST(native_paired_readout,
     atomically_owns_ordered_xr_with_exact_native_and_mapping_identity) {
    auto x = values(3, 2, 1.0);
    auto r = values(3, 2, 101.0);
    const auto *x_storage = x.data();
    const auto *r_storage = r.data();
    auto network = pipeline::NativePairedReadoutNetwork::admit(
        axis(0, 40, {10.0, 11.0, 12.0}, {100, 102, 103}),
        detectors(0, "apt-relation:148670", 2), mapping(0),
        std::move(x), std::move(r), states(6), states(6));

    EXPECT_EQ(network.values(pipeline::NativeReadoutCoordinate::x).data(),
              x_storage);
    EXPECT_EQ(network.values(pipeline::NativeReadoutCoordinate::r).data(),
              r_storage);
    EXPECT_DOUBLE_EQ(
        network.value(pipeline::NativeReadoutCoordinate::x, 42, 1), 22.0);
    EXPECT_DOUBLE_EQ(
        network.value(pipeline::NativeReadoutCoordinate::r, 42, 1), 122.0);
    EXPECT_EQ(network.detector(1).detector_occurrence_id, "detector:0:1");
    EXPECT_EQ(network.detector(1).detector_association_record_id,
              "apt-relation:148670");
    EXPECT_EQ(network.detector(1).tone_or_channel_id, "tone:0:1");

    const auto &native_axis = network.occurrence_axis();
    EXPECT_EQ(native_axis.native_identity(42).native_row(), 42);
    EXPECT_EQ(native_axis.packet_counter(42), 103);
    EXPECT_EQ(native_axis.occurrence(42).parent_readout_occurrence_key,
              10042);
    EXPECT_EQ(native_axis.occurrence(42).paired_xr_occurrence_key, 20042);
    EXPECT_NEAR(
        native_axis.occurrence(42).integration_support.duration_sec(), 0.4,
        1.0e-12);
    const auto runs = native_axis.contiguous_runs();
    ASSERT_EQ(runs.size(), 2U);
    EXPECT_EQ(runs[0].first_native_row, 40);
    EXPECT_EQ(runs[0].past_last_native_row, 41);
    EXPECT_EQ(runs[1].first_native_row, 41);
    EXPECT_EQ(runs[1].past_last_native_row, 43);

    const auto &authority = network.mapping_authority();
    EXPECT_EQ(authority.producer_interface_id,
              pipeline::native_paired_xr_producer_interface_id);
    EXPECT_EQ(authority.mapping_record_id, "mapping-record:0");
    EXPECT_EQ(authority.transform_representation_id, "jacobian-record:0");
    EXPECT_EQ(authority.event_time_epoch_meaning_id,
              "integration-centroid-unix:0");
    EXPECT_EQ(authority.native_time_unit_id, "s:0");
    EXPECT_EQ(authority.native_cadence_record_id, "native-cadence:0");
    EXPECT_EQ(authority.native_time_validity_state_id,
              "native-time-validity:0");
    EXPECT_EQ(authority.x.unit_or_scale_id, "x:0:unit-or-scale");
    EXPECT_EQ(authority.x.sign_convention_id, "x:0:sign");
    EXPECT_EQ(authority.r.reference_point_id, "r:0:reference");
    EXPECT_EQ(authority.r.normalization_id, "r:0:normalization");
}

TEST(native_paired_readout,
     preserves_coordinate_local_availability_validity_and_pair_causes) {
    auto x = values(2, 1, 1.0);
    auto r = values(2, 1, 101.0);
    r(1, 0) = std::numeric_limits<double>::quiet_NaN();
    auto x_states = states(2);
    auto r_states = states(2);
    x_states[0] = pipeline::NativeReadoutCoordinateState::measured(
        true, false, true, true);
    r_states[1] = pipeline::NativeReadoutCoordinateState::measured(
        false, false, true, false);
    auto network = pipeline::NativePairedReadoutNetwork::admit(
        axis(7, 70, {20.0, 21.0}, {700, 701}),
        detectors(7, "association:7", 1), mapping(7, "7"),
        std::move(x), std::move(r), std::move(x_states),
        std::move(r_states));

    EXPECT_TRUE(network.state(
        pipeline::NativeReadoutCoordinate::x, 71, 0).valid());
    EXPECT_FALSE(network.state(
        pipeline::NativeReadoutCoordinate::r, 71, 0).payload_available());
    EXPECT_FALSE(network.pair_available(71, 0));
    EXPECT_FALSE(network.pair_valid(71, 0));
    EXPECT_TRUE(network.occurrence_axis()
                    .occurrence(71)
                    .integration_support.duration_sec() > 0.0);
    EXPECT_EQ(network.occurrence_axis()
                  .occurrence(71)
                  .paired_xr_occurrence_key,
              20071);

    const auto r_causes = network.state(
        pipeline::NativeReadoutCoordinate::r, 71, 0).causes();
    EXPECT_TRUE(pipeline::has_cause(
        r_causes,
        pipeline::NativeReadoutCoordinateCause::producer_unavailable));
    EXPECT_TRUE(pipeline::has_cause(
        r_causes, pipeline::NativeReadoutCoordinateCause::producer_invalid));
    EXPECT_TRUE(pipeline::has_cause(
        r_causes, pipeline::NativeReadoutCoordinateCause::nonfinite_payload));
    const auto pair_causes = network.pair_causes(71, 0);
    EXPECT_TRUE(pipeline::has_cause(
        pair_causes, pipeline::NativePairedReadoutCause::r_unavailable));
    EXPECT_TRUE(pipeline::has_cause(
        pair_causes,
        pipeline::NativePairedReadoutCause::r_producer_invalid));
    EXPECT_TRUE(pipeline::has_cause(
        pair_causes, pipeline::NativePairedReadoutCause::r_nonfinite));
    EXPECT_FALSE(pipeline::has_cause(
        pair_causes, pipeline::NativePairedReadoutCause::x_unavailable));
    EXPECT_EQ(network.state(
                  pipeline::NativeReadoutCoordinate::r, 71, 0).origin(),
              pipeline::NativeReadoutOrigin::measured);
}

TEST(native_paired_readout,
     retains_independent_network_native_axes_without_a_common_grid) {
    const pipeline::NativeObservationScope scope{152390, 0, 4};
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(
        make_network(7, 700, {100.15, 101.35}, {10, 12}, 1));
    networks.push_back(
        make_network(0, 40, {100.0, 100.8, 101.6}, {1, 2, 3}, 2));
    auto observation = pipeline::NativePairedReadoutObservation::admit(
        scope, {0, 7}, std::move(networks));

    EXPECT_EQ(observation.scope(), scope);
    EXPECT_EQ(observation.participant_network_ids(),
              (std::vector<pipeline::TimestreamNetworkId>{0, 7}));
    EXPECT_EQ(observation.network(0).occurrence_count(), 3);
    EXPECT_EQ(observation.network(7).occurrence_count(), 2);
    EXPECT_DOUBLE_EQ(observation.network(0)
                         .occurrence_axis()
                         .native_identity(41)
                         .reconstructed_time_unix_sec(),
                     100.8);
    EXPECT_DOUBLE_EQ(observation.network(7)
                         .occurrence_axis()
                         .native_identity(701)
                         .reconstructed_time_unix_sec(),
                     101.35);
    EXPECT_EQ(observation.cardinality().network_count, 2U);
    EXPECT_EQ(observation.cardinality().detector_count, 3U);
    EXPECT_EQ(observation.cardinality().native_occurrence_count, 5U);
    EXPECT_EQ(observation.cardinality().detector_occurrence_count, 8U);
    EXPECT_THROW(observation.network(3), std::out_of_range);
}

TEST(native_paired_readout,
     rejects_incomplete_identity_ambiguous_axes_and_payload_disagreement) {
    auto incomplete =
        std::make_shared<pipeline::NativeReadoutMappingAuthority>(*mapping(0));
    incomplete->x.sign_convention_id.clear();
    EXPECT_THROW(
        pipeline::NativePairedReadoutNetwork::admit(
            axis(0, 10, {1.0}, {1}), detectors(0, "association:0", 1),
            std::move(incomplete), values(1, 1, 1.0),
            values(1, 1, 101.0), states(1), states(1)),
        std::invalid_argument);

    auto ambiguous_detectors = detectors(0, "association:0", 2);
    ambiguous_detectors[1].tone_or_channel_id =
        ambiguous_detectors[0].tone_or_channel_id;
    EXPECT_THROW(
        pipeline::NativePairedReadoutNetwork::admit(
            axis(0, 10, {1.0}, {1}), std::move(ambiguous_detectors),
            mapping(0), values(1, 2, 1.0), values(1, 2, 101.0),
            states(2), states(2)),
        std::invalid_argument);

    auto nonfinite = values(1, 1, 1.0);
    nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        pipeline::NativePairedReadoutNetwork::admit(
            axis(0, 10, {1.0}, {1}), detectors(0, "association:0", 1),
            mapping(0), std::move(nonfinite), values(1, 1, 101.0),
            states(1), states(1)),
        std::invalid_argument);

    auto timing =
        std::make_shared<const pipeline::NativeNetworkAlignment>(
            0, 10, vector({1.0}), std::vector<std::int64_t>{1});
    EXPECT_THROW(
        pipeline::NativePairedReadoutOccurrenceAxis(
            std::move(timing), 10,
            {{100, 200, {0.0, 1.0}}}),
        std::invalid_argument);
}

TEST(native_paired_readout,
     rejects_missing_participants_and_cross_network_detector_ambiguity) {
    const pipeline::NativeObservationScope scope{152390, 0, 4};
    std::vector<pipeline::NativePairedReadoutNetwork> incomplete;
    incomplete.push_back(make_network(0, 40, {1.0}, {1}, 1));
    EXPECT_THROW(
        pipeline::NativePairedReadoutObservation::admit(
            scope, {0, 7}, std::move(incomplete)),
        std::invalid_argument);

    auto first = make_network(0, 40, {1.0}, {1}, 1);
    auto second_axis = axis(7, 70, {1.1}, {7});
    auto duplicate = detectors(7, "association:0", 1);
    duplicate[0].detector_occurrence_id = "detector:0:0";
    auto second = pipeline::NativePairedReadoutNetwork::admit(
        std::move(second_axis), std::move(duplicate), mapping(7, "7"),
        values(1, 1, 1.0), values(1, 1, 101.0), states(1), states(1));
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(std::move(first));
    networks.push_back(std::move(second));
    EXPECT_THROW(
        pipeline::NativePairedReadoutObservation::admit(
            scope, {0, 7}, std::move(networks)),
        std::invalid_argument);
}

TEST(native_paired_readout,
     reports_compact_bounded_owned_memory_without_per_cell_identity) {
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(make_network(0, 40, {1.0, 2.0}, {1, 2}, 2));
    auto observation = pipeline::NativePairedReadoutObservation::admit(
        pipeline::NativeObservationScope{152390, 0, 4}, {0},
        std::move(networks));
    const auto memory = observation.memory_evidence();

    EXPECT_EQ(memory.numeric_payload_bytes, 8U * sizeof(double));
    EXPECT_EQ(memory.coordinate_state_bytes,
              8U * sizeof(pipeline::NativeReadoutCoordinateState));
    EXPECT_EQ(memory.occurrence_axis_bytes,
              2U * sizeof(
                       pipeline::NativePairedReadoutOccurrenceBinding));
    EXPECT_EQ(memory.detector_axis_bytes,
              2U * sizeof(pipeline::NativeReadoutDetectorBinding));
    EXPECT_GT(memory.identity_text_bytes, 0U);
    EXPECT_EQ(memory.referenced_native_axis_count, 1U);
    EXPECT_GT(memory.logical_owned_bytes(), memory.numeric_payload_bytes);
}

}  // namespace
