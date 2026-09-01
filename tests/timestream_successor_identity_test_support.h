#pragma once

#include <citlali/core/pipeline/timestream_native_paired_readout.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace citlali::test::timestream_successor {

namespace pipeline = citlali::pipeline;

inline Eigen::VectorXd time_vector(const std::vector<double> &values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    for (std::size_t index = 0; index < values.size(); ++index) {
        result(static_cast<Eigen::Index>(index)) = values[index];
    }
    return result;
}

inline pipeline::NativeReadoutCoordinateAuthority coordinate_authority(
    const std::string &name) {
    return {name + ":meaning", name + ":unit-or-scale", name + ":sign",
            name + ":reference", name + ":normalization",
            name + ":metric", name + ":validity-domain",
            name + ":uncertainty-state"};
}

inline std::shared_ptr<const pipeline::NativeReadoutMappingAuthority>
mapping_authority(pipeline::TimestreamNetworkId network_id,
                  const std::string &suffix) {
    auto result =
        std::make_shared<pipeline::NativeReadoutMappingAuthority>();
    result->network_id = network_id;
    result->producer_id = "kids-producer:" + suffix;
    result->producer_instance_id = "kids-result:" + suffix;
    result->producer_interface_id =
        std::string{pipeline::native_paired_xr_producer_interface_id};
    result->mapping_record_id = "mapping-record:" + suffix;
    result->mapping_revision_id = "mapping-revision:" + suffix;
    result->tune_id = "tune:" + suffix;
    result->readout_interface_id = "readout-interface:" + suffix;
    result->input_coordinate_record_id = "native-iq:" + suffix;
    result->transform_id = "iq-to-xr:" + suffix;
    result->transform_representation_id = "jacobian:" + suffix;
    result->applicability_domain_id = "domain:" + suffix;
    result->event_time_epoch_meaning_id =
        "integration-centroid-unix:" + suffix;
    result->native_time_unit_id = "s:" + suffix;
    result->native_cadence_record_id = "cadence:" + suffix;
    result->native_time_validity_state_id = "time-validity:" + suffix;
    result->timing_uncertainty_state_id = "time-uncertainty:" + suffix;
    result->parent_readout_record_id = "readout-occurrences:" + suffix;
    result->paired_xr_record_id = "paired-occurrences:" + suffix;
    result->runtime_binding_rule_id = "runtime-binding:" + suffix;
    result->compatibility_rule_id = "compatibility:" + suffix;
    result->failure_semantics_id = "fail-closed:" + suffix;
    result->x = coordinate_authority("x:" + suffix);
    result->r = coordinate_authority("r:" + suffix);
    return result;
}

inline std::shared_ptr<const pipeline::NativePairedReadoutOccurrenceAxis>
occurrence_axis(pipeline::TimestreamNetworkId network_id,
                pipeline::TimestreamNativeRow first_row,
                const std::vector<double> &times,
                std::vector<pipeline::TimestreamPacketCounter> counters) {
    auto timing =
        std::make_shared<const pipeline::NativeNetworkAlignment>(
            network_id, first_row, time_vector(times), std::move(counters));
    std::vector<pipeline::NativePairedReadoutOccurrenceBinding> occurrences;
    occurrences.reserve(times.size());
    for (std::size_t index = 0; index < times.size(); ++index) {
        const auto key = static_cast<std::int64_t>(index) + first_row;
        occurrences.push_back(
            {10000 + key, 20000 + key,
             {times[index] - 0.004, times[index] + 0.004}});
    }
    return std::make_shared<const
                            pipeline::NativePairedReadoutOccurrenceAxis>(
        std::move(timing), first_row, std::move(occurrences));
}

inline std::vector<pipeline::NativeReadoutDetectorBinding> detector_axis(
    pipeline::TimestreamNetworkId network_id,
    Eigen::Index detector_count) {
    std::vector<pipeline::NativeReadoutDetectorBinding> result;
    result.reserve(static_cast<std::size_t>(detector_count));
    for (Eigen::Index column = 0; column < detector_count; ++column) {
        const auto suffix = std::to_string(network_id) + ":" +
            std::to_string(column);
        result.push_back(
            {network_id, column, "detector-occurrence:" + suffix,
             "detector-association:" + std::to_string(network_id),
             "tone:" + suffix});
    }
    return result;
}

inline pipeline::NativePairedReadoutMatrix matrix(
    Eigen::Index rows,
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

inline pipeline::NativeReadoutCoordinateState valid_state() {
    return pipeline::NativeReadoutCoordinateState::measured(
        true, true, true, true);
}

inline std::vector<pipeline::NativeReadoutCoordinateState> valid_states(
    std::size_t count) {
    return std::vector<pipeline::NativeReadoutCoordinateState>(
        count, valid_state());
}

inline pipeline::NativePairedReadoutNetwork make_network(
    pipeline::TimestreamNetworkId network_id,
    pipeline::TimestreamNativeRow first_row,
    const std::vector<double> &times,
    std::vector<pipeline::TimestreamPacketCounter> counters,
    Eigen::Index detector_count,
    double x_base,
    double r_base,
    std::vector<pipeline::NativeReadoutCoordinateState> x_states = {},
    std::vector<pipeline::NativeReadoutCoordinateState> r_states = {}) {
    const auto rows = static_cast<Eigen::Index>(times.size());
    const auto cells = static_cast<std::size_t>(rows * detector_count);
    if (x_states.empty()) x_states = valid_states(cells);
    if (r_states.empty()) r_states = valid_states(cells);
    return pipeline::NativePairedReadoutNetwork::admit(
        occurrence_axis(network_id, first_row, times, std::move(counters)),
        detector_axis(network_id, detector_count),
        mapping_authority(network_id, std::to_string(network_id)),
        matrix(rows, detector_count, x_base),
        matrix(rows, detector_count, r_base), std::move(x_states),
        std::move(r_states));
}

inline std::shared_ptr<const pipeline::NativePairedReadoutObservation>
make_observation(
    std::vector<pipeline::NativePairedReadoutNetwork> networks,
    std::vector<pipeline::TimestreamNetworkId> required_networks) {
    auto admitted = pipeline::NativePairedReadoutObservation::admit(
        pipeline::NativeObservationScope{152390, 0, 4},
        std::move(required_networks),
        std::move(networks));
    return std::make_shared<const pipeline::NativePairedReadoutObservation>(
        std::move(admitted));
}

}  // namespace citlali::test::timestream_successor
