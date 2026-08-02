#pragma once

#include <citlali/core/pipeline/sci_align_contract.h>

#include <Eigen/Core>
#include <fmt/format.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::engine_detail {

struct PacketSlotConsistencySummary {
    std::uint64_t gap_event_count = 0;
    std::uint64_t missing_packet_count = 0;
};

inline PacketSlotConsistencySummary require_packet_slot_consistency(
    const std::string &interface_id, const Eigen::VectorXd &assigned_times,
    const std::vector<citlali::pipeline::sci_align::PacketGap> &packet_gaps,
    double phase_seconds,
    double cadence_seconds) {
    citlali::pipeline::sci_align::require_strictly_increasing(
        assigned_times, "assigned detector times");
    citlali::pipeline::sci_align::require_finite_positive(
        cadence_seconds, "detector cadence");
    if (!std::isfinite(phase_seconds)) {
        throw std::runtime_error(
            "detector packet/slot consistency input is malformed for " +
            interface_id);
    }

    PacketSlotConsistencySummary result;
    std::size_t next_gap = 0;
    auto previous_slot = citlali::pipeline::sci_align::round_half_up_slot(
        (assigned_times[0] - phase_seconds) / cadence_seconds);
    for (Eigen::Index row = 1; row < assigned_times.size(); ++row) {
        const auto current_slot =
            citlali::pipeline::sci_align::round_half_up_slot(
                (assigned_times[row] - phase_seconds) / cadence_seconds);
        if (current_slot <= previous_slot) {
            throw std::runtime_error(fmt::format(
                "{} assigned common-grid slots are not increasing at row {}",
                interface_id, row));
        }
        std::uint64_t packet_delta = 1;
        if (next_gap < packet_gaps.size() &&
            packet_gaps[next_gap].row_before == row - 1) {
            const auto &gap = packet_gaps[next_gap];
            if (gap.missing_packet_count == 0 ||
                gap.missing_packet_count ==
                    std::numeric_limits<std::uint64_t>::max()) {
                throw std::runtime_error(
                    interface_id +
                    " has a malformed compact PacketCount gap identity");
            }
            packet_delta = gap.missing_packet_count + 1;
            ++result.gap_event_count;
            if (result.missing_packet_count >
                std::numeric_limits<std::uint64_t>::max() -
                    gap.missing_packet_count) {
                throw std::runtime_error(
                    interface_id +
                    " missing PacketCount summary exceeds uint64 range");
            }
            result.missing_packet_count += gap.missing_packet_count;
            ++next_gap;
        }
        const auto slot_delta =
            static_cast<std::uint64_t>(current_slot) -
            static_cast<std::uint64_t>(previous_slot);
        if (packet_delta != slot_delta) {
            throw std::runtime_error(fmt::format(
                "{} PacketCount delta {} conflicts with assigned common-grid slot delta {} at row {}",
                interface_id, packet_delta, slot_delta, row));
        }
        previous_slot = current_slot;
    }
    if (next_gap != packet_gaps.size()) {
        throw std::runtime_error(
            interface_id +
            " has an out-of-order or out-of-support compact PacketCount gap identity");
    }
    return result;
}

}  // namespace citlali::engine_detail
