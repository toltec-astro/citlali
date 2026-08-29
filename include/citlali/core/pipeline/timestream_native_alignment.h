#pragma once

#include <citlali/core/pipeline/timestream_coincidence_cohort.h>
#include <citlali/core/pipeline/timestream_native_timing.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct NativeSlotAssociation {
    TimestreamNativeRow native_row = -1;
    CoincidenceAbsenceReason absence_reason =
        CoincidenceAbsenceReason::no_candidate;

    bool mapped() const noexcept { return native_row >= 0; }

    friend bool operator==(const NativeSlotAssociation &,
                           const NativeSlotAssociation &) = default;
};

class NativeAlignmentPlan {
public:
    NativeAlignmentPlan(
        NativeObservationScope scope,
        std::vector<NativeNetworkAlignment> networks,
        Eigen::VectorXd common_slot_reference_times_unix_sec,
        std::map<TimestreamNetworkId,
                 std::vector<NativeSlotAssociation>> associations_by_network)
        : scope_{scope}, networks_{std::move(networks)},
          common_slot_reference_times_unix_sec_{
              std::move(common_slot_reference_times_unix_sec)} {
        if (networks_.empty() ||
            common_slot_reference_times_unix_sec_.size() <= 0) {
            throw std::invalid_argument(
                "native alignment plan requires networks and relational slots");
        }
        for (Eigen::Index slot = 0;
             slot < common_slot_reference_times_unix_sec_.size(); ++slot) {
            const double value =
                common_slot_reference_times_unix_sec_(slot);
            if (!std::isfinite(value) ||
                (slot > 0 &&
                 !(value >
                   common_slot_reference_times_unix_sec_(slot - 1)))) {
                throw std::invalid_argument(
                    "common-slot reference times must be finite and strictly increasing");
            }
        }

        std::sort(networks_.begin(), networks_.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.network_id() < rhs.network_id();
                  });
        if (associations_by_network.size() != networks_.size()) {
            throw std::invalid_argument(
                "native association and network counts differ");
        }
        associations_.reserve(networks_.size());
        for (std::size_t index = 0; index < networks_.size(); ++index) {
            const auto network_id = networks_[index].network_id();
            if (!network_index_by_id_.emplace(network_id, index).second) {
                throw std::invalid_argument(
                    "native alignment plan repeats a network ID");
            }
            const auto found = associations_by_network.find(network_id);
            if (found == associations_by_network.end()) {
                throw std::invalid_argument(
                    "native alignment plan lacks a network association");
            }
            validate_associations(networks_[index], found->second);
            participant_network_ids_.push_back(network_id);
            associations_.push_back(found->second);
        }
    }

    const NativeObservationScope &scope() const noexcept { return scope_; }
    const std::vector<NativeNetworkAlignment> &networks() const noexcept {
        return networks_;
    }
    const std::vector<TimestreamNetworkId> &participant_network_ids() const
        noexcept {
        return participant_network_ids_;
    }
    const Eigen::VectorXd &common_slot_reference_times_unix_sec() const
        noexcept {
        return common_slot_reference_times_unix_sec_;
    }
    std::size_t slot_count() const noexcept {
        return static_cast<std::size_t>(
            common_slot_reference_times_unix_sec_.size());
    }
    const NativeNetworkAlignment &network(
        TimestreamNetworkId network_id) const {
        return networks_.at(network_index(network_id));
    }
    const NativeSlotAssociation &association(
        TimestreamNetworkId network_id, std::size_t common_slot) const {
        return associations_.at(network_index(network_id)).at(common_slot);
    }

private:
    std::size_t network_index(TimestreamNetworkId network_id) const {
        const auto found = network_index_by_id_.find(network_id);
        if (found == network_index_by_id_.end()) {
            throw std::out_of_range(
                "network is absent from the native alignment plan");
        }
        return found->second;
    }

    void validate_associations(
        const NativeNetworkAlignment &network,
        const std::vector<NativeSlotAssociation> &associations) const {
        if (associations.size() != slot_count()) {
            throw std::invalid_argument(
                "native associations do not match common-slot count");
        }
        std::set<TimestreamNativeRow> used_rows;
        for (const auto &association : associations) {
            if (!association.mapped()) continue;
            if (association.native_row < network.first_native_row() ||
                association.native_row >= network.past_last_native_row()) {
                throw std::invalid_argument(
                    "mapped native row is outside its network support");
            }
            if (!used_rows.insert(association.native_row).second) {
                throw std::logic_error(
                    "one native row cannot populate two relational slots");
            }
        }
    }

    NativeObservationScope scope_;
    std::vector<NativeNetworkAlignment> networks_;
    Eigen::VectorXd common_slot_reference_times_unix_sec_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::map<TimestreamNetworkId, std::size_t> network_index_by_id_;
    std::vector<std::vector<NativeSlotAssociation>> associations_;
};

inline std::vector<NativeSlotAssociation>
make_direct_native_slot_associations(
    TimestreamNativeRow first_native_row, std::size_t slot_count) {
    if (first_native_row < 0 || slot_count == 0 ||
        slot_count > static_cast<std::size_t>(
            std::numeric_limits<TimestreamNativeRow>::max() -
            first_native_row)) {
        throw std::invalid_argument(
            "direct native slot association interval is invalid");
    }
    std::vector<NativeSlotAssociation> result(slot_count);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
        result[slot].native_row =
            first_native_row + static_cast<TimestreamNativeRow>(slot);
    }
    return result;
}

// Preserve the established gap association except at the corrected scientific
// boundary: std::round selects the single candidate slot (half-way cases away
// from zero), and admission requires abs(delta) < dt/2. The result must have
// exact presence parity with the compatibility mask.
inline std::vector<NativeSlotAssociation>
make_gap_native_slot_associations(
    const NativeNetworkAlignment &network,
    const Eigen::VectorXd &common_slot_reference_times,
    const Eigen::VectorXi &legacy_presence_mask,
    double realized_dt_sec) {
    const auto &network_times =
        network.reconstructed_times_unix_sec();
    if (common_slot_reference_times.size() <= 0 ||
        legacy_presence_mask.size() !=
            common_slot_reference_times.size() ||
        !std::isfinite(realized_dt_sec) || realized_dt_sec <= 0.0) {
        throw std::invalid_argument(
            "gap native slot association inputs are invalid");
    }
    for (Eigen::Index slot = 0;
         slot < common_slot_reference_times.size(); ++slot) {
        const double value = common_slot_reference_times(slot);
        if (!std::isfinite(value) ||
            (slot > 0 &&
             !(value > common_slot_reference_times(slot - 1)))) {
            throw std::invalid_argument(
                "gap reference times must be finite and strictly increasing");
        }
    }
    for (Eigen::Index slot = 0; slot < legacy_presence_mask.size(); ++slot) {
        if (legacy_presence_mask(slot) != 0 &&
            legacy_presence_mask(slot) != 1) {
            throw std::invalid_argument(
                "legacy presence mask must contain only zero or one");
        }
    }

    const double max_init_time = common_slot_reference_times(0);
    const double tolerance = realized_dt_sec / 2.0;
    std::vector<NativeSlotAssociation> result(
        static_cast<std::size_t>(common_slot_reference_times.size()));
    std::set<Eigen::Index> mapped_slots;
    for (Eigen::Index local_row = 0; local_row < network_times.size();
         ++local_row) {
        const double value = network_times(local_row);
        const double grid_position =
            (value - max_init_time) / realized_dt_sec;
        if (!std::isfinite(grid_position)) {
            throw std::invalid_argument(
                "native grid position must be finite");
        }
        const double rounded = std::round(grid_position);
        if (rounded < 0.0 ||
            rounded >= static_cast<double>(
                common_slot_reference_times.size())) {
            continue;
        }
        const auto slot = static_cast<Eigen::Index>(rounded);
        if (std::abs(value - common_slot_reference_times(slot)) >=
            tolerance) {
            continue;
        }
        if (!mapped_slots.insert(slot).second) {
            throw std::logic_error(
                "two native rows map to one relational slot");
        }
        result[static_cast<std::size_t>(slot)].native_row =
            network.first_native_row() +
            static_cast<TimestreamNativeRow>(local_row);
    }

    const double native_min = network_times(0);
    const double native_max = network_times(network_times.size() - 1);
    for (Eigen::Index slot = 0;
         slot < common_slot_reference_times.size(); ++slot) {
        const bool legacy_present = legacy_presence_mask(slot) == 1;
        auto &association = result[static_cast<std::size_t>(slot)];
        if (legacy_present != association.mapped()) {
            throw std::logic_error(
                "native association disagrees with legacy presence mask");
        }
        if (!association.mapped()) {
            const double reference = common_slot_reference_times(slot);
            association.absence_reason =
                reference < native_min || reference > native_max
                    ? CoincidenceAbsenceReason::outside_native_support
                    : CoincidenceAbsenceReason::no_candidate;
        }
    }
    return result;
}

}  // namespace citlali::pipeline
