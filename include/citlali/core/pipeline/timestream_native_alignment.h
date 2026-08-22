#pragma once

#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/pipeline/timestream_coincidence_cohort.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citlali::pipeline {

using TimestreamPacketCounter = std::int64_t;

// Observation identity is carried independently of the APT implementation so
// native timing remains a distinct authority.  The observation owner binds it
// to the verified compact-v2 relation before publication.
struct NativeObservationScope {
    std::int64_t observation = 0;
    std::int64_t subobservation = 0;
    std::int64_t scan = 0;

    explicit NativeObservationScope(std::int64_t observation_,
                                    std::int64_t subobservation_,
                                    std::int64_t scan_)
        : observation{observation_}, subobservation{subobservation_},
          scan{scan_} {
        if (observation <= 0 || subobservation < 0 || scan < 0) {
            throw std::invalid_argument(
                "native observation scope is invalid");
        }
    }

    friend bool operator==(const NativeObservationScope &,
                           const NativeObservationScope &) = default;
};

// This transition records delivered packet-counter provenance only.  It does
// not infer a duration, a rollover policy, or a missing detector sample.
struct NativeCounterDiscontinuity {
    TimestreamNativeRow before_native_row = -1;
    TimestreamNativeRow after_native_row = -1;
    TimestreamPacketCounter before_counter = 0;
    TimestreamPacketCounter after_counter = 0;

    friend bool operator==(const NativeCounterDiscontinuity &,
                           const NativeCounterDiscontinuity &) = default;
};

struct NativeRunBoundary {
    bool stream_boundary = false;
    bool scan_boundary = false;
    std::optional<NativeCounterDiscontinuity> counter_discontinuity;

    friend bool operator==(const NativeRunBoundary &,
                           const NativeRunBoundary &) = default;
};

struct NativeContiguousRun {
    TimestreamNetworkId network_id = -1;
    TimestreamNativeRow first_native_row = -1;
    TimestreamNativeRow past_last_native_row = -1;
    NativeRunBoundary boundary_before;
    NativeRunBoundary boundary_after;

    TimestreamNativeRow row_count() const noexcept {
        return past_last_native_row - first_native_row;
    }
};

inline bool packet_counters_are_contiguous(
    TimestreamPacketCounter before,
    TimestreamPacketCounter after) noexcept {
    return before != std::numeric_limits<TimestreamPacketCounter>::max() &&
           after == before + 1;
}

template <class TimeMatrix>
std::vector<TimestreamPacketCounter>
packet_counters_from_timestream_matrix(const TimeMatrix &timestamps) {
    if (timestamps.rows() <= 0 || timestamps.cols() <= 3) {
        throw std::invalid_argument(
            "native packet-counter source requires rows and column 3");
    }

    std::vector<TimestreamPacketCounter> result;
    result.reserve(static_cast<std::size_t>(timestamps.rows()));
    for (Eigen::Index row = 0; row < timestamps.rows(); ++row) {
        const long double value =
            static_cast<long double>(timestamps(row, 3));
        if (!std::isfinite(value) || std::floor(value) != value ||
            value < static_cast<long double>(
                        std::numeric_limits<TimestreamPacketCounter>::min()) ||
            value > static_cast<long double>(
                        std::numeric_limits<TimestreamPacketCounter>::max())) {
            throw std::invalid_argument(
                "native packet counter must be a finite representable integer");
        }
        result.push_back(static_cast<TimestreamPacketCounter>(value));
    }
    return result;
}

class NativeNetworkAlignment {
public:
    NativeNetworkAlignment(
        TimestreamNetworkId network_id,
        TimestreamNativeRow first_native_row,
        Eigen::VectorXd reconstructed_times_unix_sec,
        std::vector<TimestreamPacketCounter> packet_counters)
        : network_id_{network_id}, first_native_row_{first_native_row},
          reconstructed_times_unix_sec_{
              std::move(reconstructed_times_unix_sec)},
          packet_counters_{std::move(packet_counters)} {
        if (network_id_ < 0 || first_native_row_ < 0) {
            throw std::invalid_argument(
                "native network alignment requires nonnegative identity");
        }
        if (reconstructed_times_unix_sec_.size() <= 0 ||
            static_cast<std::size_t>(
                reconstructed_times_unix_sec_.size()) !=
                packet_counters_.size()) {
            throw std::invalid_argument(
                "native times and counters require equal nonzero cardinality");
        }
        if (reconstructed_times_unix_sec_.size() >
            std::numeric_limits<TimestreamNativeRow>::max() -
                first_native_row_) {
            throw std::length_error("native row interval would overflow");
        }
        for (Eigen::Index row = 0;
             row < reconstructed_times_unix_sec_.size(); ++row) {
            const double value = reconstructed_times_unix_sec_(row);
            if (!std::isfinite(value) ||
                (row > 0 &&
                 !(value > reconstructed_times_unix_sec_(row - 1)))) {
                throw std::invalid_argument(
                    "native reconstructed times must be finite and strictly increasing");
            }
        }
    }

    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    TimestreamNativeRow first_native_row() const noexcept {
        return first_native_row_;
    }
    TimestreamNativeRow past_last_native_row() const noexcept {
        return first_native_row_ + static_cast<TimestreamNativeRow>(
            reconstructed_times_unix_sec_.size());
    }
    Eigen::Index row_count() const noexcept {
        return reconstructed_times_unix_sec_.size();
    }
    const Eigen::VectorXd &reconstructed_times_unix_sec() const noexcept {
        return reconstructed_times_unix_sec_;
    }
    const std::vector<TimestreamPacketCounter> &packet_counters() const
        noexcept {
        return packet_counters_;
    }

    NativeSampleIdentity identity(TimestreamNativeRow native_row) const {
        const auto offset = checked_offset(native_row);
        return NativeSampleIdentity{
            network_id_, native_row,
            reconstructed_times_unix_sec_(offset)};
    }

    TimestreamPacketCounter packet_counter(
        TimestreamNativeRow native_row) const {
        return packet_counters_.at(
            static_cast<std::size_t>(checked_offset(native_row)));
    }

    std::optional<NativeCounterDiscontinuity> discontinuity_between(
        TimestreamNativeRow before_native_row,
        TimestreamNativeRow after_native_row) const {
        if (before_native_row ==
            std::numeric_limits<TimestreamNativeRow>::max() ||
            after_native_row != before_native_row + 1) {
            throw std::invalid_argument(
                "native discontinuity query requires adjacent delivered rows");
        }
        const auto before = packet_counter(before_native_row);
        const auto after = packet_counter(after_native_row);
        if (packet_counters_are_contiguous(before, after)) {
            return std::nullopt;
        }
        return NativeCounterDiscontinuity{
            before_native_row, after_native_row, before, after};
    }

private:
    Eigen::Index checked_offset(TimestreamNativeRow native_row) const {
        if (native_row < first_native_row_ ||
            native_row >= past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside the delivered network alignment");
        }
        return static_cast<Eigen::Index>(native_row - first_native_row_);
    }

    TimestreamNetworkId network_id_;
    TimestreamNativeRow first_native_row_;
    Eigen::VectorXd reconstructed_times_unix_sec_;
    std::vector<TimestreamPacketCounter> packet_counters_;
};

template <class TimeMatrix>
NativeNetworkAlignment make_native_network_alignment(
    TimestreamNetworkId network_id,
    TimestreamNativeRow first_native_row,
    const TimeMatrix &timestamps,
    double fpga_frequency_hz,
    double interface_sync_offset_sec) {
    if (timestamps.rows() <= 0 || timestamps.cols() <= 5 ||
        !std::isfinite(fpga_frequency_hz) || fpga_frequency_hz <= 0.0 ||
        !std::isfinite(interface_sync_offset_sec)) {
        throw std::invalid_argument(
            "native timestamp reconstruction inputs are invalid");
    }
    for (Eigen::Index row = 0; row < timestamps.rows(); ++row) {
        for (Eigen::Index column : {Eigen::Index{0}, Eigen::Index{1},
                                    Eigen::Index{2}, Eigen::Index{4},
                                    Eigen::Index{5}}) {
            if (!std::isfinite(
                    static_cast<long double>(timestamps(row, column)))) {
                throw std::invalid_argument(
                    "native timestamp reconstruction source must be finite");
            }
        }
    }
    Eigen::MatrixXd source = timestamps.template cast<double>();
    auto reconstructed = network_time_from_timestream_matrix(
        source, fpga_frequency_hz, interface_sync_offset_sec);
    auto counters = packet_counters_from_timestream_matrix(timestamps);
    return NativeNetworkAlignment{
        network_id, first_native_row, std::move(reconstructed),
        std::move(counters)};
}

inline std::vector<NativeContiguousRun> partition_native_contiguous_runs(
    const NativeNetworkAlignment &network,
    TimestreamNativeRow first_native_row,
    TimestreamNativeRow past_last_native_row) {
    if (first_native_row < network.first_native_row() ||
        past_last_native_row > network.past_last_native_row() ||
        first_native_row >= past_last_native_row) {
        throw std::invalid_argument(
            "native scan row window must be a nonempty delivered interval");
    }

    auto boundary_before = [&](TimestreamNativeRow row) {
        NativeRunBoundary result;
        result.scan_boundary = row == first_native_row;
        result.stream_boundary = row == network.first_native_row();
        if (row > network.first_native_row()) {
            result.counter_discontinuity =
                network.discontinuity_between(row - 1, row);
        }
        return result;
    };
    auto boundary_after = [&](TimestreamNativeRow row) {
        NativeRunBoundary result;
        result.scan_boundary = row == past_last_native_row;
        result.stream_boundary = row == network.past_last_native_row();
        if (row < network.past_last_native_row()) {
            result.counter_discontinuity =
                network.discontinuity_between(row - 1, row);
        }
        return result;
    };

    std::vector<NativeContiguousRun> runs;
    TimestreamNativeRow run_begin = first_native_row;
    for (TimestreamNativeRow row = first_native_row + 1;
         row < past_last_native_row; ++row) {
        const auto discontinuity =
            network.discontinuity_between(row - 1, row);
        if (!discontinuity.has_value()) continue;
        auto before = boundary_before(run_begin);
        auto after = boundary_after(row);
        after.counter_discontinuity = discontinuity;
        runs.push_back(NativeContiguousRun{
            network.network_id(), run_begin, row,
            std::move(before), std::move(after)});
        run_begin = row;
    }
    auto before = boundary_before(run_begin);
    if (run_begin > first_native_row) {
        before.counter_discontinuity =
            network.discontinuity_between(run_begin - 1, run_begin);
    }
    runs.push_back(NativeContiguousRun{
        network.network_id(), run_begin, past_last_native_row,
        std::move(before), boundary_after(past_last_native_row)});
    return runs;
}

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

// Preserve the established gap association exactly: std::round selects the
// single candidate slot (half-way cases away from zero), the edge is inclusive
// at abs(delta) == dt/2, and the result must have exact presence parity with
// the compatibility mask.
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
        if (std::abs(value - common_slot_reference_times(slot)) >
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
