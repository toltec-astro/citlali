#pragma once

#include <citlali/core/pipeline/timestream_coincidence_cohort.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

using TimestreamDetectorColumn = Eigen::Index;
using TimestreamDetectorUid = std::int64_t;
using TimestreamPacketCounter = std::int64_t;
using NativeDetectorFlagBits = std::uint64_t;
using NativeDetectorFlagBitsMatrix =
    Eigen::Matrix<NativeDetectorFlagBits, Eigen::Dynamic, Eigen::Dynamic>;
using NativeDetectorBooleanMatrix =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

// This transition records delivered packet-counter provenance only.  It does
// not infer a duration or synthesize any missing detector row.
struct NativeCounterDiscontinuity {
    TimestreamNativeRow before_native_row = -1;
    TimestreamNativeRow after_native_row = -1;
    TimestreamPacketCounter before_counter = 0;
    TimestreamPacketCounter after_counter = 0;

    friend bool operator==(const NativeCounterDiscontinuity &lhs,
                           const NativeCounterDiscontinuity &rhs) noexcept {
        return lhs.before_native_row == rhs.before_native_row &&
               lhs.after_native_row == rhs.after_native_row &&
               lhs.before_counter == rhs.before_counter &&
               lhs.after_counter == rhs.after_counter;
    }
};

struct NativeRunBoundary {
    bool stream_boundary = false;
    bool scan_boundary = false;
    // A complete shared-consumer cohort may end even when one participant's
    // delivered rows remain contiguous (for example, because another
    // participant is absent).  This boundary is relational only; it does not
    // declare a detector-time gap or a physical event boundary.
    bool cohort_boundary = false;
    std::optional<NativeCounterDiscontinuity> counter_discontinuity;

    friend bool operator==(const NativeRunBoundary &lhs,
                           const NativeRunBoundary &rhs) noexcept {
        return lhs.stream_boundary == rhs.stream_boundary &&
               lhs.scan_boundary == rhs.scan_boundary &&
               lhs.cohort_boundary == rhs.cohort_boundary &&
               lhs.counter_discontinuity == rhs.counter_discontinuity;
    }
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
    // B1 does not infer a packet-counter wrap policy.  Only an exactly
    // representable +1 transition may share native RTC support.
    return before != std::numeric_limits<TimestreamPacketCounter>::max() &&
           after == before + 1;
}

template <class TimeMatrix>
std::vector<TimestreamPacketCounter>
packet_counters_from_timestream_matrix(const TimeMatrix &ts) {
    if (ts.rows() <= 0) {
        throw std::invalid_argument(
            "native packet-counter source must contain at least one row");
    }
    if (ts.cols() <= 3) {
        throw std::invalid_argument(
            "native packet-counter source must contain Data.Toltec.Ts column 3");
    }

    std::vector<TimestreamPacketCounter> result;
    result.reserve(static_cast<std::size_t>(ts.rows()));
    for (Eigen::Index row = 0; row < ts.rows(); ++row) {
        const long double value = static_cast<long double>(ts(row, 3));
        if (!std::isfinite(static_cast<double>(value)) ||
            std::floor(value) != value ||
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

// Compact observation-owned authority for one delivered network stream.  A
// reconstructed time is attached to a delivered native row; neither this
// class nor its consumers assign physical integration-event semantics to it.
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
        if (network_id_ < 0) {
            throw std::invalid_argument(
                "native network alignment requires a nonnegative network ID");
        }
        if (first_native_row_ < 0) {
            throw std::invalid_argument(
                "native network alignment requires a nonnegative first row");
        }
        if (reconstructed_times_unix_sec_.size() <= 0) {
            throw std::invalid_argument(
                "native network alignment requires delivered rows");
        }
        if (static_cast<std::size_t>(
                reconstructed_times_unix_sec_.size()) !=
            packet_counters_.size()) {
            throw std::invalid_argument(
                "native times and packet counters must have identical row counts");
        }
        if (reconstructed_times_unix_sec_.size() >
            std::numeric_limits<TimestreamNativeRow>::max() -
                first_native_row_) {
            throw std::length_error("native row interval would overflow");
        }
        for (Eigen::Index row = 0;
             row < reconstructed_times_unix_sec_.size(); ++row) {
            const double value = reconstructed_times_unix_sec_(row);
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "native reconstructed timestamp must be finite");
            }
        }
    }

    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    TimestreamNativeRow first_native_row() const noexcept {
        return first_native_row_;
    }
    TimestreamNativeRow past_last_native_row() const noexcept {
        return first_native_row_ +
               static_cast<TimestreamNativeRow>(
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
        if (after_native_row != before_native_row + 1) {
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
        if (!discontinuity.has_value()) {
            continue;
        }
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
};

class NativeCohortSelection {
public:
    NativeCohortSelection(CoincidenceCohort cohort,
                          std::vector<std::size_t> relational_common_slots)
        : cohort_{std::move(cohort)},
          relational_common_slots_{std::move(relational_common_slots)} {
        if (relational_common_slots_.size() != cohort_.slot_count()) {
            throw std::invalid_argument(
                "cohort common-slot identities must match its row count");
        }
        if (!std::is_sorted(relational_common_slots_.begin(),
                            relational_common_slots_.end()) ||
            std::adjacent_find(relational_common_slots_.begin(),
                               relational_common_slots_.end()) !=
                relational_common_slots_.end()) {
            throw std::invalid_argument(
                "cohort common-slot identities must be strictly increasing");
        }
    }

    const CoincidenceCohort &cohort() const noexcept { return cohort_; }
    const std::vector<std::size_t> &relational_common_slots() const
        noexcept {
        return relational_common_slots_;
    }

private:
    CoincidenceCohort cohort_;
    std::vector<std::size_t> relational_common_slots_;
};

// Common-slot reference times are approximate coincidence coordinates only.
// The injective mapping below never promotes them to detector sample times or
// physical telescope-time authority.
class NativeAlignmentPlan {
public:
    NativeAlignmentPlan(
        std::vector<NativeNetworkAlignment> networks,
        Eigen::VectorXd common_slot_reference_times_unix_sec,
        std::vector<std::vector<NativeSlotAssociation>> associations)
        : networks_{std::move(networks)},
          common_slot_reference_times_unix_sec_{
              std::move(common_slot_reference_times_unix_sec)},
          associations_{std::move(associations)} {
        if (networks_.empty()) {
            throw std::invalid_argument(
                "native alignment plan requires at least one network");
        }
        if (common_slot_reference_times_unix_sec_.size() <= 0) {
            throw std::invalid_argument(
                "native alignment plan requires relational common slots");
        }
        for (Eigen::Index slot = 0;
             slot < common_slot_reference_times_unix_sec_.size(); ++slot) {
            const double value =
                common_slot_reference_times_unix_sec_(slot);
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "common-slot reference time must be finite");
            }
            if (slot > 0 &&
                !(value > common_slot_reference_times_unix_sec_(slot - 1))) {
                throw std::invalid_argument(
                    "common-slot reference times must increase strictly");
            }
        }

        if (associations_.size() != networks_.size()) {
            throw std::invalid_argument(
                "native slot associations must match the network count");
        }
        for (std::size_t network_index = 0;
             network_index < networks_.size(); ++network_index) {
            const auto network_id = networks_[network_index].network_id();
            if (!network_index_by_id_.emplace(network_id, network_index)
                     .second) {
                throw std::invalid_argument(
                    "native alignment plan contains a duplicate network ID");
            }
            participant_network_ids_.push_back(network_id);
            validate_associations(
                networks_[network_index], associations_[network_index]);
        }
    }

    const std::vector<NativeNetworkAlignment> &networks() const noexcept {
        return networks_;
    }
    const NativeNetworkAlignment &network(
        TimestreamNetworkId network_id) const {
        const auto it = network_index_by_id_.find(network_id);
        if (it == network_index_by_id_.end()) {
            throw std::out_of_range(
                "network is absent from the native alignment plan");
        }
        return networks_.at(it->second);
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

    const NativeSlotAssociation &association(
        TimestreamNetworkId network_id, std::size_t common_slot) const {
        const auto it = network_index_by_id_.find(network_id);
        if (it == network_index_by_id_.end()) {
            throw std::out_of_range(
                "network is absent from the native alignment plan");
        }
        return associations_.at(it->second).at(common_slot);
    }

    NativeCohortSelection select_cohort(
        NativeOperationIdentity operation,
        std::size_t first_common_slot,
        std::size_t past_last_common_slot,
        TimestreamNativeRevision expected_revision,
        const std::map<std::pair<TimestreamNetworkId,
                                 TimestreamNativeRow>,
                       NativeInvalidityProvenance> &row_invalidities = {})
        const {
        if (first_common_slot >= past_last_common_slot ||
            past_last_common_slot > slot_count()) {
            throw std::invalid_argument(
                "native cohort selection requires a nonempty common-slot interval");
        }
        std::vector<std::size_t> common_slots;
        common_slots.reserve(past_last_common_slot - first_common_slot);
        for (std::size_t common_slot = first_common_slot;
             common_slot < past_last_common_slot; ++common_slot) {
            common_slots.push_back(common_slot);
        }
        return select_cohort_exact(
            operation, std::move(common_slots), expected_revision,
            row_invalidities);
    }

    NativeCohortSelection select_cohort_exact(
        NativeOperationIdentity operation,
        std::vector<std::size_t> common_slots,
        TimestreamNativeRevision expected_revision,
        const std::map<std::pair<TimestreamNetworkId,
                                 TimestreamNativeRow>,
                       NativeInvalidityProvenance> &row_invalidities = {})
        const {
        if (common_slots.empty()) {
            throw std::invalid_argument(
                "native exact cohort selection requires common slots");
        }
        for (std::size_t index = 0; index < common_slots.size(); ++index) {
            if (common_slots[index] >= slot_count() ||
                (index > 0 &&
                 common_slots[index] <= common_slots[index - 1])) {
                throw std::invalid_argument(
                    "native exact cohort common slots must be in-range and increase strictly");
            }
        }
        const auto selected_slots = common_slots.size();
        CoincidenceCohortBuilder builder{
            operation, participant_network_ids_, selected_slots};
        for (std::size_t local_slot = 0; local_slot < selected_slots;
             ++local_slot) {
            const auto common_slot = common_slots[local_slot];
            for (const auto network_id : participant_network_ids_) {
                const auto &mapped = association(network_id, common_slot);
                if (!mapped.mapped()) {
                    builder.assign_absent(
                        network_id, local_slot, mapped.absence_reason);
                    continue;
                }
                auto identity =
                    network(network_id).identity(mapped.native_row);
                const auto invalidity = row_invalidities.find(
                    std::make_pair(network_id, mapped.native_row));
                if (invalidity == row_invalidities.end()) {
                    builder.assign_mapped_valid(
                        network_id, local_slot, std::move(identity),
                        expected_revision);
                }
                else {
                    builder.assign_mapped_invalid(
                        network_id, local_slot, std::move(identity),
                        expected_revision, invalidity->second);
                }
            }
        }
        return NativeCohortSelection{
            std::move(builder).finish(), std::move(common_slots)};
    }

private:
    void validate_associations(
        const NativeNetworkAlignment &network,
        const std::vector<NativeSlotAssociation> &associations) const {
        if (associations.size() != slot_count()) {
            throw std::invalid_argument(
                "native slot associations must match the common-slot count");
        }
        std::set<TimestreamNativeRow> used_native_rows;
        for (const auto &association : associations) {
            if (!association.mapped()) {
                continue;
            }
            if (association.native_row < network.first_native_row() ||
                association.native_row >= network.past_last_native_row()) {
                throw std::invalid_argument(
                    "mapped native slot row is outside its network support");
            }
            if (!used_native_rows.insert(association.native_row).second) {
                throw std::logic_error(
                    "one delivered native row cannot populate two common slots");
            }
        }
    }

    std::vector<NativeNetworkAlignment> networks_;
    Eigen::VectorXd common_slot_reference_times_unix_sec_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::map<TimestreamNetworkId, std::size_t> network_index_by_id_;
    std::vector<std::vector<NativeSlotAssociation>> associations_;
};

// A shared RTC/PTC operation is admitted only on a rectangular temporal
// cohort in which every participant names one delivered, mapped-valid native
// row.  Detector-level exclusions remain a separate PCA concern and do not
// weaken this temporal admission rule.
inline void require_complete_native_cohort(
    const NativeCohortSelection &selection) {
    const auto &cohort = selection.cohort();
    if (cohort.slot_count() == 0 || cohort.participant_count() == 0 ||
        selection.relational_common_slots().size() != cohort.slot_count()) {
        throw std::logic_error(
            "native shared-consumer cohort is empty or has unequal cardinality");
    }
    std::set<NativeSampleKey> destinations;
    for (std::size_t slot = 0; slot < cohort.slot_count(); ++slot) {
        for (std::size_t participant = 0;
             participant < cohort.participant_count(); ++participant) {
            const auto &cell = cohort.cell(slot, participant);
            if (cell.state() != CoincidenceCellState::mapped_valid ||
                !cell.identity().has_value()) {
                throw std::logic_error(
                    "native shared-consumer cohort must be complete and mapped-valid");
            }
            const auto expected_network =
                cohort.participant_network_ids().at(participant);
            if (cell.identity()->network_id() != expected_network ||
                !destinations.insert(cell.identity()->key()).second) {
                throw std::logic_error(
                    "native shared-consumer cohort identity is noninjective");
            }
        }
    }
}

struct NativeCompleteCohortRun {
    std::size_t run_ordinal = 0;
    std::size_t first_common_slot = 0;
    std::size_t past_last_common_slot = 0;
    NativeCohortSelection selection;
    std::vector<NativeContiguousRun> participant_runs;
};

// Partition one scan-bounded common-slot interval into maximal shared-support
// runs.  An absent/mapped-invalid cell or any participant row/counter
// discontinuity closes the candidate before it can be published.  Omitted
// common slots create no detector value, output row, pointing, or ledger
// destination.
inline std::vector<NativeCompleteCohortRun>
partition_complete_native_cohort_runs(
    const NativeAlignmentPlan &plan,
    NativeOperationIdentity operation,
    std::size_t first_common_slot,
    std::size_t past_last_common_slot,
    TimestreamNativeRevision expected_revision,
    const std::map<std::pair<TimestreamNetworkId, TimestreamNativeRow>,
                   NativeInvalidityProvenance> &row_invalidities = {}) {
    if (first_common_slot >= past_last_common_slot ||
        past_last_common_slot > plan.slot_count()) {
        throw std::invalid_argument(
            "native shared-consumer scan interval is invalid");
    }

    const auto &network_ids = plan.participant_network_ids();
    auto slot_is_complete = [&](std::size_t slot) {
        for (const auto network_id : network_ids) {
            const auto &association = plan.association(network_id, slot);
            if (!association.mapped() ||
                row_invalidities.contains(
                    {network_id, association.native_row})) {
                return false;
            }
        }
        return true;
    };
    auto slots_are_contiguous = [&](std::size_t before,
                                    std::size_t after) {
        for (const auto network_id : network_ids) {
            const auto before_row =
                plan.association(network_id, before).native_row;
            const auto after_row =
                plan.association(network_id, after).native_row;
            if (after_row != before_row + 1 ||
                plan.network(network_id)
                    .discontinuity_between(before_row, after_row)
                    .has_value()) {
                return false;
            }
        }
        return true;
    };

    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    std::optional<std::size_t> run_begin;
    for (std::size_t slot = first_common_slot;
         slot < past_last_common_slot; ++slot) {
        if (!slot_is_complete(slot)) {
            if (run_begin.has_value()) {
                intervals.emplace_back(*run_begin, slot);
                run_begin.reset();
            }
            continue;
        }
        if (!run_begin.has_value()) {
            run_begin = slot;
            continue;
        }
        if (!slots_are_contiguous(slot - 1, slot)) {
            intervals.emplace_back(*run_begin, slot);
            run_begin = slot;
        }
    }
    if (run_begin.has_value()) {
        intervals.emplace_back(*run_begin, past_last_common_slot);
    }

    std::vector<NativeCompleteCohortRun> result;
    result.reserve(intervals.size());
    for (std::size_t ordinal = 0; ordinal < intervals.size(); ++ordinal) {
        const auto [begin, end] = intervals[ordinal];
        auto selection = plan.select_cohort(
            operation, begin, end, expected_revision, row_invalidities);
        require_complete_native_cohort(selection);

        std::vector<NativeContiguousRun> participant_runs;
        participant_runs.reserve(network_ids.size());
        for (const auto network_id : network_ids) {
            const auto &network = plan.network(network_id);
            const auto first_row =
                plan.association(network_id, begin).native_row;
            const auto past_row =
                plan.association(network_id, end - 1).native_row + 1;
            NativeRunBoundary before;
            before.stream_boundary =
                first_row == network.first_native_row();
            before.scan_boundary = begin == first_common_slot;
            before.cohort_boundary = begin != first_common_slot;
            if (first_row > network.first_native_row()) {
                before.counter_discontinuity =
                    network.discontinuity_between(first_row - 1, first_row);
            }
            NativeRunBoundary after;
            after.stream_boundary =
                past_row == network.past_last_native_row();
            after.scan_boundary = end == past_last_common_slot;
            after.cohort_boundary = end != past_last_common_slot;
            if (past_row < network.past_last_native_row()) {
                after.counter_discontinuity =
                    network.discontinuity_between(past_row - 1, past_row);
            }
            participant_runs.push_back(NativeContiguousRun{
                network_id, first_row, past_row,
                std::move(before), std::move(after)});
        }
        result.push_back(NativeCompleteCohortRun{
            ordinal, begin, end, std::move(selection),
            std::move(participant_runs)});
    }
    return result;
}

inline std::vector<NativeSlotAssociation>
make_direct_native_slot_associations(
    TimestreamNativeRow first_native_row,
    std::size_t slot_count) {
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

inline std::vector<NativeSlotAssociation>
make_gap_native_slot_associations(
    const NativeNetworkAlignment &network,
    const Eigen::VectorXd &common_slot_reference_times,
    const Eigen::VectorXi &legacy_presence_mask,
    double max_init_time,
    double dt,
    double tolerance) {
    const auto &network_times =
        network.reconstructed_times_unix_sec();
    if (network_times.size() <= 0 ||
        common_slot_reference_times.size() <= 0 ||
        legacy_presence_mask.size() !=
            common_slot_reference_times.size() ||
        !std::isfinite(max_init_time) || !std::isfinite(dt) ||
        !std::isfinite(tolerance) || dt <= 0.0 || tolerance < 0.0) {
        throw std::invalid_argument(
            "gap native slot association inputs are invalid");
    }

    std::vector<NativeSlotAssociation> result(
        static_cast<std::size_t>(common_slot_reference_times.size()));
    std::set<Eigen::Index> mapped_slots;
    for (Eigen::Index native_row = 0;
         native_row < network_times.size(); ++native_row) {
        const double value = network_times(native_row);
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "gap native reconstructed timestamp must be finite");
        }
        const double grid_position = (value - max_init_time) / dt;
        if (!std::isfinite(grid_position)) {
            throw std::invalid_argument(
                "gap native grid position must be finite");
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
                "legacy gap grouping maps two native rows to one common slot");
        }
        result[static_cast<std::size_t>(slot)].native_row =
            network.first_native_row() +
            static_cast<TimestreamNativeRow>(native_row);
    }

    for (Eigen::Index slot = 0;
         slot < common_slot_reference_times.size(); ++slot) {
        const bool legacy_present = legacy_presence_mask(slot) != 0;
        const bool mapped = result[static_cast<std::size_t>(slot)].mapped();
        if (legacy_present != mapped) {
            throw std::logic_error(
                "native gap association disagrees with the established common-grid mask");
        }
        if (!mapped) {
            const double reference = common_slot_reference_times(slot);
            const double native_time_min = network_times.minCoeff();
            const double native_time_max = network_times.maxCoeff();
            result[static_cast<std::size_t>(slot)].absence_reason =
                reference < native_time_min || reference > native_time_max
                    ? CoincidenceAbsenceReason::outside_native_support
                    : CoincidenceAbsenceReason::no_candidate;
        }
    }
    return result;
}

struct NativeDetectorSampleKey {
    NativeSampleKey native_sample;
    TimestreamDetectorColumn detector_column = -1;

    friend bool operator==(const NativeDetectorSampleKey &lhs,
                           const NativeDetectorSampleKey &rhs) noexcept {
        return lhs.native_sample == rhs.native_sample &&
               lhs.detector_column == rhs.detector_column;
    }
    friend bool operator<(const NativeDetectorSampleKey &lhs,
                          const NativeDetectorSampleKey &rhs) noexcept {
        if (lhs.native_sample < rhs.native_sample) {
            return true;
        }
        if (rhs.native_sample < lhs.native_sample) {
            return false;
        }
        return lhs.detector_column < rhs.detector_column;
    }
};

class NativeDetectorBlock {
public:
    NativeDetectorBlock(
        const NativeNetworkAlignment &network,
        TimestreamNativeRow first_native_row,
        TimestreamDetectorColumn first_detector_column,
        Eigen::MatrixXd measured_values,
        NativeDetectorFlagBitsMatrix original_flag_bits)
        : network_id_{network.network_id()},
          first_native_row_{first_native_row},
          first_detector_column_{first_detector_column},
          measured_values_{std::move(measured_values)},
          original_flag_bits_{std::move(original_flag_bits)} {
        if (first_detector_column_ < 0) {
            throw std::invalid_argument(
                "native detector block requires a nonnegative first column");
        }
        if (measured_values_.rows() <= 0 || measured_values_.cols() <= 0) {
            throw std::invalid_argument(
                "native detector block must contain measured cells");
        }
        if (original_flag_bits_.rows() != measured_values_.rows() ||
            original_flag_bits_.cols() != measured_values_.cols()) {
            throw std::invalid_argument(
                "native detector values and flags must have identical shape");
        }
        if (first_native_row_ < network.first_native_row() ||
            measured_values_.rows() >
                network.past_last_native_row() - first_native_row_) {
            throw std::invalid_argument(
                "native detector block row interval is outside its network");
        }
        reconstructed_times_unix_sec_.resize(measured_values_.rows());
        packet_counters_.reserve(
            static_cast<std::size_t>(measured_values_.rows()));
        for (Eigen::Index row = 0; row < measured_values_.rows(); ++row) {
            const auto native_row =
                first_native_row_ + static_cast<TimestreamNativeRow>(row);
            reconstructed_times_unix_sec_(row) =
                network.identity(native_row).reconstructed_time_unix_sec();
            packet_counters_.push_back(network.packet_counter(native_row));
        }
    }

    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    TimestreamNativeRow first_native_row() const noexcept {
        return first_native_row_;
    }
    TimestreamNativeRow past_last_native_row() const noexcept {
        return first_native_row_ +
               static_cast<TimestreamNativeRow>(measured_values_.rows());
    }
    TimestreamDetectorColumn first_detector_column() const noexcept {
        return first_detector_column_;
    }
    TimestreamDetectorColumn past_last_detector_column() const noexcept {
        return first_detector_column_ + measured_values_.cols();
    }
    const Eigen::MatrixXd &measured_values() const noexcept {
        return measured_values_;
    }
    const NativeDetectorFlagBitsMatrix &original_flag_bits() const noexcept {
        return original_flag_bits_;
    }
    const Eigen::VectorXd &reconstructed_times_unix_sec() const noexcept {
        return reconstructed_times_unix_sec_;
    }
    const std::vector<TimestreamPacketCounter> &packet_counters() const
        noexcept {
        return packet_counters_;
    }

    NativeSampleIdentity identity(Eigen::Index local_row) const {
        if (local_row < 0 || local_row >= measured_values_.rows()) {
            throw std::out_of_range(
                "native detector block row is out of range");
        }
        return NativeSampleIdentity{
            network_id_,
            first_native_row_ +
                static_cast<TimestreamNativeRow>(local_row),
            reconstructed_times_unix_sec_(local_row)};
    }

    Eigen::Index local_row(TimestreamNativeRow native_row) const {
        if (native_row < first_native_row_ ||
            native_row >= past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside the measured detector block");
        }
        return static_cast<Eigen::Index>(native_row - first_native_row_);
    }

private:
    TimestreamNetworkId network_id_;
    TimestreamNativeRow first_native_row_;
    TimestreamDetectorColumn first_detector_column_;
    Eigen::MatrixXd measured_values_;
    NativeDetectorFlagBitsMatrix original_flag_bits_;
    Eigen::VectorXd reconstructed_times_unix_sec_;
    std::vector<TimestreamPacketCounter> packet_counters_;
};

enum class NativePreparedPcaGroupRole {
    pca_clean,
    pass_through,
};

struct NativeDetectorCoincidenceProvenance {
    std::size_t common_slot = 0;
    std::size_t participant_index = 0;
    TimestreamNetworkId participant_network_id = -1;
    TimestreamDetectorColumn detector_column = -1;
    TimestreamDetectorUid detector_uid = -1;
    std::string effective_grouping;
    Eigen::Index group_key = -1;
    Eigen::Index subgroup_index = -1;
    NativePreparedPcaGroupRole group_role =
        NativePreparedPcaGroupRole::pca_clean;
    NativeDetectorFlagBits delivered_flag_bits = 0;
    NativeDetectorFlagBits operation_exclusion_bits = 0;
    std::int64_t apt_flag_value = 0;
    std::string exclusion_reason;
};

enum class NativeDetectorRevisionAction {
    replaced_by_pca_result,
    preserved_pca_invalid,
    preserved_corr_ungrouped,
};

struct NativeDetectorRevisionRecord {
    NativeOperationIdentity operation;
    TimestreamNativeRevision input_revision = 0;
    TimestreamNativeRevision output_revision = 0;
    NativeDetectorRevisionAction action =
        NativeDetectorRevisionAction::replaced_by_pca_result;
    NativeDetectorCoincidenceProvenance coincidence_provenance;
};

class NativeDetectorLedger {
public:
    struct Seed {
        NativeSampleIdentity identity;
        TimestreamDetectorColumn detector_column = -1;
        double measured_value = 0.0;
        NativeDetectorFlagBits original_flag_bits = 0;
        std::string original_flag_reason;
    };

    struct Record {
        NativeSampleIdentity identity;
        TimestreamDetectorColumn detector_column = -1;
        double measured_value = 0.0;
        double current_value = 0.0;
        NativeDetectorFlagBits original_flag_bits = 0;
        std::string original_flag_reason;
        TimestreamNativeRevision revision = 0;
        std::vector<NativeDetectorRevisionRecord> lineage;
    };

    struct SnapshotEntry {
        NativeDetectorSampleKey key;
        double measured_value = 0.0;
        double current_value = 0.0;
        NativeDetectorFlagBits original_flag_bits = 0;
        std::string original_flag_reason;
        TimestreamNativeRevision revision = 0;
        std::vector<NativeDetectorRevisionRecord> lineage;
    };

    class Update {
    public:
        static Update replacement(
            NativeSampleIdentity identity,
            TimestreamDetectorColumn detector_column,
            TimestreamNativeRevision expected_revision,
            double value,
            NativeDetectorCoincidenceProvenance provenance) {
            return Update{
                std::move(identity), detector_column, expected_revision,
                NativeDetectorRevisionAction::replaced_by_pca_result,
                value, std::move(provenance)};
        }

        static Update preserve_invalid(
            NativeSampleIdentity identity,
            TimestreamDetectorColumn detector_column,
            TimestreamNativeRevision expected_revision,
            NativeDetectorCoincidenceProvenance provenance) {
            return Update{
                std::move(identity), detector_column, expected_revision,
                NativeDetectorRevisionAction::preserved_pca_invalid,
                std::nullopt, std::move(provenance)};
        }

        static Update preserve_corr_ungrouped(
            NativeSampleIdentity identity,
            TimestreamDetectorColumn detector_column,
            TimestreamNativeRevision expected_revision,
            NativeDetectorCoincidenceProvenance provenance) {
            return Update{
                std::move(identity), detector_column, expected_revision,
                NativeDetectorRevisionAction::preserved_corr_ungrouped,
                std::nullopt, std::move(provenance)};
        }

        const NativeSampleIdentity &identity() const noexcept {
            return identity_;
        }
        TimestreamDetectorColumn detector_column() const noexcept {
            return detector_column_;
        }
        TimestreamNativeRevision expected_revision() const noexcept {
            return expected_revision_;
        }
        NativeDetectorRevisionAction action() const noexcept {
            return action_;
        }
        const std::optional<double> &replacement_value() const noexcept {
            return replacement_value_;
        }
        const NativeDetectorCoincidenceProvenance &provenance() const
            noexcept {
            return provenance_;
        }

    private:
        Update(NativeSampleIdentity identity,
               TimestreamDetectorColumn detector_column,
               TimestreamNativeRevision expected_revision,
               NativeDetectorRevisionAction action,
               std::optional<double> replacement_value,
               NativeDetectorCoincidenceProvenance provenance)
            : identity_{std::move(identity)},
              detector_column_{detector_column},
              expected_revision_{expected_revision}, action_{action},
              replacement_value_{replacement_value},
              provenance_{std::move(provenance)} {}

        NativeSampleIdentity identity_;
        TimestreamDetectorColumn detector_column_;
        TimestreamNativeRevision expected_revision_;
        NativeDetectorRevisionAction action_;
        std::optional<double> replacement_value_;
        NativeDetectorCoincidenceProvenance provenance_;
    };

    explicit NativeDetectorLedger(std::vector<Seed> seeds) {
        for (auto &seed : seeds) {
            if (seed.detector_column < 0) {
                throw std::invalid_argument(
                    "native detector ledger requires nonnegative detector columns");
            }
            const NativeDetectorSampleKey key{
                seed.identity.key(), seed.detector_column};
            Record record{
                seed.identity, seed.detector_column, seed.measured_value,
                seed.measured_value, seed.original_flag_bits,
                std::move(seed.original_flag_reason), 0, {}};
            if (!records_.emplace(key, std::move(record)).second) {
                throw std::invalid_argument(
                    "duplicate native detector sample in ledger seed");
            }
        }
    }

    const Record &at(const NativeDetectorSampleKey &key) const {
        const auto it = records_.find(key);
        if (it == records_.end()) {
            throw std::out_of_range(
                "native detector sample is absent from the ledger");
        }
        return it->second;
    }
    bool contains(const NativeDetectorSampleKey &key) const noexcept {
        return records_.find(key) != records_.end();
    }
    std::size_t size() const noexcept { return records_.size(); }
    std::vector<SnapshotEntry> snapshot() const {
        std::vector<SnapshotEntry> result;
        result.reserve(records_.size());
        for (const auto &[key, record] : records_) {
            result.push_back(SnapshotEntry{
                key, record.measured_value, record.current_value,
                record.original_flag_bits, record.original_flag_reason,
                record.revision, record.lineage});
        }
        return result;
    }
    const std::optional<NativeOperationIdentity> &last_operation() const
        noexcept {
        return last_operation_;
    }

    void apply_transaction(const NativeOperationIdentity &operation,
                           const std::vector<Update> &updates) {
        if (last_operation_.has_value() &&
            operation.sequence <= last_operation_->sequence) {
            throw std::logic_error(
                "native detector operation sequence must increase monotonically");
        }

        std::set<NativeDetectorSampleKey> destinations;
        for (const auto &update : updates) {
            const NativeDetectorSampleKey key{
                update.identity().key(), update.detector_column()};
            if (!destinations.insert(key).second) {
                throw std::logic_error(
                    "native detector scatter contains a duplicate destination");
            }
            const auto it = records_.find(key);
            if (it == records_.end()) {
                throw std::logic_error(
                    "native detector scatter destination is absent");
            }
            if (!(it->second.identity == update.identity()) ||
                it->second.detector_column != update.detector_column()) {
                throw std::logic_error(
                    "native detector scatter identity or timestamp changed");
            }
            if (it->second.revision != update.expected_revision()) {
                throw std::logic_error(
                    "native detector scatter expected revision is stale");
            }
            if (it->second.revision ==
                std::numeric_limits<TimestreamNativeRevision>::max()) {
                throw std::overflow_error(
                    "native detector sample revision would overflow");
            }
            const bool replacement =
                update.action() ==
                NativeDetectorRevisionAction::replaced_by_pca_result;
            if (replacement != update.replacement_value().has_value()) {
                throw std::logic_error(
                    "native detector scatter action and value disagree");
            }
            if (replacement &&
                !std::isfinite(*update.replacement_value())) {
                throw std::logic_error(
                    "native detector PCA replacement must be finite");
            }
            const auto &provenance = update.provenance();
            if (provenance.participant_network_id !=
                    update.identity().network_id() ||
                provenance.detector_column != update.detector_column() ||
                provenance.detector_uid < 0 ||
                provenance.effective_grouping.empty() ||
                provenance.group_key < 0 ||
                provenance.subgroup_index < 0) {
                throw std::logic_error(
                    "native detector scatter provenance changed identity or group");
            }
            const bool has_invalidity =
                provenance.delivered_flag_bits != 0 ||
                provenance.operation_exclusion_bits != 0 ||
                provenance.apt_flag_value != 0 ||
                !provenance.exclusion_reason.empty();
            if (replacement &&
                (has_invalidity || provenance.group_role !=
                    NativePreparedPcaGroupRole::pca_clean)) {
                throw std::logic_error(
                    "PCA-valid detector scatter has inconsistent group provenance");
            }
            const bool pass_through =
                update.action() ==
                NativeDetectorRevisionAction::preserved_corr_ungrouped;
            if (!replacement && !pass_through && !has_invalidity) {
                throw std::logic_error(
                    "PCA-invalid detector scatter requires provenance");
            }
            if (pass_through &&
                (has_invalidity || provenance.group_role !=
                    NativePreparedPcaGroupRole::pass_through)) {
                throw std::logic_error(
                    "corr_nw pass-through scatter has inconsistent provenance");
            }
        }

        // Stage every allocating lineage payload and the affected candidate
        // records before live mutation.  Swapping the fully built affected
        // records below is noexcept, so validation, allocation, or copy
        // failure leaves the complete ledger unchanged without copying the
        // unrelated observation-sized map.
        std::vector<std::pair<NativeDetectorSampleKey, Record>> candidates;
        static_assert(std::is_nothrow_swappable_v<Record>);
        candidates.reserve(updates.size());
        for (const auto &update : updates) {
            const NativeDetectorSampleKey key{
                update.identity().key(), update.detector_column()};
            auto record = records_.at(key);
            const auto input_revision = record.revision;
            const auto output_revision = input_revision + 1;
            if (update.replacement_value().has_value()) {
                record.current_value = *update.replacement_value();
            }
            record.revision = output_revision;
            record.lineage.push_back(NativeDetectorRevisionRecord{
                operation, input_revision, output_revision,
                update.action(), update.provenance()});
            candidates.emplace_back(key, std::move(record));
        }
        for (auto &[key, candidate] : candidates) {
            using std::swap;
            swap(records_.at(key), candidate);
        }
        last_operation_ = operation;
    }

private:
    std::map<NativeDetectorSampleKey, Record> records_;
    std::optional<NativeOperationIdentity> last_operation_;
};

inline bool native_detector_revision_record_equal(
    const NativeDetectorRevisionRecord &lhs,
    const NativeDetectorRevisionRecord &rhs) {
    const auto &lp = lhs.coincidence_provenance;
    const auto &rp = rhs.coincidence_provenance;
    return lhs.operation == rhs.operation &&
           lhs.input_revision == rhs.input_revision &&
           lhs.output_revision == rhs.output_revision &&
           lhs.action == rhs.action &&
           lp.common_slot == rp.common_slot &&
           lp.participant_index == rp.participant_index &&
           lp.participant_network_id == rp.participant_network_id &&
           lp.detector_column == rp.detector_column &&
           lp.detector_uid == rp.detector_uid &&
           lp.effective_grouping == rp.effective_grouping &&
           lp.group_key == rp.group_key &&
           lp.subgroup_index == rp.subgroup_index &&
           lp.group_role == rp.group_role &&
           lp.delivered_flag_bits == rp.delivered_flag_bits &&
           lp.operation_exclusion_bits ==
               rp.operation_exclusion_bits &&
           lp.apt_flag_value == rp.apt_flag_value &&
           lp.exclusion_reason == rp.exclusion_reason;
}

inline NativeDetectorLedger seed_native_detector_ledger(
    const std::vector<NativeDetectorBlock> &blocks) {
    std::vector<NativeDetectorLedger::Seed> seeds;
    std::size_t reserve = 0;
    for (const auto &block : blocks) {
        const auto rows = static_cast<std::size_t>(
            block.measured_values().rows());
        const auto cols = static_cast<std::size_t>(
            block.measured_values().cols());
        if (cols != 0 && rows >
                (std::numeric_limits<std::size_t>::max() - reserve) / cols) {
            throw std::length_error(
                "native detector ledger cardinality would overflow");
        }
        reserve += rows * cols;
    }
    seeds.reserve(reserve);
    for (const auto &block : blocks) {
        for (Eigen::Index row = 0;
             row < block.measured_values().rows(); ++row) {
            const auto identity = block.identity(row);
            for (Eigen::Index col = 0;
                 col < block.measured_values().cols(); ++col) {
                const auto flag_bits = block.original_flag_bits()(row, col);
                seeds.push_back(NativeDetectorLedger::Seed{
                    identity, block.first_detector_column() + col,
                    block.measured_values()(row, col), flag_bits,
                    flag_bits == 0
                        ? std::string{}
                        : std::string{"delivered detector flag bits"}});
            }
        }
    }
    return NativeDetectorLedger{std::move(seeds)};
}

struct NativeDetectorColumnBinding {
    TimestreamDetectorColumn detector_column = -1;
    TimestreamDetectorUid detector_uid = -1;
    TimestreamNetworkId network_id = -1;

    friend bool operator==(const NativeDetectorColumnBinding &lhs,
                           const NativeDetectorColumnBinding &rhs) noexcept {
        return lhs.detector_column == rhs.detector_column &&
               lhs.detector_uid == rhs.detector_uid &&
               lhs.network_id == rhs.network_id;
    }
};

struct NativeDetectorInvalidity {
    NativeDetectorFlagBits delivered_flag_bits = 0;
    NativeDetectorFlagBits operation_exclusion_bits = 0;
    std::int64_t apt_flag_value = 0;
    std::string reason;

    friend bool operator==(const NativeDetectorInvalidity &lhs,
                           const NativeDetectorInvalidity &rhs) {
        return lhs.delivered_flag_bits == rhs.delivered_flag_bits &&
               lhs.operation_exclusion_bits ==
                   rhs.operation_exclusion_bits &&
               lhs.apt_flag_value == rhs.apt_flag_value &&
               lhs.reason == rhs.reason;
    }
};

inline bool native_detector_invalidity_present(
    const NativeDetectorInvalidity &invalidity) noexcept {
    return invalidity.delivered_flag_bits != 0 ||
           invalidity.operation_exclusion_bits != 0 ||
           invalidity.apt_flag_value != 0 ||
           !invalidity.reason.empty();
}

class NativeDetectorPcaWorkingSet {
public:
    const NativeOperationIdentity &operation() const noexcept {
        return operation_;
    }
    const std::vector<std::size_t> &relational_common_slots() const
        noexcept {
        return relational_common_slots_;
    }
    const std::vector<TimestreamNetworkId> &participant_network_ids() const
        noexcept {
        return participant_network_ids_;
    }
    const std::vector<NativeDetectorColumnBinding> &detector_bindings() const
        noexcept {
        return detector_bindings_;
    }
    Eigen::Index slot_count() const noexcept { return values_.rows(); }
    Eigen::Index detector_count() const noexcept { return values_.cols(); }
    const Eigen::MatrixXd &values() const noexcept { return values_; }
    Eigen::MatrixXd &mutable_values_for_pca() noexcept { return values_; }
    bool binding_finalized() const noexcept { return binding_finalized_; }
    const NativeDetectorBooleanMatrix &exclusion_flags() const noexcept {
        return exclusion_flags_;
    }
    const std::vector<CoincidenceCellState> &provenance_states() const
        noexcept {
        return provenance_states_;
    }
    const std::vector<std::optional<NativeSampleIdentity>> &
    mapped_identities() const noexcept {
        return mapped_identities_;
    }
    const std::vector<TimestreamNativeRevision> &expected_revisions() const
        noexcept {
        return expected_revisions_;
    }
    const std::vector<std::optional<NativeDetectorInvalidity>> &
    invalidity_provenance() const noexcept {
        return invalidity_provenance_;
    }
    const std::vector<std::optional<CoincidenceAbsenceReason>> &
    absence_reasons() const noexcept {
        return absence_reasons_;
    }
    const std::vector<std::size_t> &participant_indices() const noexcept {
        return participant_indices_;
    }

    const std::vector<CoincidenceCellState> &frozen_provenance_states()
        const noexcept {
        return frozen_provenance_states_;
    }
    const std::vector<std::optional<NativeSampleIdentity>> &
    frozen_mapped_identities() const noexcept {
        return frozen_mapped_identities_;
    }
    const std::vector<TimestreamNativeRevision> &
    frozen_expected_revisions() const noexcept {
        return frozen_expected_revisions_;
    }
    const std::vector<std::optional<NativeDetectorInvalidity>> &
    frozen_invalidity_provenance() const noexcept {
        return frozen_invalidity_provenance_;
    }
    const std::vector<std::optional<CoincidenceAbsenceReason>> &
    frozen_absence_reasons() const noexcept {
        return frozen_absence_reasons_;
    }
    const NativeDetectorBooleanMatrix &frozen_exclusion_flags() const
        noexcept {
        return frozen_exclusion_flags_;
    }

    void require_all_values_finite_for_pca() const {
        if (!values_.array().isFinite().all()) {
            throw std::logic_error(
                "native detector PCA working matrix must remain finite");
        }
    }

private:
    friend NativeDetectorPcaWorkingSet gather_native_detector_pca_working_set(
        const NativeDetectorLedger &, const NativeCohortSelection &,
        std::vector<NativeDetectorColumnBinding>,
        const NativeDetectorFlagBitsMatrix &, FinitePcaPlaceholder);
    friend void finalize_native_detector_pca_binding(
        NativeDetectorPcaWorkingSet &, const Eigen::VectorXi &,
        FinitePcaPlaceholder);

    void apply_apt_detector_exclusion(Eigen::Index detector,
                                      std::int64_t apt_flag_value,
                                      FinitePcaPlaceholder placeholder) {
        if (detector < 0 || detector >= detector_count() ||
            apt_flag_value == 0) {
            throw std::invalid_argument(
                "APT detector exclusion requires a valid column and nonzero flag");
        }
        for (Eigen::Index slot = 0; slot < slot_count(); ++slot) {
            const auto flat =
                static_cast<std::size_t>(slot) *
                    static_cast<std::size_t>(detector_count()) +
                static_cast<std::size_t>(detector);
            if (provenance_states_.at(flat) ==
                CoincidenceCellState::absent) {
                continue;
            }
            provenance_states_.at(flat) =
                CoincidenceCellState::mapped_invalid;
            exclusion_flags_(slot, detector) = true;
            values_(slot, detector) = placeholder.value();
            if (!invalidity_provenance_.at(flat).has_value()) {
                invalidity_provenance_.at(flat) =
                    NativeDetectorInvalidity{};
            }
            auto &invalidity = *invalidity_provenance_.at(flat);
            invalidity.apt_flag_value = apt_flag_value;
            if (!invalidity.reason.empty()) {
                invalidity.reason += "; ";
            }
            invalidity.reason += "production APT detector exclusion";
        }
    }

    NativeDetectorPcaWorkingSet(
        NativeOperationIdentity operation,
        std::vector<std::size_t> relational_common_slots,
        std::vector<TimestreamNetworkId> participant_network_ids,
        std::vector<NativeDetectorColumnBinding> detector_bindings,
        Eigen::MatrixXd values,
        NativeDetectorBooleanMatrix exclusion_flags,
        std::vector<CoincidenceCellState> provenance_states,
        std::vector<std::optional<NativeSampleIdentity>> mapped_identities,
        std::vector<TimestreamNativeRevision> expected_revisions,
        std::vector<std::optional<NativeDetectorInvalidity>>
            invalidity_provenance,
        std::vector<std::optional<CoincidenceAbsenceReason>> absence_reasons,
        std::vector<std::size_t> participant_indices)
        : operation_{operation},
          relational_common_slots_{std::move(relational_common_slots)},
          participant_network_ids_{std::move(participant_network_ids)},
          detector_bindings_{std::move(detector_bindings)},
          values_{std::move(values)},
          exclusion_flags_{std::move(exclusion_flags)},
          provenance_states_{std::move(provenance_states)},
          mapped_identities_{std::move(mapped_identities)},
          expected_revisions_{std::move(expected_revisions)},
          invalidity_provenance_{std::move(invalidity_provenance)},
          absence_reasons_{std::move(absence_reasons)},
          participant_indices_{std::move(participant_indices)},
          frozen_exclusion_flags_{exclusion_flags_},
          frozen_provenance_states_{provenance_states_},
          frozen_mapped_identities_{mapped_identities_},
          frozen_expected_revisions_{expected_revisions_},
          frozen_invalidity_provenance_{invalidity_provenance_},
          frozen_absence_reasons_{absence_reasons_} {}

    void freeze_binding() {
        frozen_exclusion_flags_ = exclusion_flags_;
        frozen_provenance_states_ = provenance_states_;
        frozen_mapped_identities_ = mapped_identities_;
        frozen_expected_revisions_ = expected_revisions_;
        frozen_invalidity_provenance_ = invalidity_provenance_;
        frozen_absence_reasons_ = absence_reasons_;
    }

    NativeOperationIdentity operation_;
    std::vector<std::size_t> relational_common_slots_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::vector<NativeDetectorColumnBinding> detector_bindings_;
    Eigen::MatrixXd values_;
    NativeDetectorBooleanMatrix exclusion_flags_;
    std::vector<CoincidenceCellState> provenance_states_;
    std::vector<std::optional<NativeSampleIdentity>> mapped_identities_;
    std::vector<TimestreamNativeRevision> expected_revisions_;
    std::vector<std::optional<NativeDetectorInvalidity>>
        invalidity_provenance_;
    std::vector<std::optional<CoincidenceAbsenceReason>> absence_reasons_;
    std::vector<std::size_t> participant_indices_;
    NativeDetectorBooleanMatrix frozen_exclusion_flags_;
    std::vector<CoincidenceCellState> frozen_provenance_states_;
    std::vector<std::optional<NativeSampleIdentity>>
        frozen_mapped_identities_;
    std::vector<TimestreamNativeRevision> frozen_expected_revisions_;
    std::vector<std::optional<NativeDetectorInvalidity>>
        frozen_invalidity_provenance_;
    std::vector<std::optional<CoincidenceAbsenceReason>>
        frozen_absence_reasons_;
    bool binding_finalized_ = false;
};

inline void finalize_native_detector_pca_binding(
    NativeDetectorPcaWorkingSet &working_set,
    const Eigen::VectorXi &apt_flags,
    FinitePcaPlaceholder placeholder) {
    if (working_set.binding_finalized_ ||
        apt_flags.size() != working_set.detector_count()) {
        throw std::logic_error(
            "native detector PCA binding finalization is invalid");
    }
    for (Eigen::Index detector = 0; detector < apt_flags.size(); ++detector) {
        if (apt_flags(detector) != 0) {
            working_set.apply_apt_detector_exclusion(
                detector, apt_flags(detector), placeholder);
        }
    }
    working_set.freeze_binding();
    working_set.binding_finalized_ = true;
}

inline NativeDetectorPcaWorkingSet gather_native_detector_pca_working_set(
    const NativeDetectorLedger &ledger,
    const NativeCohortSelection &selection,
    std::vector<NativeDetectorColumnBinding> detector_bindings,
    const NativeDetectorFlagBitsMatrix &actual_exclusion_bits,
    FinitePcaPlaceholder excluded_placeholder) {
    const auto &cohort = selection.cohort();
    const auto n_slots = static_cast<Eigen::Index>(cohort.slot_count());
    const auto n_dets = static_cast<Eigen::Index>(detector_bindings.size());
    if (n_dets <= 0) {
        throw std::invalid_argument(
            "native detector PCA group must contain detector columns");
    }
    if (actual_exclusion_bits.rows() != n_slots ||
        actual_exclusion_bits.cols() != n_dets) {
        throw std::invalid_argument(
            "actual PCA exclusion mask must match cohort/group shape");
    }

    std::set<TimestreamDetectorColumn> seen_columns;
    std::vector<std::size_t> participant_index_by_detector;
    participant_index_by_detector.reserve(detector_bindings.size());
    for (const auto &binding : detector_bindings) {
        if (binding.detector_column < 0 || binding.detector_uid < 0 ||
            binding.network_id < 0) {
            throw std::invalid_argument(
                "native detector binding requires nonnegative identities");
        }
        if (!seen_columns.insert(binding.detector_column).second) {
            throw std::invalid_argument(
                "native detector PCA group contains a duplicate column");
        }
        const auto it = std::find(
            cohort.participant_network_ids().begin(),
            cohort.participant_network_ids().end(), binding.network_id);
        if (it == cohort.participant_network_ids().end()) {
            throw std::invalid_argument(
                "detector network is absent from the frozen cohort");
        }
        participant_index_by_detector.push_back(
            static_cast<std::size_t>(
                it - cohort.participant_network_ids().begin()));
    }

    Eigen::MatrixXd values(n_slots, n_dets);
    NativeDetectorBooleanMatrix exclusion_flags(n_slots, n_dets);
    exclusion_flags.setConstant(true);
    std::vector<CoincidenceCellState> states;
    std::vector<std::optional<NativeSampleIdentity>> identities;
    std::vector<TimestreamNativeRevision> revisions;
    std::vector<std::optional<NativeDetectorInvalidity>> invalidities;
    std::vector<std::optional<CoincidenceAbsenceReason>> absences;
    std::vector<std::size_t> participant_indices;
    const auto flat_size = static_cast<std::size_t>(n_slots) *
                           static_cast<std::size_t>(n_dets);
    states.reserve(flat_size);
    identities.reserve(flat_size);
    revisions.reserve(flat_size);
    invalidities.reserve(flat_size);
    absences.reserve(flat_size);
    participant_indices.reserve(flat_size);

    for (Eigen::Index slot = 0; slot < n_slots; ++slot) {
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            const auto participant_index =
                participant_index_by_detector.at(
                    static_cast<std::size_t>(det));
            participant_indices.push_back(participant_index);
            const auto &cell = cohort.cell(
                static_cast<std::size_t>(slot), participant_index);
            identities.push_back(cell.identity());
            revisions.push_back(cell.expected_revision());
            if (!cell.is_mapped()) {
                states.push_back(CoincidenceCellState::absent);
                invalidities.push_back(std::nullopt);
                absences.push_back(cell.absence_reason());
                values(slot, det) = excluded_placeholder.value();
                continue;
            }

            const auto &identity = *cell.identity();
            const auto &binding = detector_bindings.at(
                static_cast<std::size_t>(det));
            const NativeDetectorSampleKey key{
                identity.key(), binding.detector_column};
            const auto &record = ledger.at(key);
            if (!(record.identity == identity) ||
                record.detector_column != binding.detector_column) {
                throw std::logic_error(
                    "native detector cohort timestamp or column changed");
            }
            if (record.revision != cell.expected_revision()) {
                throw std::logic_error(
                    "native detector cohort revision is stale during gather");
            }

            NativeDetectorFlagBits invalid_bits =
                actual_exclusion_bits(slot, det);
            const NativeDetectorFlagBits delivered_flag_bits =
                record.original_flag_bits;
            std::string invalid_reason = record.original_flag_reason;
            if (!cell.pca_valid()) {
                invalid_bits |= cell.invalidity()->original_flag_bits;
                if (!invalid_reason.empty() &&
                    !cell.invalidity()->reason.empty()) {
                    invalid_reason += "; ";
                }
                invalid_reason += cell.invalidity()->reason;
            }
            if (actual_exclusion_bits(slot, det) != 0) {
                if (!invalid_reason.empty()) {
                    invalid_reason += "; ";
                }
                invalid_reason += "actual production PCA exclusion mask";
            }

            if (!cell.pca_valid() || delivered_flag_bits != 0 ||
                invalid_bits != 0 ||
                !invalid_reason.empty()) {
                if (delivered_flag_bits == 0 && invalid_bits == 0 &&
                    invalid_reason.empty()) {
                    throw std::logic_error(
                        "mapped-invalid detector cell lacks provenance");
                }
                states.push_back(CoincidenceCellState::mapped_invalid);
                invalidities.push_back(NativeDetectorInvalidity{
                    delivered_flag_bits, invalid_bits, 0,
                    std::move(invalid_reason)});
                absences.push_back(std::nullopt);
                values(slot, det) = excluded_placeholder.value();
                continue;
            }

            if (!std::isfinite(record.current_value)) {
                throw std::logic_error(
                    "PCA-valid native detector value must be finite");
            }
            states.push_back(CoincidenceCellState::mapped_valid);
            invalidities.push_back(std::nullopt);
            absences.push_back(std::nullopt);
            exclusion_flags(slot, det) = false;
            values(slot, det) = record.current_value;
        }
    }

    NativeDetectorPcaWorkingSet result{
        cohort.operation(), selection.relational_common_slots(),
        cohort.participant_network_ids(), std::move(detector_bindings),
        std::move(values), std::move(exclusion_flags), std::move(states),
        std::move(identities), std::move(revisions),
        std::move(invalidities), std::move(absences),
        std::move(participant_indices)};
    result.require_all_values_finite_for_pca();
    return result;
}

class NativeDetectorPcaCompatibilityClassification {
public:
    bool compatible() const noexcept { return hazards_ == 0; }
    bool has(PcaCompatibilityHazard hazard) const noexcept {
        return (hazards_ & static_cast<std::uint8_t>(hazard)) != 0;
    }

private:
    friend NativeDetectorPcaCompatibilityClassification
    classify_native_detector_pca_compatibility(
        const NativeDetectorPcaWorkingSet &,
        const PcaCompatibilityInputs &);
    friend NativeDetectorPcaCompatibilityClassification
    classify_native_detector_pca_compatibility(
        bool, const PcaCompatibilityInputs &);
    explicit NativeDetectorPcaCompatibilityClassification(
        std::uint8_t hazards)
        : hazards_{hazards} {}
    std::uint8_t hazards_ = 0;
};

inline NativeDetectorPcaCompatibilityClassification
classify_native_detector_pca_compatibility(
    bool has_excluded_cells,
    const PcaCompatibilityInputs &inputs) {
    std::uint8_t hazards = 0;
    if (has_excluded_cells && inputs.null_model_active_for_operation) {
        hazards |=
            static_cast<std::uint8_t>(PcaCompatibilityHazard::null_model);
    }
    if (has_excluded_cells &&
        inputs.adaptive_selector_active_for_operation) {
        hazards |= static_cast<std::uint8_t>(
            PcaCompatibilityHazard::adaptive_selector);
    }
    if (has_excluded_cells &&
        inputs.marchenko_pastur_active_for_operation &&
        inputs.marchenko_pastur_band_requested) {
        hazards |= static_cast<std::uint8_t>(
            PcaCompatibilityHazard::band_limited_marchenko_pastur);
    }
    return NativeDetectorPcaCompatibilityClassification{hazards};
}

inline NativeDetectorPcaCompatibilityClassification
classify_native_detector_pca_compatibility(
    const NativeDetectorPcaWorkingSet &working_set,
    const PcaCompatibilityInputs &inputs) {
    return classify_native_detector_pca_compatibility(
        working_set.exclusion_flags().count() != 0, inputs);
}

inline void require_native_detector_pca_compatibility(
    const NativeDetectorPcaCompatibilityClassification &classification) {
    if (!classification.compatible()) {
        throw std::logic_error(
            "actual native detector exclusions are incompatible with the "
            "requested optional PCA mode; no fallback is selected");
    }
}

struct NativePreparedPcaGroup {
    NativePreparedPcaGroup(
        std::string effective_grouping_, Eigen::Index group_key_,
        Eigen::Index subgroup_index_, NativePreparedPcaGroupRole role_,
        std::vector<TimestreamDetectorColumn> detector_columns_,
        std::vector<TimestreamDetectorUid> detector_uids_,
        Eigen::VectorXi apt_flags_,
        NativeDetectorPcaWorkingSet working_set_)
        : effective_grouping{std::move(effective_grouping_)},
          group_key{group_key_},
          subgroup_index{subgroup_index_}, role{role_},
          detector_columns{std::move(detector_columns_)},
          detector_uids{std::move(detector_uids_)},
          apt_flags{std::move(apt_flags_)},
          working_set{std::move(working_set_)},
          frozen_effective_grouping_{effective_grouping},
          frozen_group_key_{group_key},
          frozen_subgroup_index_{subgroup_index}, frozen_role_{role},
          frozen_detector_columns_{detector_columns},
          frozen_detector_uids_{detector_uids},
          frozen_apt_flags_{apt_flags} {}

    std::string effective_grouping;
    Eigen::Index group_key = -1;
    Eigen::Index subgroup_index = -1;
    NativePreparedPcaGroupRole role = NativePreparedPcaGroupRole::pca_clean;
    std::vector<TimestreamDetectorColumn> detector_columns;
    std::vector<TimestreamDetectorUid> detector_uids;
    Eigen::VectorXi apt_flags;
    NativeDetectorPcaWorkingSet working_set;
    bool group_identity_is_frozen() const {
        return effective_grouping == frozen_effective_grouping_ &&
               group_key == frozen_group_key_ &&
               subgroup_index == frozen_subgroup_index_ &&
               role == frozen_role_ &&
               detector_columns == frozen_detector_columns_ &&
               detector_uids == frozen_detector_uids_ &&
               apt_flags.size() == frozen_apt_flags_.size() &&
               !(apt_flags.array() != frozen_apt_flags_.array()).any();
    }

private:
    std::string frozen_effective_grouping_;
    Eigen::Index frozen_group_key_ = -1;
    Eigen::Index frozen_subgroup_index_ = -1;
    NativePreparedPcaGroupRole frozen_role_ =
        NativePreparedPcaGroupRole::pca_clean;
    std::vector<TimestreamDetectorColumn> frozen_detector_columns_;
    std::vector<TimestreamDetectorUid> frozen_detector_uids_;
    Eigen::VectorXi frozen_apt_flags_;
};

struct NativePreparedPcaOperation {
    NativePreparedPcaOperation(NativeOperationIdentity operation_,
                               std::string grouping_,
                               Eigen::Index detector_count_)
        : operation{operation_}, grouping{std::move(grouping_)},
          detector_count{detector_count_},
          frozen_grouping_{grouping},
          frozen_detector_count_{detector_count} {}

    NativeOperationIdentity operation;
    std::string grouping;
    Eigen::Index detector_count = 0;
    std::vector<NativePreparedPcaGroup> groups;

    void require_complete_detector_partition() const {
        if (grouping != frozen_grouping_ ||
            detector_count != frozen_detector_count_ ||
            detector_count <= 0 || groups.empty()) {
            throw std::logic_error(
                "native PCA preparation lacks a detector partition");
        }
        std::vector<bool> seen(
            static_cast<std::size_t>(detector_count), false);
        for (const auto &group : groups) {
            if (!(group.working_set.operation() == operation) ||
                !group.working_set.binding_finalized() ||
                !group.group_identity_is_frozen() ||
                group.effective_grouping != grouping ||
                group.group_key < 0 || group.subgroup_index < 0 ||
                group.detector_columns.size() !=
                    static_cast<std::size_t>(
                        group.working_set.detector_count()) ||
                group.detector_uids.size() !=
                    group.detector_columns.size() ||
                group.apt_flags.size() !=
                    group.working_set.detector_count()) {
                throw std::logic_error(
                    "native PCA prepared group has inconsistent identity or shape");
            }
            for (std::size_t binding_index = 0;
                 binding_index < group.detector_columns.size();
                 ++binding_index) {
                const auto &binding =
                    group.working_set.detector_bindings().at(
                        binding_index);
                if (group.detector_columns.at(binding_index) !=
                        binding.detector_column ||
                    group.detector_uids.at(binding_index) !=
                        binding.detector_uid) {
                    throw std::logic_error(
                        "native PCA group detector identity changed");
                }
                if (binding.detector_column < 0 ||
                    binding.detector_column >= detector_count ||
                    seen.at(static_cast<std::size_t>(
                        binding.detector_column))) {
                    throw std::logic_error(
                        "native PCA detector partition is not injective");
                }
                seen[static_cast<std::size_t>(
                    binding.detector_column)] = true;
            }
        }
        if (!std::all_of(seen.begin(), seen.end(),
                         [](bool value) { return value; })) {
            throw std::logic_error(
                "native PCA detector partition is incomplete");
        }
    }

private:
    std::string frozen_grouping_;
    Eigen::Index frozen_detector_count_ = 0;
};

inline void scatter_native_detector_pca_results_transactionally(
    NativeDetectorLedger &ledger,
    const NativeCohortSelection &selection,
    const NativePreparedPcaOperation &prepared) {
    const auto &cohort = selection.cohort();
    if (!(prepared.operation == cohort.operation()) ||
        prepared.grouping.empty()) {
        throw std::logic_error(
            "native PCA prepared operation does not match the cohort");
    }
    prepared.require_complete_detector_partition();

    std::vector<NativeDetectorLedger::Update> updates;
    for (const auto &group : prepared.groups) {
        const auto &working = group.working_set;
        if (!(working.operation() == cohort.operation()) ||
            working.relational_common_slots() !=
                selection.relational_common_slots() ||
            working.participant_network_ids() !=
                cohort.participant_network_ids() ||
            working.slot_count() !=
                static_cast<Eigen::Index>(cohort.slot_count())) {
            throw std::logic_error(
                "native PCA working result changed cohort identity");
        }
        if (working.exclusion_flags().rows() !=
                working.frozen_exclusion_flags().rows() ||
            working.exclusion_flags().cols() !=
                working.frozen_exclusion_flags().cols() ||
            (working.exclusion_flags().array() !=
             working.frozen_exclusion_flags().array())
                    .any() ||
            working.provenance_states() !=
                working.frozen_provenance_states() ||
            working.mapped_identities() !=
                working.frozen_mapped_identities() ||
            working.expected_revisions() !=
                working.frozen_expected_revisions() ||
            working.invalidity_provenance() !=
                working.frozen_invalidity_provenance() ||
            working.absence_reasons() !=
                working.frozen_absence_reasons()) {
            throw std::logic_error(
                "native PCA result mapping or exclusion binding changed");
        }
        if (group.detector_columns.size() !=
            working.detector_bindings().size()) {
            throw std::logic_error(
                "native PCA result detector-group shape changed");
        }
        for (std::size_t detector = 0;
             detector < group.detector_columns.size(); ++detector) {
            if (group.detector_columns[detector] !=
                    working.detector_bindings()[detector].detector_column ||
                group.detector_uids[detector] !=
                    working.detector_bindings()[detector].detector_uid) {
                throw std::logic_error(
                    "native PCA result detector-group binding changed");
            }
        }

        const auto n_dets = working.detector_count();
        const auto expected_flat =
            static_cast<std::size_t>(working.slot_count()) *
            static_cast<std::size_t>(n_dets);
        if (working.provenance_states().size() != expected_flat ||
            working.mapped_identities().size() != expected_flat ||
            working.expected_revisions().size() != expected_flat ||
            working.invalidity_provenance().size() != expected_flat ||
            working.absence_reasons().size() != expected_flat ||
            working.participant_indices().size() != expected_flat) {
            throw std::logic_error(
                "native PCA result provenance shape changed");
        }

        for (Eigen::Index slot = 0; slot < working.slot_count(); ++slot) {
            for (Eigen::Index det = 0; det < n_dets; ++det) {
                const auto flat =
                    static_cast<std::size_t>(slot) *
                        static_cast<std::size_t>(n_dets) +
                    static_cast<std::size_t>(det);
                const auto participant_index =
                    working.participant_indices().at(flat);
                const auto &cell = cohort.cell(
                    static_cast<std::size_t>(slot), participant_index);
                if (!(working.mapped_identities().at(flat) ==
                      cell.identity()) ||
                    working.expected_revisions().at(flat) !=
                        cell.expected_revision()) {
                    throw std::logic_error(
                        "native PCA result mapping or revision changed");
                }

                const auto state =
                    working.provenance_states().at(flat);
                if (!cell.is_mapped()) {
                    if (state != CoincidenceCellState::absent ||
                        working.absence_reasons().at(flat) !=
                            cell.absence_reason()) {
                        throw std::logic_error(
                            "native PCA absence provenance changed");
                    }
                    continue;
                }

                const auto &identity = *cell.identity();
                const auto &binding = working.detector_bindings().at(
                    static_cast<std::size_t>(det));
                NativeDetectorCoincidenceProvenance provenance{
                    working.relational_common_slots().at(
                        static_cast<std::size_t>(slot)),
                    participant_index, binding.network_id,
                    binding.detector_column, binding.detector_uid,
                    group.effective_grouping, group.group_key,
                    group.subgroup_index, group.role, 0, 0, 0, {}};

                if (state == CoincidenceCellState::mapped_invalid) {
                    const auto &invalidity =
                        working.invalidity_provenance().at(flat);
                    if (!invalidity.has_value() ||
                        !native_detector_invalidity_present(*invalidity)) {
                        throw std::logic_error(
                            "native PCA invalid result lost provenance");
                    }
                    provenance.delivered_flag_bits =
                        invalidity->delivered_flag_bits;
                    provenance.operation_exclusion_bits =
                        invalidity->operation_exclusion_bits;
                    provenance.apt_flag_value =
                        invalidity->apt_flag_value;
                    provenance.exclusion_reason = invalidity->reason;
                    updates.push_back(
                        NativeDetectorLedger::Update::preserve_invalid(
                            identity, binding.detector_column,
                            working.expected_revisions().at(flat),
                            std::move(provenance)));
                    continue;
                }
                if (group.role ==
                    NativePreparedPcaGroupRole::pass_through) {
                    if (state != CoincidenceCellState::mapped_valid ||
                        working.exclusion_flags()(slot, det)) {
                        throw std::logic_error(
                            "corr_nw pass-through cell is not PCA-valid");
                    }
                    updates.push_back(
                        NativeDetectorLedger::Update::
                            preserve_corr_ungrouped(
                                identity, binding.detector_column,
                                working.expected_revisions().at(flat),
                                std::move(provenance)));
                    continue;
                }
                if (group.role != NativePreparedPcaGroupRole::pca_clean) {
                    throw std::logic_error(
                        "native PCA group role is invalid");
                }
                if (state != CoincidenceCellState::mapped_valid ||
                    working.exclusion_flags()(slot, det)) {
                    throw std::logic_error(
                        "native PCA valid/excluded state changed");
                }
                const double value = working.values()(slot, det);
                if (!std::isfinite(value)) {
                    throw std::logic_error(
                        "native detector PCA result must be finite");
                }
                updates.push_back(
                    NativeDetectorLedger::Update::replacement(
                        identity, binding.detector_column,
                        working.expected_revisions().at(flat), value,
                        std::move(provenance)));
            }
        }
    }
    ledger.apply_transaction(prepared.operation, updates);
}

struct NativeStrideSupport {
    std::size_t run_ordinal = 0;
    TimestreamNativeRow run_output_row = -1;
    NativeSampleIdentity selected_anchor;
    int factor = 1;
    TimestreamNativeRow first_support_native_row = -1;
    TimestreamNativeRow past_last_support_native_row = -1;
    std::vector<NativeSampleIdentity> exact_support_rows;
    bool final_short_support = false;
    std::vector<TimestreamDetectorColumn> detector_columns;
    std::vector<NativeDetectorFlagBits> ored_flag_support;
};

struct NativeRunDownsampleResult {
    NativeContiguousRun run;
    Eigen::MatrixXd selected_values;
    NativeDetectorBooleanMatrix ored_flags;
    std::vector<NativeStrideSupport> support;
};

inline std::vector<NativeStrideSupport> make_native_stride_support(
    const NativeDetectorBlock &block,
    const NativeContiguousRun &run,
    const NativeDetectorFlagBitsMatrix &actual_run_flag_bits,
    int factor,
    std::size_t run_ordinal) {
    if (factor <= 0) {
        throw std::invalid_argument(
            "native RTC downsample factor must be positive");
    }
    if (run.network_id != block.network_id() ||
        run.first_native_row < block.first_native_row() ||
        run.past_last_native_row > block.past_last_native_row() ||
        run.first_native_row >= run.past_last_native_row) {
        throw std::invalid_argument(
            "native RTC run is outside its measured detector block");
    }
    if (actual_run_flag_bits.rows() != run.row_count() ||
        actual_run_flag_bits.cols() != block.measured_values().cols()) {
        throw std::invalid_argument(
            "native RTC actual flag support must match the run shape");
    }

    const auto run_rows = run.row_count();
    const auto output_rows =
        (run_rows + static_cast<TimestreamNativeRow>(factor) - 1) /
        static_cast<TimestreamNativeRow>(factor);
    std::vector<NativeStrideSupport> result;
    result.reserve(static_cast<std::size_t>(output_rows));
    for (TimestreamNativeRow output_row = 0;
         output_row < output_rows; ++output_row) {
        const auto first =
            run.first_native_row +
            output_row * static_cast<TimestreamNativeRow>(factor);
        const auto past = std::min(
            first + static_cast<TimestreamNativeRow>(factor),
            run.past_last_native_row);
        std::vector<NativeSampleIdentity> identities;
        identities.reserve(static_cast<std::size_t>(past - first));
        for (auto native_row = first; native_row < past; ++native_row) {
            identities.push_back(
                block.identity(block.local_row(native_row)));
        }
        std::vector<TimestreamDetectorColumn> detector_columns;
        std::vector<NativeDetectorFlagBits> ored_flags(
            static_cast<std::size_t>(block.measured_values().cols()), 0);
        detector_columns.reserve(
            static_cast<std::size_t>(block.measured_values().cols()));
        for (Eigen::Index det = 0;
             det < block.measured_values().cols(); ++det) {
            detector_columns.push_back(
                block.first_detector_column() + det);
            for (auto native_row = first; native_row < past; ++native_row) {
                ored_flags.at(static_cast<std::size_t>(det)) |=
                    actual_run_flag_bits(
                        static_cast<Eigen::Index>(
                            native_row - run.first_native_row),
                        det);
            }
        }
        result.push_back(NativeStrideSupport{
            run_ordinal, output_row, identities.front(), factor, first, past,
            std::move(identities), past - first < factor,
            std::move(detector_columns), std::move(ored_flags)});
    }
    return result;
}

}  // namespace citlali::pipeline
