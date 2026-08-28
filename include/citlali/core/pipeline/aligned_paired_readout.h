#pragma once

#include <citlali/core/pipeline/paired_readout.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

namespace citlali::pipeline {

struct AlignedReadoutCellIdentity {
    TimestreamNetworkId network_id = -1;
    std::size_t common_slot = 0;
    std::int64_t detector_uid = 0;
    TimestreamNativeRow native_row = -1;

    bool has_native_occurrence() const noexcept { return native_row >= 0; }

    friend bool operator==(const AlignedReadoutCellIdentity &,
                           const AlignedReadoutCellIdentity &) = default;
    friend bool operator<(const AlignedReadoutCellIdentity &lhs,
                          const AlignedReadoutCellIdentity &rhs) noexcept {
        if (lhs.network_id != rhs.network_id) {
            return lhs.network_id < rhs.network_id;
        }
        if (lhs.common_slot != rhs.common_slot) {
            return lhs.common_slot < rhs.common_slot;
        }
        return lhs.detector_uid < rhs.detector_uid;
    }
};

// A read-only ALIGN relation over PairedReadout.  It adds no numerical plane,
// per-cell identity storage, or AST coordinate.  Values and native axes remain
// owned by the paired producer product; slot admission remains ALIGN-owned.
class AlignedPairedReadout {
public:
    static std::shared_ptr<const AlignedPairedReadout> admit(
        std::shared_ptr<const PairedReadout> native_parent,
        std::shared_ptr<const NativeAlignmentPlan> alignment,
        std::size_t first_common_slot,
        std::size_t past_last_common_slot) {
        if (!native_parent || !alignment ||
            !(native_parent->scope() == alignment->scope()) ||
            first_common_slot >= past_last_common_slot ||
            past_last_common_slot > alignment->slot_count() ||
            native_parent->network_count() != alignment->networks().size()) {
            throw std::invalid_argument(
                "aligned paired readout parent or slot interval is incomplete");
        }

        std::size_t detector_count = 0;
        std::size_t aligned_cell_count = 0;
        std::size_t mapped_cell_count = 0;
        for (const auto network_id : alignment->participant_network_ids()) {
            const auto &native = native_parent->network(network_id);
            const auto &paired_axis = *native.occurrence_axis_handle();
            const auto &alignment_axis = alignment->network(network_id);
            if (paired_axis.network_id() != alignment_axis.network_id()) {
                throw std::invalid_argument(
                    "paired readout network differs from ALIGN timing authority");
            }
            for (auto row = paired_axis.first_native_row();
                 row < paired_axis.past_last_native_row(); ++row) {
                if (!(paired_axis.identity(row) == alignment_axis.identity(row)) ||
                    paired_axis.native_timing_handle()->packet_counter(row) !=
                        alignment_axis.packet_counter(row)) {
                    throw std::invalid_argument(
                        "paired readout native timing differs from ALIGN authority");
                }
            }
            const auto detectors =
                static_cast<std::size_t>(native.detector_count());
            if (detector_count >
                std::numeric_limits<std::size_t>::max() - detectors) {
                throw std::length_error(
                    "aligned paired readout detector count would overflow");
            }
            detector_count += detectors;
            const auto slots = past_last_common_slot - first_common_slot;
            if (detectors != 0 &&
                slots > std::numeric_limits<std::size_t>::max() / detectors) {
                throw std::length_error(
                    "aligned paired readout cardinality would overflow");
            }
            const auto cells = slots * detectors;
            if (aligned_cell_count >
                std::numeric_limits<std::size_t>::max() - cells) {
                throw std::length_error(
                    "aligned paired readout cell count would overflow");
            }
            aligned_cell_count += cells;
            for (std::size_t slot = first_common_slot;
                 slot < past_last_common_slot; ++slot) {
                const auto &association = alignment->association(
                    network_id, slot);
                if (!association.mapped()) continue;
                if (association.native_row < paired_axis.first_native_row() ||
                    association.native_row >=
                        paired_axis.past_last_native_row()) {
                    throw std::invalid_argument(
                        "ALIGN slot maps outside paired native support");
                }
                if (mapped_cell_count >
                    std::numeric_limits<std::size_t>::max() - detectors) {
                    throw std::length_error(
                        "aligned paired readout mapped count would overflow");
                }
                mapped_cell_count += detectors;
            }
        }

        return std::shared_ptr<const AlignedPairedReadout>(
            new AlignedPairedReadout{
                std::move(native_parent), std::move(alignment),
                first_common_slot, past_last_common_slot, detector_count,
                aligned_cell_count, mapped_cell_count});
    }

    const std::shared_ptr<const PairedReadout> &native_parent_handle()
        const noexcept {
        return native_parent_;
    }
    const std::shared_ptr<const NativeAlignmentPlan> &alignment_handle()
        const noexcept {
        return alignment_;
    }
    const NativeObservationScope &scope() const noexcept {
        return native_parent_->scope();
    }
    std::size_t first_common_slot() const noexcept {
        return first_common_slot_;
    }
    std::size_t past_last_common_slot() const noexcept {
        return past_last_common_slot_;
    }
    std::size_t common_slot_count() const noexcept {
        return past_last_common_slot_ - first_common_slot_;
    }
    std::size_t detector_count() const noexcept { return detector_count_; }
    std::size_t aligned_cell_count() const noexcept {
        return aligned_cell_count_;
    }
    std::size_t mapped_cell_count() const noexcept {
        return mapped_cell_count_;
    }
    double common_slot_time_unix_sec(std::size_t common_slot) const {
        require_slot(common_slot);
        return alignment_->common_slot_reference_times_unix_sec()(
            static_cast<Eigen::Index>(common_slot));
    }
    const PairedReadoutNetwork &network(
        TimestreamNetworkId network_id) const {
        return native_parent_->network(network_id);
    }
    bool mapped(TimestreamNetworkId network_id,
                std::size_t common_slot) const {
        require_slot(common_slot);
        return alignment_->association(network_id, common_slot).mapped();
    }
    CoincidenceAbsenceReason absence_reason(
        TimestreamNetworkId network_id, std::size_t common_slot) const {
        require_slot(common_slot);
        const auto &association = alignment_->association(
            network_id, common_slot);
        if (association.mapped()) {
            throw std::logic_error(
                "mapped aligned pair has no absence reason");
        }
        return association.absence_reason;
    }
    AlignedReadoutCellIdentity identity(
        TimestreamNetworkId network_id, std::size_t common_slot,
        Eigen::Index detector_index) const {
        require_slot(common_slot);
        const auto &native = network(network_id);
        const auto &detector = native.detector(detector_index);
        const auto &association = alignment_->association(
            network_id, common_slot);
        return AlignedReadoutCellIdentity{
            network_id, common_slot, detector.output_uid,
            association.mapped() ? association.native_row : -1};
    }
    std::optional<NativeSampleIdentity> representative_native_identity(
        TimestreamNetworkId network_id, std::size_t common_slot) const {
        require_slot(common_slot);
        const auto &association = alignment_->association(
            network_id, common_slot);
        if (!association.mapped()) return std::nullopt;
        return network(network_id).occurrence_axis_handle()->identity(
            association.native_row);
    }
    std::optional<double> value(
        ReadoutMember member, TimestreamNetworkId network_id,
        std::size_t common_slot, Eigen::Index detector_index) const {
        require_slot(common_slot);
        const auto &association = alignment_->association(
            network_id, common_slot);
        if (!association.mapped()) return std::nullopt;
        return network(network_id).value(
            member, association.native_row, detector_index);
    }
    std::optional<ReadoutMemberState> state(
        ReadoutMember member, TimestreamNetworkId network_id,
        std::size_t common_slot, Eigen::Index detector_index) const {
        require_slot(common_slot);
        const auto &association = alignment_->association(
            network_id, common_slot);
        if (!association.mapped()) return std::nullopt;
        return network(network_id).state(
            member, association.native_row, detector_index);
    }
    std::optional<PairedReadoutCause> native_pair_causes(
        TimestreamNetworkId network_id, std::size_t common_slot,
        Eigen::Index detector_index) const {
        require_slot(common_slot);
        const auto &association = alignment_->association(
            network_id, common_slot);
        if (!association.mapped()) return std::nullopt;
        return network(network_id).pair_causes(
            association.native_row, detector_index);
    }

private:
    AlignedPairedReadout(
        std::shared_ptr<const PairedReadout> native_parent,
        std::shared_ptr<const NativeAlignmentPlan> alignment,
        std::size_t first_common_slot,
        std::size_t past_last_common_slot,
        std::size_t detector_count,
        std::size_t aligned_cell_count,
        std::size_t mapped_cell_count)
        : native_parent_{std::move(native_parent)},
          alignment_{std::move(alignment)},
          first_common_slot_{first_common_slot},
          past_last_common_slot_{past_last_common_slot},
          detector_count_{detector_count},
          aligned_cell_count_{aligned_cell_count},
          mapped_cell_count_{mapped_cell_count} {}

    void require_slot(std::size_t common_slot) const {
        if (common_slot < first_common_slot_ ||
            common_slot >= past_last_common_slot_) {
            throw std::out_of_range(
                "common slot is outside aligned paired readout support");
        }
    }

    std::shared_ptr<const PairedReadout> native_parent_;
    std::shared_ptr<const NativeAlignmentPlan> alignment_;
    std::size_t first_common_slot_;
    std::size_t past_last_common_slot_;
    std::size_t detector_count_;
    std::size_t aligned_cell_count_;
    std::size_t mapped_cell_count_;
};

}  // namespace citlali::pipeline
