#pragma once

#include <citlali/core/pipeline/paired_readout.h>
#include <citlali/core/pipeline/timestream_native_alignment.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct CommonAnalysisGridCellIdentity {
    TimestreamNetworkId network_id = -1;
    std::size_t analysis_slot = 0;
    std::int64_t detector_uid = 0;
    TimestreamNativeRow native_row = -1;

    bool has_native_occurrence() const noexcept { return native_row >= 0; }

    friend bool operator==(const CommonAnalysisGridCellIdentity &,
                           const CommonAnalysisGridCellIdentity &) = default;
    friend bool operator<(const CommonAnalysisGridCellIdentity &lhs,
                          const CommonAnalysisGridCellIdentity &rhs) noexcept {
        if (lhs.network_id != rhs.network_id) {
            return lhs.network_id < rhs.network_id;
        }
        if (lhs.analysis_slot != rhs.analysis_slot) {
            return lhs.analysis_slot < rhs.analysis_slot;
        }
        return lhs.detector_uid < rhs.detector_uid;
    }
};

// Explicit ALIGN-owned cross-network relation. The legacy shared-slot carrier
// remains an implementation detail behind this terminology-correct boundary.
// Construction is opt-in and does not modify any source network axis.
class CommonAnalysisGridRelation {
public:
    static std::shared_ptr<const CommonAnalysisGridRelation>
    admit(NativeObservationScope scope,
          std::vector<NativeNetworkAlignment> network_timings,
          Eigen::VectorXd grid_times_unix_sec,
          std::map<TimestreamNetworkId, std::vector<NativeSlotAssociation>>
              source_associations) {
        auto implementation = std::make_shared<const NativeAlignmentPlan>(
            std::move(scope), std::move(network_timings),
            std::move(grid_times_unix_sec), std::move(source_associations));
        return std::shared_ptr<const CommonAnalysisGridRelation>(
            new CommonAnalysisGridRelation{std::move(implementation)});
    }

    const NativeObservationScope &scope() const noexcept {
        return implementation_->scope();
    }
    std::span<const TimestreamNetworkId>
    participant_network_ids() const noexcept {
        return implementation_->participant_network_ids();
    }
    std::size_t slot_count() const noexcept {
        return implementation_->slot_count();
    }
    double grid_time_unix_sec(std::size_t analysis_slot) const {
        if (analysis_slot >= slot_count()) {
            throw std::out_of_range(
                "common-analysis-grid slot is outside relation support");
        }
        return implementation_->common_slot_reference_times_unix_sec()(
            static_cast<Eigen::Index>(analysis_slot));
    }
    const NativeNetworkAlignment &
    network(TimestreamNetworkId network_id) const {
        return implementation_->network(network_id);
    }
    const NativeSlotAssociation &association(TimestreamNetworkId network_id,
                                             std::size_t analysis_slot) const {
        return implementation_->association(network_id, analysis_slot);
    }

private:
    explicit CommonAnalysisGridRelation(
        std::shared_ptr<const NativeAlignmentPlan> implementation)
        : implementation_{std::move(implementation)} {}

    std::shared_ptr<const NativeAlignmentPlan> implementation_;
};

// An explicitly requested, read-only cross-network common-analysis-grid view.
// ALIGN owns the relation because it owns timing knowledge. The view is
// non-destructive: it adds no numerical plane, per-cell identity plane, or AST
// coordinate, and every mapped cell retains its source network occurrence and
// exact source time. Ordinary network-timed RTC does not consume this type.
class CommonAnalysisGridPairedReadoutView {
public:
    static std::shared_ptr<const CommonAnalysisGridPairedReadoutView>
    admit(std::shared_ptr<const PairedReadout> native_parent,
          std::shared_ptr<const CommonAnalysisGridRelation> relation,
          std::size_t first_analysis_slot,
          std::size_t past_last_analysis_slot) {
        if (!native_parent || !relation ||
            !(native_parent->scope() == relation->scope()) ||
            first_analysis_slot >= past_last_analysis_slot ||
            past_last_analysis_slot > relation->slot_count() ||
            native_parent->network_count() !=
                relation->participant_network_ids().size()) {
            throw std::invalid_argument(
                "common-analysis-grid paired view is incomplete");
        }

        std::size_t detector_count = 0;
        std::size_t view_cell_count = 0;
        std::size_t mapped_cell_count = 0;
        for (const auto network_id : relation->participant_network_ids()) {
            const auto &native = native_parent->network(network_id);
            const auto &paired_axis = *native.occurrence_axis_handle();
            const auto &alignment_axis = relation->network(network_id);
            if (paired_axis.network_id() != alignment_axis.network_id()) {
                throw std::invalid_argument("paired readout network differs "
                                            "from ALIGN timing authority");
            }
            for (auto row = paired_axis.first_native_row();
                 row < paired_axis.past_last_native_row(); ++row) {
                if (!(paired_axis.identity(row) ==
                      alignment_axis.identity(row)) ||
                    paired_axis.native_timing_handle()->packet_counter(row) !=
                        alignment_axis.packet_counter(row)) {
                    throw std::invalid_argument("paired readout native timing "
                                                "differs from ALIGN authority");
                }
            }
            const auto detectors =
                static_cast<std::size_t>(native.detector_count());
            if (detector_count >
                std::numeric_limits<std::size_t>::max() - detectors) {
                throw std::length_error(
                    "common-analysis-grid detector count would overflow");
            }
            detector_count += detectors;
            const auto slots = past_last_analysis_slot - first_analysis_slot;
            if (detectors != 0 &&
                slots > std::numeric_limits<std::size_t>::max() / detectors) {
                throw std::length_error(
                    "common-analysis-grid cardinality would overflow");
            }
            const auto cells = slots * detectors;
            if (view_cell_count >
                std::numeric_limits<std::size_t>::max() - cells) {
                throw std::length_error(
                    "common-analysis-grid cell count would overflow");
            }
            view_cell_count += cells;
            for (std::size_t slot = first_analysis_slot;
                 slot < past_last_analysis_slot; ++slot) {
                const auto &association =
                    relation->association(network_id, slot);
                if (!association.mapped())
                    continue;
                if (association.native_row < paired_axis.first_native_row() ||
                    association.native_row >=
                        paired_axis.past_last_native_row()) {
                    throw std::invalid_argument(
                        "ALIGN slot maps outside paired native support");
                }
                if (mapped_cell_count >
                    std::numeric_limits<std::size_t>::max() - detectors) {
                    throw std::length_error(
                        "common-analysis-grid mapped count would overflow");
                }
                mapped_cell_count += detectors;
            }
        }

        return std::shared_ptr<const CommonAnalysisGridPairedReadoutView>(
            new CommonAnalysisGridPairedReadoutView{
                std::move(native_parent), std::move(relation),
                first_analysis_slot, past_last_analysis_slot, detector_count,
                view_cell_count, mapped_cell_count});
    }

    const std::shared_ptr<const PairedReadout> &
    native_parent_handle() const noexcept {
        return native_parent_;
    }
    const std::shared_ptr<const CommonAnalysisGridRelation> &
    relation_handle() const noexcept {
        return relation_;
    }
    const NativeObservationScope &scope() const noexcept {
        return native_parent_->scope();
    }
    std::size_t first_analysis_slot() const noexcept {
        return first_analysis_slot_;
    }
    std::size_t past_last_analysis_slot() const noexcept {
        return past_last_analysis_slot_;
    }
    std::size_t analysis_slot_count() const noexcept {
        return past_last_analysis_slot_ - first_analysis_slot_;
    }
    std::size_t detector_count() const noexcept { return detector_count_; }
    std::size_t view_cell_count() const noexcept { return view_cell_count_; }
    std::size_t mapped_cell_count() const noexcept {
        return mapped_cell_count_;
    }
    double grid_time_unix_sec(std::size_t analysis_slot) const {
        require_slot(analysis_slot);
        return relation_->grid_time_unix_sec(analysis_slot);
    }
    const PairedReadoutNetwork &network(TimestreamNetworkId network_id) const {
        return native_parent_->network(network_id);
    }
    bool mapped(TimestreamNetworkId network_id,
                std::size_t analysis_slot) const {
        require_slot(analysis_slot);
        return relation_->association(network_id, analysis_slot).mapped();
    }
    CoincidenceAbsenceReason absence_reason(TimestreamNetworkId network_id,
                                            std::size_t analysis_slot) const {
        require_slot(analysis_slot);
        const auto &association =
            relation_->association(network_id, analysis_slot);
        if (association.mapped()) {
            throw std::logic_error(
                "mapped common-analysis-grid cell has no absence reason");
        }
        return association.absence_reason;
    }
    CommonAnalysisGridCellIdentity identity(TimestreamNetworkId network_id,
                                            std::size_t analysis_slot,
                                            Eigen::Index detector_index) const {
        require_slot(analysis_slot);
        const auto &native = network(network_id);
        const auto &detector = native.detector(detector_index);
        const auto &association =
            relation_->association(network_id, analysis_slot);
        return CommonAnalysisGridCellIdentity{
            network_id, analysis_slot, detector.output_uid,
            association.mapped() ? association.native_row : -1};
    }
    std::optional<NativeSampleIdentity>
    representative_native_identity(TimestreamNetworkId network_id,
                                   std::size_t analysis_slot) const {
        require_slot(analysis_slot);
        const auto &association =
            relation_->association(network_id, analysis_slot);
        if (!association.mapped())
            return std::nullopt;
        return network(network_id)
            .occurrence_axis_handle()
            ->identity(association.native_row);
    }
    std::optional<double>
    source_network_time_unix_sec(TimestreamNetworkId network_id,
                                 std::size_t analysis_slot) const {
        const auto source =
            representative_native_identity(network_id, analysis_slot);
        return source
                   ? std::optional<double>{source
                                               ->reconstructed_time_unix_sec()}
                   : std::nullopt;
    }
    std::optional<double>
    source_time_residual_sec(TimestreamNetworkId network_id,
                             std::size_t analysis_slot) const {
        const auto source_time =
            source_network_time_unix_sec(network_id, analysis_slot);
        return source_time
                   ? std::optional<double>{*source_time -
                                           grid_time_unix_sec(analysis_slot)}
                   : std::nullopt;
    }
    const NativeOccurrenceInterval *
    source_occurrence_interval(TimestreamNetworkId network_id,
                               std::size_t analysis_slot) const {
        require_slot(analysis_slot);
        const auto &association =
            relation_->association(network_id, analysis_slot);
        if (!association.mapped())
            return nullptr;
        return &network(network_id)
                    .occurrence_axis_handle()
                    ->interval(association.native_row);
    }
    std::optional<double> value(ReadoutMember member,
                                TimestreamNetworkId network_id,
                                std::size_t analysis_slot,
                                Eigen::Index detector_index) const {
        require_slot(analysis_slot);
        const auto &association =
            relation_->association(network_id, analysis_slot);
        if (!association.mapped())
            return std::nullopt;
        return network(network_id)
            .value(member, association.native_row, detector_index);
    }
    std::optional<ReadoutMemberState> state(ReadoutMember member,
                                            TimestreamNetworkId network_id,
                                            std::size_t analysis_slot,
                                            Eigen::Index detector_index) const {
        require_slot(analysis_slot);
        const auto &association =
            relation_->association(network_id, analysis_slot);
        if (!association.mapped())
            return std::nullopt;
        return network(network_id)
            .state(member, association.native_row, detector_index);
    }
    std::optional<PairedReadoutCause>
    native_pair_causes(TimestreamNetworkId network_id,
                       std::size_t analysis_slot,
                       Eigen::Index detector_index) const {
        require_slot(analysis_slot);
        const auto &association =
            relation_->association(network_id, analysis_slot);
        if (!association.mapped())
            return std::nullopt;
        return network(network_id)
            .pair_causes(association.native_row, detector_index);
    }

private:
    CommonAnalysisGridPairedReadoutView(
        std::shared_ptr<const PairedReadout> native_parent,
        std::shared_ptr<const CommonAnalysisGridRelation> relation,
        std::size_t first_analysis_slot, std::size_t past_last_analysis_slot,
        std::size_t detector_count, std::size_t view_cell_count,
        std::size_t mapped_cell_count)
        : native_parent_{std::move(native_parent)},
          relation_{std::move(relation)},
          first_analysis_slot_{first_analysis_slot},
          past_last_analysis_slot_{past_last_analysis_slot},
          detector_count_{detector_count}, view_cell_count_{view_cell_count},
          mapped_cell_count_{mapped_cell_count} {}

    void require_slot(std::size_t analysis_slot) const {
        if (analysis_slot < first_analysis_slot_ ||
            analysis_slot >= past_last_analysis_slot_) {
            throw std::out_of_range(
                "slot is outside common-analysis-grid view support");
        }
    }

    std::shared_ptr<const PairedReadout> native_parent_;
    std::shared_ptr<const CommonAnalysisGridRelation> relation_;
    std::size_t first_analysis_slot_;
    std::size_t past_last_analysis_slot_;
    std::size_t detector_count_;
    std::size_t view_cell_count_;
    std::size_t mapped_cell_count_;
};

} // namespace citlali::pipeline
