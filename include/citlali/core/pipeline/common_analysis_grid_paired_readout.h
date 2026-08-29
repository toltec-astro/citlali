#pragma once

#include <citlali/core/pipeline/paired_readout.h>
#include <citlali/core/pipeline/timestream_native_alignment.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct CommonAnalysisGridRequestIdentity {
    std::uint64_t request = 0;

    friend bool operator==(const CommonAnalysisGridRequestIdentity &,
                           const CommonAnalysisGridRequestIdentity &) = default;
};

enum class CommonAnalysisGridAdmissionRule : std::uint8_t {
    strict_half_cadence,
};

enum class CommonAnalysisGridSupportPolicy : std::uint8_t {
    preserve_partial_network_support,
};

enum class CommonAnalysisGridFailurePolicy : std::uint8_t {
    fail_closed,
};

// Lightweight identity-bound request for the exceptional cross-network view.
// Epoch samples remain relation-owned; the request names their stable set and
// cardinality instead of copying the full axis.
class CommonAnalysisGridRequest {
public:
    static std::shared_ptr<const CommonAnalysisGridRequest>
    admit(CommonAnalysisGridRequestIdentity identity,
          NativeObservationScope scope, std::string consuming_method_id,
          std::vector<TimestreamNetworkId> participant_network_ids,
          std::string analysis_epoch_set_id, std::size_t analysis_epoch_count,
          double admission_cadence_sec,
          CommonAnalysisGridAdmissionRule admission_rule,
          CommonAnalysisGridSupportPolicy support_policy,
          CommonAnalysisGridFailurePolicy failure_policy) {
        if (identity.request == 0 || consuming_method_id.empty() ||
            participant_network_ids.size() < 2 ||
            analysis_epoch_set_id.empty() || analysis_epoch_count == 0 ||
            !std::isfinite(admission_cadence_sec) ||
            admission_cadence_sec <= 0.0 ||
            admission_rule !=
                CommonAnalysisGridAdmissionRule::strict_half_cadence ||
            support_policy != CommonAnalysisGridSupportPolicy::
                                  preserve_partial_network_support ||
            failure_policy != CommonAnalysisGridFailurePolicy::fail_closed) {
            throw std::invalid_argument(
                "common-analysis-grid request is incomplete");
        }
        std::sort(participant_network_ids.begin(),
                  participant_network_ids.end());
        if (std::adjacent_find(participant_network_ids.begin(),
                               participant_network_ids.end()) !=
            participant_network_ids.end()) {
            throw std::invalid_argument(
                "common-analysis-grid request repeats a network");
        }
        return std::shared_ptr<const CommonAnalysisGridRequest>(
            new CommonAnalysisGridRequest{
                identity, scope, std::move(consuming_method_id),
                std::move(participant_network_ids),
                std::move(analysis_epoch_set_id), analysis_epoch_count,
                admission_cadence_sec, admission_rule, support_policy,
                failure_policy});
    }

    const CommonAnalysisGridRequestIdentity &identity() const noexcept {
        return identity_;
    }
    const NativeObservationScope &scope() const noexcept { return scope_; }
    const std::string &consuming_method_id() const noexcept {
        return consuming_method_id_;
    }
    std::span<const TimestreamNetworkId>
    participant_network_ids() const noexcept {
        return participant_network_ids_;
    }
    const std::string &analysis_epoch_set_id() const noexcept {
        return analysis_epoch_set_id_;
    }
    std::size_t analysis_epoch_count() const noexcept {
        return analysis_epoch_count_;
    }
    double admission_cadence_sec() const noexcept {
        return admission_cadence_sec_;
    }
    CommonAnalysisGridAdmissionRule admission_rule() const noexcept {
        return admission_rule_;
    }
    CommonAnalysisGridSupportPolicy support_policy() const noexcept {
        return support_policy_;
    }
    CommonAnalysisGridFailurePolicy failure_policy() const noexcept {
        return failure_policy_;
    }

private:
    CommonAnalysisGridRequest(
        CommonAnalysisGridRequestIdentity identity,
        NativeObservationScope scope, std::string consuming_method_id,
        std::vector<TimestreamNetworkId> participant_network_ids,
        std::string analysis_epoch_set_id, std::size_t analysis_epoch_count,
        double admission_cadence_sec,
        CommonAnalysisGridAdmissionRule admission_rule,
        CommonAnalysisGridSupportPolicy support_policy,
        CommonAnalysisGridFailurePolicy failure_policy)
        : identity_{identity}, scope_{scope},
          consuming_method_id_{std::move(consuming_method_id)},
          participant_network_ids_{std::move(participant_network_ids)},
          analysis_epoch_set_id_{std::move(analysis_epoch_set_id)},
          analysis_epoch_count_{analysis_epoch_count},
          admission_cadence_sec_{admission_cadence_sec},
          admission_rule_{admission_rule}, support_policy_{support_policy},
          failure_policy_{failure_policy} {}

    CommonAnalysisGridRequestIdentity identity_;
    NativeObservationScope scope_;
    std::string consuming_method_id_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::string analysis_epoch_set_id_;
    std::size_t analysis_epoch_count_;
    double admission_cadence_sec_;
    CommonAnalysisGridAdmissionRule admission_rule_;
    CommonAnalysisGridSupportPolicy support_policy_;
    CommonAnalysisGridFailurePolicy failure_policy_;
};

struct CommonAnalysisGridRelationIdentity {
    CommonAnalysisGridRequestIdentity request;
    std::uint64_t align_plan = 0;

    friend bool
    operator==(const CommonAnalysisGridRelationIdentity &,
               const CommonAnalysisGridRelationIdentity &) = default;
};

struct CommonAnalysisGridCellIdentity {
    CommonAnalysisGridRelationIdentity relation;
    TimestreamNetworkId network_id = -1;
    std::size_t analysis_slot = 0;
    std::int64_t detector_uid = 0;
    TimestreamNativeRow native_row = -1;

    bool has_native_occurrence() const noexcept { return native_row >= 0; }

    friend bool operator==(const CommonAnalysisGridCellIdentity &,
                           const CommonAnalysisGridCellIdentity &) = default;
    friend bool operator<(const CommonAnalysisGridCellIdentity &lhs,
                          const CommonAnalysisGridCellIdentity &rhs) noexcept {
        if (lhs.relation.request.request != rhs.relation.request.request) {
            return lhs.relation.request.request < rhs.relation.request.request;
        }
        if (lhs.relation.align_plan != rhs.relation.align_plan) {
            return lhs.relation.align_plan < rhs.relation.align_plan;
        }
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
    admit(CommonAnalysisGridRelationIdentity identity,
          std::shared_ptr<const CommonAnalysisGridRequest> request,
          std::vector<NativeNetworkAlignment> network_timings,
          Eigen::VectorXd grid_times_unix_sec) {
        if (!request || identity.align_plan == 0 ||
            !(identity.request == request->identity()) ||
            grid_times_unix_sec.size() !=
                static_cast<Eigen::Index>(request->analysis_epoch_count()) ||
            network_timings.size() !=
                request->participant_network_ids().size()) {
            throw std::invalid_argument(
                "common-analysis-grid relation identity is incomplete");
        }
        validate_grid(grid_times_unix_sec, request->admission_cadence_sec());
        std::sort(network_timings.begin(), network_timings.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.network_id() < rhs.network_id();
                  });
        std::map<TimestreamNetworkId, std::vector<NativeSlotAssociation>>
            source_associations;
        const auto requested_networks = request->participant_network_ids();
        for (std::size_t index = 0; index < network_timings.size(); ++index) {
            if (network_timings[index].network_id() !=
                requested_networks[index]) {
                throw std::invalid_argument(
                    "common-analysis-grid relation network inventory differs "
                    "from its request");
            }
            source_associations.emplace(network_timings[index].network_id(),
                                        derive_strict_half_associations(
                                            network_timings[index],
                                            grid_times_unix_sec,
                                            request->admission_cadence_sec()));
        }
        auto implementation = std::make_shared<const NativeAlignmentPlan>(
            request->scope(), std::move(network_timings),
            std::move(grid_times_unix_sec), std::move(source_associations));
        return std::shared_ptr<const CommonAnalysisGridRelation>(
            new CommonAnalysisGridRelation{identity, std::move(request),
                                           std::move(implementation)});
    }

    const CommonAnalysisGridRelationIdentity &identity() const noexcept {
        return identity_;
    }
    const std::shared_ptr<const CommonAnalysisGridRequest> &
    request_handle() const noexcept {
        return request_;
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
    static void validate_grid(const Eigen::VectorXd &grid_times_unix_sec,
                              double admission_cadence_sec) {
        if (grid_times_unix_sec.size() <= 0) {
            throw std::invalid_argument(
                "common-analysis-grid relation requires analysis epochs");
        }
        for (Eigen::Index slot = 0; slot < grid_times_unix_sec.size(); ++slot) {
            const auto epoch = grid_times_unix_sec(slot);
            if (!std::isfinite(epoch)) {
                throw std::invalid_argument(
                    "common-analysis-grid epochs must be finite");
            }
            if (slot == 0)
                continue;
            const auto prior = grid_times_unix_sec(slot - 1);
            const auto spacing = epoch - prior;
            const auto scale = std::max({1.0, std::abs(epoch), std::abs(prior),
                                         std::abs(admission_cadence_sec)});
            const auto roundoff =
                16.0 * std::numeric_limits<double>::epsilon() * scale;
            if (!(spacing > 0.0) ||
                spacing + roundoff < admission_cadence_sec) {
                throw std::invalid_argument(
                    "common-analysis-grid epochs do not provide unique "
                    "strict-half support");
            }
        }
    }

    static std::vector<NativeSlotAssociation>
    derive_strict_half_associations(const NativeNetworkAlignment &network,
                                    const Eigen::VectorXd &grid_times_unix_sec,
                                    double admission_cadence_sec) {
        const auto tolerance = admission_cadence_sec / 2.0;
        std::vector<NativeSlotAssociation> result(
            static_cast<std::size_t>(grid_times_unix_sec.size()));
        std::set<std::size_t> mapped_slots;
        const auto *grid_begin = grid_times_unix_sec.data();
        const auto *grid_end = grid_begin + grid_times_unix_sec.size();
        for (auto row = network.first_native_row();
             row < network.past_last_native_row(); ++row) {
            const auto value =
                network.identity(row).reconstructed_time_unix_sec();
            const auto *upper = std::lower_bound(grid_begin, grid_end, value);
            std::optional<std::size_t> candidate;
            const auto consider = [&](const double *epoch) {
                if (epoch < grid_begin || epoch >= grid_end ||
                    !(std::abs(value - *epoch) < tolerance)) {
                    return;
                }
                const auto slot = static_cast<std::size_t>(epoch - grid_begin);
                if (candidate && *candidate != slot) {
                    throw std::logic_error(
                        "native occurrence satisfies two strict-half slots");
                }
                candidate = slot;
            };
            if (upper != grid_end)
                consider(upper);
            if (upper != grid_begin)
                consider(upper - 1);
            if (!candidate)
                continue;
            if (!mapped_slots.insert(*candidate).second) {
                throw std::logic_error(
                    "two native occurrences satisfy one strict-half slot");
            }
            result[*candidate].native_row = row;
        }

        const auto native_min = network.identity(network.first_native_row())
                                    .reconstructed_time_unix_sec();
        const auto native_max =
            network.identity(network.past_last_native_row() - 1)
                .reconstructed_time_unix_sec();
        for (std::size_t slot = 0; slot < result.size(); ++slot) {
            if (result[slot].mapped())
                continue;
            const auto epoch =
                grid_times_unix_sec(static_cast<Eigen::Index>(slot));
            result[slot].absence_reason =
                epoch < native_min || epoch > native_max
                    ? CoincidenceAbsenceReason::outside_native_support
                    : CoincidenceAbsenceReason::no_candidate;
        }
        return result;
    }

    CommonAnalysisGridRelation(
        CommonAnalysisGridRelationIdentity identity,
        std::shared_ptr<const CommonAnalysisGridRequest> request,
        std::shared_ptr<const NativeAlignmentPlan> implementation)
        : identity_{identity}, request_{std::move(request)},
          implementation_{std::move(implementation)} {}

    CommonAnalysisGridRelationIdentity identity_;
    std::shared_ptr<const CommonAnalysisGridRequest> request_;
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
            past_last_analysis_slot > relation->slot_count()) {
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
        (void)relation_->network(network_id);
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
            relation_->identity(), network_id, analysis_slot,
            detector.output_uid,
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
