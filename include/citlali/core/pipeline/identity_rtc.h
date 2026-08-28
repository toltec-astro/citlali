#pragma once

#include <citlali/core/pipeline/aligned_paired_readout.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citlali::pipeline {

enum class RtcEvidenceOrigin : std::uint8_t {
    none = 0,
    x = 1U << 0,
    r = 1U << 1,
    x_and_r = (1U << 0) | (1U << 1),
    alignment = 1U << 2,
};

constexpr RtcEvidenceOrigin operator|(RtcEvidenceOrigin lhs,
                                      RtcEvidenceOrigin rhs) noexcept {
    return static_cast<RtcEvidenceOrigin>(
        static_cast<std::uint8_t>(lhs) |
        static_cast<std::uint8_t>(rhs));
}

constexpr bool has_origin(RtcEvidenceOrigin value,
                          RtcEvidenceOrigin origin) noexcept {
    return (static_cast<std::uint8_t>(value) &
            static_cast<std::uint8_t>(origin)) != 0;
}

enum class RtcEvidenceClass : std::uint8_t {
    input_admission,
};

// A sparse derived record.  Immutable timing, detector, mapping, validity,
// support, and payload facts remain in the referenced aligned parent.
struct RtcEvidenceEvent {
    AlignedReadoutCellIdentity identity;
    RtcEvidenceOrigin origin = RtcEvidenceOrigin::none;
    RtcEvidenceClass evidence_class = RtcEvidenceClass::input_admission;
    PairedReadoutCause member_local_causes = PairedReadoutCause::none;
    std::optional<CoincidenceAbsenceReason> alignment_absence;

    bool direct_x() const noexcept {
        return has_origin(origin, RtcEvidenceOrigin::x);
    }
    bool direct_r() const noexcept {
        return has_origin(origin, RtcEvidenceOrigin::r);
    }
    bool joint_alignment() const noexcept {
        return has_origin(origin, RtcEvidenceOrigin::alignment);
    }

    friend bool operator==(const RtcEvidenceEvent &,
                           const RtcEvidenceEvent &) = default;
};

struct RtcEvidenceIdentity {
    std::uint64_t attempt = 0;

    friend bool operator==(const RtcEvidenceIdentity &,
                           const RtcEvidenceIdentity &) = default;
};

struct RtcEvidenceSummary {
    std::size_t examined_cell_count = 0;
    std::size_t mapped_cell_count = 0;
    std::size_t accepted_event_count = 0;
    std::size_t direct_x_event_count = 0;
    std::size_t direct_r_event_count = 0;
    std::size_t x_and_r_event_count = 0;
    std::size_t alignment_absence_event_count = 0;
};

struct RtcEvidenceMemoryEvidence {
    std::size_t derived_event_bytes = 0;
    std::size_t referenced_parent_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return derived_event_bytes;
    }
};

class RtcEvidence {
public:
    const RtcEvidenceIdentity &identity() const noexcept { return identity_; }
    const NativeObservationScope &scope() const noexcept {
        return input_->scope();
    }
    const std::shared_ptr<const AlignedPairedReadout> &input_handle()
        const noexcept {
        return input_;
    }
    std::span<const RtcEvidenceEvent> events() const noexcept {
        return events_;
    }
    const RtcEvidenceSummary &summary() const noexcept { return summary_; }
    RtcEvidenceMemoryEvidence memory_evidence() const noexcept {
        return {events_.size() * sizeof(RtcEvidenceEvent), 1};
    }
    std::optional<std::size_t> find_index(
        const AlignedReadoutCellIdentity &identity) const noexcept {
        const auto found = std::lower_bound(
            events_.begin(), events_.end(), identity,
            [](const RtcEvidenceEvent &event,
               const AlignedReadoutCellIdentity &candidate) {
                return event.identity < candidate;
            });
        if (found == events_.end() || !(found->identity == identity)) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(found - events_.begin());
    }
    const RtcEvidenceEvent *find(
        const AlignedReadoutCellIdentity &identity) const noexcept {
        const auto index = find_index(identity);
        return index ? &events_[*index] : nullptr;
    }

private:
    friend std::shared_ptr<const RtcEvidence> learn_identity_rtc(
        std::shared_ptr<const AlignedPairedReadout>, std::uint64_t);

    RtcEvidence(RtcEvidenceIdentity identity,
                std::shared_ptr<const AlignedPairedReadout> input,
                std::vector<RtcEvidenceEvent> events,
                RtcEvidenceSummary summary)
        : identity_{identity}, input_{std::move(input)},
          events_{std::move(events)}, summary_{summary} {}

    RtcEvidenceIdentity identity_;
    std::shared_ptr<const AlignedPairedReadout> input_;
    std::vector<RtcEvidenceEvent> events_;
    RtcEvidenceSummary summary_;
};

inline std::shared_ptr<const RtcEvidence> learn_identity_rtc(
    std::shared_ptr<const AlignedPairedReadout> input,
    std::uint64_t attempt) {
    if (!input || attempt == 0) {
        throw std::invalid_argument(
            "identity RTC learning requires input and an attempt identity");
    }

    RtcEvidenceSummary summary;
    summary.examined_cell_count = input->aligned_cell_count();
    summary.mapped_cell_count = input->mapped_cell_count();
    std::vector<RtcEvidenceEvent> events;
    events.reserve(summary.examined_cell_count - summary.mapped_cell_count);

    for (const auto network_id :
         input->alignment_handle()->participant_network_ids()) {
        const auto &network = input->network(network_id);
        for (auto slot = input->first_common_slot();
             slot < input->past_last_common_slot(); ++slot) {
            const auto mapped = input->mapped(network_id, slot);
            for (Eigen::Index detector = 0;
                 detector < network.detector_count(); ++detector) {
                const auto identity = input->identity(
                    network_id, slot, detector);
                if (!mapped) {
                    events.push_back(RtcEvidenceEvent{
                        identity, RtcEvidenceOrigin::alignment,
                        RtcEvidenceClass::input_admission,
                        PairedReadoutCause::none,
                        input->absence_reason(network_id, slot)});
                    ++summary.alignment_absence_event_count;
                    continue;
                }

                const auto x_state = *input->state(
                    ReadoutMember::x, network_id, slot, detector);
                const auto r_state = *input->state(
                    ReadoutMember::r, network_id, slot, detector);
                const bool x_event = !x_state.valid();
                const bool r_event = !r_state.valid();
                if (!x_event && !r_event) continue;

                RtcEvidenceOrigin origin = RtcEvidenceOrigin::none;
                if (x_event) origin = origin | RtcEvidenceOrigin::x;
                if (r_event) origin = origin | RtcEvidenceOrigin::r;
                events.push_back(RtcEvidenceEvent{
                    identity, origin, RtcEvidenceClass::input_admission,
                    *input->native_pair_causes(network_id, slot, detector),
                    std::nullopt});
                if (x_event) ++summary.direct_x_event_count;
                if (r_event) ++summary.direct_r_event_count;
                if (x_event && r_event) ++summary.x_and_r_event_count;
            }
        }
    }
    summary.accepted_event_count = events.size();

    return std::shared_ptr<const RtcEvidence>(new RtcEvidence{
        RtcEvidenceIdentity{attempt}, std::move(input), std::move(events),
        summary});
}

enum class RtcPairDecision : std::uint8_t {
    eligible,
    ineligible,
};

struct RtcPlanIdentity {
    RtcEvidenceIdentity evidence;
    std::uint64_t resolution = 0;

    friend bool operator==(const RtcPlanIdentity &,
                           const RtcPlanIdentity &) = default;
};

struct RtcIdentityOperator {
    std::size_t sampling_factor = 1;
    std::size_t sampling_phase = 0;
    double x_from_x = 1.0;
    double x_from_r = 0.0;
    double r_from_x = 0.0;
    double r_from_r = 1.0;

    friend bool operator==(const RtcIdentityOperator &,
                           const RtcIdentityOperator &) = default;
};

struct RtcPairAction {
    std::size_t evidence_index = 0;
    RtcPairDecision decision = RtcPairDecision::ineligible;

    friend bool operator==(const RtcPairAction &,
                           const RtcPairAction &) = default;
};

struct RtcPlanMemoryEvidence {
    std::size_t derived_action_bytes = 0;
    std::size_t referenced_evidence_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return derived_action_bytes;
    }
};

class RtcPlan {
public:
    const RtcPlanIdentity &identity() const noexcept { return identity_; }
    const std::shared_ptr<const RtcEvidence> &evidence_handle()
        const noexcept {
        return evidence_;
    }
    const std::shared_ptr<const AlignedPairedReadout> &input_handle()
        const noexcept {
        return evidence_->input_handle();
    }
    const RtcIdentityOperator &operator_spec() const noexcept {
        return operator_;
    }
    std::span<const RtcPairAction> actions() const noexcept {
        return actions_;
    }
    RtcPlanMemoryEvidence memory_evidence() const noexcept {
        return {actions_.size() * sizeof(RtcPairAction), 1};
    }
    RtcPairDecision decision(
        const AlignedReadoutCellIdentity &identity) const noexcept {
        const auto evidence_index = evidence_->find_index(identity);
        if (!evidence_index) return RtcPairDecision::eligible;
        const auto action = std::lower_bound(
            actions_.begin(), actions_.end(), *evidence_index,
            [](const RtcPairAction &candidate, std::size_t index) {
                return candidate.evidence_index < index;
            });
        return action == actions_.end() ||
                       action->evidence_index != *evidence_index
                   ? RtcPairDecision::eligible
                   : action->decision;
    }
    const RtcEvidenceEvent *causal_evidence(
        const AlignedReadoutCellIdentity &identity) const noexcept {
        return decision(identity) == RtcPairDecision::ineligible
                   ? evidence_->find(identity)
                   : nullptr;
    }

private:
    friend std::shared_ptr<const RtcPlan> consider_identity_rtc(
        std::shared_ptr<const RtcEvidence>, std::uint64_t);

    RtcPlan(RtcPlanIdentity identity,
            std::shared_ptr<const RtcEvidence> evidence,
            std::vector<RtcPairAction> actions)
        : identity_{identity}, evidence_{std::move(evidence)},
          actions_{std::move(actions)} {}

    RtcPlanIdentity identity_;
    std::shared_ptr<const RtcEvidence> evidence_;
    RtcIdentityOperator operator_;
    std::vector<RtcPairAction> actions_;
};

inline std::shared_ptr<const RtcPlan> consider_identity_rtc(
    std::shared_ptr<const RtcEvidence> evidence,
    std::uint64_t resolution) {
    if (!evidence || resolution == 0) {
        throw std::invalid_argument(
            "identity RTC consideration requires evidence and a resolution identity");
    }
    std::vector<RtcPairAction> actions;
    actions.reserve(evidence->events().size());
    for (std::size_t index = 0; index < evidence->events().size(); ++index) {
        // The identity witness has one conservative policy: every accepted
        // input-admission event makes the paired occurrence ineligible.  The
        // evidence record retains whether x, r, both, or ALIGN caused it.
        actions.push_back({index, RtcPairDecision::ineligible});
    }
    return std::shared_ptr<const RtcPlan>(new RtcPlan{
        RtcPlanIdentity{evidence->identity(), resolution},
        std::move(evidence), std::move(actions)});
}

struct RtcTimestreamMemoryEvidence {
    std::size_t owned_numeric_bytes = 0;
    std::size_t owned_state_plane_bytes = 0;
    std::size_t referenced_parent_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return owned_numeric_bytes + owned_state_plane_bytes;
    }
};

struct RtcApplyResult;

// Identity RTC is a view product: it owns neither a duplicate TOD plane nor a
// duplicate support plane.  Scientific values, axes, and primitive support
// are read through the immutable aligned parent; decisions come from the
// immutable plan.
class RtcTimestream {
public:
    const std::shared_ptr<const RtcPlan> &plan_handle() const noexcept {
        return plan_;
    }
    const std::shared_ptr<const AlignedPairedReadout> &input_handle()
        const noexcept {
        return input_;
    }
    const NativeObservationScope &scope() const noexcept {
        return input_->scope();
    }
    std::size_t first_common_slot() const noexcept {
        return input_->first_common_slot();
    }
    std::size_t past_last_common_slot() const noexcept {
        return input_->past_last_common_slot();
    }
    std::size_t output_slot_count() const noexcept {
        return input_->common_slot_count();
    }
    std::size_t output_cell_count() const noexcept {
        return input_->aligned_cell_count();
    }
    const RtcIdentityOperator &realized_operator() const noexcept {
        return plan_->operator_spec();
    }
    double output_time_unix_sec(std::size_t common_slot) const {
        return input_->common_slot_time_unix_sec(common_slot);
    }
    AlignedReadoutCellIdentity identity(
        TimestreamNetworkId network_id, std::size_t common_slot,
        Eigen::Index detector_index) const {
        return input_->identity(network_id, common_slot, detector_index);
    }
    std::optional<NativeSampleIdentity> representative_native_identity(
        TimestreamNetworkId network_id, std::size_t common_slot) const {
        return input_->representative_native_identity(network_id,
                                                       common_slot);
    }
    std::optional<NativeOccurrenceInterval> representative_interval(
        TimestreamNetworkId network_id, std::size_t common_slot) const {
        const auto native_identity = representative_native_identity(
            network_id, common_slot);
        if (!native_identity) return std::nullopt;
        return input_->network(network_id)
            .occurrence_axis_handle()
            ->interval(native_identity->native_row());
    }
    std::optional<double> value(
        ReadoutMember member, TimestreamNetworkId network_id,
        std::size_t common_slot, Eigen::Index detector_index) const {
        return input_->value(member, network_id, common_slot,
                             detector_index);
    }
    bool member_numerically_valid(
        ReadoutMember member, TimestreamNetworkId network_id,
        std::size_t common_slot, Eigen::Index detector_index) const {
        const auto state = input_->state(
            member, network_id, common_slot, detector_index);
        return state.has_value() && state->valid();
    }
    ReadoutMemberCause member_local_causes(
        ReadoutMember member, TimestreamNetworkId network_id,
        std::size_t common_slot, Eigen::Index detector_index) const {
        const auto state = input_->state(
            member, network_id, common_slot, detector_index);
        return state ? state->causes()
                     : ReadoutMemberCause::producer_unavailable;
    }
    RtcPairDecision pair_decision(
        TimestreamNetworkId network_id, std::size_t common_slot,
        Eigen::Index detector_index) const {
        return plan_->decision(
            identity(network_id, common_slot, detector_index));
    }
    const RtcEvidenceEvent *pair_causal_evidence(
        TimestreamNetworkId network_id, std::size_t common_slot,
        Eigen::Index detector_index) const {
        return plan_->causal_evidence(
            identity(network_id, common_slot, detector_index));
    }
    RtcTimestreamMemoryEvidence memory_evidence() const noexcept {
        return {0, 0, 1};
    }

private:
    friend RtcApplyResult apply_identity_rtc(
        std::shared_ptr<const RtcPlan>,
        std::shared_ptr<const AlignedPairedReadout>);

    RtcTimestream(std::shared_ptr<const RtcPlan> plan,
                  std::shared_ptr<const AlignedPairedReadout> input)
        : plan_{std::move(plan)}, input_{std::move(input)} {}

    std::shared_ptr<const RtcPlan> plan_;
    std::shared_ptr<const AlignedPairedReadout> input_;
};

enum class RtcCompletionState : std::uint8_t {
    complete,
};

// A compact execution record.  It deliberately does not contain TOD values,
// support planes, event histories, or generalized provenance.
struct RtcRealization {
    RtcPlanIdentity plan_identity;
    RtcCompletionState completion = RtcCompletionState::complete;
    std::size_t output_slot_count = 0;
    std::size_t output_cell_count = 0;
    std::size_t pair_ineligible_cell_count = 0;
    std::size_t x_numerically_valid_cell_count = 0;
    std::size_t r_numerically_valid_cell_count = 0;
    std::size_t realized_sampling_factor = 1;
};

struct RtcApplyResult {
    std::shared_ptr<const RtcTimestream> product;
    RtcRealization realization;
};

inline RtcApplyResult apply_identity_rtc(
    std::shared_ptr<const RtcPlan> plan,
    std::shared_ptr<const AlignedPairedReadout> input) {
    if (!plan || !input || plan->input_handle().get() != input.get()) {
        throw std::invalid_argument(
            "identity RTC apply requires the exact plan-bound input");
    }
    if (!(plan->input_handle()->scope() == input->scope()) ||
        plan->operator_spec().sampling_factor != 1 ||
        plan->operator_spec().sampling_phase != 0) {
        throw std::logic_error(
            "identity RTC plan binding or operator is inconsistent");
    }

    RtcRealization realization{
        plan->identity(), RtcCompletionState::complete,
        input->common_slot_count(), input->aligned_cell_count(),
        plan->actions().size(), 0, 0, 1};
    for (const auto network_id :
         input->alignment_handle()->participant_network_ids()) {
        const auto &network = input->network(network_id);
        for (auto slot = input->first_common_slot();
             slot < input->past_last_common_slot(); ++slot) {
            for (Eigen::Index detector = 0;
                 detector < network.detector_count(); ++detector) {
                const auto x_state = input->state(
                    ReadoutMember::x, network_id, slot, detector);
                const auto r_state = input->state(
                    ReadoutMember::r, network_id, slot, detector);
                if (x_state && x_state->valid()) {
                    ++realization.x_numerically_valid_cell_count;
                }
                if (r_state && r_state->valid()) {
                    ++realization.r_numerically_valid_cell_count;
                }
            }
        }
    }

    return RtcApplyResult{
        std::shared_ptr<const RtcTimestream>(
            new RtcTimestream{std::move(plan), std::move(input)}),
        realization};
}

}  // namespace citlali::pipeline
