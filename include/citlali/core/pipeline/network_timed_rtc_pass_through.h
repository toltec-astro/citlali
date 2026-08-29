#pragma once

#include <citlali/core/pipeline/identity_rtc.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

enum class RtcApplicationUse : std::uint8_t {
    rtc_terminal,
    sci_cal_handoff,
};

struct RtcApplicationContextIdentity {
    std::uint64_t request = 0;

    friend bool operator==(const RtcApplicationContextIdentity &,
                           const RtcApplicationContextIdentity &) = default;
};

// The producer mapping already owns unit identity. Context owns only the
// request-side statements needed to interpret that raw coordinate.
struct RtcRawCoordinateMeaning {
    std::string sign_id;
    std::string reference_id;
    std::string baseline_id;

    bool complete() const noexcept {
        return !sign_id.empty() && !reference_id.empty() &&
               !baseline_id.empty();
    }
};

// Immutable request facts for ordinary RTC. The input is a bounded view over
// network-scoped native occurrence axes. Values, timing, intervals, validity,
// causes, support, detector identity, and mapping identity remain parent-owned.
class RtcApplicationContext {
public:
    static std::shared_ptr<const RtcApplicationContext>
    admit(RtcApplicationContextIdentity identity,
          std::shared_ptr<const NativePairedReadoutView> input,
          RtcApplicationUse requested_use, std::string interval_id,
          RtcRawCoordinateMeaning x_meaning, RtcRawCoordinateMeaning r_meaning,
          std::string valid_domain_id, std::vector<std::string> named_consumers,
          std::vector<std::string> external_constraint_ids,
          bool conditioned_r_requested) {
        if (identity.request == 0 || !input || interval_id.empty() ||
            !x_meaning.complete() || !r_meaning.complete() ||
            valid_domain_id.empty() || named_consumers.empty()) {
            throw std::invalid_argument(
                "ordinary RTC application context is incomplete");
        }
        normalize_unique_nonempty(named_consumers, "RTC consumer");
        normalize_unique_nonempty(external_constraint_ids,
                                  "RTC external constraint");
        return std::shared_ptr<const RtcApplicationContext>(
            new RtcApplicationContext{
                identity, std::move(input), requested_use,
                std::move(interval_id), std::move(x_meaning),
                std::move(r_meaning), std::move(valid_domain_id),
                std::move(named_consumers), std::move(external_constraint_ids),
                conditioned_r_requested});
    }

    const RtcApplicationContextIdentity &identity() const noexcept {
        return identity_;
    }
    const std::shared_ptr<const NativePairedReadoutView> &
    input_handle() const noexcept {
        return input_;
    }
    RtcApplicationUse requested_use() const noexcept { return requested_use_; }
    const std::string &interval_id() const noexcept { return interval_id_; }
    const RtcRawCoordinateMeaning &x_meaning() const noexcept {
        return x_meaning_;
    }
    const RtcRawCoordinateMeaning &r_meaning() const noexcept {
        return r_meaning_;
    }
    const std::string &valid_domain_id() const noexcept {
        return valid_domain_id_;
    }
    std::span<const std::string> named_consumers() const noexcept {
        return named_consumers_;
    }
    std::span<const std::string> external_constraint_ids() const noexcept {
        return external_constraint_ids_;
    }
    bool conditioned_r_requested() const noexcept {
        return conditioned_r_requested_;
    }

private:
    static void normalize_unique_nonempty(std::vector<std::string> &values,
                                          const char *label) {
        for (const auto &value : values) {
            if (value.empty()) {
                throw std::invalid_argument(std::string{label} +
                                            " identity must be nonempty");
            }
        }
        std::sort(values.begin(), values.end());
        if (std::adjacent_find(values.begin(), values.end()) != values.end()) {
            throw std::invalid_argument(std::string{label} +
                                        " identities must be unique");
        }
    }

    RtcApplicationContext(RtcApplicationContextIdentity identity,
                          std::shared_ptr<const NativePairedReadoutView> input,
                          RtcApplicationUse requested_use,
                          std::string interval_id,
                          RtcRawCoordinateMeaning x_meaning,
                          RtcRawCoordinateMeaning r_meaning,
                          std::string valid_domain_id,
                          std::vector<std::string> named_consumers,
                          std::vector<std::string> external_constraint_ids,
                          bool conditioned_r_requested)
        : identity_{identity}, input_{std::move(input)},
          requested_use_{requested_use}, interval_id_{std::move(interval_id)},
          x_meaning_{std::move(x_meaning)}, r_meaning_{std::move(r_meaning)},
          valid_domain_id_{std::move(valid_domain_id)},
          named_consumers_{std::move(named_consumers)},
          external_constraint_ids_{std::move(external_constraint_ids)},
          conditioned_r_requested_{conditioned_r_requested} {}

    RtcApplicationContextIdentity identity_;
    std::shared_ptr<const NativePairedReadoutView> input_;
    RtcApplicationUse requested_use_;
    std::string interval_id_;
    RtcRawCoordinateMeaning x_meaning_;
    RtcRawCoordinateMeaning r_meaning_;
    std::string valid_domain_id_;
    std::vector<std::string> named_consumers_;
    std::vector<std::string> external_constraint_ids_;
    bool conditioned_r_requested_ = false;
};

struct NetworkTimedRtcEvidenceIdentity {
    RtcApplicationContextIdentity context;
    RtcEvidenceIdentity learned;

    friend bool operator==(const NetworkTimedRtcEvidenceIdentity &,
                           const NetworkTimedRtcEvidenceIdentity &) = default;
};

// This layer adds context binding only. The accepted sparse identity evidence
// remains the sole owner of derived per-cell events.
class NetworkTimedRtcEvidence {
public:
    const NetworkTimedRtcEvidenceIdentity &identity() const noexcept {
        return identity_;
    }
    const std::shared_ptr<const RtcApplicationContext> &
    context_handle() const noexcept {
        return context_;
    }
    const std::shared_ptr<const RtcEvidence> &
    identity_evidence_handle() const noexcept {
        return identity_evidence_;
    }
    std::span<const RtcEvidenceEvent> events() const noexcept {
        return identity_evidence_->events();
    }
    const RtcEvidenceSummary &summary() const noexcept {
        return identity_evidence_->summary();
    }
    RtcEvidenceMemoryEvidence memory_evidence() const noexcept {
        return identity_evidence_->memory_evidence();
    }
    const RtcEvidenceEvent *find(TimestreamNetworkId network_id,
                                 TimestreamNativeRow native_row,
                                 Eigen::Index detector_index) const noexcept {
        return identity_evidence_->find(network_id, native_row, detector_index);
    }
    RtcNativeCellIdentity
    scientific_identity(const RtcEvidenceEvent &event) const {
        return identity_evidence_->scientific_identity(event);
    }
    PairedReadoutCause
    member_local_causes(const RtcEvidenceEvent &event) const {
        return identity_evidence_->member_local_causes(event);
    }

private:
    friend std::shared_ptr<const NetworkTimedRtcEvidence>
    learn_network_timed_rtc_pass_through(
        std::shared_ptr<const RtcApplicationContext>,
        std::span<const std::shared_ptr<const NativePairedReadoutView>>,
        std::uint64_t);

    NetworkTimedRtcEvidence(
        NetworkTimedRtcEvidenceIdentity identity,
        std::shared_ptr<const RtcApplicationContext> context,
        std::shared_ptr<const RtcEvidence> identity_evidence)
        : identity_{identity}, context_{std::move(context)},
          identity_evidence_{std::move(identity_evidence)} {}

    NetworkTimedRtcEvidenceIdentity identity_;
    std::shared_ptr<const RtcApplicationContext> context_;
    std::shared_ptr<const RtcEvidence> identity_evidence_;
};

inline std::shared_ptr<const NetworkTimedRtcEvidence>
learn_network_timed_rtc_pass_through(
    std::shared_ptr<const RtcApplicationContext> context,
    std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions,
    std::uint64_t attempt) {
    if (!context || attempt == 0) {
        throw std::invalid_argument(
            "ordinary RTC learning requires context and attempt identity");
    }
    auto learned = learn_identity_rtc_partitioned(context->input_handle(),
                                                  partitions, attempt);
    return std::shared_ptr<const NetworkTimedRtcEvidence>(
        new NetworkTimedRtcEvidence{{context->identity(), learned->identity()},
                                    std::move(context),
                                    std::move(learned)});
}

inline std::shared_ptr<const NetworkTimedRtcEvidence>
learn_network_timed_rtc_pass_through(
    std::shared_ptr<const RtcApplicationContext> context,
    std::uint64_t attempt) {
    if (!context) {
        throw std::invalid_argument("ordinary RTC learning requires context");
    }
    const std::vector<std::shared_ptr<const NativePairedReadoutView>>
        partitions{context->input_handle()};
    return learn_network_timed_rtc_pass_through(std::move(context), partitions,
                                                attempt);
}

enum class NetworkTimedRtcPairDisposition : std::uint8_t {
    eligible,
    ineligible,
};

enum class NetworkTimedRtcMemberAvailability : std::uint8_t {
    available,
    unavailable,
    not_requested,
};

enum class NetworkTimedRtcPairCauseRole : std::uint8_t {
    none,
    direct,
    inferred_from_x,
    inferred_from_r,
};

enum class NetworkTimedRtcOperationDisposition : std::uint8_t {
    not_selected,
    identity,
};

// Exact bounded policy for the M=1 witness. It cannot be mistaken for an
// executed despiker, level-shift repair, donor replacement, or filter.
struct NetworkTimedRtcPassThroughPolicy {
    NetworkTimedRtcOperationDisposition despiking =
        NetworkTimedRtcOperationDisposition::not_selected;
    NetworkTimedRtcOperationDisposition level_shift_correction =
        NetworkTimedRtcOperationDisposition::not_selected;
    NetworkTimedRtcOperationDisposition donor_replacement =
        NetworkTimedRtcOperationDisposition::not_selected;
    NetworkTimedRtcOperationDisposition temporal_filter =
        NetworkTimedRtcOperationDisposition::identity;
    NetworkTimedRtcOperationDisposition phase_zero_sampling =
        NetworkTimedRtcOperationDisposition::identity;
    NetworkTimedRtcOperationDisposition leakage_diagnostic =
        NetworkTimedRtcOperationDisposition::not_selected;
    NetworkTimedRtcOperationDisposition atmospheric_diagnostic =
        NetworkTimedRtcOperationDisposition::not_selected;
    bool coordinate_dependent_operation = false;

    friend bool operator==(const NetworkTimedRtcPassThroughPolicy &,
                           const NetworkTimedRtcPassThroughPolicy &) = default;
};

struct NetworkTimedRtcPlanIdentity {
    NetworkTimedRtcEvidenceIdentity evidence;
    std::uint64_t resolution = 0;

    friend bool operator==(const NetworkTimedRtcPlanIdentity &,
                           const NetworkTimedRtcPlanIdentity &) = default;
};

class NetworkTimedRtcPlan {
public:
    const NetworkTimedRtcPlanIdentity &identity() const noexcept {
        return identity_;
    }
    const std::shared_ptr<const NetworkTimedRtcEvidence> &
    evidence_handle() const noexcept {
        return evidence_;
    }
    const std::shared_ptr<const RtcApplicationContext> &
    context_handle() const noexcept {
        return evidence_->context_handle();
    }
    const std::shared_ptr<const RtcPlan> &
    identity_plan_handle() const noexcept {
        return identity_plan_;
    }
    const RtcIdentityOperator &operator_spec() const noexcept {
        return identity_plan_->operator_spec();
    }
    const NetworkTimedRtcPassThroughPolicy &policy() const noexcept {
        return policy_;
    }
    bool conditioned_r_requested() const noexcept {
        return context_handle()->conditioned_r_requested();
    }
    NetworkTimedRtcPairDisposition
    pair_disposition(TimestreamNetworkId network_id,
                     TimestreamNativeRow native_row,
                     Eigen::Index detector_index) const {
        require_cell(network_id, native_row, detector_index);
        return identity_plan_->decision(network_id, native_row,
                                        detector_index) ==
                       RtcPairDecision::eligible
                   ? NetworkTimedRtcPairDisposition::eligible
                   : NetworkTimedRtcPairDisposition::ineligible;
    }
    NetworkTimedRtcMemberAvailability
    member_availability(ReadoutMember member, TimestreamNetworkId network_id,
                        TimestreamNativeRow native_row,
                        Eigen::Index detector_index) const {
        require_cell(network_id, native_row, detector_index);
        if (member == ReadoutMember::r && !conditioned_r_requested()) {
            return NetworkTimedRtcMemberAvailability::not_requested;
        }
        return context_handle()
                       ->input_handle()
                       ->network(network_id)
                       .state(member, native_row, detector_index)
                       .available()
                   ? NetworkTimedRtcMemberAvailability::available
                   : NetworkTimedRtcMemberAvailability::unavailable;
    }
    bool member_numerically_valid(ReadoutMember member,
                                  TimestreamNetworkId network_id,
                                  TimestreamNativeRow native_row,
                                  Eigen::Index detector_index) const {
        require_cell(network_id, native_row, detector_index);
        return context_handle()
            ->input_handle()
            ->network(network_id)
            .state(member, native_row, detector_index)
            .valid();
    }
    NetworkTimedRtcPairCauseRole
    pair_cause_role(ReadoutMember member, TimestreamNetworkId network_id,
                    TimestreamNativeRow native_row,
                    Eigen::Index detector_index) const {
        require_cell(network_id, native_row, detector_index);
        const auto *event =
            evidence_->find(network_id, native_row, detector_index);
        if (!event)
            return NetworkTimedRtcPairCauseRole::none;
        if ((member == ReadoutMember::x && event->direct_x()) ||
            (member == ReadoutMember::r && event->direct_r())) {
            return NetworkTimedRtcPairCauseRole::direct;
        }
        return member == ReadoutMember::x
                   ? NetworkTimedRtcPairCauseRole::inferred_from_r
                   : NetworkTimedRtcPairCauseRole::inferred_from_x;
    }

private:
    friend std::shared_ptr<const NetworkTimedRtcPlan>
    resolve_network_timed_rtc_pass_through(
        std::shared_ptr<const NetworkTimedRtcEvidence>, std::uint64_t);

    NetworkTimedRtcPlan(NetworkTimedRtcPlanIdentity identity,
                        std::shared_ptr<const NetworkTimedRtcEvidence> evidence,
                        std::shared_ptr<const RtcPlan> identity_plan)
        : identity_{identity}, evidence_{std::move(evidence)},
          identity_plan_{std::move(identity_plan)} {}

    void require_cell(TimestreamNetworkId network_id,
                      TimestreamNativeRow native_row,
                      Eigen::Index detector_index) const {
        const auto &input = *context_handle()->input_handle();
        const auto &support = input.span(network_id);
        if (native_row < support.first_native_row ||
            native_row >= support.past_last_native_row) {
            throw std::out_of_range(
                "native row is outside ordinary RTC plan support");
        }
        (void)input.network(network_id).detector(detector_index);
    }

    NetworkTimedRtcPlanIdentity identity_;
    std::shared_ptr<const NetworkTimedRtcEvidence> evidence_;
    std::shared_ptr<const RtcPlan> identity_plan_;
    NetworkTimedRtcPassThroughPolicy policy_;
};

inline std::shared_ptr<const NetworkTimedRtcPlan>
resolve_network_timed_rtc_pass_through(
    std::shared_ptr<const NetworkTimedRtcEvidence> evidence,
    std::uint64_t resolution) {
    if (!evidence || resolution == 0) {
        throw std::invalid_argument("ordinary RTC resolution requires evidence "
                                    "and resolution identity");
    }
    auto identity_plan =
        consider_identity_rtc(evidence->identity_evidence_handle(), resolution);
    return std::shared_ptr<const NetworkTimedRtcPlan>(
        new NetworkTimedRtcPlan{{evidence->identity(), resolution},
                                std::move(evidence),
                                std::move(identity_plan)});
}

struct NetworkTimedRtcApplyResult;

// Inspectable in-memory M=1 product. It is a view over the immutable native
// inputs and therefore preserves each network's occurrence identity, exact
// time, support, x/r values, validity, and local causes without copied planes.
class NetworkTimedRtcTimestream {
public:
    const std::shared_ptr<const NetworkTimedRtcPlan> &
    plan_handle() const noexcept {
        return plan_;
    }
    const std::shared_ptr<const RtcApplicationContext> &
    context_handle() const noexcept {
        return plan_->context_handle();
    }
    const std::shared_ptr<const NativePairedReadoutView> &
    input_handle() const noexcept {
        return context_handle()->input_handle();
    }
    const std::shared_ptr<const PairedReadout> &
    native_parent_handle() const noexcept {
        return input_handle()->parent_handle();
    }
    std::span<const NativeOccurrenceSpan> network_spans() const noexcept {
        return identity_product_->network_spans();
    }
    std::size_t output_native_occurrence_count() const noexcept {
        return identity_product_->output_native_occurrence_count();
    }
    std::size_t output_cell_count() const noexcept {
        return identity_product_->output_cell_count();
    }
    const RtcIdentityOperator &realized_operator() const noexcept {
        return identity_product_->realized_operator();
    }
    bool conditioned_r_requested() const noexcept {
        return plan_->conditioned_r_requested();
    }
    double output_time_unix_sec(TimestreamNetworkId network_id,
                                TimestreamNativeRow native_row) const {
        return identity_product_->output_time_unix_sec(network_id, native_row);
    }
    RtcNativeCellIdentity identity(TimestreamNetworkId network_id,
                                   TimestreamNativeRow native_row,
                                   Eigen::Index detector_index) const {
        return identity_product_->identity(network_id, native_row,
                                           detector_index);
    }
    NativeSampleIdentity
    representative_native_identity(TimestreamNetworkId network_id,
                                   TimestreamNativeRow native_row) const {
        return identity_product_->representative_native_identity(network_id,
                                                                 native_row);
    }
    const NativeOccurrenceInterval &
    representative_interval(TimestreamNetworkId network_id,
                            TimestreamNativeRow native_row) const {
        return identity_product_->representative_interval(network_id,
                                                          native_row);
    }
    NetworkTimedRtcPairDisposition
    pair_disposition(TimestreamNetworkId network_id,
                     TimestreamNativeRow native_row,
                     Eigen::Index detector_index) const {
        return plan_->pair_disposition(network_id, native_row, detector_index);
    }
    NetworkTimedRtcMemberAvailability
    member_availability(ReadoutMember member, TimestreamNetworkId network_id,
                        TimestreamNativeRow native_row,
                        Eigen::Index detector_index) const {
        return plan_->member_availability(member, network_id, native_row,
                                          detector_index);
    }
    bool member_numerically_valid(ReadoutMember member,
                                  TimestreamNetworkId network_id,
                                  TimestreamNativeRow native_row,
                                  Eigen::Index detector_index) const {
        return plan_->member_numerically_valid(member, network_id, native_row,
                                               detector_index);
    }
    NetworkTimedRtcPairCauseRole
    pair_cause_role(ReadoutMember member, TimestreamNetworkId network_id,
                    TimestreamNativeRow native_row,
                    Eigen::Index detector_index) const {
        return plan_->pair_cause_role(member, network_id, native_row,
                                      detector_index);
    }
    std::optional<double> conditioned_value(ReadoutMember member,
                                            TimestreamNetworkId network_id,
                                            TimestreamNativeRow native_row,
                                            Eigen::Index detector_index) const {
        if (member_availability(member, network_id, native_row,
                                detector_index) !=
            NetworkTimedRtcMemberAvailability::available) {
            return std::nullopt;
        }
        return identity_product_->value(member, network_id, native_row,
                                        detector_index);
    }
    double raw_parent_value(ReadoutMember member,
                            TimestreamNetworkId network_id,
                            TimestreamNativeRow native_row,
                            Eigen::Index detector_index) const {
        return identity_product_->value(member, network_id, native_row,
                                        detector_index);
    }
    ReadoutMemberCause raw_member_local_causes(
        ReadoutMember member, TimestreamNetworkId network_id,
        TimestreamNativeRow native_row, Eigen::Index detector_index) const {
        return identity_product_->member_local_causes(
            member, network_id, native_row, detector_index);
    }
    ReadoutMemberState raw_member_state(ReadoutMember member,
                                        TimestreamNetworkId network_id,
                                        TimestreamNativeRow native_row,
                                        Eigen::Index detector_index) const {
        return input_handle()
            ->network(network_id)
            .state(member, native_row, detector_index);
    }
    const RtcEvidenceEvent *
    pair_causal_evidence(TimestreamNetworkId network_id,
                         TimestreamNativeRow native_row,
                         Eigen::Index detector_index) const {
        return identity_product_->pair_causal_evidence(network_id, native_row,
                                                       detector_index);
    }
    RtcTimestreamMemoryEvidence memory_evidence() const noexcept {
        return identity_product_->memory_evidence();
    }

private:
    friend NetworkTimedRtcApplyResult apply_network_timed_rtc_pass_through(
        std::shared_ptr<const NetworkTimedRtcPlan>,
        std::span<const std::shared_ptr<const NativePairedReadoutView>>);

    NetworkTimedRtcTimestream(
        std::shared_ptr<const NetworkTimedRtcPlan> plan,
        std::shared_ptr<const RtcTimestream> identity_product)
        : plan_{std::move(plan)},
          identity_product_{std::move(identity_product)} {}

    std::shared_ptr<const NetworkTimedRtcPlan> plan_;
    std::shared_ptr<const RtcTimestream> identity_product_;
};

enum class NetworkTimedRtcCompletionState : std::uint8_t {
    complete,
};

// Compact record of what the immutable plan realized. Full values, support,
// event history, and provenance remain referenced through product and plan.
struct NetworkTimedRtcRealization {
    NetworkTimedRtcPlanIdentity plan_identity;
    NetworkTimedRtcCompletionState completion =
        NetworkTimedRtcCompletionState::complete;
    std::size_t engineering_partition_count = 0;
    std::size_t output_native_occurrence_count = 0;
    std::size_t output_cell_count = 0;
    std::size_t pair_ineligible_cell_count = 0;
    std::size_t x_payload_available_cell_count = 0;
    std::size_t r_payload_available_cell_count = 0;
    std::size_t x_numerically_valid_cell_count = 0;
    std::size_t r_numerically_valid_cell_count = 0;
    std::size_t realized_sampling_factor = 1;
    bool conditioned_r_requested = false;
};

struct NetworkTimedRtcApplyResult {
    std::shared_ptr<const NetworkTimedRtcTimestream> product;
    NetworkTimedRtcRealization realization;
};

inline NetworkTimedRtcApplyResult apply_network_timed_rtc_pass_through(
    std::shared_ptr<const NetworkTimedRtcPlan> plan,
    std::span<const std::shared_ptr<const NativePairedReadoutView>>
        partitions) {
    if (!plan) {
        throw std::invalid_argument(
            "ordinary RTC apply requires an immutable plan");
    }
    const auto &operator_spec = plan->operator_spec();
    if (operator_spec.sampling_factor != 1 ||
        operator_spec.sampling_phase != 0 || operator_spec.x_from_x != 1.0 ||
        operator_spec.x_from_r != 0.0 || operator_spec.r_from_x != 0.0 ||
        operator_spec.r_from_r != 1.0) {
        throw std::logic_error(
            "ordinary RTC pass-through plan has a nonidentity operator");
    }
    auto applied = apply_identity_rtc_partitioned(
        plan->identity_plan_handle(), plan->context_handle()->input_handle(),
        partitions);
    const auto identity_realization = applied.realization;
    std::size_t x_unavailable_cell_count = 0;
    std::size_t r_unavailable_cell_count = 0;
    for (const auto &event : plan->evidence_handle()->events()) {
        const auto causes = plan->evidence_handle()->member_local_causes(event);
        if (has_cause(causes, PairedReadoutCause::x_unavailable))
            ++x_unavailable_cell_count;
        if (has_cause(causes, PairedReadoutCause::r_unavailable))
            ++r_unavailable_cell_count;
    }
    const auto x_payload_available_cell_count =
        identity_realization.output_cell_count - x_unavailable_cell_count;
    const auto r_payload_available_cell_count =
        identity_realization.output_cell_count - r_unavailable_cell_count;
    NetworkTimedRtcRealization realization{
        plan->identity(),
        NetworkTimedRtcCompletionState::complete,
        partitions.size(),
        identity_realization.output_native_occurrence_count,
        identity_realization.output_cell_count,
        identity_realization.pair_ineligible_cell_count,
        x_payload_available_cell_count,
        r_payload_available_cell_count,
        identity_realization.x_numerically_valid_cell_count,
        identity_realization.r_numerically_valid_cell_count,
        identity_realization.realized_sampling_factor,
        plan->conditioned_r_requested()};
    return {std::shared_ptr<const NetworkTimedRtcTimestream>(
                new NetworkTimedRtcTimestream{std::move(plan),
                                              std::move(applied.product)}),
            realization};
}

inline NetworkTimedRtcApplyResult apply_network_timed_rtc_pass_through(
    std::shared_ptr<const NetworkTimedRtcPlan> plan) {
    if (!plan) {
        throw std::invalid_argument(
            "ordinary RTC apply requires an immutable plan");
    }
    const std::vector<std::shared_ptr<const NativePairedReadoutView>>
        partitions{plan->context_handle()->input_handle()};
    return apply_network_timed_rtc_pass_through(std::move(plan), partitions);
}

} // namespace citlali::pipeline
