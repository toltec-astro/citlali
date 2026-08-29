#pragma once

#include <citlali/core/pipeline/identity_rtc.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct RtcOnlyRunIdentity {
    std::uint64_t run = 0;

    friend bool operator==(const RtcOnlyRunIdentity &,
                           const RtcOnlyRunIdentity &) = default;
};

enum class RtcOnlyTerminalState : std::uint8_t {
    complete,
    input_admission_failed,
    learning_failed,
    consideration_failed,
    apply_failed,
    publication_failed,
};

constexpr const char *rtc_only_terminal_state_name(
    RtcOnlyTerminalState state) noexcept {
    switch (state) {
        case RtcOnlyTerminalState::complete:
            return "complete";
        case RtcOnlyTerminalState::input_admission_failed:
            return "input_admission_failed";
        case RtcOnlyTerminalState::learning_failed:
            return "learning_failed";
        case RtcOnlyTerminalState::consideration_failed:
            return "consideration_failed";
        case RtcOnlyTerminalState::apply_failed:
            return "apply_failed";
        case RtcOnlyTerminalState::publication_failed:
            return "publication_failed";
    }
    return "unknown";
}

enum class RtcOnlyFailureCause : std::uint8_t {
    none,
    invalid_run_identity,
    input_contract_rejected,
    learning_contract_rejected,
    consideration_contract_rejected,
    apply_contract_rejected,
    publication_candidate_incomplete,
    publication_slot_occupied,
    publication_contract_rejected,
};

constexpr const char *rtc_only_failure_cause_name(
    RtcOnlyFailureCause cause) noexcept {
    switch (cause) {
        case RtcOnlyFailureCause::none:
            return "none";
        case RtcOnlyFailureCause::invalid_run_identity:
            return "invalid_run_identity";
        case RtcOnlyFailureCause::input_contract_rejected:
            return "input_contract_rejected";
        case RtcOnlyFailureCause::learning_contract_rejected:
            return "learning_contract_rejected";
        case RtcOnlyFailureCause::consideration_contract_rejected:
            return "consideration_contract_rejected";
        case RtcOnlyFailureCause::apply_contract_rejected:
            return "apply_contract_rejected";
        case RtcOnlyFailureCause::publication_candidate_incomplete:
            return "publication_candidate_incomplete";
        case RtcOnlyFailureCause::publication_slot_occupied:
            return "publication_slot_occupied";
        case RtcOnlyFailureCause::publication_contract_rejected:
            return "publication_contract_rejected";
    }
    return "unknown";
}

struct RtcOnlyDiagnostics {
    std::size_t network_count = 0;
    std::size_t detector_count = 0;
    std::size_t native_occurrence_count = 0;
    std::size_t detector_occurrence_count = 0;
    std::size_t evidence_event_count = 0;
    std::size_t direct_x_event_count = 0;
    std::size_t direct_r_event_count = 0;
    std::size_t x_and_r_event_count = 0;
    std::size_t pair_ineligible_cell_count = 0;
    std::size_t x_numerically_valid_cell_count = 0;
    std::size_t r_numerically_valid_cell_count = 0;
    std::size_t derived_evidence_bytes = 0;
    std::size_t derived_plan_bytes = 0;
    std::size_t rtc_owned_numeric_bytes = 0;
    std::size_t native_admission_entry_count = 0;
    std::size_t learn_entry_count = 0;
    std::size_t consider_entry_count = 0;
    std::size_t apply_entry_count = 0;
    std::size_t publication_entry_count = 0;
};

struct RtcOnlyTerminalResult {
    RtcOnlyRunIdentity identity;
    RtcOnlyTerminalState state =
        RtcOnlyTerminalState::input_admission_failed;
    RtcOnlyFailureCause failure_cause = RtcOnlyFailureCause::none;
    std::string failure_detail;
    RtcOnlyDiagnostics diagnostics;

    bool complete() const noexcept {
        return state == RtcOnlyTerminalState::complete;
    }
};

class RtcOnlyProductSlot;
struct RtcOnlyRouteRequest;
struct RtcOnlyRouteOutcome;

class RtcOnlyTerminalProduct {
public:
    const RtcOnlyTerminalResult &terminal_result() const noexcept {
        return terminal_;
    }
    const std::shared_ptr<const RtcTimestream> &timestream_handle()
        const noexcept {
        return applied_.product;
    }
    const RtcRealization &realization() const noexcept {
        return applied_.realization;
    }
    const std::shared_ptr<const RtcPlan> &plan_handle() const noexcept {
        return applied_.product->plan_handle();
    }
    const std::shared_ptr<const RtcEvidence> &evidence_handle()
        const noexcept {
        return applied_.product->plan_handle()->evidence_handle();
    }

private:
    friend class RtcOnlyProductSlot;
    friend RtcOnlyRouteOutcome run_identity_rtc_only(
        const RtcOnlyRouteRequest &, RtcOnlyProductSlot &);

    RtcOnlyTerminalProduct(RtcApplyResult applied,
                           RtcOnlyTerminalResult terminal)
        : applied_{std::move(applied)}, terminal_{terminal} {}

    RtcApplyResult applied_;
    RtcOnlyTerminalResult terminal_;
};

class RtcOnlyProductSlot {
public:
    std::shared_ptr<const RtcOnlyTerminalProduct> snapshot() const {
        std::scoped_lock lock{mutex_};
        return product_;
    }

private:
    friend RtcOnlyRouteOutcome run_identity_rtc_only(
        const RtcOnlyRouteRequest &, RtcOnlyProductSlot &);

    void publish(
        std::shared_ptr<const RtcOnlyTerminalProduct> candidate) {
        if (!candidate || !candidate->terminal_result().complete() ||
            !candidate->timestream_handle() ||
            candidate->realization().completion !=
                RtcCompletionState::complete) {
            throw std::invalid_argument(
                "RTC-only publication candidate is incomplete");
        }
        std::scoped_lock lock{mutex_};
        if (product_) {
            throw std::logic_error(
                "RTC-only product slot already contains a completion");
        }
        product_ = std::move(candidate);
    }

    mutable std::mutex mutex_;
    std::shared_ptr<const RtcOnlyTerminalProduct> product_;
};

// This is an explicit application-boundary request, not a YAML or persistent
// TOD schema. The caller supplies the already-authoritative native pair and
// one engineering partition over its per-network native occurrence axes.
struct RtcOnlyRouteRequest {
    RtcOnlyRunIdentity identity;
    std::shared_ptr<const PairedReadout> native_input;
    std::vector<NativeOccurrenceSpan> native_spans;
};

struct RtcOnlyRouteOutcome {
    RtcOnlyTerminalResult terminal;
    std::shared_ptr<const RtcOnlyTerminalProduct> published_product;

    bool complete() const noexcept {
        return terminal.complete() && published_product != nullptr;
    }
};

inline RtcOnlyRouteOutcome run_identity_rtc_only(
    const RtcOnlyRouteRequest &request, RtcOnlyProductSlot &publication) {
    RtcOnlyTerminalResult terminal;
    terminal.identity = request.identity;
    if (request.identity.run == 0) {
        terminal.state = RtcOnlyTerminalState::input_admission_failed;
        terminal.failure_cause =
            RtcOnlyFailureCause::invalid_run_identity;
        terminal.failure_detail =
            "RTC-only route requires a nonzero run identity";
        return {terminal, nullptr};
    }

    std::shared_ptr<const NativePairedReadoutView> native_view;
    try {
        ++terminal.diagnostics.native_admission_entry_count;
        native_view = NativePairedReadoutView::admit(
            request.native_input, request.native_spans);
        terminal.diagnostics.network_count = native_view->network_count();
        terminal.diagnostics.detector_count = native_view->detector_count();
        terminal.diagnostics.native_occurrence_count =
            native_view->native_occurrence_count();
        terminal.diagnostics.detector_occurrence_count =
            native_view->detector_occurrence_count();
    } catch (const std::exception &error) {
        terminal.state = RtcOnlyTerminalState::input_admission_failed;
        terminal.failure_cause =
            RtcOnlyFailureCause::input_contract_rejected;
        terminal.failure_detail = error.what();
        return {terminal, nullptr};
    }

    std::shared_ptr<const RtcEvidence> evidence;
    try {
        ++terminal.diagnostics.learn_entry_count;
        evidence = learn_identity_rtc(native_view, request.identity.run);
        const auto &summary = evidence->summary();
        terminal.diagnostics.evidence_event_count =
            summary.accepted_event_count;
        terminal.diagnostics.direct_x_event_count =
            summary.direct_x_event_count;
        terminal.diagnostics.direct_r_event_count =
            summary.direct_r_event_count;
        terminal.diagnostics.x_and_r_event_count =
            summary.x_and_r_event_count;
        terminal.diagnostics.derived_evidence_bytes =
            evidence->memory_evidence().logical_owned_bytes();
    } catch (const std::exception &error) {
        terminal.state = RtcOnlyTerminalState::learning_failed;
        terminal.failure_cause =
            RtcOnlyFailureCause::learning_contract_rejected;
        terminal.failure_detail = error.what();
        return {terminal, nullptr};
    }

    std::shared_ptr<const RtcPlan> plan;
    try {
        ++terminal.diagnostics.consider_entry_count;
        plan = consider_identity_rtc(evidence, request.identity.run);
        terminal.diagnostics.derived_plan_bytes =
            plan->memory_evidence().logical_owned_bytes();
    } catch (const std::exception &error) {
        terminal.state = RtcOnlyTerminalState::consideration_failed;
        terminal.failure_cause =
            RtcOnlyFailureCause::consideration_contract_rejected;
        terminal.failure_detail = error.what();
        return {terminal, nullptr};
    }

    RtcApplyResult applied;
    try {
        ++terminal.diagnostics.apply_entry_count;
        applied = apply_identity_rtc(plan, native_view);
        terminal.diagnostics.pair_ineligible_cell_count =
            applied.realization.pair_ineligible_cell_count;
        terminal.diagnostics.x_numerically_valid_cell_count =
            applied.realization.x_numerically_valid_cell_count;
        terminal.diagnostics.r_numerically_valid_cell_count =
            applied.realization.r_numerically_valid_cell_count;
        terminal.diagnostics.rtc_owned_numeric_bytes =
            applied.product->memory_evidence().owned_numeric_bytes;
    } catch (const std::exception &error) {
        terminal.state = RtcOnlyTerminalState::apply_failed;
        terminal.failure_cause =
            RtcOnlyFailureCause::apply_contract_rejected;
        terminal.failure_detail = error.what();
        return {terminal, nullptr};
    }

    terminal.state = RtcOnlyTerminalState::complete;
    ++terminal.diagnostics.publication_entry_count;
    auto candidate = std::shared_ptr<const RtcOnlyTerminalProduct>(
        new RtcOnlyTerminalProduct{std::move(applied), terminal});
    try {
        publication.publish(candidate);
    } catch (const std::invalid_argument &error) {
        terminal.state = RtcOnlyTerminalState::publication_failed;
        terminal.failure_cause =
            RtcOnlyFailureCause::publication_candidate_incomplete;
        terminal.failure_detail = error.what();
        return {terminal, nullptr};
    } catch (const std::logic_error &error) {
        terminal.state = RtcOnlyTerminalState::publication_failed;
        terminal.failure_cause =
            RtcOnlyFailureCause::publication_slot_occupied;
        terminal.failure_detail = error.what();
        return {terminal, nullptr};
    } catch (const std::exception &error) {
        terminal.state = RtcOnlyTerminalState::publication_failed;
        terminal.failure_cause =
            RtcOnlyFailureCause::publication_contract_rejected;
        terminal.failure_detail = error.what();
        return {terminal, nullptr};
    }
    return {terminal, std::move(candidate)};
}

}  // namespace citlali::pipeline
