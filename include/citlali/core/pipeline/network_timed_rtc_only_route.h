#pragma once

#include <citlali/core/pipeline/network_timed_rtc_pass_through.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct NetworkTimedRtcOnlyRunIdentity {
  std::uint64_t run = 0;

  friend bool operator==(const NetworkTimedRtcOnlyRunIdentity &,
                         const NetworkTimedRtcOnlyRunIdentity &) = default;
};

// Compact application-owned completion facts for one logical network-timed
// product. They bind finalization to the admitted context and are checked
// against the immutable product before publication. They do not duplicate
// occurrence axes, support, values, or provenance history.
struct NetworkTimedRtcLogicalFinalization {
  std::uint64_t finalization = 0;
  RtcApplicationContextIdentity context_identity;
  std::size_t completed_native_occurrence_count = 0;
  std::size_t completed_cell_count = 0;
  bool observation_facts_finalized = false;
};

enum class NetworkTimedRtcOnlyTerminalState : std::uint8_t {
  complete,
  context_admission_failed,
  learning_failed,
  resolution_failed,
  apply_failed,
  finalization_failed,
  publication_failed,
};

constexpr const char *network_timed_rtc_only_terminal_state_name(
    NetworkTimedRtcOnlyTerminalState state) noexcept {
  switch (state) {
  case NetworkTimedRtcOnlyTerminalState::complete:
    return "complete";
  case NetworkTimedRtcOnlyTerminalState::context_admission_failed:
    return "context_admission_failed";
  case NetworkTimedRtcOnlyTerminalState::learning_failed:
    return "learning_failed";
  case NetworkTimedRtcOnlyTerminalState::resolution_failed:
    return "resolution_failed";
  case NetworkTimedRtcOnlyTerminalState::apply_failed:
    return "apply_failed";
  case NetworkTimedRtcOnlyTerminalState::finalization_failed:
    return "finalization_failed";
  case NetworkTimedRtcOnlyTerminalState::publication_failed:
    return "publication_failed";
  }
  return "unknown";
}

enum class NetworkTimedRtcOnlyFailureCause : std::uint8_t {
  none,
  invalid_run_identity,
  context_contract_rejected,
  nonterminal_application_context,
  incomplete_logical_support,
  learning_contract_rejected,
  resolution_contract_rejected,
  apply_contract_rejected,
  finalization_identity_mismatch,
  required_logical_content_incomplete,
  observation_facts_incomplete,
  publication_candidate_incomplete,
  publication_slot_occupied,
  publication_contract_rejected,
};

constexpr const char *network_timed_rtc_only_failure_cause_name(
    NetworkTimedRtcOnlyFailureCause cause) noexcept {
  switch (cause) {
  case NetworkTimedRtcOnlyFailureCause::none:
    return "none";
  case NetworkTimedRtcOnlyFailureCause::invalid_run_identity:
    return "invalid_run_identity";
  case NetworkTimedRtcOnlyFailureCause::context_contract_rejected:
    return "context_contract_rejected";
  case NetworkTimedRtcOnlyFailureCause::nonterminal_application_context:
    return "nonterminal_application_context";
  case NetworkTimedRtcOnlyFailureCause::incomplete_logical_support:
    return "incomplete_logical_support";
  case NetworkTimedRtcOnlyFailureCause::learning_contract_rejected:
    return "learning_contract_rejected";
  case NetworkTimedRtcOnlyFailureCause::resolution_contract_rejected:
    return "resolution_contract_rejected";
  case NetworkTimedRtcOnlyFailureCause::apply_contract_rejected:
    return "apply_contract_rejected";
  case NetworkTimedRtcOnlyFailureCause::finalization_identity_mismatch:
    return "finalization_identity_mismatch";
  case NetworkTimedRtcOnlyFailureCause::required_logical_content_incomplete:
    return "required_logical_content_incomplete";
  case NetworkTimedRtcOnlyFailureCause::observation_facts_incomplete:
    return "observation_facts_incomplete";
  case NetworkTimedRtcOnlyFailureCause::publication_candidate_incomplete:
    return "publication_candidate_incomplete";
  case NetworkTimedRtcOnlyFailureCause::publication_slot_occupied:
    return "publication_slot_occupied";
  case NetworkTimedRtcOnlyFailureCause::publication_contract_rejected:
    return "publication_contract_rejected";
  }
  return "unknown";
}

struct NetworkTimedRtcOnlyDiagnostics {
  std::size_t network_count = 0;
  std::size_t engineering_partition_count = 0;
  std::size_t detector_count = 0;
  std::size_t native_occurrence_count = 0;
  std::size_t detector_occurrence_count = 0;
  std::size_t evidence_event_count = 0;
  std::size_t direct_x_event_count = 0;
  std::size_t direct_r_event_count = 0;
  std::size_t x_and_r_event_count = 0;
  std::size_t pair_ineligible_cell_count = 0;
  std::size_t x_payload_available_cell_count = 0;
  std::size_t r_payload_available_cell_count = 0;
  std::size_t x_numerically_valid_cell_count = 0;
  std::size_t r_numerically_valid_cell_count = 0;
  std::size_t derived_evidence_bytes = 0;
  std::size_t derived_plan_bytes = 0;
  std::size_t rtc_owned_numeric_bytes = 0;
  std::size_t context_admission_entry_count = 0;
  std::size_t learn_entry_count = 0;
  std::size_t resolve_entry_count = 0;
  std::size_t apply_entry_count = 0;
  std::size_t finalization_entry_count = 0;
  std::size_t publication_entry_count = 0;
};

struct NetworkTimedRtcOnlyTerminalResult {
  NetworkTimedRtcOnlyRunIdentity identity;
  NetworkTimedRtcOnlyTerminalState state =
      NetworkTimedRtcOnlyTerminalState::context_admission_failed;
  NetworkTimedRtcOnlyFailureCause failure_cause =
      NetworkTimedRtcOnlyFailureCause::none;
  std::string failure_detail;
  NetworkTimedRtcOnlyDiagnostics diagnostics;

  bool complete() const noexcept {
    return state == NetworkTimedRtcOnlyTerminalState::complete;
  }
};

class NetworkTimedRtcOnlyProductSlot;
struct NetworkTimedRtcOnlyRouteRequest;
struct NetworkTimedRtcOnlyRouteOutcome;

class NetworkTimedRtcOnlyTerminalProduct {
public:
  const NetworkTimedRtcOnlyTerminalResult &terminal_result() const noexcept {
    return terminal_;
  }
  const NetworkTimedRtcLogicalFinalization &finalization() const noexcept {
    return finalization_;
  }
  const std::shared_ptr<const NetworkTimedRtcTimestream> &
  timestream_handle() const noexcept {
    return applied_.product;
  }
  const NetworkTimedRtcRealization &realization() const noexcept {
    return applied_.realization;
  }
  const std::shared_ptr<const NetworkTimedRtcPlan> &
  plan_handle() const noexcept {
    return applied_.product->plan_handle();
  }
  const std::shared_ptr<const NetworkTimedRtcEvidence> &
  evidence_handle() const noexcept {
    return plan_handle()->evidence_handle();
  }
  const std::shared_ptr<const RtcApplicationContext> &
  context_handle() const noexcept {
    return plan_handle()->context_handle();
  }

private:
  friend class NetworkTimedRtcOnlyProductSlot;
  friend NetworkTimedRtcOnlyRouteOutcome
  run_network_timed_rtc_only(const NetworkTimedRtcOnlyRouteRequest &,
                             NetworkTimedRtcOnlyProductSlot &);

  NetworkTimedRtcOnlyTerminalProduct(
      NetworkTimedRtcApplyResult applied,
      NetworkTimedRtcLogicalFinalization finalization,
      NetworkTimedRtcOnlyTerminalResult terminal)
      : applied_{std::move(applied)}, finalization_{finalization},
        terminal_{std::move(terminal)} {}

  NetworkTimedRtcApplyResult applied_;
  NetworkTimedRtcLogicalFinalization finalization_;
  NetworkTimedRtcOnlyTerminalResult terminal_;
};

// One in-memory, no-replace publication slot. This is deliberately not a
// persistent RTC TOD schema.
class NetworkTimedRtcOnlyProductSlot {
public:
  std::shared_ptr<const NetworkTimedRtcOnlyTerminalProduct> snapshot() const {
    std::scoped_lock lock{mutex_};
    return product_;
  }

private:
  friend NetworkTimedRtcOnlyRouteOutcome
  run_network_timed_rtc_only(const NetworkTimedRtcOnlyRouteRequest &,
                             NetworkTimedRtcOnlyProductSlot &);

  void
  publish(std::shared_ptr<const NetworkTimedRtcOnlyTerminalProduct> candidate) {
    if (!candidate || !candidate->terminal_result().complete() ||
        !candidate->timestream_handle() ||
        candidate->realization().completion !=
            NetworkTimedRtcCompletionState::complete ||
        candidate->finalization().finalization == 0 ||
        !candidate->finalization().observation_facts_finalized ||
        candidate->finalization().context_identity !=
            candidate->context_handle()->identity()) {
      throw std::invalid_argument(
          "network-timed RTC-only publication candidate is incomplete");
    }
    std::scoped_lock lock{mutex_};
    if (product_) {
      throw std::logic_error(
          "network-timed RTC-only product slot already contains a "
          "completion");
    }
    product_ = std::move(candidate);
  }

  mutable std::mutex mutex_;
  std::shared_ptr<const NetworkTimedRtcOnlyTerminalProduct> product_;
};

// Application-boundary request for one complete logical native product.
// Context owns that logical support. Partitions are an ordered exact-cover
// execution schedule and never become separately publishable products.
struct NetworkTimedRtcOnlyRouteRequest {
  NetworkTimedRtcOnlyRunIdentity identity;
  std::shared_ptr<const RtcApplicationContext> context;
  std::vector<std::shared_ptr<const NativePairedReadoutView>>
      engineering_partitions;
  NetworkTimedRtcLogicalFinalization finalization;
};

struct NetworkTimedRtcOnlyRouteOutcome {
  NetworkTimedRtcOnlyTerminalResult terminal;
  std::shared_ptr<const NetworkTimedRtcOnlyTerminalProduct> published_product;

  bool complete() const noexcept {
    return terminal.complete() && published_product != nullptr;
  }
};

inline NetworkTimedRtcOnlyRouteOutcome
run_network_timed_rtc_only(const NetworkTimedRtcOnlyRouteRequest &request,
                           NetworkTimedRtcOnlyProductSlot &publication) {
  NetworkTimedRtcOnlyTerminalResult terminal;
  terminal.identity = request.identity;
  if (request.identity.run == 0) {
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::invalid_run_identity;
    terminal.failure_detail =
        "network-timed RTC-only route requires a nonzero run identity";
    return {terminal, nullptr};
  }

  try {
    ++terminal.diagnostics.context_admission_entry_count;
    if (!request.context) {
      throw std::invalid_argument(
          "network-timed RTC-only route requires an application "
          "context");
    }
    if (request.context->requested_use() != RtcApplicationUse::rtc_terminal) {
      terminal.failure_cause =
          NetworkTimedRtcOnlyFailureCause::nonterminal_application_context;
      terminal.failure_detail =
          "network-timed RTC-only route requires rtc_terminal "
          "application use";
      return {terminal, nullptr};
    }
    require_exact_native_partition_schedule(*request.context->input_handle(),
                                            request.engineering_partitions);
    const auto &input = *request.context->input_handle();
    terminal.diagnostics.network_count = input.network_count();
    terminal.diagnostics.engineering_partition_count =
        request.engineering_partitions.size();
    terminal.diagnostics.detector_count = input.detector_count();
    terminal.diagnostics.native_occurrence_count =
        input.native_occurrence_count();
    terminal.diagnostics.detector_occurrence_count =
        input.detector_occurrence_count();
  } catch (const IncompleteNativePartitionSchedule &error) {
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::incomplete_logical_support;
    terminal.failure_detail = error.what();
    return {terminal, nullptr};
  } catch (const std::exception &error) {
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::context_contract_rejected;
    terminal.failure_detail = error.what();
    return {terminal, nullptr};
  }

  std::shared_ptr<const NetworkTimedRtcEvidence> evidence;
  try {
    ++terminal.diagnostics.learn_entry_count;
    evidence = learn_network_timed_rtc_pass_through(
        request.context, request.engineering_partitions, request.identity.run);
    const auto &summary = evidence->summary();
    terminal.diagnostics.evidence_event_count = summary.accepted_event_count;
    terminal.diagnostics.direct_x_event_count = summary.direct_x_event_count;
    terminal.diagnostics.direct_r_event_count = summary.direct_r_event_count;
    terminal.diagnostics.x_and_r_event_count = summary.x_and_r_event_count;
    terminal.diagnostics.derived_evidence_bytes =
        evidence->memory_evidence().logical_owned_bytes();
  } catch (const std::exception &error) {
    terminal.state = NetworkTimedRtcOnlyTerminalState::learning_failed;
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::learning_contract_rejected;
    terminal.failure_detail = error.what();
    return {terminal, nullptr};
  }

  std::shared_ptr<const NetworkTimedRtcPlan> plan;
  try {
    ++terminal.diagnostics.resolve_entry_count;
    plan =
        resolve_network_timed_rtc_pass_through(evidence, request.identity.run);
    terminal.diagnostics.derived_plan_bytes =
        plan->identity_plan_handle()->memory_evidence().logical_owned_bytes();
  } catch (const std::exception &error) {
    terminal.state = NetworkTimedRtcOnlyTerminalState::resolution_failed;
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::resolution_contract_rejected;
    terminal.failure_detail = error.what();
    return {terminal, nullptr};
  }

  NetworkTimedRtcApplyResult applied;
  try {
    ++terminal.diagnostics.apply_entry_count;
    applied = apply_network_timed_rtc_pass_through(
        plan, request.engineering_partitions);
    terminal.diagnostics.pair_ineligible_cell_count =
        applied.realization.pair_ineligible_cell_count;
    terminal.diagnostics.x_payload_available_cell_count =
        applied.realization.x_payload_available_cell_count;
    terminal.diagnostics.r_payload_available_cell_count =
        applied.realization.r_payload_available_cell_count;
    terminal.diagnostics.x_numerically_valid_cell_count =
        applied.realization.x_numerically_valid_cell_count;
    terminal.diagnostics.r_numerically_valid_cell_count =
        applied.realization.r_numerically_valid_cell_count;
    terminal.diagnostics.rtc_owned_numeric_bytes =
        applied.product->memory_evidence().owned_numeric_bytes;
  } catch (const std::exception &error) {
    terminal.state = NetworkTimedRtcOnlyTerminalState::apply_failed;
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::apply_contract_rejected;
    terminal.failure_detail = error.what();
    return {terminal, nullptr};
  }

  ++terminal.diagnostics.finalization_entry_count;
  terminal.state = NetworkTimedRtcOnlyTerminalState::finalization_failed;
  if (request.finalization.finalization == 0 ||
      !request.finalization.observation_facts_finalized) {
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::observation_facts_incomplete;
    terminal.failure_detail =
        "network-timed RTC-only observation facts are incomplete";
    return {terminal, nullptr};
  }
  if (request.finalization.context_identity != request.context->identity()) {
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::finalization_identity_mismatch;
    terminal.failure_detail =
        "network-timed RTC-only finalization does not bind the admitted "
        "context";
    return {terminal, nullptr};
  }
  if (request.finalization.completed_native_occurrence_count !=
          applied.product->output_native_occurrence_count() ||
      request.finalization.completed_cell_count !=
          applied.product->output_cell_count()) {
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::required_logical_content_incomplete;
    terminal.failure_detail =
        "network-timed RTC-only completed counts do not match the logical "
        "product";
    return {terminal, nullptr};
  }

  terminal.state = NetworkTimedRtcOnlyTerminalState::complete;
  terminal.failure_cause = NetworkTimedRtcOnlyFailureCause::none;
  ++terminal.diagnostics.publication_entry_count;
  auto candidate = std::shared_ptr<const NetworkTimedRtcOnlyTerminalProduct>(
      new NetworkTimedRtcOnlyTerminalProduct{std::move(applied),
                                             request.finalization, terminal});
  try {
    publication.publish(candidate);
  } catch (const std::invalid_argument &error) {
    terminal.state = NetworkTimedRtcOnlyTerminalState::publication_failed;
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::publication_candidate_incomplete;
    terminal.failure_detail = error.what();
    return {terminal, nullptr};
  } catch (const std::logic_error &error) {
    terminal.state = NetworkTimedRtcOnlyTerminalState::publication_failed;
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::publication_slot_occupied;
    terminal.failure_detail = error.what();
    return {terminal, nullptr};
  } catch (const std::exception &error) {
    terminal.state = NetworkTimedRtcOnlyTerminalState::publication_failed;
    terminal.failure_cause =
        NetworkTimedRtcOnlyFailureCause::publication_contract_rejected;
    terminal.failure_detail = error.what();
    return {terminal, nullptr};
  }
  return {terminal, std::move(candidate)};
}

} // namespace citlali::pipeline
