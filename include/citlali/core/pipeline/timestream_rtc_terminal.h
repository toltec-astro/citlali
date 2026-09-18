#pragma once

#include <citlali/core/pipeline/timestream_val_rtc_output.h>

namespace citlali::pipeline {

enum class RtcTerminalCoordinateRole : std::uint8_t { not_requested };
enum class RtcTerminalResponseStatus : std::uint8_t {
    unavailable_complete_conditioned_response_not_realized
};
enum class RtcTerminalUncertaintyStatus : std::uint8_t {
    unavailable_no_admitted_covariance
};

struct RtcPipelineFinalization {
    RtcOnlyRunIdentity run;
    std::size_t detector_count = 0, scheduled_slots = 0;
    std::size_t x_available = 0, r_available = 0;
    std::size_t directly_excluded = 0, replacement_influenced = 0;
};

// An explicit RTC-only request. No ordinary-science request can be converted
// to this type by a failed handoff. A later consumer must admit its own roles.
class RtcPipelineTerminal;
class RtcPipelineTerminalSlot;
struct RtcPipelineTerminalOutcome;
RtcPipelineTerminalOutcome finalize_rtc_only(
    RtcOnlyRunIdentity, std::shared_ptr<const RtcOutputGrid>, RtcPipelineTerminalSlot &,
    std::shared_ptr<const RtcPipelineDecision> = {});

class RtcPipelineTerminal {
public:
    const auto &grid_handle() const noexcept { return grid_; }
    const auto &val_snapshot_handle() const noexcept { return val_; }
    const auto &finalization() const noexcept { return final_; }
    const auto &final_decision_handle() const noexcept { return decision_; }
    static constexpr auto detector_coordinate_role = RtcTerminalCoordinateRole::not_requested;
    static constexpr auto x_response = RtcTerminalResponseStatus::unavailable_complete_conditioned_response_not_realized;
    static constexpr auto r_response = x_response;
    static constexpr auto uncertainty = RtcTerminalUncertaintyStatus::unavailable_no_admitted_covariance;
    static constexpr bool calibrated = false, science_qualified = false;
    static constexpr unsigned rtc_time_correction_application_count = 0;

    // Filter support is local. These exact retained donor records additionally
    // preserve median contributors/endpoints, background-fit support, factors,
    // masks and evidence/selection parents. Nothing is presented as a scalar
    // complete response or as new independent exposure.
    struct Support {
        std::optional<RtcEventRange> filter;
        std::vector<std::shared_ptr<const RtcDonorFillPlan>> donors;
        std::shared_ptr<const RtcNotchRecoveryPlan> plan;
    };
    Support support(std::size_t detector, std::size_t slot) const {
        const auto occurrence = grid_->occurrence(detector, slot);
        auto plan = grid_->applied_handle()->detector_results().at(detector)->plan_handle();
        Support result{occurrence.filter_footprint, {}, plan};
        if (result.filter) for (const auto &donor : plan->donor_plans()) {
            const auto affected = donor->selection().affected;
            if (affected.first < result.filter->past_last && affected.past_last > result.filter->first)
                result.donors.push_back(donor);
        }
        return result;
    }
private:
    friend RtcPipelineTerminalOutcome finalize_rtc_only(
        RtcOnlyRunIdentity, std::shared_ptr<const RtcOutputGrid>, RtcPipelineTerminalSlot &,
        std::shared_ptr<const RtcPipelineDecision>);
    RtcPipelineTerminal(std::shared_ptr<const RtcOutputGrid> grid,
        std::shared_ptr<const ValSnapshot> val, RtcPipelineFinalization final,
        std::shared_ptr<const RtcPipelineDecision> decision)
        : grid_{std::move(grid)}, val_{std::move(val)}, final_{final}, decision_{std::move(decision)} {}
    std::shared_ptr<const RtcOutputGrid> grid_;
    std::shared_ptr<const ValSnapshot> val_;
    RtcPipelineFinalization final_;
    std::shared_ptr<const RtcPipelineDecision> decision_;
};

class RtcPipelineTerminalSlot {
public:
    std::shared_ptr<const RtcPipelineTerminal> snapshot() const {
        std::scoped_lock lock{mutex_}; return product_;
    }
private:
    friend RtcPipelineTerminalOutcome finalize_rtc_only(
        RtcOnlyRunIdentity, std::shared_ptr<const RtcOutputGrid>, RtcPipelineTerminalSlot &,
        std::shared_ptr<const RtcPipelineDecision>);
    mutable std::mutex mutex_;
    std::shared_ptr<const RtcPipelineTerminal> product_;
};

struct RtcPipelineTerminalOutcome {
    RtcOnlyTerminalState state = RtcOnlyTerminalState::finalization_failed;
    RtcOnlyFailureCause failure_cause = RtcOnlyFailureCause::required_logical_content_incomplete;
    std::string reason;
    std::shared_ptr<const RtcPipelineTerminal> product;
    bool complete() const noexcept { return state == RtcOnlyTerminalState::complete && bool(product); }
};

inline RtcPipelineTerminalOutcome finalize_rtc_only(
    RtcOnlyRunIdentity run, std::shared_ptr<const RtcOutputGrid> grid,
    RtcPipelineTerminalSlot &slot, std::shared_ptr<const RtcPipelineDecision> decision) {
    RtcPipelineTerminalOutcome outcome;
    try {
        if (!run.run) {
            outcome.failure_cause = RtcOnlyFailureCause::invalid_run_identity;
            throw std::invalid_argument("RTC finalization requires a nonzero run identity");
        }
        if (!grid) throw std::invalid_argument("RTC finalization requires a complete grid");
        const auto &plan = grid->applied_handle()->plan_handle();
        if ((plan->reassessment_handle() && !decision) ||
            (decision && (decision->disposition() == RtcPipelineDisposition::unavailable ||
                decision->selected_plan().get() != plan.get() ||
                decision->snapshot_handle().get() != grid->input_val_snapshot_handle().get() ||
                (decision->disposition() == RtcPipelineDisposition::retain &&
                 decision->reassessment_handle()->previous_handle().get() != grid->applied_handle().get())))) {
            outcome.failure_cause = RtcOnlyFailureCause::consideration_contract_rejected;
            throw std::invalid_argument("RTC finalization lacks its exact admitted final decision");
        }
        const auto &parent = *grid->align_handle()->paired_handle();
        const auto &input = *grid->applied_handle()->plan_handle()->input_handle();
        if (input.spans().size() != parent.participant_network_ids().size())
            throw std::invalid_argument("RTC terminal omits a required network");
        for (auto network : parent.participant_network_ids()) {
            const auto &span = input.span(network);
            const auto &axis = parent.network(network).occurrence_axis();
            if (span.first_native_row != axis.first_native_row() ||
                span.past_last_native_row != axis.past_last_native_row())
                throw std::invalid_argument("RTC terminal logical support is incomplete");
        }
        RtcPipelineFinalization final{run};
        final.detector_count = grid->detectors().size();
        if (final.detector_count != parent.cardinality().detector_count)
            throw std::invalid_argument("RTC terminal detector domain is incomplete");
        for (std::size_t d = 0; d < grid->detectors().size(); ++d) {
            for (std::size_t s = 0; s < grid->detectors()[d].scheduled_count; ++s) {
                const auto fact = grid->state(d, s);
                ++final.scheduled_slots;
                final.x_available += fact.x_available; final.r_available += fact.r_available;
                final.directly_excluded += fact.representative_excluded;
                final.replacement_influenced += fact.replacement_influence;
                // Validate every advertised value before exposing any terminal.
                for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r}) {
                    const auto value = grid->value(d, s, c);
                    if (value && !std::isfinite(*value))
                        throw std::invalid_argument("RTC terminal contains a nonfinite advertised value");
                }
            }
        }
        auto val = ValSnapshot::commit_rtc_output(grid->input_val_snapshot_handle(), ValRtcOutputFacts::preserve(grid));
        auto candidate = std::shared_ptr<const RtcPipelineTerminal>(new RtcPipelineTerminal{grid, std::move(val), final, std::move(decision)});
        outcome.state = RtcOnlyTerminalState::publication_failed;
        outcome.failure_cause = RtcOnlyFailureCause::publication_slot_occupied;
        std::scoped_lock lock{slot.mutex_};
        if (slot.product_) throw std::logic_error("RTC terminal slot is already occupied");
        slot.product_ = candidate;
        outcome.product = std::move(candidate);
        outcome.state = RtcOnlyTerminalState::complete;
        outcome.failure_cause = RtcOnlyFailureCause::none;
    } catch (const std::exception &error) { outcome.reason = error.what(); }
    return outcome;
}

} // namespace citlali::pipeline
