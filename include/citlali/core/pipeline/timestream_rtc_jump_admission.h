#pragma once

#include <citlali/core/pipeline/timestream_rtc_jump_reassessment.h>

namespace citlali::pipeline {

// Owner-selected use of the preserved empirical measurements, 2026-09-13.
// This is exclusion evidence, not an offset repair or a tail probability.
struct RtcJumpAdmissionPolicy {
    static constexpr std::string_view identity = "rtc-jump-exclusion-2026-09-13-v1";
    static constexpr std::size_t observation_group_count = 3;
};

enum class RtcJumpAdmissionCause : std::uint8_t {
    measurement_unavailable, refinement_unresolved, recovery_unavailable,
    confirmed_recovery, source_protection_unavailable,
    protected_optical_test_unavailable, admitted
};

struct RtcJumpAdmittedCoordinate {
    RtcJumpAdmissionCause cause = RtcJumpAdmissionCause::measurement_unavailable;
    bool reassessed = false;
    // These are references into the immutable evidence chain, not new fits.
    RtcEventRange affected;
    NativeReadoutIntegrationSupport physical_bound;
    bool admitted() const noexcept { return cause == RtcJumpAdmissionCause::admitted; }
};

struct RtcJumpAdmissionGroup {
    std::size_t original_group = 0;
    TimestreamNetworkId network = -1;
    std::uint32_t detector = 0;
    std::array<RtcJumpAdmittedCoordinate, 2> coordinates;
    bool admitted() const noexcept { return coordinates[0].admitted() || coordinates[1].admitted(); }
};

// RTC Consider: final measurements become class-specific exclusion evidence.
// The exact original VAL and source membership remain in the parent chain.
// No beam/motion ceiling is used to classify a protected astronomical signal.
class RtcJumpAdmissionDecision {
public:
    static std::shared_ptr<const RtcJumpAdmissionDecision> consider(
        std::shared_ptr<const RtcJumpReassessmentDecision> parent,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id) {
        if (!parent) throw std::invalid_argument("RTC jump admission requires final reassessment");
        const auto &remeasured = *parent->evidence_handle();
        const auto &refit = *remeasured.request_handle()->refit_handle();
        const auto &audit = *refit.request_handle()->audit_handle();
        const auto &initial = *audit.parent_handle();
        const auto &assessment = rtc_jump_reassessment_detail::assessment(initial);
        rtc_jump_detail::require_snapshot(assessment, snapshot, id);
        auto result = std::shared_ptr<RtcJumpAdmissionDecision>(new RtcJumpAdmissionDecision);
        result->parent_ = std::move(parent);
        result->id_ = id;
        result->groups_.reserve(assessment.events().size());
        for (std::size_t i = 0; i < assessment.events().size(); ++i) {
            const auto &e = assessment.events()[i];
            result->groups_.push_back({i, e.network, e.detector, {}});
        }
        for (std::size_t i = 0; i < audit.audits().size(); ++i) {
            const auto group = audit.audits()[i].event;
            const auto &event = assessment.events()[group];
            for (std::size_t c = 0; c < 2; ++c) {
                auto &out = result->groups_[group].coordinates[c];
                const auto cause = result->parent_->coordinates()[i][c];
                if (event.refinement_limited) {
                    out.cause = RtcJumpAdmissionCause::refinement_unresolved;
                    continue;
                }
                if (cause != RtcJumpReassessmentCause::unchanged_measured &&
                    cause != RtcJumpReassessmentCause::reassessed_measured) continue;
                out.reassessed = cause == RtcJumpReassessmentCause::reassessed_measured;
                const auto &transition = out.reassessed ? remeasured.coordinates()[i][c].transition : initial.coordinates()[group][c];
                const auto &recovery = out.reassessed ? refit.coordinates()[i][c].recovery : event.recovery[c];
                const auto &background = out.reassessed ? refit.coordinates()[i][c].primary : event.background[c];
                if (recovery.recovered()) {
                    out.cause = RtcJumpAdmissionCause::confirmed_recovery;
                    continue;
                }
                // Only a complete, valid fixed search is usable negative
                // recovery evidence. Invalid/gap/end/unavailable is not "no return".
                if (recovery.cause != RtcEventRecoveryCause::search_limit || !recovery.examined.present()) {
                    out.cause = RtcJumpAdmissionCause::recovery_unavailable;
                    continue;
                }
                if (!transition.available() || !transition.affected.present() ||
                    !transition.confirmations[1].rows.present()) continue;

                // Require explicit outside-source authority over the evidence
                // context, including fit flanks and recovery/transition search.
                // This does not change the protected samples in noise learning.
                auto first = std::min(recovery.examined.first, transition.examined.first);
                auto last = std::max(recovery.examined.past_last, transition.examined.past_last);
                for (const auto &side : background.support) if (side.usable) {
                    first = std::min(first, side.first_used);
                    last = std::max(last, side.last_used + 1);
                }
                bool unknown = false, protected_source = false;
                const auto &protection = *assessment.spike_handle()->protection_handle();
                for (auto row = first; row < last; ++row) {
                    const auto state = protection.state(event.network, event.detector, row);
                    unknown |= state == RtcSpikeProtection::unavailable;
                    protected_source |= state == RtcSpikeProtection::protected_source;
                }
                out.cause = unknown ? RtcJumpAdmissionCause::source_protection_unavailable :
                    protected_source ? RtcJumpAdmissionCause::protected_optical_test_unavailable :
                    RtcJumpAdmissionCause::admitted;
                if (out.admitted()) {
                    out.affected = transition.affected;
                    out.physical_bound = transition.physical_bound;
                }
            }
        }
        return result;
    }

    const auto &parent_handle() const noexcept { return parent_; }
    const auto &groups() const noexcept { return groups_; }
    std::uint64_t consideration() const noexcept { return id_; }
    const RtcEventAssessmentEvidence &assessment() const {
        return rtc_jump_reassessment_detail::assessment(*parent_->evidence_handle()->request_handle()->refit_handle()->request_handle()->audit_handle()->parent_handle());
    }
    const auto &input_handle() const { return assessment().spike_handle()->input_handle(); }
    const auto &val_snapshot_handle() const { return assessment().spike_handle()->val_snapshot_handle(); }
    std::size_t logical_owned_bytes() const noexcept { return groups_.size() * sizeof(RtcJumpAdmissionGroup); }

private:
    RtcJumpAdmissionDecision() = default;
    std::shared_ptr<const RtcJumpReassessmentDecision> parent_;
    std::vector<RtcJumpAdmissionGroup> groups_;
    std::uint64_t id_ = 0;
};

} // namespace citlali::pipeline
