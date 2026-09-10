#pragma once

#include <citlali/core/pipeline/timestream_rtc_jump_consistency.h>

namespace citlali::pipeline {

// Owner-selected transition-evidence trial, 2026-09-10. These constants are
// deliberately a separate use binding from recovery and fitting exclusion.
struct RtcJumpTransitionPolicy {
    static constexpr std::string_view identity = "rtc-jump-transition-2026-09-10-v1";
    static constexpr double residual_sigma = 4.0;
    static constexpr double confirmation_seconds = 0.05;
    static constexpr double search_seconds = 2.0;
};

enum class RtcJumpTransitionRequestCause : std::uint8_t {
    consistency_not_passed, confirmed_recovery, requested
};

// Consider selects numerical work; it does not admit a physical event. The
// exact parent chain retains original VAL, source protection, and x/r origins.
class RtcJumpTransitionRequest {
public:
    static std::shared_ptr<const RtcJumpTransitionRequest> consider(
        std::shared_ptr<const RtcJumpConsistencyDecision> consistency,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id) {
        if (!consistency) throw std::invalid_argument("RTC transition requires consistency decision");
        rtc_jump_detail::require_snapshot(
            *consistency->evidence_handle()->amplitude_handle()->review_handle()->evidence_handle(), snapshot, id);
        auto out = std::shared_ptr<RtcJumpTransitionRequest>(new RtcJumpTransitionRequest);
        out->consistency_ = std::move(consistency); out->id_ = id;
        for (const auto &pair : out->consistency_->coordinates()) {
            std::array<RtcJumpTransitionRequestCause, 2> causes;
            for (std::size_t c = 0; c < 2; ++c)
                causes[c] = !pair[c].passes() ? RtcJumpTransitionRequestCause::consistency_not_passed :
                    pair[c].confirmed_recovery_excludes_persistent_shift ? RtcJumpTransitionRequestCause::confirmed_recovery :
                    RtcJumpTransitionRequestCause::requested;
            out->coordinates_.push_back(causes);
        }
        return out;
    }
    const auto &consistency_handle() const noexcept { return consistency_; }
    const auto &coordinates() const noexcept { return coordinates_; }
    std::uint64_t consideration() const noexcept { return id_; }
private:
    RtcJumpTransitionRequest() = default;
    std::shared_ptr<const RtcJumpConsistencyDecision> consistency_;
    std::vector<std::array<RtcJumpTransitionRequestCause, 2>> coordinates_;
    std::uint64_t id_ = 0;
};

enum class RtcJumpTransitionCause : std::uint8_t {
    not_requested, measured, background_unavailable, nonfinite,
    pre_confirmation_missing, post_confirmation_missing, ambiguous_reference,
    invalid_transition_support, competing_exclusion,
    support_geometry_unavailable
};

struct RtcJumpConfirmation {
    RtcEventRange rows;
    NativeReadoutIntegrationSupport physical;
    bool also_matches_other_reference = false;
};

struct RtcJumpTransition {
    RtcJumpTransitionCause cause = RtcJumpTransitionCause::not_requested;
    std::array<RtcJumpConfirmation, 2> confirmations;
    RtcEventRange examined;
    // Half-open original native cells between confirmations. No scan identity
    // or hard-class union is inferred from this coordinate-local measurement.
    RtcEventRange affected;
    NativeReadoutIntegrationSupport physical_bound;
    double frozen_residual_scale = std::numeric_limits<double>::quiet_NaN();
    std::size_t examined_rows = 0, invalid_rows = 0, excluded_rows = 0;
    bool observation_truncated = false, acquisition_truncated = false;
    bool exceeds_fitting_exclusion = false;
    bool multiple_candidate_edges = false;
    static constexpr bool physical_event_identity_resolved = false;
    // Parent mapping retains the declared timing-uncertainty authority. This
    // trial does not estimate its magnitude or certify total physical coverage.
    static constexpr bool timing_uncertainty_quantified = false;
    bool available() const noexcept { return cause == RtcJumpTransitionCause::measured; }
};

namespace rtc_jump_transition_detail {
// The earlier merged fitting-mask list also contains this event's own trial
// mask. Reconstruct other candidates' unchanged masks by exact membership;
// subtracting the own mask from a merged union would erase overlapping peers.
inline std::vector<RtcEventRange> neighbor_masks(
    const RtcSpikeEvidence &spikes, const RtcAssessedEvent &event,
    const NativeContiguousRun &run, std::span<const std::size_t> candidates) {
    using namespace rtc_event_assessment_detail;
    const auto &axis = spikes.input_handle()->network(event.network).occurrence_axis();
    const auto &seed = spikes.candidates()[event.seed];
    const double center = std::midpoint(time(axis, seed.earlier_row), time(axis, seed.later_row));
    const double low = center - RtcJumpTransitionPolicy::search_seconds - RtcEventAssessmentPolicy::trial_half_width_seconds;
    const double high = center + RtcJumpTransitionPolicy::search_seconds + RtcEventAssessmentPolicy::trial_half_width_seconds;
    const RtcEventRange range{run.first_native_row, run.past_last_native_row};
    const auto first = lower(axis, range, low);
    auto it = std::lower_bound(candidates.begin(), candidates.end(), first - 1,
        [&](auto i, auto row) { return spikes.candidates()[i].earlier_row < row; });
    std::vector<RtcEventRange> masks;
    for (; it != candidates.end(); ++it) {
        const auto &candidate = spikes.candidates()[*it];
        if (candidate.earlier_row >= run.past_last_native_row) break;
        if (candidate.earlier_row < run.first_native_row) continue;
        if (time(axis, candidate.earlier_row) > high) break;
        if (std::find(event.candidates.begin(), event.candidates.end(), *it) != event.candidates.end()) continue;
        masks.push_back(trial(axis, range, candidate));
    }
    return merge(std::move(masks));
}

inline bool agrees(double sample, double prediction, double scale) {
    const double limit = RtcJumpTransitionPolicy::residual_sigma * scale;
    return std::isfinite(sample) && std::isfinite(prediction) &&
        std::isfinite(scale) && scale > 0 && std::isfinite(limit) &&
        std::isfinite(sample - prediction) && std::abs(sample - prediction) <= limit;
}

inline RtcJumpTransition measure(const RtcSpikeEvidence &spikes,
                                 const RtcAssessedEvent &event, std::size_t c,
                                 const NativeContiguousRun &run,
                                 const std::vector<RtcEventRange> &neighbors) {
    using namespace rtc_event_assessment_detail;
    RtcJumpTransition out;
    const auto &fit = event.background[c];
    if (!fit.available()) { out.cause = RtcJumpTransitionCause::background_unavailable; return out; }
    out.frozen_residual_scale = fit.pre_scale_fit.scale;
    const auto &net = spikes.input_handle()->network(event.network);
    const auto &axis = net.occurrence_axis();
    if (event.seed >= spikes.candidates().size()) throw std::invalid_argument("RTC transition seed is out of range");
    const auto &seed = spikes.candidates()[event.seed];
    const double center = std::midpoint(time(axis, seed.earlier_row), time(axis, seed.later_row));
    const double low = center - RtcJumpTransitionPolicy::search_seconds;
    const double high = center + RtcJumpTransitionPolicy::search_seconds;
    if (!std::isfinite(low) || !std::isfinite(high) || !std::isfinite(event.origin) ||
        !std::isfinite(event.time_scale) || event.time_scale <= 0 ||
        !std::isfinite(out.frozen_residual_scale) || out.frozen_residual_scale <= 0 ||
        !std::isfinite(RtcJumpTransitionPolicy::residual_sigma * out.frozen_residual_scale)) {
        out.cause = RtcJumpTransitionCause::nonfinite; return out;
    }
    auto first_edge = seed.earlier_row, last_edge = seed.later_row;
    for (auto i : event.candidates) {
        if (i >= spikes.candidates().size()) throw std::invalid_argument("RTC transition member is out of range");
        const auto &candidate = spikes.candidates()[i];
        first_edge = std::min(first_edge, candidate.earlier_row);
        last_edge = std::max(last_edge, candidate.later_row);
    }
    out.multiple_candidate_edges = last_edge != first_edge + 1;
    const auto run_begin = axis.occurrence(run.first_native_row).integration_support.begin_unix_sec;
    const auto run_end = axis.occurrence(run.past_last_native_row - 1).integration_support.end_unix_sec;
    out.observation_truncated = (run_begin > low && run.first_native_row == axis.first_native_row()) ||
        (run_end < high && run.past_last_native_row == axis.past_last_native_row());
    out.acquisition_truncated = (run_begin > low && run.first_native_row != axis.first_native_row()) ||
        (run_end < high && run.past_last_native_row != axis.past_last_native_row());
    const RtcEventRange range{run.first_native_row, run.past_last_native_row};
    const auto begin = lower(axis, range, low), end = lower(axis, range, high);
    out.examined = {begin, begin};
    RtcJumpConfirmation current;
    bool nonfinite = false, geometry = false;
    double previous_end = -INFINITY;
    auto reset = [&] { current = {}; };
    for (auto row = begin; row < end; ++row) {
        out.examined.past_last = row + 1;
        const auto &cell = axis.occurrence(row).integration_support;
        // Complete integration support must lie inside the physical search.
        if (cell.begin_unix_sec < low || cell.end_unix_sec > high) { reset(); continue; }
        ++out.examined_rows;
        const double tolerance = 8 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(cell.begin_unix_sec), std::abs(previous_end)});
        if (std::isfinite(previous_end) && std::abs(cell.begin_unix_sec - previous_end) > tolerance) {
            geometry = true; reset();
        }
        previous_end = cell.end_unix_sec;
        if (row >= first_edge && row <= last_edge) { reset(); continue; }
        if (contains(neighbors, row)) { ++out.excluded_rows; reset(); continue; }
        if (!net.state(coord(c), row, event.detector).valid()) { ++out.invalid_rows; reset(); continue; }
        const double y = net.value(coord(c), row, event.detector);
        const double pre = polynomial(fit.cubic_with_offset, (time(axis, row) - event.origin) / event.time_scale);
        const double post = pre + fit.cubic_with_offset.offset;
        if (!std::isfinite(y) || !std::isfinite(pre) || !std::isfinite(post) ||
            !std::isfinite(y - pre) || !std::isfinite(y - post)) {
            nonfinite = true; reset(); continue;
        }
        const std::size_t side = row < first_edge ? 0 : 1;
        if (!agrees(y, side == 0 ? pre : post, out.frozen_residual_scale)) { reset(); continue; }
        const bool other = agrees(y, side == 0 ? post : pre, out.frozen_residual_scale);
        if (!current.rows.present()) {
            current.rows = {row, row + 1}; current.physical = cell;
            current.also_matches_other_reference = other;
        } else {
            current.rows.past_last = row + 1;
            current.physical.end_unix_sec = cell.end_unix_sec;
            current.also_matches_other_reference &= other;
        }
        if (current.physical.duration_sec() >= RtcJumpTransitionPolicy::confirmation_seconds) {
            out.confirmations[side] = current;
            if (side == 1) break; // First post confirmation, last pre confirmation.
        }
    }
    const auto &pre = out.confirmations[0], &post = out.confirmations[1];
    if (nonfinite) out.cause = RtcJumpTransitionCause::nonfinite;
    else if (geometry) out.cause = RtcJumpTransitionCause::support_geometry_unavailable;
    else if (!pre.rows.present()) out.cause = RtcJumpTransitionCause::pre_confirmation_missing;
    else if (!post.rows.present()) out.cause = RtcJumpTransitionCause::post_confirmation_missing;
    else if (pre.also_matches_other_reference || post.also_matches_other_reference)
        out.cause = RtcJumpTransitionCause::ambiguous_reference;
    else {
        out.affected = {pre.rows.past_last, post.rows.first};
        out.physical_bound = {pre.physical.end_unix_sec, post.physical.begin_unix_sec};
        out.cause = RtcJumpTransitionCause::measured;
        if (!out.affected.present() || !(out.physical_bound.begin_unix_sec < out.physical_bound.end_unix_sec))
            out.cause = RtcJumpTransitionCause::support_geometry_unavailable;
        for (auto row = out.affected.first; row < out.affected.past_last; ++row) {
            const auto &cell = axis.occurrence(row).integration_support;
            if (!(cell.begin_unix_sec < out.physical_bound.end_unix_sec &&
                  cell.end_unix_sec > out.physical_bound.begin_unix_sec))
                out.cause = RtcJumpTransitionCause::support_geometry_unavailable;
            else if (contains(neighbors, row))
                out.cause = RtcJumpTransitionCause::competing_exclusion;
            else if (!net.state(coord(c), row, event.detector).valid())
                out.cause = RtcJumpTransitionCause::invalid_transition_support;
            else if (!std::isfinite(net.value(coord(c), row, event.detector)))
                out.cause = RtcJumpTransitionCause::nonfinite;
        }
        out.exceeds_fitting_exclusion = out.affected.first < event.trial_exclusion.first ||
            out.affected.past_last > event.trial_exclusion.past_last;
    }
    return out;
}
} // namespace rtc_jump_transition_detail

// Learn produces compact, coordinate-origin evidence for later complete RTC
// Consider. A measured bracket is conditional on the parent timing/model and
// is not an accepted event, a qualified uncertainty bound, or an Apply plan.
class RtcJumpTransitionEvidence {
public:
    static std::shared_ptr<const RtcJumpTransitionEvidence> learn(
        std::shared_ptr<const RtcJumpTransitionRequest> request, std::uint64_t id) {
        if (!request || id == 0) throw std::invalid_argument("RTC transition Learn requires request and identity");
        auto out = std::shared_ptr<RtcJumpTransitionEvidence>(new RtcJumpTransitionEvidence);
        out->request_ = std::move(request); out->id_ = id;
        const auto &assessment = *out->request_->consistency_handle()->evidence_handle()->amplitude_handle()->review_handle()->evidence_handle();
        const auto &spikes = *assessment.spike_handle();
        std::map<TimestreamNetworkId, std::vector<NativeContiguousRun>> runs;
        for (const auto &span : spikes.input_handle()->spans())
            runs[span.network_id] = spikes.input_handle()->network(span.network_id).occurrence_axis().contiguous_runs();
        std::map<std::pair<TimestreamNetworkId, std::uint32_t>, std::vector<std::size_t>> indices;
        for (std::size_t i = 0; i < assessment.events().size(); ++i)
            if (std::ranges::find(out->request_->coordinates()[i], RtcJumpTransitionRequestCause::requested) != out->request_->coordinates()[i].end()) {
                const auto &e = assessment.events()[i]; indices.try_emplace({e.network, e.detector});
            }
        for (std::size_t i = 0; i < spikes.candidates().size(); ++i) {
            const auto &b = spikes.blocks()[spikes.candidates()[i].noise_block_index];
            const auto found = indices.find({b.network_id, b.detector_index});
            if (found != indices.end()) { found->second.push_back(i); ++out->index_entries_; }
        }
        for (auto &[key, list] : indices)
            std::sort(list.begin(), list.end(), [&](auto a, auto b) {
                return std::tie(spikes.candidates()[a].earlier_row, a) < std::tie(spikes.candidates()[b].earlier_row, b);
            });
        out->coordinates_.resize(assessment.events().size());
        for (std::size_t i = 0; i < assessment.events().size(); ++i) {
            const auto &event = assessment.events()[i];
            if (std::ranges::find(out->request_->coordinates()[i], RtcJumpTransitionRequestCause::requested) == out->request_->coordinates()[i].end()) continue;
            const auto row = spikes.candidates()[event.seed].earlier_row;
            const auto &network_runs = runs.at(event.network);
            const auto found = std::upper_bound(network_runs.begin(), network_runs.end(), row,
                [](auto r, const auto &run) { return r < run.first_native_row; });
            if (found == network_runs.begin() || row >= std::prev(found)->past_last_native_row)
                throw std::invalid_argument("RTC transition candidate has no physical native run");
            const auto neighbors = rtc_jump_transition_detail::neighbor_masks(spikes, event, *std::prev(found), indices.at({event.network, event.detector}));
            out->peak_neighbor_ranges_ = std::max(out->peak_neighbor_ranges_, neighbors.size());
            for (std::size_t c = 0; c < 2; ++c) {
                if (out->request_->coordinates()[i][c] != RtcJumpTransitionRequestCause::requested) continue;
                ++out->requested_;
                out->coordinates_[i][c] = rtc_jump_transition_detail::measure(spikes, event, c, *std::prev(found), neighbors);
                out->examined_ += out->coordinates_[i][c].examined_rows;
            }
        }
        return out;
    }
    const auto &request_handle() const noexcept { return request_; }
    const auto &coordinates() const noexcept { return coordinates_; }
    std::size_t requested_coordinates() const noexcept { return requested_; }
    std::size_t examined_rows() const noexcept { return examined_; }
    std::size_t indexed_candidates() const noexcept { return index_entries_; }
    std::size_t peak_neighbor_ranges() const noexcept { return peak_neighbor_ranges_; }
    std::uint64_t attempt() const noexcept { return id_; }
    static constexpr bool hard_event_accepted = false;
    static constexpr bool apply_authorized = false;
private:
    RtcJumpTransitionEvidence() = default;
    std::shared_ptr<const RtcJumpTransitionRequest> request_;
    std::vector<std::array<RtcJumpTransition, 2>> coordinates_;
    std::uint64_t id_ = 0;
    std::size_t requested_ = 0, examined_ = 0;
    std::size_t index_entries_ = 0, peak_neighbor_ranges_ = 0;
};

} // namespace citlali::pipeline
