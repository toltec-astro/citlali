#pragma once

#include <citlali/core/pipeline/timestream_rtc_event_assessment.h>
#include <optional>

namespace citlali::pipeline {

// Owner-selected empirical checks, RTC-JUMP-CONSISTENCY-001, 2026-09-10.
// These overlapping fits are not independent uncertainty measurements.
struct RtcJumpConsistencyPolicy {
    static constexpr std::string_view identity = "rtc-jump-consistency-2026-09-10-v1";
    static constexpr double amplitude_sigma = 5.0;
    static constexpr double short_flank_seconds = 1.0;
    static constexpr double agreement_sigma = 2.0;
};

enum class RtcJumpAmplitudeCause : std::uint8_t {
    not_seeded, background_unavailable, noise_unavailable, arithmetic_nonfinite,
    below_threshold, passes
};

namespace rtc_jump_detail {
inline RtcJumpAmplitudeCause amplitude(double offset, double sigma) {
    if (!std::isfinite(sigma) || sigma <= 0) return RtcJumpAmplitudeCause::noise_unavailable;
    const double bound = RtcJumpConsistencyPolicy::amplitude_sigma * sigma;
    if (!std::isfinite(offset) || !std::isfinite(bound) || !std::isfinite(offset / sigma))
        return RtcJumpAmplitudeCause::arithmetic_nonfinite;
    return std::abs(offset) >= bound ? RtcJumpAmplitudeCause::passes : RtcJumpAmplitudeCause::below_threshold;
}
inline void require_snapshot(const RtcEventAssessmentEvidence &e,
                             const std::shared_ptr<const ValSnapshot> &snapshot,
                             std::uint64_t id) {
    if (!snapshot || id == 0 || e.spike_handle()->val_snapshot_handle().get() != snapshot.get())
        throw std::invalid_argument("RTC jump decision requires original VAL snapshot and nonzero identity");
}
} // namespace rtc_jump_detail

struct RtcJumpAmplitudeCoordinate {
    std::optional<std::size_t> candidate;
    std::optional<std::size_t> noise_block;
    double sigma_delta = std::numeric_limits<double>::quiet_NaN();
    double offset_sigma = std::numeric_limits<double>::quiet_NaN();
    RtcJumpAmplitudeCause cause = RtcJumpAmplitudeCause::not_seeded;
    bool passes() const noexcept { return cause == RtcJumpAmplitudeCause::passes; }
};

// Consider explicitly selects the subset on which the additional Learn runs.
// One original candidate/block per coordinate, using existing onset ordering.
class RtcJumpAmplitudeDecision {
public:
    static std::shared_ptr<const RtcJumpAmplitudeDecision> consider(
        std::shared_ptr<const RtcEventAssessmentDecision> review,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id) {
        if (!review) throw std::invalid_argument("RTC jump amplitude requires assessment decision");
        const auto &assessment = *review->evidence_handle();
        rtc_jump_detail::require_snapshot(assessment, snapshot, id);
        auto result = std::shared_ptr<RtcJumpAmplitudeDecision>(new RtcJumpAmplitudeDecision);
        result->review_ = std::move(review); result->id_ = id;
        const auto &spikes = *assessment.spike_handle();
        result->coordinates_.resize(assessment.events().size());
        for (std::size_t i = 0; i < assessment.events().size(); ++i) {
            const auto &event = assessment.events()[i];
            for (auto candidate : event.candidates) {
                if (candidate >= spikes.candidates().size()) throw std::invalid_argument("RTC jump candidate index out of range");
                const auto &seed = spikes.candidates()[candidate];
                const auto c = seed.coordinate == NativeReadoutCoordinate::x ? 0U : 1U;
                auto &out = result->coordinates_[i][c];
                if (out.candidate) continue;
                if (seed.noise_block_index >= spikes.blocks().size()) throw std::invalid_argument("RTC jump block index out of range");
                const auto &block = spikes.blocks()[seed.noise_block_index];
                if (block.network_id != event.network || block.detector_index != event.detector)
                    throw std::invalid_argument("RTC jump candidate/block differs from assessed detector");
                out.candidate = candidate; out.noise_block = seed.noise_block_index;
                const auto &noise = block.coordinates[c];
                if (!noise.available() || !std::isfinite(noise.scale) || noise.scale <= 0) {
                    out.cause = RtcJumpAmplitudeCause::noise_unavailable; continue;
                }
                out.sigma_delta = noise.scale;
                if (!event.background[c].available()) {
                    out.cause = RtcJumpAmplitudeCause::background_unavailable; continue;
                }
                const double offset = event.background[c].cubic_with_offset.offset;
                out.cause = rtc_jump_detail::amplitude(offset, noise.scale);
                if (out.cause != RtcJumpAmplitudeCause::arithmetic_nonfinite)
                    out.offset_sigma = offset / noise.scale;
            }
        }
        return result;
    }
    const auto &review_handle() const noexcept { return review_; }
    const auto &coordinates() const noexcept { return coordinates_; }
    std::uint64_t consideration() const noexcept { return id_; }
private:
    RtcJumpAmplitudeDecision() = default;
    std::shared_ptr<const RtcEventAssessmentDecision> review_;
    std::vector<std::array<RtcJumpAmplitudeCoordinate, 2>> coordinates_;
    std::uint64_t id_ = 0;
};

enum class RtcJumpShortFitCause : std::uint8_t {
    not_requested, none, context_truncated, insufficient_samples, nonfinite
};

struct RtcJumpShortFit {
    RtcJumpShortFitCause cause = RtcJumpShortFitCause::not_requested;
    std::array<RtcEventFitSupport, 2> support;
    std::array<std::size_t, 2> neighbor_excluded{};
    RtcEventCubicFit pre_scale_fit, cubic_with_offset;
    std::size_t scratch_rows = 0;
    // Coefficients use the primary event's unchanged origin/time_scale basis.
    // All samples, units, signs and reference orientation are original.
    bool available() const noexcept {
        return cause == RtcJumpShortFitCause::none && pre_scale_fit.available() && cubic_with_offset.available();
    }
};

struct RtcJumpFitCounts {
    std::size_t requested_coordinates = 0, pre_fit_calls = 0, joint_fit_calls = 0;
    // IRLS loop entries on successful and failed fits; zero means failure
    // before the loop. Call counts also include failed numerical attempts.
    std::size_t available_coordinates = 0, pre_iterations = 0, joint_iterations = 0;
    std::size_t peak_scratch_rows = 0;
};

namespace rtc_jump_detail {
inline RtcJumpShortFit short_fit(const RtcSpikeEvidence &spikes,
                                const RtcAssessedEvent &event, std::size_t c,
                                const NativeContiguousRun &run, RtcJumpFitCounts &counts) {
    using namespace rtc_event_assessment_detail;
    RtcJumpShortFit out; out.cause = RtcJumpShortFitCause::none;
    const auto &net = spikes.input_handle()->network(event.network);
    const auto &axis = net.occurrence_axis();
    double begin = INFINITY, end = -INFINITY;
    for (auto row = event.trial_exclusion.first; row < event.trial_exclusion.past_last; ++row) {
        const auto &cell = axis.occurrence(row).integration_support;
        begin = std::min(begin, cell.begin_unix_sec); end = std::max(end, cell.end_unix_sec);
    }
    const double low = begin - RtcJumpConsistencyPolicy::short_flank_seconds;
    const double high = end + RtcJumpConsistencyPolicy::short_flank_seconds;
    if (!std::isfinite(low) || !std::isfinite(high) || !std::isfinite(event.origin) ||
        !std::isfinite(event.time_scale) || event.time_scale <= 0) {
        out.cause = RtcJumpShortFitCause::nonfinite; return out;
    }
    if (axis.occurrence(run.first_native_row).integration_support.begin_unix_sec > low ||
        axis.occurrence(run.past_last_native_row - 1).integration_support.end_unix_sec < high) {
        out.cause = RtcJumpShortFitCause::context_truncated; return out;
    }
    std::vector<std::array<double, 3>> samples;
    for (std::size_t side = 0; side < 2; ++side) {
        // Visit only the primary flank: no copied full-observation mask/plane.
        const auto &parent_support = event.background[c].support[side];
        for (auto row = parent_support.first_used; row <= parent_support.last_used; ++row) {
            const auto &cell = axis.occurrence(row).integration_support;
            const bool admitted = side == 0 ? row < event.trial_exclusion.first &&
                cell.begin_unix_sec >= low && cell.end_unix_sec <= begin :
                row >= event.trial_exclusion.past_last && cell.begin_unix_sec >= end && cell.end_unix_sec <= high;
            if (!admitted) continue;
            if (contains(event.neighbor_exclusions, row)) { ++out.neighbor_excluded[side]; continue; }
            auto &support = out.support[side];
            if (!net.state(coord(c), row, event.detector).valid()) { ++support.invalid; continue; }
            const double y = net.value(coord(c), row, event.detector);
            if (!std::isfinite(y)) { out.cause = RtcJumpShortFitCause::nonfinite; continue; }
            if (support.usable == 0) {
                support.first_used = row; support.begin_unix_sec = cell.begin_unix_sec;
                support.end_unix_sec = cell.end_unix_sec;
            }
            ++support.usable; support.last_used = row;
            support.begin_unix_sec = std::min(support.begin_unix_sec, cell.begin_unix_sec);
            support.end_unix_sec = std::max(support.end_unix_sec, cell.end_unix_sec);
            samples.push_back({(time(axis, row) - event.origin) / event.time_scale, y, static_cast<double>(side)});
        }
    }
    out.scratch_rows = samples.size();
    counts.peak_scratch_rows = std::max(counts.peak_scratch_rows, out.scratch_rows);
    if (out.cause != RtcJumpShortFitCause::none) return out;
    if (out.support[0].usable < RtcEventBackgroundPolicy::minimum_samples ||
        out.support[1].usable < RtcEventBackgroundPolicy::minimum_samples) {
        out.cause = RtcJumpShortFitCause::insufficient_samples; return out;
    }
    Eigen::MatrixXd design(samples.size(), 5); Eigen::VectorXd values(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto &s = samples[i];
        design.row(i) << 1, s[0], s[0]*s[0], s[0]*s[0]*s[0], s[2]; values[i] = s[1];
    }
    const auto n = static_cast<Eigen::Index>(out.support[0].usable);
    ++counts.pre_fit_calls;
    out.pre_scale_fit = rtc_event_background_detail::fit(design.topLeftCorner(n, 4), values.head(n));
    counts.pre_iterations += out.pre_scale_fit.iterations;
    if (!out.pre_scale_fit.available()) return out;
    ++counts.joint_fit_calls;
    out.cubic_with_offset = rtc_event_background_detail::fit(design, values, out.pre_scale_fit.scale);
    counts.joint_iterations += out.cubic_with_offset.iterations;
    if (out.available()) ++counts.available_coordinates;
    return out;
}
} // namespace rtc_jump_detail

class RtcJumpConsistencyEvidence {
public:
    static std::shared_ptr<const RtcJumpConsistencyEvidence> learn(
        std::shared_ptr<const RtcJumpAmplitudeDecision> amplitude, std::uint64_t id) {
        if (!amplitude || id == 0) throw std::invalid_argument("RTC short-fit Learn requires amplitude decision and identity");
        auto result = std::shared_ptr<RtcJumpConsistencyEvidence>(new RtcJumpConsistencyEvidence);
        result->amplitude_ = std::move(amplitude); result->id_ = id;
        const auto &assessment = *result->amplitude_->review_handle()->evidence_handle();
        const auto &spikes = *assessment.spike_handle();
        std::map<TimestreamNetworkId, std::vector<NativeContiguousRun>> runs;
        for (const auto &span : spikes.input_handle()->spans())
            runs[span.network_id] = spikes.input_handle()->network(span.network_id).occurrence_axis().contiguous_runs();
        result->coordinates_.resize(assessment.events().size());
        for (std::size_t i = 0; i < assessment.events().size(); ++i) {
            const auto &event = assessment.events()[i];
            for (std::size_t c = 0; c < 2; ++c) {
                const auto &gate = result->amplitude_->coordinates()[i][c];
                if (!gate.passes()) continue;
                ++result->counts_.requested_coordinates;
                const auto row = spikes.candidates()[*gate.candidate].earlier_row;
                const auto &network_runs = runs.at(event.network);
                const auto found = std::upper_bound(network_runs.begin(), network_runs.end(), row,
                    [](auto r, const auto &run) { return r < run.first_native_row; });
                if (found == network_runs.begin() || row >= std::prev(found)->past_last_native_row)
                    throw std::invalid_argument("RTC jump candidate has no physical native run");
                result->coordinates_[i][c] = rtc_jump_detail::short_fit(spikes, event, c, *std::prev(found), result->counts_);
            }
        }
        return result;
    }
    const auto &amplitude_handle() const noexcept { return amplitude_; }
    const auto &coordinates() const noexcept { return coordinates_; }
    const auto &counts() const noexcept { return counts_; }
    std::uint64_t attempt() const noexcept { return id_; }
private:
    RtcJumpConsistencyEvidence() = default;
    std::shared_ptr<const RtcJumpAmplitudeDecision> amplitude_;
    std::vector<std::array<RtcJumpShortFit, 2>> coordinates_;
    RtcJumpFitCounts counts_;
    std::uint64_t id_ = 0;
};

enum class RtcJumpConsistencyCause : std::uint8_t {
    primary_not_passed, short_fit_unavailable, arithmetic_nonfinite,
    short_below_threshold, sign_disagreement, offset_disagreement, passes
};
namespace rtc_jump_detail {
inline RtcJumpConsistencyCause consistency(double primary, double shorter, double sigma) {
    const auto a = amplitude(primary, sigma), b = amplitude(shorter, sigma);
    if (a == RtcJumpAmplitudeCause::below_threshold) return RtcJumpConsistencyCause::primary_not_passed;
    if (a != RtcJumpAmplitudeCause::passes || b == RtcJumpAmplitudeCause::noise_unavailable ||
        b == RtcJumpAmplitudeCause::arithmetic_nonfinite) return RtcJumpConsistencyCause::arithmetic_nonfinite;
    if (b == RtcJumpAmplitudeCause::below_threshold) return RtcJumpConsistencyCause::short_below_threshold;
    if (std::signbit(primary) != std::signbit(shorter)) return RtcJumpConsistencyCause::sign_disagreement;
    const double difference = std::abs(primary - shorter);
    const double bound = RtcJumpConsistencyPolicy::agreement_sigma * sigma;
    if (!std::isfinite(difference) || !std::isfinite(bound)) return RtcJumpConsistencyCause::arithmetic_nonfinite;
    return difference <= bound ? RtcJumpConsistencyCause::passes : RtcJumpConsistencyCause::offset_disagreement;
}
} // namespace rtc_jump_detail

struct RtcJumpCoordinateDecision {
    RtcJumpConsistencyCause cause = RtcJumpConsistencyCause::primary_not_passed;
    double short_offset_sigma = std::numeric_limits<double>::quiet_NaN();
    double offset_difference_sigma = std::numeric_limits<double>::quiet_NaN();
    bool confirmed_recovery_excludes_persistent_shift = false;
    bool passes() const noexcept { return cause == RtcJumpConsistencyCause::passes; }
};

class RtcJumpConsistencyDecision {
public:
    static std::shared_ptr<const RtcJumpConsistencyDecision> consider(
        std::shared_ptr<const RtcJumpConsistencyEvidence> evidence,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id) {
        if (!evidence) throw std::invalid_argument("RTC jump consideration requires short-fit evidence");
        const auto &assessment = *evidence->amplitude_handle()->review_handle()->evidence_handle();
        rtc_jump_detail::require_snapshot(assessment, snapshot, id);
        auto result = std::shared_ptr<RtcJumpConsistencyDecision>(new RtcJumpConsistencyDecision);
        result->evidence_ = std::move(evidence); result->id_ = id;
        result->coordinates_.resize(assessment.events().size());
        for (std::size_t i = 0; i < assessment.events().size(); ++i) for (std::size_t c = 0; c < 2; ++c) {
            auto &out = result->coordinates_[i][c];
            out.confirmed_recovery_excludes_persistent_shift = assessment.events()[i].recovery[c].recovered();
            const auto &amplitude = result->evidence_->amplitude_handle()->coordinates()[i][c];
            if (!amplitude.passes()) continue;
            const auto &fit = result->evidence_->coordinates()[i][c];
            if (!fit.available()) { out.cause = RtcJumpConsistencyCause::short_fit_unavailable; continue; }
            const double primary = assessment.events()[i].background[c].cubic_with_offset.offset;
            const double shorter = fit.cubic_with_offset.offset;
            out.cause = rtc_jump_detail::consistency(primary, shorter, amplitude.sigma_delta);
            if (out.cause != RtcJumpConsistencyCause::arithmetic_nonfinite) {
                out.short_offset_sigma = shorter / amplitude.sigma_delta;
                const double difference_sigma = std::abs(primary - shorter) / amplitude.sigma_delta;
                if (std::isfinite(difference_sigma)) out.offset_difference_sigma = difference_sigma;
            }
        }
        return result;
    }
    const auto &evidence_handle() const noexcept { return evidence_; }
    const auto &coordinates() const noexcept { return coordinates_; }
    std::uint64_t consideration() const noexcept { return id_; }
    // Consistency is one empirical predicate, never complete hard-event admission.
    static constexpr bool hard_event_accepted = false;
    static constexpr bool apply_authorized = false;
private:
    RtcJumpConsistencyDecision() = default;
    std::shared_ptr<const RtcJumpConsistencyEvidence> evidence_;
    std::vector<std::array<RtcJumpCoordinateDecision, 2>> coordinates_;
    std::uint64_t id_ = 0;
};

} // namespace citlali::pipeline
