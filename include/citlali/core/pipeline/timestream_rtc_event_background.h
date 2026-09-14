#pragma once

#include <citlali/core/pipeline/timestream_rtc_spike_learn.h>
#include <Eigen/QR>

namespace citlali::pipeline {

// Owner decisions and numerical binding:
// TIMESTREAM_SUCCESSOR_RTC_EVENT_BACKGROUND_001_2026-09-08.
// This is descriptive original-sample Learn evidence, not a shift classifier.
struct RtcEventBackgroundPolicy {
    static constexpr std::string_view identity = "rtc-event-joint-cubic-offset-v1";
    static constexpr double flank_seconds = 2.0;
    static constexpr std::size_t minimum_samples = 64;
    static constexpr double huber_tuning = 1.345;
    static constexpr std::size_t maximum_iterations = 256;
    static constexpr double convergence_tolerance = 1e-8;
};

enum class RtcEventFitCause : std::uint8_t {
    none, not_attempted, insufficient_samples, additional_candidate,
    nonfinite, zero_scale, rank_deficient, iteration_limit
};

// The caller supplies a fitting exclusion, NOT accepted physical event extent.
// Both original candidate endpoints must be inside this half-open native range.
struct RtcEventFitRequest {
    std::size_t candidate_index = 0;
    TimestreamNativeRow excluded_first = -1;
    TimestreamNativeRow excluded_past_last = -1;
};

struct RtcEventFitSupport {
    std::size_t usable = 0;
    std::size_t invalid = 0;
    TimestreamNativeRow first_used = -1;
    TimestreamNativeRow last_used = -1;
    double begin_unix_sec = std::numeric_limits<double>::quiet_NaN();
    double end_unix_sec = std::numeric_limits<double>::quiet_NaN();
};

struct RtcEventCubicFit {
    // p(u) = coefficients[0] + ... + coefficients[3]*u^3.
    // u=(native midpoint time - evidence.origin_unix_sec())/time_scale_seconds.
    // All coefficients and offset retain this coordinate's ORIGINAL units,
    // scale, reference and sign through the exact parent mapping authority.
    std::array<double, 4> coefficients{
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    double offset = std::numeric_limits<double>::quiet_NaN();
    double scale = std::numeric_limits<double>::quiet_NaN();
    // Sum rho(residual/scale), Huber rho with k=1.345. Descriptive only;
    // neither a likelihood nor a covariance/independence assumption.
    double huber_loss = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    RtcEventFitCause cause = RtcEventFitCause::not_attempted;
    bool available() const noexcept { return cause == RtcEventFitCause::none; }
};

struct RtcEventCoordinateBackground {
    // pre, post; support eligibility is coordinate-local. x/r are never pooled.
    std::array<RtcEventFitSupport, 2> support;
    RtcEventFitCause support_cause = RtcEventFitCause::not_attempted;
    RtcEventCubicFit pre_scale_fit;
    RtcEventCubicFit cubic;
    RtcEventCubicFit cubic_with_offset;
    bool available() const noexcept {
        return support_cause == RtcEventFitCause::none && pre_scale_fit.available() &&
               cubic.available() && cubic_with_offset.available();
    }
};

namespace rtc_event_background_detail {
inline RtcEventCubicFit fit(const Eigen::MatrixXd &design, const Eigen::VectorXd &values,
                           double fixed_scale = std::numeric_limits<double>::quiet_NaN()) {
    RtcEventCubicFit result;
    auto fail = [&](RtcEventFitCause cause) { result.cause = cause; return result; };
    if (design.rows() != values.size() || design.rows() < design.cols() ||
        (design.cols() != 4 && design.cols() != 5))
        throw std::invalid_argument("RTC event fit design shape");
    if (!design.allFinite() || !values.allFinite()) return fail(RtcEventFitCause::nonfinite);
    const bool estimate_scale = std::isnan(fixed_scale);
    if (!estimate_scale && (!std::isfinite(fixed_scale) || fixed_scale <= 0))
        return fail(RtcEventFitCause::zero_scale);
    std::vector<double> scratch(values.data(), values.data() + values.size());
    const double center = rtc_spike_detail::median(scratch);
    const Eigen::VectorXd y = values.array() - center;
    if (!y.allFinite()) return fail(RtcEventFitCause::nonfinite);
    const double rank_tolerance = 128 * std::numeric_limits<double>::epsilon() *
                                  static_cast<double>(design.rows());
    auto solve = [&](const Eigen::MatrixXd &a, const Eigen::VectorXd &b, Eigen::VectorXd &coef) {
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a);
        qr.setThreshold(rank_tolerance);
        if (qr.rank() != a.cols()) return RtcEventFitCause::rank_deficient;
        coef = qr.solve(b);
        return coef.allFinite() ? RtcEventFitCause::none : RtcEventFitCause::nonfinite;
    };
    Eigen::VectorXd coef;
    if (const auto cause = solve(design, y, coef); cause != RtcEventFitCause::none) return fail(cause);
    Eigen::VectorXd predicted = design * coef;
    Eigen::VectorXd residual = y - predicted;
    auto scale_for = [&](const Eigen::VectorXd &e) {
        scratch.assign(e.data(), e.data() + e.size());
        const double middle = rtc_spike_detail::median(scratch);
        for (double &v : scratch) v = std::abs(v - middle);
        if (std::any_of(scratch.begin(), scratch.end(), [](double v) { return !std::isfinite(v); }))
            return std::numeric_limits<double>::quiet_NaN();
        return RtcSpikeLearnPolicy::mad_scale * rtc_spike_detail::median(scratch);
    };
    if (!predicted.allFinite() || !residual.allFinite()) return fail(RtcEventFitCause::nonfinite);
    double scale = estimate_scale ? scale_for(residual) : fixed_scale;
    for (std::size_t iteration = 1; iteration <= RtcEventBackgroundPolicy::maximum_iterations; ++iteration) {
        result.iterations = iteration;
        if (!std::isfinite(scale)) return fail(RtcEventFitCause::nonfinite);
        if (scale <= 0) return fail(RtcEventFitCause::zero_scale);
        Eigen::MatrixXd weighted = design;
        Eigen::VectorXd rhs = y;
        for (Eigen::Index row = 0; row < design.rows(); ++row) {
            const double score = std::abs(residual[row]) / scale;
            if (!std::isfinite(score)) return fail(RtcEventFitCause::nonfinite);
            const double weight = score <= RtcEventBackgroundPolicy::huber_tuning ? 1.0 :
                std::sqrt(RtcEventBackgroundPolicy::huber_tuning / score);
            weighted.row(row) *= weight;
            rhs[row] *= weight;
        }
        Eigen::VectorXd next;
        if (const auto cause = solve(weighted, rhs, next); cause != RtcEventFitCause::none) return fail(cause);
        const Eigen::VectorXd next_prediction = design * next;
        const Eigen::VectorXd next_residual = y - next_prediction;
        if (!next_prediction.allFinite() || !next_residual.allFinite()) return fail(RtcEventFitCause::nonfinite);
        const double next_scale = estimate_scale ? scale_for(next_residual) : scale;
        if (!std::isfinite(next_scale)) return fail(RtcEventFitCause::nonfinite);
        if (next_scale <= 0) return fail(RtcEventFitCause::zero_scale);
        const double change = (next_prediction - predicted).cwiseAbs().maxCoeff();
        const double roundoff = 64 * std::numeric_limits<double>::epsilon() * y.cwiseAbs().maxCoeff();
        const bool converged = std::isfinite(change) &&
            change <= RtcEventBackgroundPolicy::convergence_tolerance * scale + roundoff &&
            std::abs(next_scale - scale) <= RtcEventBackgroundPolicy::convergence_tolerance * scale;
        coef = next;
        predicted = next_prediction;
        residual = next_residual;
        scale = next_scale;
        if (!converged) continue;
        coef[0] += center;
        if (!coef.allFinite()) return fail(RtcEventFitCause::nonfinite);
        double loss = 0;
        for (double e : residual) {
            const double z = std::abs(e) / scale;
            const double k = RtcEventBackgroundPolicy::huber_tuning;
            loss += z <= k ? 0.5*z*z : k*(z - 0.5*k);
        }
        if (!std::isfinite(loss)) return fail(RtcEventFitCause::nonfinite);
        std::copy_n(coef.data(), 4, result.coefficients.begin());
        result.offset = coef.size() == 5 ? coef[4] : 0;
        result.scale = scale;
        result.huber_loss = loss;
        result.cause = RtcEventFitCause::none;
        return result;
    }
    return fail(RtcEventFitCause::iteration_limit);
}
} // namespace rtc_event_background_detail

class RtcEventBackgroundEvidence;
std::shared_ptr<const RtcEventBackgroundEvidence> learn_rtc_event_background(
    std::shared_ptr<const RtcSpikeEvidence>, RtcEventFitRequest, std::uint64_t);

class RtcEventBackgroundEvidence {
public:
    const auto &spike_evidence_handle() const noexcept { return spikes_; }
    const auto &request() const noexcept { return request_; }
    std::uint64_t attempt() const noexcept { return attempt_; }
    const auto &coordinates() const noexcept { return coordinates_; }
    const auto &excluded_support() const noexcept { return excluded_support_; }
    double origin_unix_sec() const noexcept { return origin_; }
    double time_scale_seconds() const noexcept { return time_scale_; }
    // pre/post requested windows clipped by actual acquisition support.
    const auto &observation_truncated() const noexcept { return observation_truncated_; }
    const auto &gap_truncated() const noexcept { return gap_truncated_; }
    std::size_t peak_scratch_rows() const noexcept { return peak_scratch_rows_; }
    static constexpr bool offset_uncertainty_available() noexcept { return false; }
    static constexpr bool exclusion_containment_established() noexcept { return false; }
private:
    friend std::shared_ptr<const RtcEventBackgroundEvidence> learn_rtc_event_background(
        std::shared_ptr<const RtcSpikeEvidence>, RtcEventFitRequest, std::uint64_t);
    RtcEventBackgroundEvidence(std::shared_ptr<const RtcSpikeEvidence> spikes,
        RtcEventFitRequest request, std::uint64_t attempt)
        : spikes_{std::move(spikes)}, request_{request}, attempt_{attempt} {}
    std::shared_ptr<const RtcSpikeEvidence> spikes_;
    RtcEventFitRequest request_;
    std::uint64_t attempt_;
    std::array<RtcEventCoordinateBackground, 2> coordinates_;
    NativeReadoutIntegrationSupport excluded_support_;
    double origin_ = 0;
    double time_scale_ = 1;
    std::array<bool, 2> observation_truncated_{};
    std::array<bool, 2> gap_truncated_{};
    std::size_t peak_scratch_rows_ = 0;
};

inline std::shared_ptr<const RtcEventBackgroundEvidence> learn_rtc_event_background(
    std::shared_ptr<const RtcSpikeEvidence> spikes, RtcEventFitRequest request, std::uint64_t attempt) {
    if (!spikes || attempt == 0 || request.candidate_index >= spikes->candidates().size())
        throw std::invalid_argument("RTC background requires exact spike evidence, candidate and attempt");
    const auto &candidate = spikes->candidates()[request.candidate_index];
    const auto &block = spikes->blocks()[candidate.noise_block_index];
    const auto &network = spikes->input_handle()->network(block.network_id);
    const auto &axis = network.occurrence_axis();
    if (request.excluded_first < axis.first_native_row() ||
        request.excluded_first > candidate.earlier_row ||
        request.excluded_past_last <= candidate.later_row ||
        request.excluded_past_last > axis.past_last_native_row())
        throw std::invalid_argument("RTC background exclusion must contain original candidate endpoints");
    const auto runs = axis.contiguous_runs();
    const auto found = std::find_if(runs.begin(), runs.end(), [&](const auto &run) {
        return run.first_native_row <= request.excluded_first &&
               run.past_last_native_row >= request.excluded_past_last;
    });
    if (found == runs.end()) throw std::invalid_argument("RTC background exclusion crosses a physical native gap");
    const auto &run = *found;
    auto e = std::shared_ptr<RtcEventBackgroundEvidence>(new RtcEventBackgroundEvidence{spikes, request, attempt});
    // Include the full physical support of every excluded cell, even if a
    // producer's integration supports overlap or have nonuniform widths.
    e->excluded_support_ = axis.occurrence(request.excluded_first).integration_support;
    for (auto row = request.excluded_first + 1; row < request.excluded_past_last; ++row) {
        const auto &s = axis.occurrence(row).integration_support;
        e->excluded_support_.begin_unix_sec = std::min(e->excluded_support_.begin_unix_sec, s.begin_unix_sec);
        e->excluded_support_.end_unix_sec = std::max(e->excluded_support_.end_unix_sec, s.end_unix_sec);
    }
    const double begin = e->excluded_support_.begin_unix_sec;
    const double end = e->excluded_support_.end_unix_sec;
    const double low = begin - RtcEventBackgroundPolicy::flank_seconds;
    const double high = end + RtcEventBackgroundPolicy::flank_seconds;
    e->origin_ = std::midpoint(begin, end);
    e->time_scale_ = std::max(e->origin_ - low, high - e->origin_);
    if (!std::isfinite(low) || !std::isfinite(high) || !std::isfinite(e->time_scale_) || e->time_scale_ <= 0)
        throw std::invalid_argument("RTC background physical window is not representable");
    const bool before_clipped = axis.occurrence(run.first_native_row).integration_support.begin_unix_sec > low;
    const bool after_clipped = axis.occurrence(run.past_last_native_row - 1).integration_support.end_unix_sec < high;
    e->observation_truncated_ = {before_clipped && run.first_native_row == axis.first_native_row(),
                                after_clipped && run.past_last_native_row == axis.past_last_native_row()};
    e->gap_truncated_ = {before_clipped && run.first_native_row != axis.first_native_row(),
                        after_clipped && run.past_last_native_row != axis.past_last_native_row()};
    // Locate only the local midpoint range; inspect complete integration
    // supports below. Internal computational partitions never enter this API.
    auto lower_row = [&](double time) {
        auto first = run.first_native_row, last = run.past_last_native_row;
        while (first < last) {
            const auto mid = first + (last - first) / 2;
            if (axis.native_identity(mid).reconstructed_time_unix_sec() < time) first = mid + 1;
            else last = mid;
        }
        return first;
    };
    const auto local_first = lower_row(low), local_end = lower_row(high);
    auto flank = [&](TimestreamNativeRow row) -> int {
        if (row < local_first || row >= local_end) return -1;
        const auto &s = axis.occurrence(row).integration_support;
        if (row < request.excluded_first && s.begin_unix_sec >= low && s.end_unix_sec <= begin) return 0;
        if (row >= request.excluded_past_last && s.begin_unix_sec >= end && s.end_unix_sec <= high) return 1;
        return -1;
    };
    for (std::size_t c = 0; c < 2; ++c) {
        auto &out = e->coordinates_[c];
        const auto coordinate = c == 0 ? NativeReadoutCoordinate::x : NativeReadoutCoordinate::r;
        std::vector<std::array<double, 3>> samples; // scaled midpoint, original value, post indicator
        out.support_cause = RtcEventFitCause::none;
        for (auto row = local_first; row < local_end; ++row) {
            const int side = flank(row);
            if (side < 0) continue;
            auto &support = out.support[side];
            if (!network.state(coordinate, row, block.detector_index).valid()) { ++support.invalid; continue; }
            const double value = network.value(coordinate, row, block.detector_index);
            if (!std::isfinite(value)) { out.support_cause = RtcEventFitCause::nonfinite; continue; }
            const auto &s = axis.occurrence(row).integration_support;
            if (support.usable == 0) {
                support.first_used = row;
                support.begin_unix_sec = s.begin_unix_sec;
                support.end_unix_sec = s.end_unix_sec;
            }
            ++support.usable;
            support.last_used = row;
            support.begin_unix_sec = std::min(support.begin_unix_sec, s.begin_unix_sec);
            support.end_unix_sec = std::max(support.end_unix_sec, s.end_unix_sec);
            samples.push_back({(axis.native_identity(row).reconstructed_time_unix_sec() - e->origin_) / e->time_scale_,
                               value, static_cast<double>(side)});
        }
        e->peak_scratch_rows_ = std::max(e->peak_scratch_rows_, samples.size());
        if (out.support_cause != RtcEventFitCause::none) continue;
        if (std::any_of(out.support.begin(), out.support.end(), [](const auto &s) {
                return s.usable < RtcEventBackgroundPolicy::minimum_samples;
            })) { out.support_cause = RtcEventFitCause::insufficient_samples; continue; }
        for (const auto &other : spikes->candidates()) {
            const auto &other_block = spikes->blocks()[other.noise_block_index];
            if (other.coordinate == coordinate && other_block.network_id == block.network_id &&
                other_block.detector_index == block.detector_index &&
                (flank(other.earlier_row) >= 0 || flank(other.later_row) >= 0)) {
                out.support_cause = RtcEventFitCause::additional_candidate;
                break;
            }
        }
        if (out.support_cause != RtcEventFitCause::none) continue;
        Eigen::MatrixXd design(samples.size(), 5);
        Eigen::VectorXd values(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const auto &s = samples[i];
            design.row(i) << 1.0, s[0], s[0]*s[0], s[0]*s[0]*s[0], s[2];
            values[i] = s[1];
        }
        const auto pre = static_cast<Eigen::Index>(out.support[0].usable);
        out.pre_scale_fit = rtc_event_background_detail::fit(design.topLeftCorner(pre, 4), values.head(pre));
        if (!out.pre_scale_fit.available()) continue;
        const double frozen_scale = out.pre_scale_fit.scale;
        out.cubic = rtc_event_background_detail::fit(design.leftCols(4), values, frozen_scale);
        out.cubic_with_offset = rtc_event_background_detail::fit(design, values, frozen_scale);
    }
    return e;
}

struct RtcEventBackgroundRequirements {
    bool numerical_evidence_unavailable = true;
    bool candidate_block_pair_screening_exclusion_required = false;
    bool incomplete_observation_context = false;
    bool physical_gap_context = false;
    bool source_protection_unavailable = true;
    bool protected_optical_assessment_required = false;
    bool offset_uncertainty_and_acceptance_required = true;
    bool extent_containment_and_recovery_required = true;
    bool background_adequacy_and_coverage_required = true;
};

// Bounded runtime Consider constraints. There is deliberately no conversion to
// an RTC Apply plan or to a hard paired pathology flag from these fit facts.
class RtcEventBackgroundDecision {
public:
    static std::shared_ptr<const RtcEventBackgroundDecision> consider(
        std::shared_ptr<const RtcEventBackgroundEvidence> evidence,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t consideration) {
        if (!evidence || consideration == 0 ||
            snapshot.get() != evidence->spike_evidence_handle()->val_snapshot_handle().get())
            throw std::invalid_argument("RTC background Consider requires exact evidence snapshot and identity");
        return std::shared_ptr<const RtcEventBackgroundDecision>(
            new RtcEventBackgroundDecision{std::move(evidence), consideration});
    }
    const auto &evidence_handle() const noexcept { return evidence_; }
    // Retain ALL earlier screening decisions, including blocks beyond this
    // local fit. Later complete planning must consume both products.
    const auto &screening_decision_handle() const noexcept { return screening_; }
    std::uint64_t consideration() const noexcept { return consideration_; }
    RtcEventBackgroundRequirements requirements(NativeReadoutCoordinate coordinate) const {
        if (coordinate != NativeReadoutCoordinate::x && coordinate != NativeReadoutCoordinate::r)
            throw std::invalid_argument("RTC event background requires x or r coordinate");
        RtcEventBackgroundRequirements r;
        r.numerical_evidence_unavailable = !evidence_->coordinates()[coordinate == NativeReadoutCoordinate::x ? 0 : 1].available();
        const auto any = [](const auto &a) { return a[0] || a[1]; };
        r.incomplete_observation_context = any(evidence_->observation_truncated());
        r.physical_gap_context = any(evidence_->gap_truncated());
        // Examine the entire proposed exclusion, not just the seed edge. The
        // original authority handle is retained for any later extent expansion.
        const auto &spikes = *evidence_->spike_evidence_handle();
        const auto &candidate = spikes.candidates()[evidence_->request().candidate_index];
        const auto &block = spikes.blocks()[candidate.noise_block_index];
        r.candidate_block_pair_screening_exclusion_required =
            screening_->requires_pair_exclusion_from_mapmaking(candidate.noise_block_index);
        r.source_protection_unavailable = false;
        for (auto row = evidence_->request().excluded_first; row < evidence_->request().excluded_past_last; ++row) {
            const auto state = spikes.protection_handle()->state(block.network_id, block.detector_index, row);
            r.source_protection_unavailable |= state == RtcSpikeProtection::unavailable;
            r.protected_optical_assessment_required |= state == RtcSpikeProtection::protected_source;
        }
        return r;
    }
private:
    RtcEventBackgroundDecision(std::shared_ptr<const RtcEventBackgroundEvidence> evidence, std::uint64_t id)
        : evidence_{std::move(evidence)}, consideration_{id},
          screening_{RtcSpikeLearningDecision::consider(evidence_->spike_evidence_handle(),
              evidence_->spike_evidence_handle()->val_snapshot_handle(), id)} {}
    std::shared_ptr<const RtcEventBackgroundEvidence> evidence_;
    std::uint64_t consideration_;
    std::shared_ptr<const RtcSpikeLearningDecision> screening_;
};

} // namespace citlali::pipeline
