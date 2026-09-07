#pragma once

#include <citlali/core/pipeline/timestream_identity_rtc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <tuple>

namespace citlali::pipeline {

// Owner disposition: TIMESTREAM_SUCCESSOR_RTC_SPIKE_LEARN_001_2026-09-07.
// A candidate is an edge of the original signal, not a classified event or
// permission to mask/replace either endpoint. No Gaussian probability is claimed.
struct RtcSpikeLearnPolicy {
    static constexpr std::string_view identity = "rtc-native-spike-learn-v1";
    static constexpr double block_seconds = 10.0;
    static constexpr std::size_t minimum_differences = 256;
    static constexpr double mad_scale = 1.4826;
    static constexpr double threshold = 5.0;
    static constexpr std::string_view readout_assumption =
        "rtc-native-readout-uniform-average-assumption-v1";
};

enum class RtcSpikeProtection : std::uint8_t {
    outside_source, protected_source, unavailable
};

struct RtcSpikeProtectionRegion {
    TimestreamNetworkId network_id = -1;
    std::uint32_t detector_index = 0;
    TimestreamNativeRow first = -1;
    TimestreamNativeRow past_last = -1;
    RtcSpikeProtection state = RtcSpikeProtection::unavailable;
};

// The source owner supplies membership; RTC does not infer sky geometry from
// values, legacy masks, or D2 processing-exclusion bits. The explicitly declared
// default covers the entire exact parent; sparse regions override it. This is
// an in-memory binding, not a source catalog or a persistent identity format.
class RtcSpikeSourceProtection {
public:
    static std::shared_ptr<const RtcSpikeSourceProtection> admit(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        std::string authority_id, RtcSpikeProtection default_state,
        std::vector<RtcSpikeProtectionRegion> regions = {}) {
        auto valid = [](RtcSpikeProtection state) {
            return state == RtcSpikeProtection::outside_source ||
                   state == RtcSpikeProtection::protected_source ||
                   state == RtcSpikeProtection::unavailable;
        };
        if (!parent || authority_id.empty() || !valid(default_state))
            throw std::invalid_argument("RTC spike protection requires exact parent and explicit authority/state");
        std::sort(regions.begin(), regions.end(), [](const auto &a, const auto &b) {
            return std::tie(a.network_id, a.detector_index, a.first) <
                   std::tie(b.network_id, b.detector_index, b.first);
        });
        const RtcSpikeProtectionRegion *previous = nullptr;
        for (const auto &region : regions) {
            const auto &network = parent->network(region.network_id);
            const auto &axis = network.occurrence_axis();
            if (!valid(region.state) ||
                region.detector_index >= static_cast<std::size_t>(network.detector_count()) ||
                region.first < axis.first_native_row() ||
                region.first >= region.past_last ||
                region.past_last > axis.past_last_native_row() ||
                (previous && previous->network_id == region.network_id &&
                 previous->detector_index == region.detector_index &&
                 previous->past_last > region.first))
                throw std::invalid_argument("RTC spike protection region is invalid or overlaps");
            previous = &region;
        }
        return std::shared_ptr<const RtcSpikeSourceProtection>(
            new RtcSpikeSourceProtection{std::move(parent), std::move(authority_id),
                                        default_state, std::move(regions)});
    }

    const auto &parent_handle() const noexcept { return parent_; }
    const std::string &authority_id() const noexcept { return authority_id_; }
    RtcSpikeProtection state(TimestreamNetworkId network, std::uint32_t detector,
                             TimestreamNativeRow row) const {
        const auto &input = parent_->network(network);
        if (detector >= static_cast<std::size_t>(input.detector_count()) ||
            row < input.occurrence_axis().first_native_row() ||
            row >= input.occurrence_axis().past_last_native_row())
            throw std::out_of_range("RTC spike protection address outside parent");
        const auto key = std::tuple{network, detector, row};
        auto found = std::upper_bound(regions_.begin(), regions_.end(), key,
            [](const auto &value, const auto &region) {
                return value < std::tie(region.network_id, region.detector_index, region.first);
            });
        if (found != regions_.begin()) {
            --found;
            if (found->network_id == network && found->detector_index == detector &&
                row < found->past_last) return found->state;
        }
        return default_state_;
    }

private:
    RtcSpikeSourceProtection(std::shared_ptr<const NativePairedReadoutObservation> parent,
        std::string authority, RtcSpikeProtection state,
        std::vector<RtcSpikeProtectionRegion> regions)
        : parent_{std::move(parent)}, authority_id_{std::move(authority)},
          default_state_{state}, regions_{std::move(regions)} {}
    std::shared_ptr<const NativePairedReadoutObservation> parent_;
    std::string authority_id_;
    RtcSpikeProtection default_state_;
    std::vector<RtcSpikeProtectionRegion> regions_;
};

enum class RtcSpikeNoiseCause : std::uint8_t {
    none = 0, insufficient_population = 1, admitted_nonfinite = 2,
    arithmetic_nonfinite = 4, zero_scale = 8
};
constexpr RtcSpikeNoiseCause operator|(RtcSpikeNoiseCause a, RtcSpikeNoiseCause b) {
    return static_cast<RtcSpikeNoiseCause>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

struct RtcSpikeNoiseEstimate {
    std::size_t admitted_differences = 0;
    std::size_t excluded_differences = 0;
    // Unavailable numerical fields stay NaN. Counts and causes remain usable.
    double center = std::numeric_limits<double>::quiet_NaN();
    double scale = std::numeric_limits<double>::quiet_NaN();
    RtcSpikeNoiseCause cause = RtcSpikeNoiseCause::none;
    bool available() const noexcept { return cause == RtcSpikeNoiseCause::none; }
};

struct RtcSpikeNoiseBlock {
    TimestreamNetworkId network_id = -1;
    std::uint32_t detector_index = 0;
    TimestreamNativeRow run_first = -1;
    TimestreamNativeRow first = -1;
    TimestreamNativeRow past_last = -1;
    double anchor_unix_sec = 0.0;
    std::uint64_t time_block_index = 0;
    std::array<RtcSpikeNoiseEstimate, 2> coordinates;
    bool pair_screening_available() const noexcept {
        return coordinates[0].available() && coordinates[1].available();
    }
};

struct RtcSpikeCandidate {
    std::size_t noise_block_index = 0;
    TimestreamNativeRow earlier_row = -1;
    TimestreamNativeRow later_row = -1;
    NativeReadoutCoordinate coordinate = NativeReadoutCoordinate::x;
    double difference = 0.0;
    double centered_difference = 0.0;
    double absolute_score = 0.0;
    RtcSpikeProtection protection = RtcSpikeProtection::unavailable;
    friend bool operator==(const RtcSpikeCandidate &, const RtcSpikeCandidate &) = default;
};

class RtcSpikeEvidence;
std::shared_ptr<const RtcSpikeEvidence> learn_rtc_spike_candidates_partitioned(
    std::shared_ptr<const NativePairedReadoutView>,
    std::span<const std::shared_ptr<const NativePairedReadoutView>>,
    std::shared_ptr<const ValSnapshot>,
    std::shared_ptr<const RtcSpikeSourceProtection>, std::uint64_t);

class RtcSpikeEvidence {
public:
    const auto &input_handle() const noexcept { return input_; }
    const auto &val_snapshot_handle() const noexcept { return val_; }
    const auto &protection_handle() const noexcept { return protection_; }
    std::uint64_t attempt() const noexcept { return attempt_; }
    std::span<const RtcSpikeNoiseBlock> blocks() const noexcept { return blocks_; }
    std::span<const RtcSpikeCandidate> candidates() const noexcept { return candidates_; }
    std::size_t logical_owned_bytes() const noexcept {
        return blocks_.size() * sizeof(RtcSpikeNoiseBlock) +
               candidates_.size() * sizeof(RtcSpikeCandidate);
    }
    std::size_t peak_scratch_differences() const noexcept { return peak_scratch_; }
    ValAddress endpoint_address(std::size_t candidate_index, bool later) const {
        const auto &candidate = candidates_.at(candidate_index);
        const auto &block = blocks_.at(candidate.noise_block_index);
        return val_->address(block.network_id,
            later ? candidate.later_row : candidate.earlier_row, block.detector_index);
    }

    // Publish only explicitly present coordinate-local noise failures. Absence
    // is not permission. Caller supplies the exact ORIGINAL numerical subject
    // descriptors; the RTC evidence author remains a separate identity.
    ValDelta noise_failure_delta(
        std::shared_ptr<const ValSnapshot> snapshot,
        std::span<const std::shared_ptr<const ValNativeRealization>> originals) const {
        if (snapshot.get() != val_.get() || originals.size() != input_->spans().size())
            throw std::invalid_argument("RTC spike publication requires original VAL snapshot and complete subjects");
        for (std::size_t i = 0; i < originals.size(); ++i) {
            const auto &subject = originals[i];
            if (!subject || subject->paired_handle().get() != input_->parent_handle().get() ||
                subject->network_id() != input_->spans()[i].network_id ||
                subject->role() != ValNativeProductRole::original_input)
                throw std::invalid_argument("RTC spike publication subject is not exact original native input");
        }
        ValDeltaBuilder builder{val_, ValProducerProductIdentity{ValProducer::rtc, attempt_}};
        for (const auto &block : blocks_) {
            const auto it = std::lower_bound(originals.begin(), originals.end(), block.network_id,
                [](const auto &subject, auto id) { return subject->network_id() < id; });
            for (std::size_t c = 0; c < 2; ++c) {
                const auto &estimate = block.coordinates[c];
                if (estimate.available()) continue;
                for (auto row = block.first; row < block.past_last; ++row)
                    builder.propose(val_->native_target(*it,
                        val_->address(block.network_id, row, block.detector_index),
                        c == 0 ? NativeReadoutCoordinate::x : NativeReadoutCoordinate::r),
                        ValFactCode{1}, ValFactState{1},
                        ValFactCause{static_cast<std::uint32_t>(estimate.cause)});
            }
        }
        return builder.freeze();
    }

private:
    friend std::shared_ptr<const RtcSpikeEvidence> learn_rtc_spike_candidates_partitioned(
        std::shared_ptr<const NativePairedReadoutView>,
        std::span<const std::shared_ptr<const NativePairedReadoutView>>,
        std::shared_ptr<const ValSnapshot>,
        std::shared_ptr<const RtcSpikeSourceProtection>, std::uint64_t);
    RtcSpikeEvidence(std::shared_ptr<const NativePairedReadoutView> input,
        std::shared_ptr<const ValSnapshot> val,
        std::shared_ptr<const RtcSpikeSourceProtection> protection, std::uint64_t attempt)
        : input_{std::move(input)}, val_{std::move(val)},
          protection_{std::move(protection)}, attempt_{attempt} {}
    std::shared_ptr<const NativePairedReadoutView> input_;
    std::shared_ptr<const ValSnapshot> val_;
    std::shared_ptr<const RtcSpikeSourceProtection> protection_;
    std::uint64_t attempt_;
    std::vector<RtcSpikeNoiseBlock> blocks_;
    std::vector<RtcSpikeCandidate> candidates_;
    std::size_t peak_scratch_ = 0;
};

namespace rtc_spike_detail {
inline double median(std::vector<double> &values) {
    const auto mid = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), mid, values.end());
    if (values.size() % 2 != 0) return *mid;
    return std::midpoint(*std::max_element(values.begin(), mid), *mid);
}
inline std::uint64_t block_index(double time, double anchor) {
    const double index = std::floor((time - anchor) / RtcSpikeLearnPolicy::block_seconds);
    if (!std::isfinite(index) || index < 0.0 || index >= 0x1p64)
        throw std::invalid_argument("RTC spike time block is outside representable domain");
    return static_cast<std::uint64_t>(index);
}
inline RtcSpikeProtection edge_protection(RtcSpikeProtection first, RtcSpikeProtection second) {
    if (first == RtcSpikeProtection::unavailable || second == RtcSpikeProtection::unavailable)
        return RtcSpikeProtection::unavailable;
    if (first == RtcSpikeProtection::protected_source || second == RtcSpikeProtection::protected_source)
        return RtcSpikeProtection::protected_source;
    return RtcSpikeProtection::outside_source;
}
} // namespace rtc_spike_detail

inline std::shared_ptr<const RtcSpikeEvidence> learn_rtc_spike_candidates_partitioned(
    std::shared_ptr<const NativePairedReadoutView> input,
    std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions,
    std::shared_ptr<const ValSnapshot> val,
    std::shared_ptr<const RtcSpikeSourceProtection> protection, std::uint64_t attempt) {
    if (!input || !val || !protection || attempt == 0 ||
        val->generation().value != 0 ||
        val->paired_handle().get() != input->parent_handle().get() ||
        protection->parent_handle().get() != input->parent_handle().get())
        throw std::invalid_argument("RTC spike Learn requires original input, initial VAL snapshot, protection and attempt");
    const auto complete = full_native_occurrence_spans(*input->parent_handle());
    if (!std::equal(complete.begin(), complete.end(), input->spans().begin(), input->spans().end()))
        throw std::invalid_argument("RTC spike logical input must retain complete parent run context");
    require_exact_native_partition_schedule(*input, partitions);
    auto result = std::shared_ptr<RtcSpikeEvidence>(new RtcSpikeEvidence{input, val, protection, attempt});
    // Partitions are execution scheduling only. Canonical per-run traversal
    // ensures each block and boundary-crossing edge is measured exactly once.
    std::vector<double> scratch;
    for (const auto &span : input->spans()) {
        const auto &network = input->network(span.network_id);
        const auto &axis = network.occurrence_axis();
        // A full view of a clipped paired axis still lacks physical run
        // context. Require the producer's complete declared timing support.
        if (axis.first_native_row() != axis.native_timing_handle()->first_native_row() ||
            axis.past_last_native_row() != axis.native_timing_handle()->past_last_native_row())
            throw std::invalid_argument("RTC spike Learn requires complete declared native timing support");
        for (const auto &run : axis.contiguous_runs()) {
            const double anchor = axis.native_identity(run.first_native_row).reconstructed_time_unix_sec();
            auto first = run.first_native_row;
            while (first < run.past_last_native_row) {
                const auto index = rtc_spike_detail::block_index(axis.native_identity(first).reconstructed_time_unix_sec(), anchor);
                auto end = first + 1;
                while (end < run.past_last_native_row &&
                    rtc_spike_detail::block_index(axis.native_identity(end).reconstructed_time_unix_sec(), anchor) == index) ++end;
                for (Eigen::Index detector = 0; detector < network.detector_count(); ++detector) {
                    RtcSpikeNoiseBlock block{span.network_id, static_cast<std::uint32_t>(detector),
                        run.first_native_row, first, end, anchor, index, {}};
                    const auto block_position = result->blocks_.size();
                    for (std::size_t c = 0; c < 2; ++c) {
                        const auto coordinate = c == 0 ? NativeReadoutCoordinate::x : NativeReadoutCoordinate::r;
                        auto &estimate = block.coordinates[c];
                        const auto edge_first = std::max(first, run.first_native_row + 1);
                        scratch.clear();
                        for (auto row = edge_first; row < end; ++row) {
                            if (!network.state(coordinate, row - 1, detector).valid() ||
                                !network.state(coordinate, row, detector).valid()) {
                                ++estimate.excluded_differences;
                                continue;
                            }
                            ++estimate.admitted_differences;
                            const double before = network.value(coordinate, row - 1, detector);
                            const double after = network.value(coordinate, row, detector);
                            if (!std::isfinite(before) || !std::isfinite(after)) {
                                estimate.cause = estimate.cause | RtcSpikeNoiseCause::admitted_nonfinite;
                                continue;
                            }
                            const double difference = after - before;
                            if (!std::isfinite(difference)) {
                                estimate.cause = estimate.cause | RtcSpikeNoiseCause::arithmetic_nonfinite;
                                continue;
                            }
                            scratch.push_back(difference);
                        }
                        result->peak_scratch_ = std::max(result->peak_scratch_, scratch.size());
                        if (estimate.admitted_differences < RtcSpikeLearnPolicy::minimum_differences)
                            estimate.cause = estimate.cause | RtcSpikeNoiseCause::insufficient_population;
                        if (!estimate.available()) continue;
                        const double center = rtc_spike_detail::median(scratch);
                        for (double &value : scratch) value = std::abs(value - center);
                        if (std::any_of(scratch.begin(), scratch.end(), [](double value) { return !std::isfinite(value); })) {
                            estimate.cause = RtcSpikeNoiseCause::arithmetic_nonfinite;
                            continue;
                        }
                        const double scale = RtcSpikeLearnPolicy::mad_scale * rtc_spike_detail::median(scratch);
                        const double cutoff = RtcSpikeLearnPolicy::threshold * scale;
                        if (!std::isfinite(scale) || !std::isfinite(cutoff)) {
                            estimate.cause = RtcSpikeNoiseCause::arithmetic_nonfinite;
                            continue;
                        }
                        if (scale == 0.0) {
                            estimate.cause = RtcSpikeNoiseCause::zero_scale;
                            continue;
                        }
                        estimate.center = center;
                        estimate.scale = scale;
                        const auto candidate_begin = result->candidates_.size();
                        for (auto row = edge_first; row < end; ++row) {
                            if (!network.state(coordinate, row - 1, detector).valid() ||
                                !network.state(coordinate, row, detector).valid()) continue;
                            const double difference = network.value(coordinate, row, detector) -
                                                      network.value(coordinate, row - 1, detector);
                            const double centered = difference - center;
                            const double score = std::abs(centered) / scale;
                            if (!std::isfinite(centered) || !std::isfinite(score)) {
                                estimate.cause = RtcSpikeNoiseCause::arithmetic_nonfinite;
                                estimate.center = estimate.scale = std::numeric_limits<double>::quiet_NaN();
                                result->candidates_.resize(candidate_begin);
                                break;
                            }
                            if (std::abs(centered) >= cutoff) {
                                const auto source = rtc_spike_detail::edge_protection(
                                    protection->state(span.network_id, block.detector_index, row - 1),
                                    protection->state(span.network_id, block.detector_index, row));
                                result->candidates_.push_back({block_position, row - 1, row, coordinate,
                                                              difference, centered, score, source});
                            }
                        }
                    }
                    result->blocks_.push_back(block);
                }
                first = end;
            }
        }
    }
    return result;
}

inline std::shared_ptr<const RtcSpikeEvidence> learn_rtc_spike_candidates(
    std::shared_ptr<const NativePairedReadoutView> input,
    std::shared_ptr<const ValSnapshot> val,
    std::shared_ptr<const RtcSpikeSourceProtection> protection, std::uint64_t attempt) {
    const std::array partitions{input};
    return learn_rtc_spike_candidates_partitioned(std::move(input), partitions,
        std::move(val), std::move(protection), attempt);
}

enum class RtcSpikeCandidateDisposition : std::uint8_t {
    event_assessment_required, protected_optical_assessment_required,
    source_protection_unavailable
};

// A bounded Consider result for these Learn facts, NOT a complete RTC apply
// plan. Successful noise estimation alone is not downstream eligibility.
// Downstream RTC resolution must still classify events, resolve shifts, and
// perform every required source/validity check before constructing its plan.
class RtcSpikeLearningDecision {
public:
    static std::shared_ptr<const RtcSpikeLearningDecision> consider(
        std::shared_ptr<const RtcSpikeEvidence> evidence,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t consideration) {
        if (!evidence || !snapshot || consideration == 0 ||
            snapshot.get() != evidence->val_snapshot_handle().get())
            throw std::invalid_argument("RTC spike Consider requires exact evidence snapshot and identity");
        return std::shared_ptr<const RtcSpikeLearningDecision>(
            new RtcSpikeLearningDecision{std::move(evidence), consideration});
    }
    const auto &evidence_handle() const noexcept { return evidence_; }
    std::uint64_t consideration() const noexcept { return consideration_; }
    bool requires_pair_exclusion_from_mapmaking(std::size_t block) const {
        if (block >= evidence_->blocks().size()) throw std::out_of_range("RTC spike block index");
        return !evidence_->blocks()[block].pair_screening_available();
    }
    RtcSpikeCandidateDisposition candidate_disposition(std::size_t index) const {
        if (index >= evidence_->candidates().size()) throw std::out_of_range("RTC spike candidate index");
        switch (evidence_->candidates()[index].protection) {
        case RtcSpikeProtection::outside_source:
            return RtcSpikeCandidateDisposition::event_assessment_required;
        case RtcSpikeProtection::protected_source:
            return RtcSpikeCandidateDisposition::protected_optical_assessment_required;
        default: return RtcSpikeCandidateDisposition::source_protection_unavailable;
        }
    }
private:
    RtcSpikeLearningDecision(std::shared_ptr<const RtcSpikeEvidence> evidence, std::uint64_t consideration)
        : evidence_{std::move(evidence)}, consideration_{consideration} {}
    std::shared_ptr<const RtcSpikeEvidence> evidence_;
    std::uint64_t consideration_;
};

} // namespace citlali::pipeline
