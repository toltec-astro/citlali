#pragma once

#include <citlali/core/pipeline/timestream_rtc_jump_exclusion.h>

namespace citlali::pipeline {

// These are RTC treatment causes, not producer validity or a PTC/VAL use
// decision. In particular, a screening failure is not a detected event.
struct RtcTransientExclusionCauses {
    std::array<RtcSpikeNoiseCause, 2> screening{};
    bool accepted_jump = false;
    bool excluded() const noexcept {
        return accepted_jump || screening[0] != RtcSpikeNoiseCause::none ||
               screening[1] != RtcSpikeNoiseCause::none;
    }
};

struct RtcTransientScreeningBlock {
    std::size_t original_block = 0;
    RtcEventRange rows;
};

struct RtcTransientDetectorExclusion {
    TimestreamNetworkId network = -1;
    std::uint32_t detector = 0;
    std::vector<RtcTransientScreeningBlock> screening;
    // Half-open original native row union; original run/time identities and
    // physical jump bounds remain in the referenced evidence products.
    std::vector<RtcEventRange> rows;
};

struct RtcTransientExclusionCounts {
    std::size_t screening_pair_cells = 0;
    std::size_t jump_pair_cells = 0;
    std::size_t overlap_pair_cells = 0;
    std::size_t union_pair_cells = 0;
    friend bool operator==(const RtcTransientExclusionCounts &,
                           const RtcTransientExclusionCounts &) = default;
};

// One immutable plan for the two already selected exclusion operations.
// This is not a complete transient-treatment or RTC plan: isolated-spike
// admission/donors, optical admission and unresolved-contamination policy
// remain separate. Not excluded here never means scientifically admitted.
class RtcTransientExclusionPlan {
public:
    static std::shared_ptr<const RtcTransientExclusionPlan> consider(
        std::shared_ptr<const RtcSpikeLearningDecision> screening,
        std::shared_ptr<const RtcJumpExclusionPlan> jumps,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id) {
        if (!screening || !jumps || !snapshot || !id)
            throw std::invalid_argument("RTC transient plan requires both decisions, VAL and identity");
        const auto &spikes = screening->evidence_handle();
        if (spikes.get() != jumps->admission_handle()->assessment().spike_handle().get())
            throw std::invalid_argument("RTC transient decisions have different original Learn evidence");
        if (snapshot.get() != spikes->val_snapshot_handle().get() ||
            snapshot.get() != jumps->val_snapshot_handle().get())
            throw StaleRtcValGeneration("RTC transient plan requires the exact original VAL snapshot");
        auto out = std::shared_ptr<RtcTransientExclusionPlan>(new RtcTransientExclusionPlan);
        out->screening_ = std::move(screening); out->jumps_ = std::move(jumps); out->id_ = id;
        std::map<std::pair<TimestreamNetworkId, std::uint32_t>, RtcTransientDetectorExclusion> records;
        for (std::size_t i = 0; i < spikes->blocks().size(); ++i) {
            if (!out->screening_->requires_pair_exclusion_from_mapmaking(i)) continue;
            const auto &b = spikes->blocks()[i];
            auto &d = records[{b.network_id, b.detector_index}];
            d.network = b.network_id; d.detector = b.detector_index;
            d.screening.push_back({i, {b.first, b.past_last}});
            d.rows.push_back({b.first, b.past_last});
            out->counts_.screening_pair_cells += static_cast<std::size_t>(b.past_last - b.first);
        }
        for (const auto &j : out->jumps_->detectors()) {
            auto &d = records[{j.network, j.detector}];
            d.network = j.network; d.detector = j.detector;
            for (const auto &b : d.screening) for (auto r : j.rows)
                out->counts_.overlap_pair_cells += static_cast<std::size_t>(
                    std::max<TimestreamNativeRow>(0, std::min(b.rows.past_last, r.past_last) -
                                                  std::max(b.rows.first, r.first)));
            d.rows.insert(d.rows.end(), j.rows.begin(), j.rows.end());
        }
        out->counts_.jump_pair_cells = out->jumps_->excluded_pair_cells();
        for (auto &[key, d] : records) {
            std::sort(d.screening.begin(), d.screening.end(),
                [](const auto &a, const auto &b) { return a.rows.first < b.rows.first; });
            d.rows = rtc_event_assessment_detail::merge(std::move(d.rows));
            for (auto r : d.rows)
                out->counts_.union_pair_cells += static_cast<std::size_t>(r.past_last - r.first);
            out->detectors_.push_back(std::move(d));
        }
        return out;
    }

    const auto &screening_handle() const noexcept { return screening_; }
    const auto &jump_plan_handle() const noexcept { return jumps_; }
    const auto &input_handle() const noexcept { return screening_->evidence_handle()->input_handle(); }
    const auto &val_snapshot_handle() const noexcept { return screening_->evidence_handle()->val_snapshot_handle(); }
    const auto &detectors() const noexcept { return detectors_; }
    const auto &counts() const noexcept { return counts_; }
    std::uint64_t consideration() const noexcept { return id_; }

    RtcTransientExclusionCauses causes(TimestreamNetworkId network,
        TimestreamNativeRow row, std::uint32_t detector) const {
        RtcTransientExclusionCauses out;
        // This also checks the complete address against the exact parent.
        out.accepted_jump = jumps_->excludes(network, row, detector);
        if (const auto *d = find(network, detector)) {
            auto b = std::upper_bound(d->screening.begin(), d->screening.end(), row,
                [](auto r, const auto &v) { return r < v.rows.first; });
            if (b != d->screening.begin() && row < std::prev(b)->rows.past_last) {
                const auto &block = screening_->evidence_handle()->blocks()[std::prev(b)->original_block];
                out.screening = {block.coordinates[0].cause, block.coordinates[1].cause};
            }
        }
        return out;
    }
    bool excludes(TimestreamNetworkId network, TimestreamNativeRow row, std::uint32_t detector) const {
        return causes(network, row, detector).excluded();
    }
    std::size_t logical_owned_bytes() const noexcept {
        std::size_t n = detectors_.size() * sizeof(RtcTransientDetectorExclusion);
        for (const auto &d : detectors_)
            n += d.screening.size() * sizeof(RtcTransientScreeningBlock) + d.rows.size() * sizeof(RtcEventRange);
        return n;
    }
private:
    RtcTransientExclusionPlan() = default;
    const RtcTransientDetectorExclusion *find(TimestreamNetworkId network, std::uint32_t detector) const {
        const auto key = std::pair{network, detector};
        const auto d = std::lower_bound(detectors_.begin(), detectors_.end(), key,
            [](const auto &r, auto k) { return std::pair{r.network, r.detector} < k; });
        return d != detectors_.end() && std::pair{d->network, d->detector} == key ? &*d : nullptr;
    }
    std::shared_ptr<const RtcSpikeLearningDecision> screening_;
    std::shared_ptr<const RtcJumpExclusionPlan> jumps_;
    std::vector<RtcTransientDetectorExclusion> detectors_;
    RtcTransientExclusionCounts counts_;
    std::uint64_t id_ = 0;
};

// Apply realizes only this selected treatment, without learning, allocating a
// replacement plane, changing producer state, or claiming downstream admission.
class RtcTransientExclusionResult {
public:
    static std::shared_ptr<const RtcTransientExclusionResult> apply(
        std::shared_ptr<const RtcTransientExclusionPlan> plan,
        std::shared_ptr<const NativePairedReadoutView> input,
        std::shared_ptr<const ValSnapshot> snapshot,
        std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions) {
        if (!plan || !input || plan->input_handle().get() != input.get())
            throw std::invalid_argument("RTC transient Apply requires its exact plan-bound input");
        if (!snapshot || plan->val_snapshot_handle().get() != snapshot.get())
            throw StaleRtcValGeneration("RTC transient Apply requires its exact plan-bound VAL snapshot");
        require_exact_native_partition_schedule(*input, partitions);
        return std::shared_ptr<const RtcTransientExclusionResult>(new RtcTransientExclusionResult{std::move(plan)});
    }
    const auto &plan_handle() const noexcept { return plan_; }
    const auto &realized_counts() const noexcept { return plan_->counts(); }
    static constexpr std::size_t owned_numeric_bytes = 0;
    static constexpr std::size_t owned_state_plane_bytes = 0;
    std::optional<double> value_if_retained(NativeReadoutCoordinate coordinate,
        TimestreamNetworkId network, TimestreamNativeRow row, std::uint32_t detector) const {
        if (coordinate != NativeReadoutCoordinate::x && coordinate != NativeReadoutCoordinate::r)
            throw std::invalid_argument("RTC transient result requires x or r coordinate");
        if (plan_->excludes(network, row, detector)) return std::nullopt;
        const auto &n = plan_->input_handle()->network(network);
        if (!n.state(NativeReadoutCoordinate::x, row, detector).valid() ||
            !n.state(NativeReadoutCoordinate::r, row, detector).valid()) return std::nullopt;
        return n.value(coordinate, row, detector);
    }
private:
    explicit RtcTransientExclusionResult(std::shared_ptr<const RtcTransientExclusionPlan> plan)
        : plan_{std::move(plan)} {}
    std::shared_ptr<const RtcTransientExclusionPlan> plan_;
};

} // namespace citlali::pipeline
