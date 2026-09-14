#pragma once

#include <citlali/core/pipeline/timestream_rtc_jump_admission.h>

namespace citlali::pipeline {

// Supplied by the existing requested processing-scan owner. These are exact
// native support intervals for that generation, not acquisition ScanNum,
// output-row indices, fitting windows or a newly chosen scan partition.
struct RtcExistingScanNativeSupport {
    std::uint64_t scan = 0;
    NativeOccurrenceSpan native;
};

enum class RtcExistingScanSupportState : std::uint8_t {
    unavailable, conservative_native_support_bound
};

class RtcJumpScanBindingUnavailable : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

class RtcExistingScanBinding {
public:
    static std::shared_ptr<const RtcExistingScanBinding> admit(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        std::string processing_generation, std::string native_relation_authority,
        std::string timing_uncertainty_authority,
        RtcExistingScanSupportState state,
        std::vector<RtcExistingScanNativeSupport> complete_support) {
        // This explicit producer assertion includes the relation's timing
        // uncertainty. An authority name alone is not usable support. The
        // binding does not estimate uncertainty or reconstruct omitted rows.
        if (state != RtcExistingScanSupportState::conservative_native_support_bound)
            throw RtcJumpScanBindingUnavailable("RTC existing scan conservative native support is unavailable");
        if (!parent || processing_generation.empty() || native_relation_authority.empty() ||
            timing_uncertainty_authority.empty() || complete_support.empty())
            throw std::invalid_argument("RTC existing scans require complete native support and explicit generation/relation/uncertainty authority");
        std::sort(complete_support.begin(), complete_support.end(), [](const auto &a, const auto &b) {
            return std::tie(a.native.network_id, a.scan, a.native.first_native_row) <
                   std::tie(b.native.network_id, b.scan, b.native.first_native_row);
        });
        const RtcExistingScanNativeSupport *previous = nullptr;
        for (const auto &s : complete_support) {
            const auto &axis = parent->network(s.native.network_id).occurrence_axis();
            if (s.native.first_native_row < axis.first_native_row() ||
                s.native.past_last_native_row > axis.past_last_native_row() ||
                s.native.first_native_row >= s.native.past_last_native_row ||
                (previous && previous->native.network_id == s.native.network_id && previous->scan == s.scan &&
                 previous->native.past_last_native_row > s.native.first_native_row))
                throw std::invalid_argument("RTC existing scan native support is invalid or repeated");
            previous = &s;
        }
        return std::shared_ptr<const RtcExistingScanBinding>(new RtcExistingScanBinding{
            std::move(parent), std::move(processing_generation), std::move(native_relation_authority),
            std::move(timing_uncertainty_authority), std::move(complete_support)});
    }
    const auto &parent_handle() const noexcept { return parent_; }
    const auto &processing_generation() const noexcept { return generation_; }
    const auto &native_relation_authority() const noexcept { return relation_; }
    const auto &timing_uncertainty_authority() const noexcept { return uncertainty_; }
    const auto &supports() const noexcept { return supports_; }
private:
    RtcExistingScanBinding(std::shared_ptr<const NativePairedReadoutObservation> parent,
        std::string generation, std::string relation, std::string uncertainty,
        std::vector<RtcExistingScanNativeSupport> supports)
        : parent_{std::move(parent)}, generation_{std::move(generation)}, relation_{std::move(relation)},
          uncertainty_{std::move(uncertainty)}, supports_{std::move(supports)} {}
    std::shared_ptr<const NativePairedReadoutObservation> parent_;
    std::string generation_, relation_, uncertainty_;
    std::vector<RtcExistingScanNativeSupport> supports_;
};

struct RtcJumpDetectorExclusion {
    TimestreamNetworkId network = -1;
    std::uint32_t detector = 0;
    bool whole_observation = false;
    std::vector<std::size_t> original_groups;
    std::vector<std::uint64_t> scans;
    // Union of selected treatment intervals, not physical transition extent.
    std::vector<RtcEventRange> rows;
};

// Complete plan for the selected jump-exclusion operation. Other RTC
// operations and PTC/VAL named-use policies are not claimed complete here.
class RtcJumpExclusionPlan {
public:
    static std::shared_ptr<const RtcJumpExclusionPlan> consider(
        std::shared_ptr<const RtcJumpAdmissionDecision> admission,
        std::shared_ptr<const RtcExistingScanBinding> scans,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id) {
        if (!admission || !snapshot || !id || admission->val_snapshot_handle().get() != snapshot.get())
            throw std::invalid_argument("RTC jump plan requires admission, exact original VAL and identity");
        if (scans && scans->parent_handle().get() != admission->input_handle()->parent_handle().get())
            throw std::invalid_argument("RTC jump scan binding has a different native parent");
        auto result = std::shared_ptr<RtcJumpExclusionPlan>(new RtcJumpExclusionPlan);
        result->admission_ = std::move(admission); result->scans_ = std::move(scans); result->id_ = id;
        std::map<std::pair<TimestreamNetworkId, std::uint32_t>, std::vector<std::size_t>> groups;
        for (const auto &group : result->admission_->groups()) if (group.admitted())
            groups[{group.network, group.detector}].push_back(group.original_group);
        if (!groups.empty() && !result->scans_)
            throw RtcJumpScanBindingUnavailable("RTC jump treatment needs the exact existing scan/native relation");
        for (const auto &[key, selected] : groups) {
            RtcJumpDetectorExclusion out;
            out.network = key.first; out.detector = key.second; out.original_groups = selected;
            out.whole_observation = selected.size() >= RtcJumpAdmissionPolicy::observation_group_count;
            std::set<std::uint64_t> scan_ids;
            for (auto group_index : selected) {
                const auto &group = result->admission_->groups().at(group_index);
                for (const auto &coordinate : group.coordinates) if (coordinate.admitted()) {
                    bool bound = false;
                    for (const auto &s : result->scans_->supports()) {
                        if (s.native.network_id != out.network) continue;
                        if (coordinate.affected.first < s.native.past_last_native_row &&
                            s.native.first_native_row < coordinate.affected.past_last) {
                            scan_ids.insert(s.scan); bound = true;
                        }
                    }
                    if (!bound) throw RtcJumpScanBindingUnavailable("RTC admitted transition has no existing scan support");
                }
            }
            if (out.whole_observation) {
                const auto &axis = result->input_handle()->network(out.network).occurrence_axis();
                out.rows.push_back({axis.first_native_row(), axis.past_last_native_row()});
                for (const auto &s : result->scans_->supports()) if (s.native.network_id == out.network) scan_ids.insert(s.scan);
            } else {
                // Once a scan is selected, include every native interval of
                // that same scan, including disjoint support across gaps.
                for (const auto &s : result->scans_->supports())
                    if (s.native.network_id == out.network && scan_ids.contains(s.scan))
                        out.rows.push_back({s.native.first_native_row, s.native.past_last_native_row});
                out.rows = rtc_event_assessment_detail::merge(std::move(out.rows));
            }
            out.scans.assign(scan_ids.begin(), scan_ids.end());
            for (auto r : out.rows) result->excluded_cells_ += static_cast<std::size_t>(r.past_last - r.first);
            result->detectors_.push_back(std::move(out));
        }
        return result;
    }
    const auto &admission_handle() const noexcept { return admission_; }
    const auto &scan_binding_handle() const noexcept { return scans_; }
    const std::shared_ptr<const NativePairedReadoutView> &input_handle() const { return admission_->input_handle(); }
    const auto &val_snapshot_handle() const { return admission_->val_snapshot_handle(); }
    const auto &detectors() const noexcept { return detectors_; }
    std::uint64_t consideration() const noexcept { return id_; }
    std::size_t excluded_pair_cells() const noexcept { return excluded_cells_; }
    std::size_t logical_owned_bytes() const noexcept {
        std::size_t n = detectors_.size() * sizeof(RtcJumpDetectorExclusion);
        for (const auto &d : detectors_) n += d.original_groups.size() * sizeof(std::size_t) +
            d.scans.size() * sizeof(std::uint64_t) + d.rows.size() * sizeof(RtcEventRange);
        return n;
    }
    bool excludes(TimestreamNetworkId network, TimestreamNativeRow row, std::uint32_t detector) const {
        const auto &n = input_handle()->network(network);
        const auto span = input_handle()->span(network);
        if (detector >= static_cast<std::size_t>(n.detector_count()) ||
            row < span.first_native_row || row >= span.past_last_native_row)
            throw std::out_of_range("RTC jump exclusion address outside exact input");
        const auto key = std::pair{network, detector};
        const auto found = std::lower_bound(detectors_.begin(), detectors_.end(), key,
            [](const auto &d, auto k) { return std::pair{d.network, d.detector} < k; });
        return found != detectors_.end() && std::pair{found->network, found->detector} == key &&
            rtc_event_assessment_detail::contains(found->rows, row);
    }
private:
    RtcJumpExclusionPlan() = default;
    std::shared_ptr<const RtcJumpAdmissionDecision> admission_;
    std::shared_ptr<const RtcExistingScanBinding> scans_;
    std::vector<RtcJumpDetectorExclusion> detectors_;
    std::uint64_t id_ = 0;
    std::size_t excluded_cells_ = 0;
};

// Apply's zero-copy paired exclusion product. Original raw values and causes
// remain reachable for provenance. Mask-aware consumption cannot read a jump-
// excluded payload. Not excluded by this operation is not downstream admission.
class RtcJumpExclusionResult {
public:
    static std::shared_ptr<const RtcJumpExclusionResult> apply(
        std::shared_ptr<const RtcJumpExclusionPlan> plan,
        std::shared_ptr<const NativePairedReadoutView> input,
        std::shared_ptr<const ValSnapshot> snapshot,
        std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions) {
        if (!plan || !input || plan->input_handle().get() != input.get())
            throw std::invalid_argument("RTC jump Apply requires exact plan-bound input");
        if (!snapshot || plan->val_snapshot_handle().get() != snapshot.get())
            throw StaleRtcValGeneration("RTC jump Apply requires the plan-bound VAL snapshot");
        require_exact_native_partition_schedule(*input, partitions);
        return std::shared_ptr<const RtcJumpExclusionResult>(new RtcJumpExclusionResult{std::move(plan)});
    }
    const auto &plan_handle() const noexcept { return plan_; }
    std::size_t realized_excluded_pair_cells() const noexcept { return plan_->excluded_pair_cells(); }
    static constexpr std::size_t owned_numeric_bytes = 0;
    static constexpr std::size_t owned_state_plane_bytes = 0;
    bool excluded(TimestreamNetworkId network, TimestreamNativeRow row, std::uint32_t detector) const {
        return plan_->excludes(network, row, detector);
    }
    std::optional<double> value_if_retained(NativeReadoutCoordinate coordinate,
        TimestreamNetworkId network, TimestreamNativeRow row, std::uint32_t detector) const {
        if (coordinate != NativeReadoutCoordinate::x && coordinate != NativeReadoutCoordinate::r)
            throw std::invalid_argument("RTC jump result requires x or r coordinate");
        if (excluded(network, row, detector)) return std::nullopt;
        const auto &n = plan_->input_handle()->network(network);
        if (!n.state(NativeReadoutCoordinate::x, row, detector).valid() ||
            !n.state(NativeReadoutCoordinate::r, row, detector).valid()) return std::nullopt;
        return n.value(coordinate, row, detector);
    }
private:
    explicit RtcJumpExclusionResult(std::shared_ptr<const RtcJumpExclusionPlan> plan) : plan_{std::move(plan)} {}
    std::shared_ptr<const RtcJumpExclusionPlan> plan_;
};

} // namespace citlali::pipeline
