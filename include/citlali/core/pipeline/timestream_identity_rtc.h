#pragma once

#include <citlali/core/pipeline/timestream_val_state.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace citlali::pipeline {

// Engineering partitions retain absolute native occurrence identity. A span
// is a bounded view over one network's immutable Paired-D1 axis; it is not a
// common-grid association or a new scientific occurrence identity.
struct NativeOccurrenceSpan {
    TimestreamNetworkId network_id = -1;
    TimestreamNativeRow first_native_row = -1;
    TimestreamNativeRow past_last_native_row = -1;

    std::size_t occurrence_count() const noexcept {
        if (first_native_row < 0 ||
            past_last_native_row <= first_native_row) {
            return 0;
        }
        return static_cast<std::size_t>(
            past_last_native_row - first_native_row);
    }

    friend bool operator==(const NativeOccurrenceSpan &,
                           const NativeOccurrenceSpan &) = default;
};

inline std::vector<NativeOccurrenceSpan> full_native_occurrence_spans(
    const NativePairedReadoutObservation &input) {
    std::vector<NativeOccurrenceSpan> spans;
    spans.reserve(input.network_count());
    for (const auto network_id : input.participant_network_ids()) {
        const auto &network = input.network(network_id);
        const auto &axis = *network.occurrence_axis_handle();
        spans.push_back({network.network_id(), axis.first_native_row(),
                         axis.past_last_native_row()});
    }
    return spans;
}

// A lightweight immutable view binds one engineering partition to its parent.
// It owns only one native row interval per participant network. Axes, timing,
// detector identity, values, validity, causes, and support remain parent-owned.
class NativePairedReadoutView {
public:
    static std::shared_ptr<const NativePairedReadoutView> admit(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        std::vector<NativeOccurrenceSpan> spans) {
        if (!parent || spans.size() != parent->network_count()) {
            throw std::invalid_argument(
                "native RTC view requires one span per participant network");
        }
        std::sort(spans.begin(), spans.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.network_id < rhs.network_id;
                  });

        std::size_t detector_count = 0;
        std::size_t native_occurrence_count = 0;
        std::size_t detector_occurrence_count = 0;
        const auto &parent_network_ids = parent->participant_network_ids();
        for (std::size_t index = 0; index < spans.size(); ++index) {
            const auto &span = spans[index];
            const auto &network = parent->network(parent_network_ids[index]);
            const auto &axis = *network.occurrence_axis_handle();
            if (span.network_id != network.network_id() ||
                span.first_native_row < axis.first_native_row() ||
                span.first_native_row >= span.past_last_native_row ||
                span.past_last_native_row > axis.past_last_native_row()) {
                throw std::invalid_argument(
                    "native RTC view span is incomplete or outside parent support");
            }
            const auto occurrences = span.occurrence_count();
            const auto detectors =
                static_cast<std::size_t>(network.detector_count());
            if (occurrences >
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max()) + 1ULL) {
                throw std::length_error(
                    "native RTC view span exceeds compact evidence range");
            }
            if (detectors >
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max()) + 1ULL) {
                throw std::length_error(
                    "native RTC detector axis exceeds compact evidence range");
            }
            if (detector_count >
                    std::numeric_limits<std::size_t>::max() - detectors ||
                native_occurrence_count >
                    std::numeric_limits<std::size_t>::max() - occurrences ||
                (detectors != 0 &&
                 occurrences >
                     std::numeric_limits<std::size_t>::max() / detectors) ||
                detector_occurrence_count >
                    std::numeric_limits<std::size_t>::max() -
                        occurrences * detectors) {
                throw std::length_error(
                    "native RTC view cardinality would overflow");
            }
            detector_count += detectors;
            native_occurrence_count += occurrences;
            detector_occurrence_count += occurrences * detectors;
        }

        return std::shared_ptr<const NativePairedReadoutView>(
            new NativePairedReadoutView{
                std::move(parent), std::move(spans), detector_count,
                native_occurrence_count, detector_occurrence_count});
    }

    static std::shared_ptr<const NativePairedReadoutView> full(
        std::shared_ptr<const NativePairedReadoutObservation> parent) {
        if (!parent) {
            throw std::invalid_argument(
                "native RTC full view requires a parent");
        }
        auto spans = full_native_occurrence_spans(*parent);
        return admit(std::move(parent), std::move(spans));
    }

    const std::shared_ptr<const NativePairedReadoutObservation> &
    parent_handle() const noexcept {
        return parent_;
    }
    const NativeObservationScope &scope() const noexcept {
        return parent_->scope();
    }
    std::span<const NativeOccurrenceSpan> spans() const noexcept {
        return spans_;
    }
    const NativeOccurrenceSpan &span(
        TimestreamNetworkId network_id) const {
        const auto found = std::lower_bound(
            spans_.begin(), spans_.end(), network_id,
            [](const auto &candidate, TimestreamNetworkId id) {
                return candidate.network_id < id;
            });
        if (found == spans_.end() || found->network_id != network_id) {
            throw std::out_of_range(
                "network is absent from native RTC view");
        }
        return *found;
    }
    const NativePairedReadoutNetwork &network(
        TimestreamNetworkId network_id) const {
        return parent_->network(network_id);
    }
    std::size_t network_count() const noexcept { return spans_.size(); }
    std::size_t detector_count() const noexcept { return detector_count_; }
    std::size_t native_occurrence_count() const noexcept {
        return native_occurrence_count_;
    }
    std::size_t detector_occurrence_count() const noexcept {
        return detector_occurrence_count_;
    }

private:
    NativePairedReadoutView(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        std::vector<NativeOccurrenceSpan> spans,
        std::size_t detector_count,
        std::size_t native_occurrence_count,
        std::size_t detector_occurrence_count)
        : parent_{std::move(parent)}, spans_{std::move(spans)},
          detector_count_{detector_count},
          native_occurrence_count_{native_occurrence_count},
          detector_occurrence_count_{detector_occurrence_count} {}

    std::shared_ptr<const NativePairedReadoutObservation> parent_;
    std::vector<NativeOccurrenceSpan> spans_;
    std::size_t detector_count_ = 0;
    std::size_t native_occurrence_count_ = 0;
    std::size_t detector_occurrence_count_ = 0;
};

class IncompleteNativePartitionSchedule : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

inline void require_exact_native_partition_schedule(
    const NativePairedReadoutView &logical_input,
    std::span<const std::shared_ptr<const NativePairedReadoutView>>
        partitions) {
    if (partitions.empty()) {
        throw IncompleteNativePartitionSchedule(
            "native RTC requires at least one engineering partition");
    }
    for (const auto &partition : partitions) {
        if (!partition ||
            partition->parent_handle().get() !=
                logical_input.parent_handle().get() ||
            !(partition->scope() == logical_input.scope()) ||
            partition->network_count() != logical_input.network_count()) {
            throw IncompleteNativePartitionSchedule(
                "native RTC partition is not bound to the logical input");
        }
    }
    for (const auto &logical_span : logical_input.spans()) {
        auto next_native_row = logical_span.first_native_row;
        for (const auto &partition : partitions) {
            const auto &partition_span =
                partition->span(logical_span.network_id);
            if (partition_span.first_native_row != next_native_row ||
                partition_span.past_last_native_row >
                    logical_span.past_last_native_row) {
                throw IncompleteNativePartitionSchedule(
                    "engineering partitions do not exactly cover declared logical support");
            }
            next_native_row = partition_span.past_last_native_row;
        }
        if (next_native_row != logical_span.past_last_native_row) {
            throw IncompleteNativePartitionSchedule(
                "engineering partitions do not exactly cover declared logical support");
        }
    }
}

struct RtcNativeCellIdentity {
    TimestreamNetworkId network_id = -1;
    TimestreamNativeRow native_row = -1;
    Eigen::Index storage_column = -1;
    std::int64_t parent_readout_occurrence_key = -1;
    std::int64_t paired_xr_occurrence_key = -1;
    std::string_view detector_occurrence_id;
    std::string_view detector_association_record_id;
    std::string_view tone_or_channel_id;
    std::string_view mapping_record_id;
    std::string_view mapping_revision_id;

    friend bool operator==(const RtcNativeCellIdentity &,
                           const RtcNativeCellIdentity &) = default;
};

inline RtcNativeCellIdentity rtc_native_cell_identity(
    const NativePairedReadoutView &input,
    TimestreamNetworkId network_id,
    TimestreamNativeRow native_row,
    Eigen::Index detector_index) {
    const auto &network = input.network(network_id);
    const auto &detector = network.detector(detector_index);
    const auto &occurrence =
        network.occurrence_axis().occurrence(native_row);
    const auto &mapping = network.mapping_authority();
    return {network_id,
            native_row,
            detector.storage_column,
            occurrence.parent_readout_occurrence_key,
            occurrence.paired_xr_occurrence_key,
            detector.detector_occurrence_id,
            detector.detector_association_record_id,
            detector.tone_or_channel_id,
            mapping.mapping_record_id,
            mapping.mapping_revision_id};
}

enum class RtcEvidenceOrigin : std::uint8_t {
    none = 0,
    x = 1U << 0,
    r = 1U << 1,
    x_and_r = (1U << 0) | (1U << 1),
};

constexpr RtcEvidenceOrigin operator|(RtcEvidenceOrigin lhs,
                                      RtcEvidenceOrigin rhs) noexcept {
    return static_cast<RtcEvidenceOrigin>(
        static_cast<std::uint8_t>(lhs) |
        static_cast<std::uint8_t>(rhs));
}

constexpr bool has_origin(RtcEvidenceOrigin value,
                          RtcEvidenceOrigin origin) noexcept {
    return (static_cast<std::uint8_t>(value) &
            static_cast<std::uint8_t>(origin)) != 0;
}

enum class RtcEvidenceClass : std::uint8_t {
    input_admission,
};

// A compact parent-local handle. The occurrence offset is relative to the
// named network span; absolute scientific identity is resolved from the view.
struct RtcEvidenceCellHandle {
    TimestreamNetworkId network_id = -1;
    std::uint32_t native_occurrence_offset = 0;
    std::uint32_t detector_index = 0;

    friend bool operator==(const RtcEvidenceCellHandle &,
                           const RtcEvidenceCellHandle &) = default;
    friend bool operator<(const RtcEvidenceCellHandle &lhs,
                          const RtcEvidenceCellHandle &rhs) noexcept {
        if (lhs.network_id != rhs.network_id) {
            return lhs.network_id < rhs.network_id;
        }
        if (lhs.native_occurrence_offset != rhs.native_occurrence_offset) {
            return lhs.native_occurrence_offset <
                   rhs.native_occurrence_offset;
        }
        return lhs.detector_index < rhs.detector_index;
    }
};

// A sparse derived record.  Only the event classification and its bounded
// parent locator are owned here.
struct RtcEvidenceEvent {
    RtcEvidenceCellHandle cell;
    RtcEvidenceOrigin origin = RtcEvidenceOrigin::none;
    RtcEvidenceClass evidence_class = RtcEvidenceClass::input_admission;

    bool direct_x() const noexcept {
        return has_origin(origin, RtcEvidenceOrigin::x);
    }
    bool direct_r() const noexcept {
        return has_origin(origin, RtcEvidenceOrigin::r);
    }
    friend bool operator==(const RtcEvidenceEvent &,
                           const RtcEvidenceEvent &) = default;
};

static_assert(sizeof(RtcEvidenceEvent) <= 16,
              "RTC evidence event must remain a compact parent handle");

struct RtcEvidenceIdentity {
    std::uint64_t attempt = 0;

    friend bool operator==(const RtcEvidenceIdentity &,
                           const RtcEvidenceIdentity &) = default;
};

struct RtcEvidenceSummary {
    std::size_t examined_cell_count = 0;
    std::size_t accepted_event_count = 0;
    std::size_t direct_x_event_count = 0;
    std::size_t direct_r_event_count = 0;
    std::size_t x_and_r_event_count = 0;
};

struct RtcEvidenceMemoryEvidence {
    std::size_t derived_event_bytes = 0;
    std::size_t referenced_parent_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return derived_event_bytes;
    }
};

class RtcEvidence {
public:
    const RtcEvidenceIdentity &identity() const noexcept { return identity_; }
    const NativeObservationScope &scope() const noexcept {
        return input_->scope();
    }
    const std::shared_ptr<const NativePairedReadoutView> &input_handle()
        const noexcept {
        return input_;
    }
    const std::shared_ptr<const ValSnapshot> &val_snapshot_handle()
        const noexcept {
        return val_snapshot_;
    }
    ValGeneration val_generation() const noexcept {
        return val_snapshot_->generation();
    }
    // Identity RTC manufactures no VAL fact. A substantive RTC producer may
    // expose producer-owned proposals from its own typed evidence contract.
    std::span<const ValFinding> proposed_val_findings() const noexcept {
        return {};
    }
    std::span<const RtcEvidenceEvent> events() const noexcept {
        return events_;
    }
    const RtcEvidenceSummary &summary() const noexcept { return summary_; }
    RtcEvidenceMemoryEvidence memory_evidence() const noexcept {
        return {events_.size() * sizeof(RtcEvidenceEvent), 1};
    }
    std::optional<std::size_t> find_index(
        TimestreamNetworkId network_id, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const noexcept {
        if (detector_index < 0 ||
            static_cast<std::uint64_t>(detector_index) >
                std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        const NativeOccurrenceSpan *network_span = nullptr;
        try {
            network_span = &input_->span(network_id);
        } catch (const std::out_of_range &) {
            return std::nullopt;
        }
        if (native_row < network_span->first_native_row ||
            native_row >= network_span->past_last_native_row) {
            return std::nullopt;
        }
        const auto offset = static_cast<std::uint64_t>(
            native_row - network_span->first_native_row);
        if (offset > std::numeric_limits<std::uint32_t>::max()) {
            return std::nullopt;
        }
        const RtcEvidenceCellHandle cell{
            network_id, static_cast<std::uint32_t>(offset),
            static_cast<std::uint32_t>(detector_index)};
        const auto found = std::lower_bound(
            events_.begin(), events_.end(), cell,
            [](const RtcEvidenceEvent &event,
               const RtcEvidenceCellHandle &candidate) {
                return event.cell < candidate;
            });
        if (found == events_.end() || !(found->cell == cell)) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(found - events_.begin());
    }
    const RtcEvidenceEvent *find(
        TimestreamNetworkId network_id, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const noexcept {
        const auto index = find_index(network_id, native_row,
                                      detector_index);
        return index ? &events_[*index] : nullptr;
    }
    TimestreamNativeRow native_row(const RtcEvidenceEvent &event) const {
        const auto &network_span = input_->span(event.cell.network_id);
        const auto row = network_span.first_native_row +
            static_cast<TimestreamNativeRow>(
                event.cell.native_occurrence_offset);
        if (row >= network_span.past_last_native_row) {
            throw std::logic_error(
                "RTC evidence handle exceeds its native parent span");
        }
        return row;
    }
    RtcNativeCellIdentity scientific_identity(
        const RtcEvidenceEvent &event) const {
        return rtc_native_cell_identity(
            *input_, event.cell.network_id, native_row(event),
            static_cast<Eigen::Index>(event.cell.detector_index));
    }
    NativePairedReadoutCause pair_local_causes(
        const RtcEvidenceEvent &event) const {
        return input_->network(event.cell.network_id)
            .pair_causes(native_row(event),
                         static_cast<Eigen::Index>(
                             event.cell.detector_index));
    }

private:
    friend std::shared_ptr<const RtcEvidence> learn_identity_rtc(
        std::shared_ptr<const NativePairedReadoutView>,
        std::shared_ptr<const ValSnapshot>, std::uint64_t);
    friend std::shared_ptr<const RtcEvidence>
    learn_identity_rtc_partitioned(
        std::shared_ptr<const NativePairedReadoutView>,
        std::span<const std::shared_ptr<const NativePairedReadoutView>>,
        std::shared_ptr<const ValSnapshot>, std::uint64_t);

    RtcEvidence(RtcEvidenceIdentity identity,
                std::shared_ptr<const NativePairedReadoutView> input,
                std::shared_ptr<const ValSnapshot> val_snapshot,
                std::vector<RtcEvidenceEvent> events,
                RtcEvidenceSummary summary)
        : identity_{identity}, input_{std::move(input)},
          val_snapshot_{std::move(val_snapshot)},
          events_{std::move(events)}, summary_{summary} {}

    RtcEvidenceIdentity identity_;
    std::shared_ptr<const NativePairedReadoutView> input_;
    std::shared_ptr<const ValSnapshot> val_snapshot_;
    std::vector<RtcEvidenceEvent> events_;
    RtcEvidenceSummary summary_;
};

inline std::shared_ptr<const RtcEvidence> learn_identity_rtc_partitioned(
    std::shared_ptr<const NativePairedReadoutView> logical_input,
    std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions,
    std::shared_ptr<const ValSnapshot> val_snapshot,
    std::uint64_t attempt) {
    if (!logical_input || !val_snapshot || attempt == 0 ||
        val_snapshot->paired_handle().get() !=
            logical_input->parent_handle().get() ||
        !(val_snapshot->scope() == logical_input->scope())) {
        throw std::invalid_argument(
            "identity RTC learning requires exact input, VAL snapshot, and attempt identity");
    }
    require_exact_native_partition_schedule(*logical_input, partitions);

    RtcEvidenceSummary summary;
    summary.examined_cell_count =
        logical_input->detector_occurrence_count();
    std::vector<RtcEvidenceEvent> events;
    for (const auto &logical_span : logical_input->spans()) {
        const auto &network =
            logical_input->network(logical_span.network_id);
        for (const auto &partition : partitions) {
            const auto &partition_span =
                partition->span(logical_span.network_id);
            for (auto row = partition_span.first_native_row;
                 row < partition_span.past_last_native_row; ++row) {
                const auto offset = static_cast<std::uint32_t>(
                    row - logical_span.first_native_row);
                for (Eigen::Index detector = 0;
                     detector < network.detector_count(); ++detector) {
                    const auto &x_state = network.state(
                        NativeReadoutCoordinate::x, row, detector);
                    const auto &r_state = network.state(
                        NativeReadoutCoordinate::r, row, detector);
                    const bool x_event = !x_state.valid();
                    const bool r_event = !r_state.valid();
                    if (!x_event && !r_event) continue;

                    RtcEvidenceOrigin origin = RtcEvidenceOrigin::none;
                    if (x_event) origin = origin | RtcEvidenceOrigin::x;
                    if (r_event) origin = origin | RtcEvidenceOrigin::r;
                    events.push_back(RtcEvidenceEvent{
                        {logical_span.network_id, offset,
                         static_cast<std::uint32_t>(detector)},
                        origin, RtcEvidenceClass::input_admission});
                    if (x_event) ++summary.direct_x_event_count;
                    if (r_event) ++summary.direct_r_event_count;
                    if (x_event && r_event) {
                        ++summary.x_and_r_event_count;
                    }
                }
            }
        }
    }
    summary.accepted_event_count = events.size();

    return std::shared_ptr<const RtcEvidence>(new RtcEvidence{
        RtcEvidenceIdentity{attempt}, std::move(logical_input),
        std::move(val_snapshot), std::move(events), summary});
}

inline std::shared_ptr<const RtcEvidence> learn_identity_rtc(
    std::shared_ptr<const NativePairedReadoutView> input,
    std::shared_ptr<const ValSnapshot> val_snapshot,
    std::uint64_t attempt) {
    const std::vector<std::shared_ptr<const NativePairedReadoutView>>
        partitions{input};
    return learn_identity_rtc_partitioned(
        std::move(input), partitions, std::move(val_snapshot), attempt);
}

enum class RtcPairDecision : std::uint8_t {
    eligible,
    ineligible,
};

struct RtcPlanIdentity {
    RtcEvidenceIdentity evidence;
    std::uint64_t consideration = 0;

    friend bool operator==(const RtcPlanIdentity &,
                           const RtcPlanIdentity &) = default;
};

struct RtcIdentityOperator {
    std::size_t sampling_factor = 1;
    std::size_t sampling_phase = 0;
    double x_from_x = 1.0;
    double x_from_r = 0.0;
    double r_from_x = 0.0;
    double r_from_r = 1.0;

    friend bool operator==(const RtcIdentityOperator &,
                           const RtcIdentityOperator &) = default;
};

enum class RtcPairPolicy : std::uint8_t {
    conservative_pair_wide,
};

// This is the identity operator's own VAL dependency contract. It makes no
// statement about another RTC operator: every different immutable generation
// invalidates identity evidence/plan use until the lifecycle is repeated.
enum class IdentityRtcValChangePolicy : std::uint8_t {
    exact_generation_requires_relearn,
};

struct RtcPlanMemoryEvidence {
    std::size_t derived_plan_bytes = 0;
    std::size_t referenced_evidence_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return derived_plan_bytes;
    }
};

class RtcPlan {
public:
    const RtcPlanIdentity &identity() const noexcept { return identity_; }
    const std::shared_ptr<const RtcEvidence> &evidence_handle()
        const noexcept {
        return evidence_;
    }
    const std::shared_ptr<const NativePairedReadoutView> &input_handle()
        const noexcept {
        return evidence_->input_handle();
    }
    const RtcIdentityOperator &operator_spec() const noexcept {
        return operator_;
    }
    const std::shared_ptr<const ValSnapshot> &val_snapshot_handle()
        const noexcept {
        return evidence_->val_snapshot_handle();
    }
    ValGeneration required_val_generation() const noexcept {
        return evidence_->val_generation();
    }
    IdentityRtcValChangePolicy val_change_policy() const noexcept {
        return IdentityRtcValChangePolicy::
            exact_generation_requires_relearn;
    }
    RtcPairPolicy pair_policy() const noexcept {
        return pair_policy_;
    }
    RtcPlanMemoryEvidence memory_evidence() const noexcept {
        return {0, 1};
    }
    RtcPairDecision decision(
        TimestreamNetworkId network_id, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        require_cell(network_id, native_row, detector_index);
        return evidence_->find_index(network_id, native_row, detector_index)
                   ? RtcPairDecision::ineligible
                   : RtcPairDecision::eligible;
    }
    const RtcEvidenceEvent *causal_evidence(
        TimestreamNetworkId network_id, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        return decision(network_id, native_row, detector_index) ==
                       RtcPairDecision::ineligible
                   ? evidence_->find(network_id, native_row,
                                     detector_index)
                   : nullptr;
    }

private:
    friend std::shared_ptr<const RtcPlan> consider_identity_rtc(
        std::shared_ptr<const RtcEvidence>,
        std::shared_ptr<const ValSnapshot>, std::uint64_t);

    RtcPlan(RtcPlanIdentity identity,
            std::shared_ptr<const RtcEvidence> evidence)
        : identity_{identity}, evidence_{std::move(evidence)} {}

    void require_cell(
        TimestreamNetworkId network_id, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        const auto &network_span = input_handle()->span(network_id);
        if (native_row < network_span.first_native_row ||
            native_row >= network_span.past_last_native_row) {
            throw std::out_of_range(
                "native row is outside RTC plan support");
        }
        (void)input_handle()->network(network_id).detector(detector_index);
    }

    RtcPlanIdentity identity_;
    std::shared_ptr<const RtcEvidence> evidence_;
    RtcIdentityOperator operator_;
    RtcPairPolicy pair_policy_ = RtcPairPolicy::conservative_pair_wide;
};

inline std::shared_ptr<const RtcPlan> consider_identity_rtc(
    std::shared_ptr<const RtcEvidence> evidence,
    std::shared_ptr<const ValSnapshot> val_snapshot,
    std::uint64_t consideration) {
    if (!evidence || !val_snapshot || consideration == 0 ||
        evidence->val_snapshot_handle().get() != val_snapshot.get()) {
        throw std::invalid_argument(
            "identity RTC consideration requires evidence, its exact VAL snapshot, and a consideration identity");
    }
    // The identity witness has one conservative policy: every accepted
    // input-admission event makes the paired occurrence ineligible. The
    // sparse evidence retains whether x, r, or both supplied the cause; the
    // plan does not duplicate one identical action per event.
    return std::shared_ptr<const RtcPlan>(new RtcPlan{
        RtcPlanIdentity{evidence->identity(), consideration},
        std::move(evidence)});
}

struct RtcTimestreamMemoryEvidence {
    std::size_t owned_numeric_bytes = 0;
    std::size_t owned_state_plane_bytes = 0;
    std::size_t referenced_parent_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return owned_numeric_bytes + owned_state_plane_bytes;
    }
};

struct RtcApplyResult;

// Identity RTC is a native-axis view product. It owns neither a duplicate TOD
// plane nor a duplicate support plane; the immutable parent owns every native
// occurrence fact and the plan references sparse evidence plus one considered
// pair policy.
class RtcTimestream {
public:
    const std::shared_ptr<const RtcPlan> &plan_handle() const noexcept {
        return plan_;
    }
    const std::shared_ptr<const NativePairedReadoutView> &input_handle()
        const noexcept {
        return input_;
    }
    const std::shared_ptr<const NativePairedReadoutObservation> &
    native_parent_handle() const noexcept {
        return input_->parent_handle();
    }
    const std::shared_ptr<const ValSnapshot> &val_snapshot_handle()
        const noexcept {
        return plan_->val_snapshot_handle();
    }
    ValGeneration val_generation() const noexcept {
        return plan_->required_val_generation();
    }
    // Identity apply produces no post-apply VAL proposal.
    std::span<const ValFinding> proposed_val_findings() const noexcept {
        return {};
    }
    const NativeObservationScope &scope() const noexcept {
        return input_->scope();
    }
    std::span<const NativeOccurrenceSpan> network_spans() const noexcept {
        return input_->spans();
    }
    std::size_t output_native_occurrence_count() const noexcept {
        return input_->native_occurrence_count();
    }
    std::size_t output_cell_count() const noexcept {
        return input_->detector_occurrence_count();
    }
    const RtcIdentityOperator &realized_operator() const noexcept {
        return plan_->operator_spec();
    }
    double output_time_unix_sec(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const {
        require_occurrence(network_id, native_row);
        return input_->network(network_id)
            .occurrence_axis_handle()->native_identity(native_row)
            .reconstructed_time_unix_sec();
    }
    RtcNativeCellIdentity identity(
        TimestreamNetworkId network_id, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        require_occurrence(network_id, native_row);
        return rtc_native_cell_identity(
            *input_, network_id, native_row, detector_index);
    }
    NativeSampleIdentity representative_native_identity(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const {
        require_occurrence(network_id, native_row);
        return input_->network(network_id)
            .occurrence_axis_handle()->native_identity(native_row);
    }
    const NativeReadoutIntegrationSupport &integration_support(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const {
        require_occurrence(network_id, native_row);
        return input_->network(network_id)
            .occurrence_axis_handle()
            ->occurrence(native_row).integration_support;
    }
    const NativePairedReadoutOccurrenceBinding &occurrence_binding(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const {
        require_occurrence(network_id, native_row);
        return input_->network(network_id)
            .occurrence_axis().occurrence(native_row);
    }
    const NativeReadoutDetectorBinding &detector_binding(
        TimestreamNetworkId network_id,
        Eigen::Index detector_index) const {
        return input_->network(network_id).detector(detector_index);
    }
    const NativeReadoutMappingAuthority &mapping_authority(
        TimestreamNetworkId network_id) const {
        return input_->network(network_id).mapping_authority();
    }
    double value(
        NativeReadoutCoordinate member, TimestreamNetworkId network_id,
        TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        require_occurrence(network_id, native_row);
        return input_->network(network_id).value(
            member, native_row, detector_index);
    }
    bool member_numerically_valid(
        NativeReadoutCoordinate member, TimestreamNetworkId network_id,
        TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        require_occurrence(network_id, native_row);
        return input_->network(network_id)
            .state(member, native_row, detector_index).valid();
    }
    NativeReadoutCoordinateCause member_local_causes(
        NativeReadoutCoordinate member, TimestreamNetworkId network_id,
        TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        require_occurrence(network_id, native_row);
        return input_->network(network_id)
            .state(member, native_row, detector_index).causes();
    }
    const NativeReadoutCoordinateState &member_state(
        NativeReadoutCoordinate member,
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        require_occurrence(network_id, native_row);
        return input_->network(network_id)
            .state(member, native_row, detector_index);
    }
    RtcPairDecision pair_decision(
        TimestreamNetworkId network_id, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        require_occurrence(network_id, native_row);
        return plan_->decision(network_id, native_row, detector_index);
    }
    const RtcEvidenceEvent *pair_causal_evidence(
        TimestreamNetworkId network_id, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        require_occurrence(network_id, native_row);
        return plan_->causal_evidence(network_id, native_row,
                                      detector_index);
    }
    RtcTimestreamMemoryEvidence memory_evidence() const noexcept {
        return {0, 0, 1};
    }

private:
    friend RtcApplyResult apply_identity_rtc(
        std::shared_ptr<const RtcPlan>,
        std::shared_ptr<const NativePairedReadoutView>,
        std::shared_ptr<const ValSnapshot>);
    friend RtcApplyResult apply_identity_rtc_partitioned(
        std::shared_ptr<const RtcPlan>,
        std::shared_ptr<const NativePairedReadoutView>,
        std::span<const std::shared_ptr<const NativePairedReadoutView>>,
        std::shared_ptr<const ValSnapshot>);

    RtcTimestream(std::shared_ptr<const RtcPlan> plan,
                  std::shared_ptr<const NativePairedReadoutView> input)
        : plan_{std::move(plan)}, input_{std::move(input)} {}

    void require_occurrence(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const {
        const auto &network_span = input_->span(network_id);
        if (native_row < network_span.first_native_row ||
            native_row >= network_span.past_last_native_row) {
            throw std::out_of_range(
                "native row is outside RTC product support");
        }
    }

    std::shared_ptr<const RtcPlan> plan_;
    std::shared_ptr<const NativePairedReadoutView> input_;
};

enum class RtcCompletionState : std::uint8_t {
    complete,
};

// A compact execution record.  It deliberately does not contain TOD values,
// support planes, event histories, or generalized provenance.
struct RtcRealization {
    RtcPlanIdentity plan_identity;
    RtcCompletionState completion = RtcCompletionState::complete;
    std::size_t output_native_occurrence_count = 0;
    std::size_t output_cell_count = 0;
    std::size_t pair_ineligible_cell_count = 0;
    std::size_t x_payload_available_cell_count = 0;
    std::size_t r_payload_available_cell_count = 0;
    std::size_t x_numerically_valid_cell_count = 0;
    std::size_t r_numerically_valid_cell_count = 0;
    std::size_t realized_sampling_factor = 1;
    ValGeneration val_generation;
};

struct RtcApplyResult {
    std::shared_ptr<const RtcTimestream> product;
    RtcRealization realization;
};

class StaleRtcValGeneration : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

inline RtcApplyResult apply_identity_rtc_partitioned(
    std::shared_ptr<const RtcPlan> plan,
    std::shared_ptr<const NativePairedReadoutView> logical_input,
    std::span<const std::shared_ptr<const NativePairedReadoutView>>
        partitions,
    std::shared_ptr<const ValSnapshot> resolved_val_snapshot) {
    if (!plan || !logical_input ||
        plan->input_handle().get() != logical_input.get()) {
        throw std::invalid_argument(
            "identity RTC apply requires the exact plan-bound input");
    }
    if (!resolved_val_snapshot ||
        plan->val_snapshot_handle().get() != resolved_val_snapshot.get()) {
        throw StaleRtcValGeneration(
            "identity RTC plan cannot apply against a different VAL generation");
    }
    if (!(plan->input_handle()->scope() == logical_input->scope()) ||
        resolved_val_snapshot->paired_handle().get() !=
            logical_input->parent_handle().get() ||
        plan->operator_spec().sampling_factor != 1 ||
        plan->operator_spec().sampling_phase != 0) {
        throw std::logic_error(
            "identity RTC plan binding or operator is inconsistent");
    }
    require_exact_native_partition_schedule(*logical_input, partitions);

    const auto pair_ineligible_cell_count =
        plan->evidence_handle()->events().size();
    RtcRealization realization{
        plan->identity(), RtcCompletionState::complete,
        logical_input->native_occurrence_count(),
        logical_input->detector_occurrence_count(),
        pair_ineligible_cell_count, 0, 0, 0, 0, 1,
        resolved_val_snapshot->generation()};
    for (const auto &partition : partitions) {
        for (const auto &network_span : partition->spans()) {
            const auto &network =
                logical_input->network(network_span.network_id);
            for (auto row = network_span.first_native_row;
                 row < network_span.past_last_native_row; ++row) {
                for (Eigen::Index detector = 0;
                     detector < network.detector_count(); ++detector) {
                    if (network.state(
                            NativeReadoutCoordinate::x, row, detector)
                            .payload_available()) {
                        ++realization.x_payload_available_cell_count;
                    }
                    if (network.state(
                            NativeReadoutCoordinate::r, row, detector)
                            .payload_available()) {
                        ++realization.r_payload_available_cell_count;
                    }
                    if (network.state(
                            NativeReadoutCoordinate::x, row, detector).valid()) {
                        ++realization.x_numerically_valid_cell_count;
                    }
                    if (network.state(
                            NativeReadoutCoordinate::r, row, detector).valid()) {
                        ++realization.r_numerically_valid_cell_count;
                    }
                }
            }
        }
    }

    return RtcApplyResult{
        std::shared_ptr<const RtcTimestream>(
            new RtcTimestream{
                std::move(plan), std::move(logical_input)}),
        realization};
}

inline RtcApplyResult apply_identity_rtc(
    std::shared_ptr<const RtcPlan> plan,
    std::shared_ptr<const NativePairedReadoutView> input,
    std::shared_ptr<const ValSnapshot> resolved_val_snapshot) {
    const std::vector<std::shared_ptr<const NativePairedReadoutView>>
        partitions{input};
    return apply_identity_rtc_partitioned(
        std::move(plan), std::move(input), partitions,
        std::move(resolved_val_snapshot));
}

}  // namespace citlali::pipeline
