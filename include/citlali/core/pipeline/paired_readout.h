#pragma once

#include <citlali/core/pipeline/timestream_native_alignment.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

// The native KIDs solver produces row-major matrices.  Retaining that layout
// makes paired ingress a move of each numerical plane rather than an implicit
// transpose or repack.  Later RTC kernels may introduce a measured, private
// layout conversion without changing this scientific product contract.
using PairedReadoutMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

enum class ReadoutMember : std::uint8_t {
    x,
    r,
};

enum class ReadoutMemberOrigin : std::uint8_t {
    measured,
};

enum class ReadoutMemberCause : std::uint16_t {
    none = 0,
    producer_unavailable = 1U << 0,
    producer_invalid = 1U << 1,
    nonfinite_payload = 1U << 2,
    outside_acquisition_support = 1U << 3,
};

constexpr ReadoutMemberCause operator|(ReadoutMemberCause lhs,
                                       ReadoutMemberCause rhs) noexcept {
    return static_cast<ReadoutMemberCause>(
        static_cast<std::uint16_t>(lhs) |
        static_cast<std::uint16_t>(rhs));
}

constexpr bool has_cause(ReadoutMemberCause value,
                         ReadoutMemberCause cause) noexcept {
    return (static_cast<std::uint16_t>(value) &
            static_cast<std::uint16_t>(cause)) != 0;
}

// One compact word preserves the independent member facts needed at ingress.
// Pair-wide consequences are derived from the two words and are not stored in
// a third dense plane.
class ReadoutMemberState {
public:
    static ReadoutMemberState measured(bool available,
                                       bool original_valid,
                                       bool in_acquisition_support,
                                       bool finite_payload) {
        if (original_valid && (!available || !finite_payload)) {
            throw std::invalid_argument(
                "original-valid readout member must be available and finite");
        }
        std::uint16_t bits = measured_origin_bit;
        if (available) bits |= available_bit;
        if (original_valid) bits |= original_valid_bit;
        if (in_acquisition_support) bits |= support_bit;
        if (!available) bits |= cause_unavailable_bit;
        if (!original_valid) bits |= cause_invalid_bit;
        if (!finite_payload) bits |= cause_nonfinite_bit;
        if (!in_acquisition_support) bits |= cause_outside_support_bit;
        return ReadoutMemberState{bits};
    }

    bool available() const noexcept { return (bits_ & available_bit) != 0; }
    bool original_valid() const noexcept {
        return (bits_ & original_valid_bit) != 0;
    }
    bool in_acquisition_support() const noexcept {
        return (bits_ & support_bit) != 0;
    }
    bool valid() const noexcept {
        return available() && original_valid() &&
               in_acquisition_support() &&
               !has_cause(causes(), ReadoutMemberCause::nonfinite_payload);
    }
    ReadoutMemberOrigin origin() const noexcept {
        return ReadoutMemberOrigin::measured;
    }
    ReadoutMemberCause causes() const noexcept {
        std::uint16_t result = 0;
        if ((bits_ & cause_unavailable_bit) != 0) {
            result |= static_cast<std::uint16_t>(
                ReadoutMemberCause::producer_unavailable);
        }
        if ((bits_ & cause_invalid_bit) != 0) {
            result |= static_cast<std::uint16_t>(
                ReadoutMemberCause::producer_invalid);
        }
        if ((bits_ & cause_nonfinite_bit) != 0) {
            result |= static_cast<std::uint16_t>(
                ReadoutMemberCause::nonfinite_payload);
        }
        if ((bits_ & cause_outside_support_bit) != 0) {
            result |= static_cast<std::uint16_t>(
                ReadoutMemberCause::outside_acquisition_support);
        }
        return static_cast<ReadoutMemberCause>(result);
    }
    bool consistent_with(double value) const noexcept {
        const bool nonfinite = !std::isfinite(value);
        return nonfinite ==
                   has_cause(causes(),
                             ReadoutMemberCause::nonfinite_payload) &&
               (!original_valid() || (available() && !nonfinite));
    }

private:
    explicit constexpr ReadoutMemberState(std::uint16_t bits)
        : bits_{bits} {}

    static constexpr std::uint16_t available_bit = 1U << 0;
    static constexpr std::uint16_t original_valid_bit = 1U << 1;
    static constexpr std::uint16_t support_bit = 1U << 2;
    static constexpr std::uint16_t measured_origin_bit = 1U << 3;
    static constexpr std::uint16_t cause_unavailable_bit = 1U << 4;
    static constexpr std::uint16_t cause_invalid_bit = 1U << 5;
    static constexpr std::uint16_t cause_nonfinite_bit = 1U << 6;
    static constexpr std::uint16_t cause_outside_support_bit = 1U << 7;

    std::uint16_t bits_ = measured_origin_bit;
};

static_assert(sizeof(ReadoutMemberState) == sizeof(std::uint16_t));

enum class PairedReadoutCause : std::uint16_t {
    none = 0,
    x_unavailable = 1U << 0,
    r_unavailable = 1U << 1,
    x_original_invalid = 1U << 2,
    r_original_invalid = 1U << 3,
    x_nonfinite = 1U << 4,
    r_nonfinite = 1U << 5,
    x_outside_support = 1U << 6,
    r_outside_support = 1U << 7,
};

constexpr PairedReadoutCause operator|(PairedReadoutCause lhs,
                                       PairedReadoutCause rhs) noexcept {
    return static_cast<PairedReadoutCause>(
        static_cast<std::uint16_t>(lhs) |
        static_cast<std::uint16_t>(rhs));
}

constexpr bool has_cause(PairedReadoutCause value,
                         PairedReadoutCause cause) noexcept {
    return (static_cast<std::uint16_t>(value) &
            static_cast<std::uint16_t>(cause)) != 0;
}

struct NativeOccurrenceInterval {
    double begin_unix_sec = 0.0;
    double end_unix_sec = 0.0;

    double duration_sec() const noexcept {
        return end_unix_sec - begin_unix_sec;
    }

    friend bool operator==(const NativeOccurrenceInterval &,
                           const NativeOccurrenceInterval &) = default;
};

// The axis references ALIGN-owned native timing and owns only the primitive
// acquisition intervals absent from that carrier.  It contains no common-slot
// association or AST coordinate.
class PairedReadoutOccurrenceAxis {
public:
    PairedReadoutOccurrenceAxis(
        std::shared_ptr<const NativeNetworkAlignment> native_timing,
        TimestreamNativeRow first_native_row,
        std::vector<NativeOccurrenceInterval> occurrence_intervals)
        : native_timing_{std::move(native_timing)},
          first_native_row_{first_native_row},
          occurrence_intervals_{std::move(occurrence_intervals)} {
        if (!native_timing_ || first_native_row_ < 0 ||
            occurrence_intervals_.empty() ||
            occurrence_intervals_.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<TimestreamNativeRow>::max() -
                    first_native_row_)) {
            throw std::invalid_argument(
                "paired readout occurrence axis is incomplete");
        }
        if (first_native_row_ < native_timing_->first_native_row() ||
            past_last_native_row() >
                native_timing_->past_last_native_row()) {
            throw std::invalid_argument(
                "paired readout occurrence axis exceeds native timing support");
        }
        for (std::size_t index = 0; index < occurrence_intervals_.size();
             ++index) {
            const auto &interval = occurrence_intervals_[index];
            const auto native_row = first_native_row_ +
                static_cast<TimestreamNativeRow>(index);
            const double event_time = native_timing_->identity(native_row)
                                          .reconstructed_time_unix_sec();
            if (!std::isfinite(interval.begin_unix_sec) ||
                !std::isfinite(interval.end_unix_sec) ||
                !(interval.begin_unix_sec < interval.end_unix_sec) ||
                event_time < interval.begin_unix_sec ||
                event_time > interval.end_unix_sec) {
                throw std::invalid_argument(
                    "native event time must lie in a finite positive occurrence interval");
            }
        }
    }

    TimestreamNetworkId network_id() const noexcept {
        return native_timing_->network_id();
    }
    TimestreamNativeRow first_native_row() const noexcept {
        return first_native_row_;
    }
    TimestreamNativeRow past_last_native_row() const noexcept {
        return first_native_row_ +
            static_cast<TimestreamNativeRow>(occurrence_intervals_.size());
    }
    std::size_t occurrence_count() const noexcept {
        return occurrence_intervals_.size();
    }
    const std::shared_ptr<const NativeNetworkAlignment> &native_timing_handle()
        const noexcept {
        return native_timing_;
    }
    NativeSampleIdentity identity(TimestreamNativeRow native_row) const {
        return native_timing_->identity(checked_index(native_row));
    }
    const NativeOccurrenceInterval &interval(
        TimestreamNativeRow native_row) const {
        return occurrence_intervals_.at(local_index(native_row));
    }
    std::span<const NativeOccurrenceInterval> intervals() const noexcept {
        return occurrence_intervals_;
    }

private:
    TimestreamNativeRow checked_index(TimestreamNativeRow native_row) const {
        (void)local_index(native_row);
        return native_row;
    }
    std::size_t local_index(TimestreamNativeRow native_row) const {
        if (native_row < first_native_row_ ||
            native_row >= past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside the paired readout occurrence axis");
        }
        return static_cast<std::size_t>(native_row - first_native_row_);
    }

    std::shared_ptr<const NativeNetworkAlignment> native_timing_;
    TimestreamNativeRow first_native_row_;
    std::vector<NativeOccurrenceInterval> occurrence_intervals_;
};

struct PairedReadoutDetectorIdentity {
    std::int64_t output_uid = 0;
    std::int64_t array_id = -1;
    TimestreamNetworkId network_id = -1;
    std::int64_t raw_source_uid = -1;
    Eigen::Index raw_channel = -1;

    friend bool operator==(const PairedReadoutDetectorIdentity &,
                           const PairedReadoutDetectorIdentity &) = default;
};

// One immutable, network-scoped reference supplies the raw coordinate meaning
// for every cell.  Full timestream content is deliberately not hashed.
struct NativeReadoutMappingIdentity {
    std::string producer_interface_id;
    std::string producer_instance_id;
    std::string tune_id;
    std::string mapping_revision;
    std::string transform_id;
    std::string x_raw_unit_id;
    std::string r_raw_unit_id;

    bool complete() const noexcept {
        return !producer_interface_id.empty() &&
               !producer_instance_id.empty() && !tune_id.empty() &&
               !mapping_revision.empty() && !transform_id.empty() &&
               !x_raw_unit_id.empty() && !r_raw_unit_id.empty();
    }

    std::size_t text_bytes() const noexcept {
        return producer_interface_id.size() + producer_instance_id.size() +
               tune_id.size() + mapping_revision.size() +
               transform_id.size() + x_raw_unit_id.size() +
               r_raw_unit_id.size();
    }
};

struct PairedReadoutCardinality {
    std::size_t network_count = 0;
    std::size_t detector_count = 0;
    std::size_t native_occurrence_count = 0;
    std::size_t detector_occurrence_count = 0;
};

// These are logical owned-content bytes, not allocator or RSS measurements.
// Referenced native timing storage is reported separately and is not counted.
struct PairedReadoutMemoryEvidence {
    std::size_t numeric_payload_bytes = 0;
    std::size_t member_state_bytes = 0;
    std::size_t occurrence_interval_bytes = 0;
    std::size_t detector_axis_bytes = 0;
    std::size_t identity_text_bytes = 0;
    std::size_t referenced_native_axis_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return numeric_payload_bytes + member_state_bytes +
               occurrence_interval_bytes + detector_axis_bytes +
               identity_text_bytes;
    }
};

class PairedReadoutNetwork {
public:
    PairedReadoutNetwork(const PairedReadoutNetwork &) = delete;
    PairedReadoutNetwork &operator=(const PairedReadoutNetwork &) = delete;
    PairedReadoutNetwork(PairedReadoutNetwork &&) noexcept = default;
    PairedReadoutNetwork &operator=(PairedReadoutNetwork &&) noexcept = default;

    static PairedReadoutNetwork admit(
        std::shared_ptr<const PairedReadoutOccurrenceAxis> occurrence_axis,
        std::vector<PairedReadoutDetectorIdentity> detectors,
        std::shared_ptr<const NativeReadoutMappingIdentity> mapping_identity,
        PairedReadoutMatrix x_values,
        PairedReadoutMatrix r_values,
        std::vector<ReadoutMemberState> x_states,
        std::vector<ReadoutMemberState> r_states) {
        if (!occurrence_axis || !mapping_identity ||
            !mapping_identity->complete() || x_values.rows() <= 0 ||
            x_values.cols() <= 0 || x_values.rows() != r_values.rows() ||
            x_values.cols() != r_values.cols() ||
            static_cast<std::size_t>(x_values.rows()) !=
                occurrence_axis->occurrence_count() ||
            static_cast<std::size_t>(x_values.cols()) != detectors.size()) {
            throw std::invalid_argument(
                "paired readout network identity or x/r shape is incomplete");
        }
        const auto rows = static_cast<std::size_t>(x_values.rows());
        const auto columns = static_cast<std::size_t>(x_values.cols());
        if (columns != 0 &&
            rows > std::numeric_limits<std::size_t>::max() / columns) {
            throw std::length_error(
                "paired readout network cardinality would overflow");
        }
        const auto cell_count = rows * columns;
        if (x_states.size() != cell_count || r_states.size() != cell_count) {
            throw std::invalid_argument(
                "paired readout member facts do not match x/r shape");
        }
        std::set<Eigen::Index> channels;
        std::set<std::int64_t> output_uids;
        for (std::size_t index = 0; index < detectors.size(); ++index) {
            const auto &detector = detectors[index];
            if (detector.output_uid <= 0 || detector.array_id < 0 ||
                detector.network_id != occurrence_axis->network_id() ||
                detector.raw_source_uid < 0 || detector.raw_channel < 0 ||
                detector.raw_channel >= x_values.cols() ||
                detector.raw_channel != static_cast<Eigen::Index>(index) ||
                !channels.insert(detector.raw_channel).second ||
                !output_uids.insert(detector.output_uid).second) {
                throw std::invalid_argument(
                    "paired readout detector axis is invalid or ambiguous");
            }
        }
        if (channels.size() != columns) {
            throw std::invalid_argument(
                "paired readout detector axis does not cover every solver column");
        }
        for (Eigen::Index row = 0; row < x_values.rows(); ++row) {
            for (Eigen::Index column = 0; column < x_values.cols(); ++column) {
                const auto index = static_cast<std::size_t>(
                    row * x_values.cols() + column);
                if (!x_states[index].consistent_with(x_values(row, column)) ||
                    !r_states[index].consistent_with(r_values(row, column))) {
                    throw std::invalid_argument(
                        "paired readout member facts disagree with payload finiteness");
                }
            }
        }
        return PairedReadoutNetwork{
            std::move(occurrence_axis), std::move(detectors),
            std::move(mapping_identity), std::move(x_values),
            std::move(r_values), std::move(x_states), std::move(r_states)};
    }

    TimestreamNetworkId network_id() const noexcept {
        return occurrence_axis_->network_id();
    }
    Eigen::Index occurrence_count() const noexcept { return x_values_.rows(); }
    Eigen::Index detector_count() const noexcept { return x_values_.cols(); }
    std::size_t cell_count() const noexcept {
        return static_cast<std::size_t>(x_values_.size());
    }
    const std::shared_ptr<const PairedReadoutOccurrenceAxis> &
    occurrence_axis_handle() const noexcept {
        return occurrence_axis_;
    }
    const std::shared_ptr<const NativeReadoutMappingIdentity> &
    mapping_identity_handle() const noexcept {
        return mapping_identity_;
    }
    std::span<const PairedReadoutDetectorIdentity> detectors() const noexcept {
        return detectors_;
    }
    const PairedReadoutDetectorIdentity &detector(
        Eigen::Index detector_index) const {
        return detectors_.at(checked_detector_index(detector_index));
    }
    const PairedReadoutMatrix &values(ReadoutMember member) const noexcept {
        return member == ReadoutMember::x ? x_values_ : r_values_;
    }
    std::span<const double> contiguous_values(
        ReadoutMember member) const noexcept {
        const auto &plane = values(member);
        return {plane.data(), static_cast<std::size_t>(plane.size())};
    }
    double value(ReadoutMember member, TimestreamNativeRow native_row,
                 Eigen::Index detector_index) const {
        return values(member)(checked_row(native_row),
                              checked_detector(detector_index));
    }
    const ReadoutMemberState &state(
        ReadoutMember member, TimestreamNativeRow native_row,
        Eigen::Index detector_index) const {
        const auto index = flat_index(native_row, detector_index);
        return member == ReadoutMember::x ? x_states_.at(index)
                                          : r_states_.at(index);
    }
    bool pair_available(TimestreamNativeRow native_row,
                        Eigen::Index detector_index) const {
        return state(ReadoutMember::x, native_row, detector_index).available() &&
               state(ReadoutMember::r, native_row, detector_index).available();
    }
    bool pair_valid(TimestreamNativeRow native_row,
                    Eigen::Index detector_index) const {
        return state(ReadoutMember::x, native_row, detector_index).valid() &&
               state(ReadoutMember::r, native_row, detector_index).valid();
    }
    PairedReadoutCause pair_causes(TimestreamNativeRow native_row,
                                   Eigen::Index detector_index) const {
        PairedReadoutCause result = PairedReadoutCause::none;
        const auto x = state(ReadoutMember::x, native_row, detector_index);
        const auto r = state(ReadoutMember::r, native_row, detector_index);
        if (!x.available()) result = result | PairedReadoutCause::x_unavailable;
        if (!r.available()) result = result | PairedReadoutCause::r_unavailable;
        if (!x.original_valid()) {
            result = result | PairedReadoutCause::x_original_invalid;
        }
        if (!r.original_valid()) {
            result = result | PairedReadoutCause::r_original_invalid;
        }
        if (has_cause(x.causes(), ReadoutMemberCause::nonfinite_payload)) {
            result = result | PairedReadoutCause::x_nonfinite;
        }
        if (has_cause(r.causes(), ReadoutMemberCause::nonfinite_payload)) {
            result = result | PairedReadoutCause::r_nonfinite;
        }
        if (!x.in_acquisition_support()) {
            result = result | PairedReadoutCause::x_outside_support;
        }
        if (!r.in_acquisition_support()) {
            result = result | PairedReadoutCause::r_outside_support;
        }
        return result;
    }

    PairedReadoutMemoryEvidence memory_evidence() const noexcept {
        return PairedReadoutMemoryEvidence{
            static_cast<std::size_t>(x_values_.size() + r_values_.size()) *
                sizeof(double),
            (x_states_.size() + r_states_.size()) *
                sizeof(ReadoutMemberState),
            occurrence_axis_->occurrence_count() *
                sizeof(NativeOccurrenceInterval),
            detectors_.size() * sizeof(PairedReadoutDetectorIdentity),
            mapping_identity_->text_bytes(),
            1};
    }

private:
    PairedReadoutNetwork(
        std::shared_ptr<const PairedReadoutOccurrenceAxis> occurrence_axis,
        std::vector<PairedReadoutDetectorIdentity> detectors,
        std::shared_ptr<const NativeReadoutMappingIdentity> mapping_identity,
        PairedReadoutMatrix x_values,
        PairedReadoutMatrix r_values,
        std::vector<ReadoutMemberState> x_states,
        std::vector<ReadoutMemberState> r_states)
        : occurrence_axis_{std::move(occurrence_axis)},
          detectors_{std::move(detectors)},
          mapping_identity_{std::move(mapping_identity)},
          x_values_{std::move(x_values)}, r_values_{std::move(r_values)},
          x_states_{std::move(x_states)}, r_states_{std::move(r_states)} {}

    Eigen::Index checked_row(TimestreamNativeRow native_row) const {
        if (native_row < occurrence_axis_->first_native_row() ||
            native_row >= occurrence_axis_->past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside paired readout network support");
        }
        return static_cast<Eigen::Index>(
            native_row - occurrence_axis_->first_native_row());
    }
    Eigen::Index checked_detector(Eigen::Index detector_index) const {
        (void)checked_detector_index(detector_index);
        return detector_index;
    }
    std::size_t checked_detector_index(Eigen::Index detector_index) const {
        if (detector_index < 0 || detector_index >= detector_count()) {
            throw std::out_of_range(
                "detector index is outside paired readout network support");
        }
        return static_cast<std::size_t>(detector_index);
    }
    std::size_t flat_index(TimestreamNativeRow native_row,
                           Eigen::Index detector_index) const {
        const auto row = checked_row(native_row);
        const auto detector = checked_detector(detector_index);
        return static_cast<std::size_t>(row * detector_count() + detector);
    }

    std::shared_ptr<const PairedReadoutOccurrenceAxis> occurrence_axis_;
    std::vector<PairedReadoutDetectorIdentity> detectors_;
    std::shared_ptr<const NativeReadoutMappingIdentity> mapping_identity_;
    PairedReadoutMatrix x_values_;
    PairedReadoutMatrix r_values_;
    std::vector<ReadoutMemberState> x_states_;
    std::vector<ReadoutMemberState> r_states_;
};

class PairedReadout {
public:
    static std::shared_ptr<const PairedReadout> admit(
        NativeObservationScope scope,
        std::vector<TimestreamNetworkId> required_network_ids,
        std::vector<PairedReadoutNetwork> networks) {
        if (required_network_ids.empty() || networks.empty()) {
            throw std::invalid_argument(
                "paired readout requires a participant inventory and data");
        }
        std::sort(required_network_ids.begin(), required_network_ids.end());
        if (std::adjacent_find(required_network_ids.begin(),
                               required_network_ids.end()) !=
                required_network_ids.end() ||
            required_network_ids.front() < 0) {
            throw std::invalid_argument(
                "paired readout participant inventory is invalid");
        }
        std::sort(networks.begin(), networks.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.network_id() < rhs.network_id();
                  });
        std::vector<TimestreamNetworkId> actual_network_ids;
        actual_network_ids.reserve(networks.size());
        std::set<std::int64_t> output_uids;
        for (const auto &network : networks) {
            actual_network_ids.push_back(network.network_id());
            for (const auto &detector : network.detectors()) {
                if (!output_uids.insert(detector.output_uid).second) {
                    throw std::invalid_argument(
                        "paired readout repeats detector identity across networks");
                }
            }
        }
        if (actual_network_ids != required_network_ids) {
            throw std::invalid_argument(
                "paired readout data do not match required participant inventory");
        }
        return std::shared_ptr<const PairedReadout>(
            new PairedReadout{scope, std::move(networks)});
    }

    const NativeObservationScope &scope() const noexcept { return scope_; }
    std::size_t network_count() const noexcept { return networks_.size(); }
    std::span<const PairedReadoutNetwork> networks() const noexcept {
        return networks_;
    }
    const PairedReadoutNetwork &network(
        TimestreamNetworkId network_id) const {
        const auto found = std::lower_bound(
            networks_.begin(), networks_.end(), network_id,
            [](const auto &candidate, TimestreamNetworkId id) {
                return candidate.network_id() < id;
            });
        if (found == networks_.end() || found->network_id() != network_id) {
            throw std::out_of_range(
                "network is absent from paired readout");
        }
        return *found;
    }
    PairedReadoutCardinality cardinality() const noexcept {
        PairedReadoutCardinality result;
        result.network_count = networks_.size();
        for (const auto &network : networks_) {
            result.detector_count +=
                static_cast<std::size_t>(network.detector_count());
            result.native_occurrence_count +=
                static_cast<std::size_t>(network.occurrence_count());
            result.detector_occurrence_count += network.cell_count();
        }
        return result;
    }
    PairedReadoutMemoryEvidence memory_evidence() const noexcept {
        PairedReadoutMemoryEvidence result;
        for (const auto &network : networks_) {
            const auto part = network.memory_evidence();
            result.numeric_payload_bytes += part.numeric_payload_bytes;
            result.member_state_bytes += part.member_state_bytes;
            result.occurrence_interval_bytes +=
                part.occurrence_interval_bytes;
            result.detector_axis_bytes += part.detector_axis_bytes;
            result.identity_text_bytes += part.identity_text_bytes;
            result.referenced_native_axis_count +=
                part.referenced_native_axis_count;
        }
        return result;
    }

private:
    PairedReadout(NativeObservationScope scope,
                  std::vector<PairedReadoutNetwork> networks)
        : scope_{scope}, networks_{std::move(networks)} {}

    NativeObservationScope scope_;
    std::vector<PairedReadoutNetwork> networks_;
};

}  // namespace citlali::pipeline
