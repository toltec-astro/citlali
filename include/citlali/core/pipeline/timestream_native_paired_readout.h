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
#include <string_view>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr std::string_view native_paired_xr_producer_interface_id =
    "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1";

// Native KIDs solver output is row-major.  Keeping that physical layout makes
// ingress an ownership transfer of each numerical plane, without imposing a
// later RTC layout or a cross-network common analysis grid.
using NativePairedReadoutMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

enum class NativeReadoutCoordinate : std::uint8_t {
    x,
    r,
};

enum class NativeReadoutOrigin : std::uint8_t {
    measured,
};

enum class NativeReadoutCoordinateCause : std::uint16_t {
    none = 0,
    producer_unavailable = 1U << 0,
    producer_invalid = 1U << 1,
    nonfinite_payload = 1U << 2,
    outside_acquisition_support = 1U << 3,
};

constexpr NativeReadoutCoordinateCause operator|(
    NativeReadoutCoordinateCause lhs,
    NativeReadoutCoordinateCause rhs) noexcept {
    return static_cast<NativeReadoutCoordinateCause>(
        static_cast<std::uint16_t>(lhs) |
        static_cast<std::uint16_t>(rhs));
}

constexpr bool has_cause(NativeReadoutCoordinateCause value,
                         NativeReadoutCoordinateCause cause) noexcept {
    return (static_cast<std::uint16_t>(value) &
            static_cast<std::uint16_t>(cause)) != 0;
}

// Availability, producer validity, acquisition support, and finiteness are
// independent native facts.  The fixed-width representation avoids a cause
// object or narrative string for every detector occurrence.
class NativeReadoutCoordinateState {
public:
    static NativeReadoutCoordinateState measured(
        bool payload_available,
        bool producer_valid,
        bool in_acquisition_support,
        bool finite_payload) {
        if (producer_valid && (!payload_available || !finite_payload)) {
            throw std::invalid_argument(
                "producer-valid native readout must be available and finite");
        }
        std::uint16_t bits = measured_origin_bit;
        if (payload_available) bits |= available_bit;
        if (producer_valid) bits |= producer_valid_bit;
        if (in_acquisition_support) bits |= support_bit;
        if (!payload_available) bits |= cause_unavailable_bit;
        if (!producer_valid) bits |= cause_invalid_bit;
        if (!finite_payload) bits |= cause_nonfinite_bit;
        if (!in_acquisition_support) bits |= cause_outside_support_bit;
        return NativeReadoutCoordinateState{bits};
    }

    bool payload_available() const noexcept {
        return (bits_ & available_bit) != 0;
    }
    bool producer_valid() const noexcept {
        return (bits_ & producer_valid_bit) != 0;
    }
    bool in_acquisition_support() const noexcept {
        return (bits_ & support_bit) != 0;
    }
    bool finite_payload() const noexcept {
        return (bits_ & cause_nonfinite_bit) == 0;
    }
    bool valid() const noexcept {
        return payload_available() && producer_valid() &&
               in_acquisition_support() && finite_payload();
    }
    NativeReadoutOrigin origin() const noexcept {
        return NativeReadoutOrigin::measured;
    }
    NativeReadoutCoordinateCause causes() const noexcept {
        std::uint16_t result = 0;
        if ((bits_ & cause_unavailable_bit) != 0) {
            result |= static_cast<std::uint16_t>(
                NativeReadoutCoordinateCause::producer_unavailable);
        }
        if ((bits_ & cause_invalid_bit) != 0) {
            result |= static_cast<std::uint16_t>(
                NativeReadoutCoordinateCause::producer_invalid);
        }
        if ((bits_ & cause_nonfinite_bit) != 0) {
            result |= static_cast<std::uint16_t>(
                NativeReadoutCoordinateCause::nonfinite_payload);
        }
        if ((bits_ & cause_outside_support_bit) != 0) {
            result |= static_cast<std::uint16_t>(
                NativeReadoutCoordinateCause::outside_acquisition_support);
        }
        return static_cast<NativeReadoutCoordinateCause>(result);
    }
    bool consistent_with(double value) const noexcept {
        return finite_payload() == std::isfinite(value) &&
               (!producer_valid() ||
                (payload_available() && finite_payload()));
    }

private:
    explicit constexpr NativeReadoutCoordinateState(std::uint16_t bits)
        : bits_{bits} {}

    static constexpr std::uint16_t available_bit = 1U << 0;
    static constexpr std::uint16_t producer_valid_bit = 1U << 1;
    static constexpr std::uint16_t support_bit = 1U << 2;
    static constexpr std::uint16_t measured_origin_bit = 1U << 3;
    static constexpr std::uint16_t cause_unavailable_bit = 1U << 4;
    static constexpr std::uint16_t cause_invalid_bit = 1U << 5;
    static constexpr std::uint16_t cause_nonfinite_bit = 1U << 6;
    static constexpr std::uint16_t cause_outside_support_bit = 1U << 7;

    std::uint16_t bits_ = measured_origin_bit;
};

static_assert(sizeof(NativeReadoutCoordinateState) == sizeof(std::uint16_t));

enum class NativePairedReadoutCause : std::uint16_t {
    none = 0,
    x_unavailable = 1U << 0,
    r_unavailable = 1U << 1,
    x_producer_invalid = 1U << 2,
    r_producer_invalid = 1U << 3,
    x_nonfinite = 1U << 4,
    r_nonfinite = 1U << 5,
    x_outside_support = 1U << 6,
    r_outside_support = 1U << 7,
};

constexpr NativePairedReadoutCause operator|(
    NativePairedReadoutCause lhs,
    NativePairedReadoutCause rhs) noexcept {
    return static_cast<NativePairedReadoutCause>(
        static_cast<std::uint16_t>(lhs) |
        static_cast<std::uint16_t>(rhs));
}

constexpr bool has_cause(NativePairedReadoutCause value,
                         NativePairedReadoutCause cause) noexcept {
    return (static_cast<std::uint16_t>(value) &
            static_cast<std::uint16_t>(cause)) != 0;
}

struct NativeReadoutIntegrationSupport {
    double begin_unix_sec = 0.0;
    double end_unix_sec = 0.0;

    double duration_sec() const noexcept {
        return end_unix_sec - begin_unix_sec;
    }

    friend bool operator==(const NativeReadoutIntegrationSupport &,
                           const NativeReadoutIntegrationSupport &) = default;
};

// Keys are stable only under the exact record references carried by the
// mapping authority.  They are not transient row numbers or timestamps.
struct NativePairedReadoutOccurrenceBinding {
    std::int64_t parent_readout_occurrence_key = -1;
    std::int64_t paired_xr_occurrence_key = -1;
    NativeReadoutIntegrationSupport integration_support;

    friend bool operator==(const NativePairedReadoutOccurrenceBinding &,
                           const NativePairedReadoutOccurrenceBinding &) =
        default;
};

// This axis references canonical network timing.  It owns only the primitive
// readout/pair occurrence relation and integration support absent from that
// authority.  It intentionally contains no common-grid relation.
class NativePairedReadoutOccurrenceAxis {
public:
    NativePairedReadoutOccurrenceAxis(
        std::shared_ptr<const NativeNetworkAlignment> native_timing,
        TimestreamNativeRow first_native_row,
        std::vector<NativePairedReadoutOccurrenceBinding> occurrences)
        : native_timing_{std::move(native_timing)},
          first_native_row_{first_native_row},
          occurrences_{std::move(occurrences)} {
        if (!native_timing_ || first_native_row_ < 0 ||
            occurrences_.empty() ||
            occurrences_.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<TimestreamNativeRow>::max() -
                    first_native_row_)) {
            throw std::invalid_argument(
                "native paired readout occurrence axis is incomplete");
        }
        if (first_native_row_ < native_timing_->first_native_row() ||
            past_last_native_row() >
                native_timing_->past_last_native_row()) {
            throw std::invalid_argument(
                "native paired readout axis exceeds timing support");
        }

        std::set<std::int64_t> parent_keys;
        std::set<std::int64_t> pair_keys;
        for (std::size_t index = 0; index < occurrences_.size(); ++index) {
            const auto &occurrence = occurrences_[index];
            const auto native_row =
                first_native_row_ + static_cast<TimestreamNativeRow>(index);
            const double event_time =
                native_timing_->identity(native_row)
                    .reconstructed_time_unix_sec();
            const auto &support = occurrence.integration_support;
            if (occurrence.parent_readout_occurrence_key < 0 ||
                occurrence.paired_xr_occurrence_key < 0 ||
                !parent_keys
                     .insert(occurrence.parent_readout_occurrence_key)
                     .second ||
                !pair_keys.insert(occurrence.paired_xr_occurrence_key)
                     .second ||
                !std::isfinite(support.begin_unix_sec) ||
                !std::isfinite(support.end_unix_sec) ||
                !(support.begin_unix_sec < support.end_unix_sec) ||
                event_time < support.begin_unix_sec ||
                !(event_time < support.end_unix_sec)) {
                throw std::invalid_argument(
                    "native readout occurrence identity or half-open support is invalid");
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
            static_cast<TimestreamNativeRow>(occurrences_.size());
    }
    std::size_t occurrence_count() const noexcept {
        return occurrences_.size();
    }
    const std::shared_ptr<const NativeNetworkAlignment> &
    native_timing_handle() const noexcept {
        return native_timing_;
    }
    NativeSampleIdentity native_identity(
        TimestreamNativeRow native_row) const {
        (void)local_index(native_row);
        return native_timing_->identity(native_row);
    }
    TimestreamPacketCounter packet_counter(
        TimestreamNativeRow native_row) const {
        (void)local_index(native_row);
        return native_timing_->packet_counter(native_row);
    }
    const NativePairedReadoutOccurrenceBinding &occurrence(
        TimestreamNativeRow native_row) const {
        return occurrences_.at(local_index(native_row));
    }
    std::span<const NativePairedReadoutOccurrenceBinding> occurrences()
        const noexcept {
        return occurrences_;
    }
    std::vector<NativeContiguousRun> contiguous_runs() const {
        return partition_native_contiguous_runs(
            *native_timing_, first_native_row_, past_last_native_row());
    }

private:
    std::size_t local_index(TimestreamNativeRow native_row) const {
        if (native_row < first_native_row_ ||
            native_row >= past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside the paired readout axis");
        }
        return static_cast<std::size_t>(native_row - first_native_row_);
    }

    std::shared_ptr<const NativeNetworkAlignment> native_timing_;
    TimestreamNativeRow first_native_row_;
    std::vector<NativePairedReadoutOccurrenceBinding> occurrences_;
};

struct NativeReadoutCoordinateAuthority {
    std::string meaning_id;
    std::string unit_or_scale_id;
    std::string sign_convention_id;
    std::string reference_point_id;
    std::string normalization_id;
    std::string metric_id;
    std::string validity_domain_id;
    std::string uncertainty_state_id;

    bool complete() const noexcept {
        return !meaning_id.empty() && !unit_or_scale_id.empty() &&
               !sign_convention_id.empty() &&
               !reference_point_id.empty() && !normalization_id.empty() &&
               !metric_id.empty() && !validity_domain_id.empty() &&
               !uncertainty_state_id.empty();
    }

    std::size_t text_bytes() const noexcept {
        return meaning_id.size() + unit_or_scale_id.size() +
               sign_convention_id.size() + reference_point_id.size() +
               normalization_id.size() + metric_id.size() +
               validity_domain_id.size() + uncertainty_state_id.size();
    }
};

// The observation-scoped binding owns stable references to the producer's
// static and realized Tune/readout authorities.  Coordinate convention fields
// are references, not locally inferred numerical policy.
struct NativeReadoutMappingAuthority {
    TimestreamNetworkId network_id = -1;
    std::string producer_id;
    std::string producer_instance_id;
    std::string producer_interface_id;
    std::string mapping_record_id;
    std::string mapping_revision_id;
    std::string tune_id;
    std::string readout_interface_id;
    std::string input_coordinate_record_id;
    std::string transform_id;
    std::string transform_representation_id;
    std::string applicability_domain_id;
    std::string event_time_epoch_meaning_id;
    std::string native_time_unit_id;
    std::string native_cadence_record_id;
    std::string native_time_validity_state_id;
    std::string timing_uncertainty_state_id;
    std::string parent_readout_record_id;
    std::string paired_xr_record_id;
    std::string runtime_binding_rule_id;
    std::string compatibility_rule_id;
    std::string failure_semantics_id;
    NativeReadoutCoordinateAuthority x;
    NativeReadoutCoordinateAuthority r;

    bool complete() const noexcept {
        return network_id >= 0 &&
               producer_interface_id ==
                   native_paired_xr_producer_interface_id &&
               !producer_id.empty() && !producer_instance_id.empty() &&
               !mapping_record_id.empty() &&
               !mapping_revision_id.empty() && !tune_id.empty() &&
               !readout_interface_id.empty() &&
               !input_coordinate_record_id.empty() &&
               !transform_id.empty() &&
               !transform_representation_id.empty() &&
               !applicability_domain_id.empty() &&
               !event_time_epoch_meaning_id.empty() &&
               !native_time_unit_id.empty() &&
               !native_cadence_record_id.empty() &&
               !native_time_validity_state_id.empty() &&
               !timing_uncertainty_state_id.empty() &&
               !parent_readout_record_id.empty() &&
               !paired_xr_record_id.empty() &&
               !runtime_binding_rule_id.empty() &&
               !compatibility_rule_id.empty() &&
               !failure_semantics_id.empty() && x.complete() &&
               r.complete();
    }

    std::size_t text_bytes() const noexcept {
        return producer_id.size() + producer_instance_id.size() +
               producer_interface_id.size() + mapping_record_id.size() +
               mapping_revision_id.size() + tune_id.size() +
               readout_interface_id.size() +
               input_coordinate_record_id.size() + transform_id.size() +
               transform_representation_id.size() +
               applicability_domain_id.size() +
               event_time_epoch_meaning_id.size() +
               native_time_unit_id.size() +
               native_cadence_record_id.size() +
               native_time_validity_state_id.size() +
               timing_uncertainty_state_id.size() +
               parent_readout_record_id.size() + paired_xr_record_id.size() +
               runtime_binding_rule_id.size() +
               compatibility_rule_id.size() + failure_semantics_id.size() +
               x.text_bytes() + r.text_bytes();
    }
};

// storage_column is local physical layout only.  The three strings carry the
// exact detector, association-record, and readout channel identities.
struct NativeReadoutDetectorBinding {
    TimestreamNetworkId network_id = -1;
    Eigen::Index storage_column = -1;
    std::string detector_occurrence_id;
    std::string detector_association_record_id;
    std::string tone_or_channel_id;

    bool complete() const noexcept {
        return network_id >= 0 && storage_column >= 0 &&
               !detector_occurrence_id.empty() &&
               !detector_association_record_id.empty() &&
               !tone_or_channel_id.empty();
    }

    std::size_t text_bytes() const noexcept {
        return detector_occurrence_id.size() +
               detector_association_record_id.size() +
               tone_or_channel_id.size();
    }
};

struct NativePairedReadoutCardinality {
    std::size_t network_count = 0;
    std::size_t detector_count = 0;
    std::size_t native_occurrence_count = 0;
    std::size_t detector_occurrence_count = 0;
};

// These values describe logical owned content, not allocator overhead or RSS.
// Canonical native timing is referenced and reported, but never double-counted.
struct NativePairedReadoutMemoryEvidence {
    std::size_t numeric_payload_bytes = 0;
    std::size_t coordinate_state_bytes = 0;
    std::size_t occurrence_axis_bytes = 0;
    std::size_t detector_axis_bytes = 0;
    std::size_t identity_text_bytes = 0;
    std::size_t referenced_native_axis_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return numeric_payload_bytes + coordinate_state_bytes +
               occurrence_axis_bytes + detector_axis_bytes +
               identity_text_bytes;
    }
};

class NativePairedReadoutNetwork {
public:
    NativePairedReadoutNetwork(const NativePairedReadoutNetwork &) = delete;
    NativePairedReadoutNetwork &operator=(
        const NativePairedReadoutNetwork &) = delete;
    NativePairedReadoutNetwork(NativePairedReadoutNetwork &&) noexcept =
        default;
    NativePairedReadoutNetwork &operator=(
        NativePairedReadoutNetwork &&) noexcept = default;

    static NativePairedReadoutNetwork admit(
        std::shared_ptr<const NativePairedReadoutOccurrenceAxis>
            occurrence_axis,
        std::vector<NativeReadoutDetectorBinding> detectors,
        std::shared_ptr<const NativeReadoutMappingAuthority>
            mapping_authority,
        NativePairedReadoutMatrix x_values,
        NativePairedReadoutMatrix r_values,
        std::vector<NativeReadoutCoordinateState> x_states,
        std::vector<NativeReadoutCoordinateState> r_states) {
        if (!occurrence_axis || !mapping_authority ||
            !mapping_authority->complete() || x_values.rows() <= 0 ||
            x_values.cols() <= 0 ||
            mapping_authority->network_id != occurrence_axis->network_id() ||
            x_values.rows() != r_values.rows() ||
            x_values.cols() != r_values.cols() ||
            static_cast<std::size_t>(x_values.rows()) !=
                occurrence_axis->occurrence_count() ||
            static_cast<std::size_t>(x_values.cols()) != detectors.size()) {
            throw std::invalid_argument(
                "native paired readout identity or x/r shape is incomplete");
        }

        const auto rows = static_cast<std::size_t>(x_values.rows());
        const auto columns = static_cast<std::size_t>(x_values.cols());
        if (rows > std::numeric_limits<std::size_t>::max() / columns) {
            throw std::length_error(
                "native paired readout cardinality would overflow");
        }
        const auto cell_count = rows * columns;
        if (x_states.size() != cell_count || r_states.size() != cell_count) {
            throw std::invalid_argument(
                "native paired readout states do not match x/r shape");
        }

        std::set<std::pair<std::string, std::string>>
            detector_occurrences;
        std::set<std::string> channel_ids;
        for (std::size_t index = 0; index < detectors.size(); ++index) {
            const auto &detector = detectors[index];
            if (!detector.complete() ||
                detector.network_id != occurrence_axis->network_id() ||
                detector.storage_column !=
                    static_cast<Eigen::Index>(index) ||
                !detector_occurrences
                     .emplace(detector.detector_association_record_id,
                              detector.detector_occurrence_id)
                     .second ||
                !channel_ids.insert(detector.tone_or_channel_id).second) {
                throw std::invalid_argument(
                    "native paired readout detector axis is invalid or ambiguous");
            }
        }

        for (Eigen::Index row = 0; row < x_values.rows(); ++row) {
            for (Eigen::Index column = 0; column < x_values.cols(); ++column) {
                const auto index = static_cast<std::size_t>(
                    row * x_values.cols() + column);
                if (!x_states[index].consistent_with(x_values(row, column)) ||
                    !r_states[index].consistent_with(r_values(row, column))) {
                    throw std::invalid_argument(
                        "native paired readout facts disagree with payload finiteness");
                }
            }
        }

        return NativePairedReadoutNetwork{
            std::move(occurrence_axis), std::move(detectors),
            std::move(mapping_authority), std::move(x_values),
            std::move(r_values), std::move(x_states), std::move(r_states)};
    }

    TimestreamNetworkId network_id() const noexcept {
        return occurrence_axis_->network_id();
    }
    Eigen::Index occurrence_count() const noexcept {
        return x_values_.rows();
    }
    Eigen::Index detector_count() const noexcept {
        return x_values_.cols();
    }
    std::size_t cell_count() const noexcept {
        return static_cast<std::size_t>(x_values_.size());
    }
    const NativePairedReadoutOccurrenceAxis &occurrence_axis() const noexcept {
        return *occurrence_axis_;
    }
    const std::shared_ptr<const NativePairedReadoutOccurrenceAxis> &
    occurrence_axis_handle() const noexcept {
        return occurrence_axis_;
    }
    const NativeReadoutMappingAuthority &mapping_authority() const noexcept {
        return *mapping_authority_;
    }
    const std::shared_ptr<const NativeReadoutMappingAuthority> &
    mapping_authority_handle() const noexcept {
        return mapping_authority_;
    }
    const std::vector<NativeReadoutDetectorBinding> &detectors() const
        noexcept {
        return detectors_;
    }
    const NativeReadoutDetectorBinding &detector(
        Eigen::Index storage_column) const {
        return detectors_.at(checked_column(storage_column));
    }
    const NativePairedReadoutMatrix &values(
        NativeReadoutCoordinate coordinate) const noexcept {
        return coordinate == NativeReadoutCoordinate::x ? x_values_ :
                                                          r_values_;
    }
    std::span<const NativeReadoutCoordinateState> states(
        NativeReadoutCoordinate coordinate) const noexcept {
        return coordinate == NativeReadoutCoordinate::x ?
            std::span<const NativeReadoutCoordinateState>{x_states_} :
            std::span<const NativeReadoutCoordinateState>{r_states_};
    }
    double value(NativeReadoutCoordinate coordinate,
                 TimestreamNativeRow native_row,
                 Eigen::Index storage_column) const {
        return values(coordinate)(checked_row(native_row),
                                  checked_column(storage_column));
    }
    const NativeReadoutCoordinateState &state(
        NativeReadoutCoordinate coordinate,
        TimestreamNativeRow native_row,
        Eigen::Index storage_column) const {
        const auto index = flat_index(native_row, storage_column);
        return coordinate == NativeReadoutCoordinate::x ?
            x_states_.at(index) : r_states_.at(index);
    }
    bool pair_available(TimestreamNativeRow native_row,
                        Eigen::Index storage_column) const {
        return state(NativeReadoutCoordinate::x, native_row, storage_column)
                   .payload_available() &&
               state(NativeReadoutCoordinate::r, native_row, storage_column)
                   .payload_available();
    }
    bool pair_valid(TimestreamNativeRow native_row,
                    Eigen::Index storage_column) const {
        return state(NativeReadoutCoordinate::x, native_row, storage_column)
                   .valid() &&
               state(NativeReadoutCoordinate::r, native_row, storage_column)
                   .valid();
    }
    NativePairedReadoutCause pair_causes(
        TimestreamNativeRow native_row,
        Eigen::Index storage_column) const {
        const auto &x = state(
            NativeReadoutCoordinate::x, native_row, storage_column);
        const auto &r = state(
            NativeReadoutCoordinate::r, native_row, storage_column);
        NativePairedReadoutCause result = NativePairedReadoutCause::none;
        if (!x.payload_available()) {
            result = result | NativePairedReadoutCause::x_unavailable;
        }
        if (!r.payload_available()) {
            result = result | NativePairedReadoutCause::r_unavailable;
        }
        if (!x.producer_valid()) {
            result = result | NativePairedReadoutCause::x_producer_invalid;
        }
        if (!r.producer_valid()) {
            result = result | NativePairedReadoutCause::r_producer_invalid;
        }
        if (!x.finite_payload()) {
            result = result | NativePairedReadoutCause::x_nonfinite;
        }
        if (!r.finite_payload()) {
            result = result | NativePairedReadoutCause::r_nonfinite;
        }
        if (!x.in_acquisition_support()) {
            result = result | NativePairedReadoutCause::x_outside_support;
        }
        if (!r.in_acquisition_support()) {
            result = result | NativePairedReadoutCause::r_outside_support;
        }
        return result;
    }

    NativePairedReadoutCardinality cardinality() const noexcept {
        return NativePairedReadoutCardinality{
            1U, static_cast<std::size_t>(detector_count()),
            static_cast<std::size_t>(occurrence_count()), cell_count()};
    }
    NativePairedReadoutMemoryEvidence memory_evidence() const noexcept {
        NativePairedReadoutMemoryEvidence result;
        result.numeric_payload_bytes =
            2U * cell_count() * sizeof(double);
        result.coordinate_state_bytes =
            2U * cell_count() * sizeof(NativeReadoutCoordinateState);
        result.occurrence_axis_bytes =
            occurrence_axis_->occurrence_count() *
            sizeof(NativePairedReadoutOccurrenceBinding);
        result.detector_axis_bytes =
            detectors_.size() * sizeof(NativeReadoutDetectorBinding);
        result.identity_text_bytes = mapping_authority_->text_bytes();
        for (const auto &detector : detectors_) {
            result.identity_text_bytes += detector.text_bytes();
        }
        result.referenced_native_axis_count = 1U;
        return result;
    }

private:
    NativePairedReadoutNetwork(
        std::shared_ptr<const NativePairedReadoutOccurrenceAxis>
            occurrence_axis,
        std::vector<NativeReadoutDetectorBinding> detectors,
        std::shared_ptr<const NativeReadoutMappingAuthority>
            mapping_authority,
        NativePairedReadoutMatrix x_values,
        NativePairedReadoutMatrix r_values,
        std::vector<NativeReadoutCoordinateState> x_states,
        std::vector<NativeReadoutCoordinateState> r_states)
        : occurrence_axis_{std::move(occurrence_axis)},
          detectors_{std::move(detectors)},
          mapping_authority_{std::move(mapping_authority)},
          x_values_{std::move(x_values)}, r_values_{std::move(r_values)},
          x_states_{std::move(x_states)}, r_states_{std::move(r_states)} {}

    Eigen::Index checked_row(TimestreamNativeRow native_row) const {
        if (native_row < occurrence_axis_->first_native_row() ||
            native_row >= occurrence_axis_->past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside the paired readout network");
        }
        return static_cast<Eigen::Index>(
            native_row - occurrence_axis_->first_native_row());
    }
    std::size_t checked_column(Eigen::Index storage_column) const {
        if (storage_column < 0 || storage_column >= detector_count()) {
            throw std::out_of_range(
                "detector column is outside the paired readout network");
        }
        return static_cast<std::size_t>(storage_column);
    }
    std::size_t flat_index(TimestreamNativeRow native_row,
                           Eigen::Index storage_column) const {
        return static_cast<std::size_t>(
            checked_row(native_row) * detector_count() +
            static_cast<Eigen::Index>(checked_column(storage_column)));
    }

    std::shared_ptr<const NativePairedReadoutOccurrenceAxis> occurrence_axis_;
    std::vector<NativeReadoutDetectorBinding> detectors_;
    std::shared_ptr<const NativeReadoutMappingAuthority> mapping_authority_;
    NativePairedReadoutMatrix x_values_;
    NativePairedReadoutMatrix r_values_;
    std::vector<NativeReadoutCoordinateState> x_states_;
    std::vector<NativeReadoutCoordinateState> r_states_;
};

class NativePairedReadoutObservation {
public:
    NativePairedReadoutObservation(
        const NativePairedReadoutObservation &) = delete;
    NativePairedReadoutObservation &operator=(
        const NativePairedReadoutObservation &) = delete;
    NativePairedReadoutObservation(
        NativePairedReadoutObservation &&) noexcept = default;
    NativePairedReadoutObservation &operator=(
        NativePairedReadoutObservation &&) noexcept = default;

    static NativePairedReadoutObservation admit(
        NativeObservationScope scope,
        std::vector<TimestreamNetworkId> required_network_ids,
        std::vector<NativePairedReadoutNetwork> networks) {
        if (required_network_ids.empty() || networks.empty()) {
            throw std::invalid_argument(
                "native paired readout observation requires networks");
        }
        std::sort(required_network_ids.begin(), required_network_ids.end());
        if (required_network_ids.front() < 0 ||
            std::adjacent_find(required_network_ids.begin(),
                               required_network_ids.end()) !=
                required_network_ids.end()) {
            throw std::invalid_argument(
                "required native paired readout network inventory is invalid");
        }
        std::sort(networks.begin(), networks.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.network_id() < rhs.network_id();
                  });
        if (networks.size() != required_network_ids.size()) {
            throw std::invalid_argument(
                "native paired readout network inventory is incomplete");
        }

        std::set<std::pair<std::string, std::string>> detector_identities;
        for (std::size_t index = 0; index < networks.size(); ++index) {
            if (networks[index].network_id() != required_network_ids[index]) {
                throw std::invalid_argument(
                    "native paired readout network inventory differs from required participants");
            }
            for (const auto &detector : networks[index].detectors()) {
                if (!detector_identities
                         .emplace(detector.detector_association_record_id,
                                  detector.detector_occurrence_id)
                         .second) {
                    throw std::invalid_argument(
                        "detector occurrence repeats across the observation");
                }
            }
        }
        return NativePairedReadoutObservation{
            scope, std::move(required_network_ids), std::move(networks)};
    }

    const NativeObservationScope &scope() const noexcept { return scope_; }
    const std::vector<TimestreamNetworkId> &participant_network_ids() const
        noexcept {
        return participant_network_ids_;
    }
    std::size_t network_count() const noexcept { return networks_.size(); }
    const NativePairedReadoutNetwork &network(
        TimestreamNetworkId network_id) const {
        const auto found = std::lower_bound(
            networks_.begin(), networks_.end(), network_id,
            [](const auto &candidate, TimestreamNetworkId requested) {
                return candidate.network_id() < requested;
            });
        if (found == networks_.end() || found->network_id() != network_id) {
            throw std::out_of_range(
                "network is absent from the native paired readout observation");
        }
        return *found;
    }
    NativePairedReadoutCardinality cardinality() const noexcept {
        NativePairedReadoutCardinality result;
        result.network_count = networks_.size();
        for (const auto &network : networks_) {
            const auto value = network.cardinality();
            result.detector_count += value.detector_count;
            result.native_occurrence_count += value.native_occurrence_count;
            result.detector_occurrence_count +=
                value.detector_occurrence_count;
        }
        return result;
    }
    NativePairedReadoutMemoryEvidence memory_evidence() const noexcept {
        NativePairedReadoutMemoryEvidence result;
        for (const auto &network : networks_) {
            const auto value = network.memory_evidence();
            result.numeric_payload_bytes += value.numeric_payload_bytes;
            result.coordinate_state_bytes += value.coordinate_state_bytes;
            result.occurrence_axis_bytes += value.occurrence_axis_bytes;
            result.detector_axis_bytes += value.detector_axis_bytes;
            result.identity_text_bytes += value.identity_text_bytes;
            result.referenced_native_axis_count +=
                value.referenced_native_axis_count;
        }
        return result;
    }

private:
    NativePairedReadoutObservation(
        NativeObservationScope scope,
        std::vector<TimestreamNetworkId> participant_network_ids,
        std::vector<NativePairedReadoutNetwork> networks)
        : scope_{scope},
          participant_network_ids_{std::move(participant_network_ids)},
          networks_{std::move(networks)} {}

    NativeObservationScope scope_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::vector<NativePairedReadoutNetwork> networks_;
};

}  // namespace citlali::pipeline
