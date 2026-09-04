#pragma once

#include <citlali/core/pipeline/timestream_val_state.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

// D2 names the application route and profile that realized the residual.  It
// does not interpret either value or infer a route from configuration state.
class D2RouteProfileIdentity {
public:
    static D2RouteProfileIdentity admit(std::string route_id,
                                        std::string profile_id,
                                        std::string profile_revision_id) {
        if (route_id.empty() || profile_id.empty() ||
            profile_revision_id.empty()) {
            throw std::invalid_argument(
                "D2 route/profile identity is incomplete");
        }
        return D2RouteProfileIdentity{
            std::move(route_id), std::move(profile_id),
            std::move(profile_revision_id)};
    }

    const std::string &route_id() const noexcept { return route_id_; }
    const std::string &profile_id() const noexcept { return profile_id_; }
    const std::string &profile_revision_id() const noexcept {
        return profile_revision_id_;
    }
    std::size_t text_bytes() const noexcept {
        return route_id_.size() + profile_id_.size() +
               profile_revision_id_.size();
    }

    friend bool operator==(const D2RouteProfileIdentity &,
                           const D2RouteProfileIdentity &) = default;

private:
    D2RouteProfileIdentity(std::string route_id, std::string profile_id,
                           std::string profile_revision_id)
        : route_id_{std::move(route_id)},
          profile_id_{std::move(profile_id)},
          profile_revision_id_{std::move(profile_revision_id)} {}

    std::string route_id_;
    std::string profile_id_;
    std::string profile_revision_id_;
};

// These identifiers describe the already-realized residual operation.  They
// carry provenance only; D2 does not design or execute the operation.
class D2ResidualRealizationIdentity {
public:
    static D2ResidualRealizationIdentity admit(
        std::string realization_id, std::string operator_id,
        std::string effective_config_id, std::string grouping_id) {
        if (realization_id.empty() || operator_id.empty() ||
            effective_config_id.empty() || grouping_id.empty()) {
            throw std::invalid_argument(
                "D2 residual realization identity is incomplete");
        }
        return D2ResidualRealizationIdentity{
            std::move(realization_id), std::move(operator_id),
            std::move(effective_config_id), std::move(grouping_id)};
    }

    const std::string &realization_id() const noexcept {
        return realization_id_;
    }
    const std::string &operator_id() const noexcept { return operator_id_; }
    const std::string &effective_config_id() const noexcept {
        return effective_config_id_;
    }
    const std::string &grouping_id() const noexcept { return grouping_id_; }
    std::size_t text_bytes() const noexcept {
        return realization_id_.size() + operator_id_.size() +
               effective_config_id_.size() + grouping_id_.size();
    }

    friend bool operator==(const D2ResidualRealizationIdentity &,
                           const D2ResidualRealizationIdentity &) = default;

private:
    D2ResidualRealizationIdentity(
        std::string realization_id, std::string operator_id,
        std::string effective_config_id, std::string grouping_id)
        : realization_id_{std::move(realization_id)},
          operator_id_{std::move(operator_id)},
          effective_config_id_{std::move(effective_config_id)},
          grouping_id_{std::move(grouping_id)} {}

    std::string realization_id_;
    std::string operator_id_;
    std::string effective_config_id_;
    std::string grouping_id_;
};

enum class D2ResidualPayloadState : std::uint8_t {
    absent,
    present_structurally_complete,
};

// This carrier reports only whether derived storage exists and has a complete
// mechanical shape.  In particular, it does not reject non-finite values and
// contains no D2-local validity or cause state.
class D2ResidualCoordinatePayload {
public:
    D2ResidualCoordinatePayload(const D2ResidualCoordinatePayload &) =
        delete;
    D2ResidualCoordinatePayload &operator=(
        const D2ResidualCoordinatePayload &) = delete;
    D2ResidualCoordinatePayload(D2ResidualCoordinatePayload &&) noexcept =
        default;
    D2ResidualCoordinatePayload &operator=(
        D2ResidualCoordinatePayload &&) noexcept = default;

    static D2ResidualCoordinatePayload absent() {
        return D2ResidualCoordinatePayload{
            std::optional<NativePairedReadoutMatrix>{}};
    }
    static D2ResidualCoordinatePayload complete(
        NativePairedReadoutMatrix values) {
        if (values.rows() <= 0 || values.cols() <= 0) {
            throw std::invalid_argument(
                "present D2 residual payload requires positive dimensions");
        }
        return D2ResidualCoordinatePayload{
            std::optional<NativePairedReadoutMatrix>{std::move(values)}};
    }

    D2ResidualPayloadState state() const noexcept {
        return values_ ?
            D2ResidualPayloadState::present_structurally_complete :
            D2ResidualPayloadState::absent;
    }
    bool present() const noexcept { return values_.has_value(); }
    bool structurally_complete() const noexcept { return values_.has_value(); }
    const NativePairedReadoutMatrix &values() const {
        if (!values_) {
            throw std::logic_error("D2 residual payload is absent");
        }
        return *values_;
    }
    std::span<const double> contiguous_values() const noexcept {
        return values_ ?
            std::span<const double>{values_->data(),
                                    static_cast<std::size_t>(values_->size())} :
            std::span<const double>{};
    }
    std::size_t owned_numeric_bytes() const noexcept {
        return contiguous_values().size_bytes();
    }

private:
    explicit D2ResidualCoordinatePayload(
        std::optional<NativePairedReadoutMatrix> values)
        : values_{std::move(values)} {}

    std::optional<NativePairedReadoutMatrix> values_;
};

enum class D2SamplingRelation : std::uint8_t {
    native_occurrence_axis_unchanged,
};

enum class D2GridRelation : std::uint8_t {
    network_native_detector_axis_unchanged,
};

// One residual realization binds the exact immutable parent and VAL snapshot
// that applied when the payload was realized.  The generation is descriptive;
// only exact in-memory handle equality admits a publication in this milestone.
class D2ResidualNetworkPayload {
public:
    D2ResidualNetworkPayload(const D2ResidualNetworkPayload &) = delete;
    D2ResidualNetworkPayload &operator=(
        const D2ResidualNetworkPayload &) = delete;
    D2ResidualNetworkPayload(D2ResidualNetworkPayload &&) noexcept = default;
    D2ResidualNetworkPayload &operator=(
        D2ResidualNetworkPayload &&) noexcept = default;

    static D2ResidualNetworkPayload realize(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        D2RouteProfileIdentity route_profile,
        std::shared_ptr<const ValSnapshot> val_snapshot,
        D2ResidualRealizationIdentity realization_identity,
        TimestreamNetworkId network_id,
        D2ResidualCoordinatePayload x_residual,
        D2ResidualCoordinatePayload r_residual) {
        if (!parent || !val_snapshot ||
            val_snapshot->paired_handle().get() != parent.get() ||
            val_snapshot->scope() != parent->scope()) {
            throw std::invalid_argument(
                "D2 residual realization requires the exact parent-bound VAL snapshot");
        }
        const auto &network = parent->network(network_id);
        require_shape(network, x_residual);
        require_shape(network, r_residual);
        return D2ResidualNetworkPayload{
            std::move(parent), std::move(route_profile),
            std::move(val_snapshot), std::move(realization_identity),
            network_id,
            std::move(x_residual), std::move(r_residual)};
    }

    const std::shared_ptr<const NativePairedReadoutObservation> &
    parent_handle() const noexcept {
        return parent_;
    }
    const D2RouteProfileIdentity &route_profile() const noexcept {
        return route_profile_;
    }
    const std::shared_ptr<const ValSnapshot> &val_snapshot_handle()
        const noexcept {
        return val_snapshot_;
    }
    ValGeneration val_generation() const noexcept {
        return val_generation_;
    }
    const D2ResidualRealizationIdentity &realization_identity()
        const noexcept {
        return realization_identity_;
    }
    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    const std::shared_ptr<const NativePairedReadoutOccurrenceAxis> &
    occurrence_axis_handle() const noexcept {
        return network().occurrence_axis_handle();
    }
    const std::vector<NativeReadoutDetectorBinding> &detectors() const noexcept {
        return network().detectors();
    }
    std::vector<NativeContiguousRun> contiguous_runs() const {
        return network().occurrence_axis().contiguous_runs();
    }
    const D2ResidualCoordinatePayload &residual(
        NativeReadoutCoordinate coordinate) const noexcept {
        return coordinate == NativeReadoutCoordinate::x ? x_residual_ :
                                                          r_residual_;
    }
    const NativePairedReadoutMatrix &prefilter_values(
        NativeReadoutCoordinate coordinate) const noexcept {
        return network().values(coordinate);
    }
    D2SamplingRelation sampling_relation() const noexcept {
        return D2SamplingRelation::native_occurrence_axis_unchanged;
    }
    D2GridRelation grid_relation() const noexcept {
        return D2GridRelation::network_native_detector_axis_unchanged;
    }
    std::size_t owned_numeric_bytes() const noexcept {
        return x_residual_.owned_numeric_bytes() +
               r_residual_.owned_numeric_bytes();
    }

private:
    D2ResidualNetworkPayload(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        D2RouteProfileIdentity route_profile,
        std::shared_ptr<const ValSnapshot> val_snapshot,
        D2ResidualRealizationIdentity realization_identity,
        TimestreamNetworkId network_id,
        D2ResidualCoordinatePayload x_residual,
        D2ResidualCoordinatePayload r_residual)
        : parent_{std::move(parent)},
          route_profile_{std::move(route_profile)},
          val_snapshot_{std::move(val_snapshot)},
          val_generation_{val_snapshot_->generation()},
          realization_identity_{std::move(realization_identity)},
          network_id_{network_id},
          x_residual_{std::move(x_residual)},
          r_residual_{std::move(r_residual)} {}

    static void require_shape(
        const NativePairedReadoutNetwork &network,
        const D2ResidualCoordinatePayload &payload) {
        if (payload.present() &&
            (payload.values().rows() != network.occurrence_count() ||
             payload.values().cols() != network.detector_count())) {
            throw std::invalid_argument(
                "D2 residual payload differs from the exact native axes");
        }
    }
    const NativePairedReadoutNetwork &network() const {
        return parent_->network(network_id_);
    }

    std::shared_ptr<const NativePairedReadoutObservation> parent_;
    D2RouteProfileIdentity route_profile_;
    std::shared_ptr<const ValSnapshot> val_snapshot_;
    ValGeneration val_generation_;
    D2ResidualRealizationIdentity realization_identity_;
    TimestreamNetworkId network_id_;
    D2ResidualCoordinatePayload x_residual_;
    D2ResidualCoordinatePayload r_residual_;
};

enum class D2SourceMaskDisposition : std::uint8_t {
    applied,
    approved_not_applicable,
};

// Source-mask evidence is processing evidence only.  A marked cell means that
// the named source-mask operation excluded that cell from its processing
// support; D2 does not translate the mark into sample invalidity.
class D2SourceMaskEvidence {
public:
    static std::shared_ptr<const D2SourceMaskEvidence> admit(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        D2RouteProfileIdentity route_profile,
        TimestreamNetworkId network_id,
        std::string evidence_id,
        D2SourceMaskDisposition disposition,
        std::vector<std::uint8_t> excluded_from_processing) {
        if (!parent || evidence_id.empty()) {
            throw std::invalid_argument(
                "D2 source-mask evidence identity is incomplete");
        }
        const auto &network = parent->network(network_id);
        if (disposition == D2SourceMaskDisposition::applied) {
            if (excluded_from_processing.size() != network.cell_count() ||
                std::any_of(excluded_from_processing.begin(),
                            excluded_from_processing.end(),
                            [](std::uint8_t value) { return value > 1U; })) {
                throw std::invalid_argument(
                    "applied D2 source-mask evidence must cover the exact "
                    "native cells");
            }
        } else if (!excluded_from_processing.empty()) {
            throw std::invalid_argument(
                "not-applicable D2 source-mask evidence cannot carry mask values");
        }
        return std::shared_ptr<const D2SourceMaskEvidence>(
            new D2SourceMaskEvidence{
                std::move(parent), std::move(route_profile), network_id,
                std::move(evidence_id), disposition,
                std::move(excluded_from_processing)});
    }

    const std::shared_ptr<const NativePairedReadoutObservation> &
    parent_handle() const noexcept {
        return parent_;
    }
    const D2RouteProfileIdentity &route_profile() const noexcept {
        return route_profile_;
    }
    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    const std::string &evidence_id() const noexcept { return evidence_id_; }
    D2SourceMaskDisposition disposition() const noexcept {
        return disposition_;
    }
    const std::shared_ptr<const NativePairedReadoutOccurrenceAxis> &
    occurrence_axis_handle() const noexcept {
        return network().occurrence_axis_handle();
    }
    const std::vector<NativeReadoutDetectorBinding> &detectors() const noexcept {
        return network().detectors();
    }
    std::vector<NativeContiguousRun> contiguous_runs() const {
        return network().occurrence_axis().contiguous_runs();
    }
    std::span<const std::uint8_t> processing_exclusions() const noexcept {
        return excluded_from_processing_;
    }
    bool excluded_from_processing(TimestreamNativeRow native_row,
                                  Eigen::Index storage_column) const {
        if (disposition_ != D2SourceMaskDisposition::applied) {
            throw std::logic_error("D2 source mask was not applied");
        }
        const auto &axis = network().occurrence_axis();
        if (native_row < axis.first_native_row() ||
            native_row >= axis.past_last_native_row() ||
            storage_column < 0 ||
            storage_column >= network().detector_count()) {
            throw std::out_of_range(
                "D2 source-mask coordinate is outside the native axes");
        }
        const auto row = static_cast<std::size_t>(
            native_row - axis.first_native_row());
        const auto column = static_cast<std::size_t>(storage_column);
        return excluded_from_processing_.at(
                   row * static_cast<std::size_t>(network().detector_count()) +
                   column) != 0;
    }
    std::size_t owned_evidence_bytes() const noexcept {
        return excluded_from_processing_.size() + evidence_id_.size() +
               route_profile_.text_bytes();
    }

private:
    D2SourceMaskEvidence(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        D2RouteProfileIdentity route_profile,
        TimestreamNetworkId network_id, std::string evidence_id,
        D2SourceMaskDisposition disposition,
        std::vector<std::uint8_t> excluded_from_processing)
        : parent_{std::move(parent)},
          route_profile_{std::move(route_profile)},
          network_id_{network_id}, evidence_id_{std::move(evidence_id)},
          disposition_{disposition},
          excluded_from_processing_{std::move(excluded_from_processing)} {}

    const NativePairedReadoutNetwork &network() const {
        return parent_->network(network_id_);
    }

    std::shared_ptr<const NativePairedReadoutObservation> parent_;
    D2RouteProfileIdentity route_profile_;
    TimestreamNetworkId network_id_;
    std::string evidence_id_;
    D2SourceMaskDisposition disposition_;
    std::vector<std::uint8_t> excluded_from_processing_;
};

enum class D2LineOperatorDisposition : std::uint8_t {
    pending,
    complete_no_lines,
    applied,
};

struct D2LineOperatorRecord {
    std::string line_id;
    double low_frequency_hz = 0.0;
    double high_frequency_hz = 0.0;
    bool effective_before_decimation = false;
    std::string operator_evidence_id;

    std::size_t text_bytes() const noexcept {
        return line_id.size() + operator_evidence_id.size();
    }
};

// Line-operator evidence records what processing was effective on the exact
// network-native support.  It remains separate from VAL and carries no sample
// validity consequence.
class D2LineOperatorEvidence {
public:
    static std::shared_ptr<const D2LineOperatorEvidence> admit(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        D2RouteProfileIdentity route_profile,
        TimestreamNetworkId network_id,
        std::string evidence_id,
        D2LineOperatorDisposition disposition,
        std::vector<D2LineOperatorRecord> records) {
        if (!parent || evidence_id.empty()) {
            throw std::invalid_argument(
                "D2 line-operator evidence identity is incomplete");
        }
        (void)parent->network(network_id);
        if (disposition != D2LineOperatorDisposition::applied) {
            if (!records.empty()) {
                throw std::invalid_argument(
                    "pending or no-lines D2 evidence cannot carry line records");
            }
        } else {
            if (records.empty()) {
                throw std::invalid_argument(
                    "applied D2 line evidence requires operator records");
            }
            std::sort(records.begin(), records.end(),
                      [](const auto &lhs, const auto &rhs) {
                          if (lhs.low_frequency_hz != rhs.low_frequency_hz) {
                              return lhs.low_frequency_hz <
                                     rhs.low_frequency_hz;
                          }
                          return lhs.high_frequency_hz <
                                 rhs.high_frequency_hz;
                      });
            std::set<std::string> line_ids;
            for (std::size_t index = 0; index < records.size(); ++index) {
                const auto &record = records[index];
                if (record.line_id.empty() ||
                    record.operator_evidence_id.empty() ||
                    !record.effective_before_decimation ||
                    !std::isfinite(record.low_frequency_hz) ||
                    !std::isfinite(record.high_frequency_hz) ||
                    record.low_frequency_hz < 0.0 ||
                    !(record.low_frequency_hz < record.high_frequency_hz) ||
                    (index > 0 &&
                     records[index - 1].high_frequency_hz >
                         record.low_frequency_hz) ||
                    !line_ids.insert(record.line_id).second) {
                    throw std::invalid_argument(
                        "applied D2 line-operator evidence is incomplete or ambiguous");
                }
            }
        }
        return std::shared_ptr<const D2LineOperatorEvidence>(
            new D2LineOperatorEvidence{
                std::move(parent), std::move(route_profile), network_id,
                std::move(evidence_id), disposition, std::move(records)});
    }

    const std::shared_ptr<const NativePairedReadoutObservation> &
    parent_handle() const noexcept {
        return parent_;
    }
    const D2RouteProfileIdentity &route_profile() const noexcept {
        return route_profile_;
    }
    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    const std::string &evidence_id() const noexcept { return evidence_id_; }
    D2LineOperatorDisposition disposition() const noexcept {
        return disposition_;
    }
    const std::shared_ptr<const NativePairedReadoutOccurrenceAxis> &
    occurrence_axis_handle() const noexcept {
        return network().occurrence_axis_handle();
    }
    std::vector<NativeContiguousRun> contiguous_runs() const {
        return network().occurrence_axis().contiguous_runs();
    }
    std::span<const D2LineOperatorRecord> records() const noexcept {
        return records_;
    }
    std::size_t owned_evidence_bytes() const noexcept {
        std::size_t result = evidence_id_.size() + route_profile_.text_bytes() +
            records_.size() * sizeof(D2LineOperatorRecord);
        for (const auto &record : records_) result += record.text_bytes();
        return result;
    }

private:
    D2LineOperatorEvidence(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        D2RouteProfileIdentity route_profile,
        TimestreamNetworkId network_id, std::string evidence_id,
        D2LineOperatorDisposition disposition,
        std::vector<D2LineOperatorRecord> records)
        : parent_{std::move(parent)},
          route_profile_{std::move(route_profile)},
          network_id_{network_id}, evidence_id_{std::move(evidence_id)},
          disposition_{disposition}, records_{std::move(records)} {}

    const NativePairedReadoutNetwork &network() const {
        return parent_->network(network_id_);
    }

    std::shared_ptr<const NativePairedReadoutObservation> parent_;
    D2RouteProfileIdentity route_profile_;
    TimestreamNetworkId network_id_;
    std::string evidence_id_;
    D2LineOperatorDisposition disposition_;
    std::vector<D2LineOperatorRecord> records_;
};

// The snapshot is an explicit peer of the residual payloads in one publication
// input.  Publication never asks VAL for a later or ambient "current" state.
class D2NativeMeasurementPublicationInput {
public:
    D2NativeMeasurementPublicationInput(
        std::shared_ptr<const NativePairedReadoutObservation> parent,
        D2RouteProfileIdentity route_profile,
        std::shared_ptr<const ValSnapshot> val_snapshot,
        D2ResidualRealizationIdentity residual_realization,
        std::vector<D2ResidualNetworkPayload> residuals,
        std::vector<std::shared_ptr<const D2SourceMaskEvidence>> source_masks,
        std::vector<std::shared_ptr<const D2LineOperatorEvidence>>
            line_operators)
        : parent{std::move(parent)},
          route_profile{std::move(route_profile)},
          val_snapshot{std::move(val_snapshot)},
          residual_realization{std::move(residual_realization)},
          residuals{std::move(residuals)},
          source_masks{std::move(source_masks)},
          line_operators{std::move(line_operators)} {}

    std::shared_ptr<const NativePairedReadoutObservation> parent;
    D2RouteProfileIdentity route_profile;
    std::shared_ptr<const ValSnapshot> val_snapshot;
    D2ResidualRealizationIdentity residual_realization;
    std::vector<D2ResidualNetworkPayload> residuals;
    std::vector<std::shared_ptr<const D2SourceMaskEvidence>> source_masks;
    std::vector<std::shared_ptr<const D2LineOperatorEvidence>> line_operators;
};

struct D2NativeMeasurementMemoryEvidence {
    std::size_t residual_numeric_bytes = 0;
    std::size_t identity_text_bytes = 0;
    std::size_t referenced_processing_evidence_bytes = 0;
    std::size_t validation_state_bytes = 0;
    std::size_t referenced_paired_product_count = 0;
    std::size_t referenced_val_snapshot_count = 0;
    std::size_t referenced_native_axis_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return residual_numeric_bytes + identity_text_bytes;
    }
};

// D2 owns the derived x/r payload and binds the immutable VAL-authored
// snapshot applicable to that measurement.  It performs no scientific
// validation, owns no validation policy, and deliberately offers no x/r
// usability-mask API while VAL lacks typed coordinate targeting.
class D2NativeMeasurement {
public:
    D2NativeMeasurement(const D2NativeMeasurement &) = delete;
    D2NativeMeasurement &operator=(const D2NativeMeasurement &) = delete;

    static std::shared_ptr<const D2NativeMeasurement> publish(
        D2NativeMeasurementPublicationInput input) {
        if (!input.parent || !input.val_snapshot ||
            input.val_snapshot->paired_handle().get() != input.parent.get() ||
            input.val_snapshot->scope() != input.parent->scope()) {
            throw std::invalid_argument(
                "D2 publication identity or VAL binding is incomplete");
        }
        const auto expected_count = input.parent->network_count();
        if (input.residuals.size() != expected_count ||
            input.source_masks.size() != expected_count ||
            input.line_operators.size() != expected_count) {
            throw std::invalid_argument(
                "D2 publication requires complete network evidence inventories");
        }
        if (std::any_of(input.source_masks.begin(), input.source_masks.end(),
                        [](const auto &handle) { return !handle; }) ||
            std::any_of(input.line_operators.begin(),
                        input.line_operators.end(),
                        [](const auto &handle) { return !handle; })) {
            throw std::invalid_argument(
                "D2 publication evidence handles cannot be null");
        }

        std::sort(input.residuals.begin(), input.residuals.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.network_id() < rhs.network_id();
                  });
        std::sort(input.source_masks.begin(), input.source_masks.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs->network_id() < rhs->network_id();
                  });
        std::sort(input.line_operators.begin(), input.line_operators.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs->network_id() < rhs->network_id();
                  });

        const auto &network_ids = input.parent->participant_network_ids();
        for (std::size_t index = 0; index < expected_count; ++index) {
            const auto expected_id = network_ids[index];
            const auto &residual = input.residuals[index];
            const auto &source_mask = input.source_masks[index];
            const auto &line_operator = input.line_operators[index];
            const auto expected_axis =
                input.parent->network(expected_id).occurrence_axis_handle();

            // Pointer equality is intentionally an in-memory admission rule.
            // A matching generation number alone is never sufficient.
            if (residual.network_id() != expected_id ||
                source_mask->network_id() != expected_id ||
                line_operator->network_id() != expected_id ||
                residual.parent_handle().get() != input.parent.get() ||
                source_mask->parent_handle().get() != input.parent.get() ||
                line_operator->parent_handle().get() != input.parent.get() ||
                residual.val_snapshot_handle().get() !=
                    input.val_snapshot.get() ||
                residual.route_profile() != input.route_profile ||
                residual.realization_identity() !=
                    input.residual_realization ||
                source_mask->route_profile() != input.route_profile ||
                line_operator->route_profile() != input.route_profile ||
                residual.occurrence_axis_handle().get() !=
                    expected_axis.get() ||
                source_mask->occurrence_axis_handle().get() !=
                    expected_axis.get() ||
                line_operator->occurrence_axis_handle().get() !=
                    expected_axis.get()) {
                throw std::invalid_argument(
                    "D2 publication evidence differs from its exact parent, "
                    "route, axes, or VAL snapshot");
            }
        }

        return std::shared_ptr<const D2NativeMeasurement>(
            new D2NativeMeasurement{std::move(input)});
    }

    const std::shared_ptr<const NativePairedReadoutObservation> &
    parent_handle() const noexcept {
        return parent_;
    }
    const NativeObservationScope &scope() const noexcept {
        return parent_->scope();
    }
    const D2RouteProfileIdentity &route_profile() const noexcept {
        return route_profile_;
    }
    const std::shared_ptr<const ValSnapshot> &val_snapshot_handle()
        const noexcept {
        return val_snapshot_;
    }
    ValGeneration val_generation() const noexcept {
        return val_generation_;
    }
    const D2ResidualRealizationIdentity &residual_realization()
        const noexcept {
        return residual_realization_;
    }
    std::size_t network_count() const noexcept { return residuals_.size(); }
    const D2ResidualNetworkPayload &network(
        TimestreamNetworkId network_id) const {
        const auto found = std::lower_bound(
            residuals_.begin(), residuals_.end(), network_id,
            [](const auto &candidate, TimestreamNetworkId requested) {
                return candidate.network_id() < requested;
            });
        if (found == residuals_.end() || found->network_id() != network_id) {
            throw std::out_of_range("network is absent from D2 measurement");
        }
        return *found;
    }
    const D2SourceMaskEvidence &source_mask(
        TimestreamNetworkId network_id) const {
        return *find_evidence(source_masks_, network_id, "source mask");
    }
    const D2LineOperatorEvidence &line_operator(
        TimestreamNetworkId network_id) const {
        return *find_evidence(line_operators_, network_id, "line operator");
    }
    const NativePairedReadoutMatrix &prefilter_values(
        TimestreamNetworkId network_id,
        NativeReadoutCoordinate coordinate) const {
        return parent_->network(network_id).values(coordinate);
    }
    D2NativeMeasurementMemoryEvidence memory_evidence() const noexcept {
        D2NativeMeasurementMemoryEvidence result;
        result.identity_text_bytes = route_profile_.text_bytes() +
                                     residual_realization_.text_bytes();
        for (const auto &residual : residuals_) {
            result.residual_numeric_bytes += residual.owned_numeric_bytes();
            result.identity_text_bytes +=
                residual.route_profile().text_bytes();
        }
        for (const auto &mask : source_masks_) {
            result.referenced_processing_evidence_bytes +=
                mask->owned_evidence_bytes();
        }
        for (const auto &line : line_operators_) {
            result.referenced_processing_evidence_bytes +=
                line->owned_evidence_bytes();
        }
        result.validation_state_bytes = 0;
        result.referenced_paired_product_count = 1;
        result.referenced_val_snapshot_count = 1;
        result.referenced_native_axis_count = residuals_.size();
        return result;
    }

private:
    explicit D2NativeMeasurement(D2NativeMeasurementPublicationInput input)
        : parent_{std::move(input.parent)},
          route_profile_{std::move(input.route_profile)},
          val_snapshot_{std::move(input.val_snapshot)},
          val_generation_{val_snapshot_->generation()},
          residual_realization_{std::move(input.residual_realization)},
          residuals_{std::move(input.residuals)},
          source_masks_{std::move(input.source_masks)},
          line_operators_{std::move(input.line_operators)} {}

    template <typename Evidence>
    static const std::shared_ptr<const Evidence> &find_evidence(
        const std::vector<std::shared_ptr<const Evidence>> &evidence,
        TimestreamNetworkId network_id, const char *kind) {
        const auto found = std::lower_bound(
            evidence.begin(), evidence.end(), network_id,
            [](const auto &candidate, TimestreamNetworkId requested) {
                return candidate->network_id() < requested;
            });
        if (found == evidence.end() || (*found)->network_id() != network_id) {
            throw std::out_of_range(std::string{"network is absent from D2 "} +
                                    kind + " evidence");
        }
        return *found;
    }

    std::shared_ptr<const NativePairedReadoutObservation> parent_;
    D2RouteProfileIdentity route_profile_;
    std::shared_ptr<const ValSnapshot> val_snapshot_;
    ValGeneration val_generation_;
    D2ResidualRealizationIdentity residual_realization_;
    std::vector<D2ResidualNetworkPayload> residuals_;
    std::vector<std::shared_ptr<const D2SourceMaskEvidence>> source_masks_;
    std::vector<std::shared_ptr<const D2LineOperatorEvidence>> line_operators_;
};

}  // namespace citlali::pipeline
