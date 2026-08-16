#pragma once

// Observation-owned SCI-ALIGN lineage.  This is a cold-boundary contract: it
// records which already-admitted native samples reached an existing map
// operation, but it neither changes numerical processing nor publishes a new
// product family.

#include <citlali/core/pipeline/apt_detector_relation.h>
#include <citlali/core/pipeline/timestream_native_consumer_bridge.h>
#include <citlali/core/pipeline/timestream_native_pointing.h>
#include <citlali/core/utils/sha256.h>

#include <Eigen/Core>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *native_cohort_product_provenance_schema_version =
    "citlali-native-cohort-product-provenance-v1";
inline constexpr const char *native_cohort_common_slot_semantics =
    "relational-coincidence-grouping-only";

inline std::string native_cohort_hex_double(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "native cohort provenance requires finite floating values");
    }
    std::ostringstream stream;
    stream << std::hexfloat << value;
    return stream.str();
}

class NativeCohortDigestBuilder {
public:
    void add(std::string_view label, std::string_view value) {
        digest_.update(label);
        digest_.update("=");
        digest_.update(std::to_string(value.size()));
        digest_.update(":");
        digest_.update(value);
        digest_.update("\n");
    }

    template <class Integer>
    void add_integer(std::string_view label, Integer value) {
        add(label, std::to_string(value));
    }

    void add_double(std::string_view label, double value) {
        add(label, native_cohort_hex_double(value));
    }

    std::string finish() {
        return "sha256:" + digest_.finish();
    }

private:
    citlali::utils::Sha256 digest_;
};

inline void native_cohort_add_artifact_identity(
    NativeCohortDigestBuilder &digest, std::string_view prefix,
    const apt_observation::ArtifactIdentity &identity) {
    digest.add(std::string(prefix) + ".schema", identity.schema);
    digest.add(std::string(prefix) + ".occurrence", identity.occurrence);
    digest.add(std::string(prefix) + ".semantic", identity.semantic_sha256);
    digest.add(std::string(prefix) + ".envelope", identity.envelope_sha256);
}

inline void native_cohort_add_observation_identity(
    NativeCohortDigestBuilder &digest, std::string_view prefix,
    const canonical_apt::ObservationIdentity &identity) {
    digest.add_integer(std::string(prefix) + ".observation",
                       identity.observation);
    digest.add_integer(std::string(prefix) + ".subobservation",
                       identity.subobservation);
    digest.add_integer(std::string(prefix) + ".scan", identity.scan);
}

inline std::string native_cohort_detector_relation_digest(
    const AptDetectorRelation &relation) {
    const auto &scope = relation.published_scope();
    if (scope.kind != PublishedAptKind::matched_observation ||
        scope.parent_content_revalidated || !scope.matched_target) {
        throw AptDetectorRelationError(
            "native cohort provenance requires a self-contained matched APT scope");
    }
    NativeCohortDigestBuilder digest;
    digest.add("schema", "citlali-native-cohort-detector-relation-v1");
    native_cohort_add_artifact_identity(digest, "artifact", scope.artifact);
    digest.add("transport.scope", scope.transport.scope);
    digest.add("transport.envelope", scope.transport.envelope_sha256);
    digest.add("transport.sha256", scope.transport.sha256);
    digest.add_integer("transport.bytes", scope.transport.byte_count);
    digest.add("receipt.sha256", scope.receipt_sha256);
    digest.add_integer("receipt.bytes", scope.receipt_byte_count);
    native_cohort_add_artifact_identity(
        digest, "baseline.artifact", scope.baseline_parent.artifact);
    digest.add("baseline.profile", scope.baseline_parent.profile);
    digest.add("baseline.descriptor", scope.baseline_parent.descriptor_sha256);
    digest.add("baseline.transport.scope", scope.baseline_parent.transport_scope);
    digest.add("baseline.transport.sha256", scope.baseline_parent.transport_sha256);
    digest.add_integer("baseline.transport.bytes", scope.baseline_parent.byte_count);
    digest.add("baseline.receipt.sha256", scope.baseline_parent.receipt_sha256);
    digest.add_integer("baseline.receipt.bytes", scope.baseline_parent.receipt_byte_count);
    native_cohort_add_artifact_identity(
        digest, "target.artifact", scope.matched_target->target_artifact);
    native_cohort_add_observation_identity(
        digest, "target.observation", scope.matched_target->observation);
    digest.add_integer("target.input.count",
                       scope.matched_target->ordered_inputs.size());
    for (const auto &input : scope.matched_target->ordered_inputs) {
        digest.add_integer("target.input.key", input.input_key);
        digest.add_integer("target.input.network", input.network);
        digest.add("target.input.interface", input.interface_name);
        digest.add_integer("target.input.channels", input.channel_count);
        digest.add_integer("target.input.raw-key", input.raw_source_key);
        digest.add("target.input.raw-sha256", input.raw_content_sha256);
        digest.add_integer("target.input.raw-bytes", input.raw_byte_count);
        native_cohort_add_observation_identity(
            digest, "target.input.raw-observation", input.raw_header_observation);
    }
    digest.add_integer("binding.count", relation.bindings().size());
    for (const auto &binding : relation.bindings()) {
        digest.add_integer("binding.column", binding.detector_column);
        digest.add_integer("binding.uid", binding.uid);
        digest.add_integer("binding.network", binding.network);
        digest.add_integer("binding.tone", binding.kids_tone);
        digest.add("binding.flag",
                   binding.flag ? std::to_string(*binding.flag) : "null");
    }
    return digest.finish();
}

inline std::string native_cohort_raw_manifest_digest(
    const AptDetectorRelation &relation) {
    const auto &target = relation.matched_target_scope();
    NativeCohortDigestBuilder digest;
    digest.add("schema", "citlali-native-cohort-raw-manifest-v1");
    native_cohort_add_artifact_identity(digest, "target", target.target_artifact);
    native_cohort_add_observation_identity(digest, "observation", target.observation);
    digest.add_integer("input.count", target.ordered_inputs.size());
    for (const auto &input : target.ordered_inputs) {
        digest.add_integer("input.key", input.input_key);
        digest.add_integer("input.network", input.network);
        digest.add("input.interface", input.interface_name);
        digest.add_integer("input.channels", input.channel_count);
        digest.add_integer("input.raw-key", input.raw_source_key);
        digest.add("input.raw-sha256", input.raw_content_sha256);
        digest.add_integer("input.raw-bytes", input.raw_byte_count);
        native_cohort_add_observation_identity(
            digest, "input.raw-observation", input.raw_header_observation);
    }
    return digest.finish();
}

inline std::string native_cohort_alignment_plan_digest(
    const NativeAlignmentPlan &plan) {
    NativeCohortDigestBuilder digest;
    digest.add("schema", "citlali-native-cohort-alignment-plan-v1");
    digest.add_integer("network.count", plan.networks().size());
    digest.add_integer("slot.count", plan.slot_count());
    for (const auto network_id : plan.participant_network_ids()) {
        const auto &network = plan.network(network_id);
        digest.add_integer("network.id", network_id);
        digest.add_integer("network.first-row", network.first_native_row());
        for (Eigen::Index row = 0; row < network.row_count(); ++row) {
            const auto native_row = network.first_native_row() + row;
            digest.add_double("network.time", network.identity(native_row).reconstructed_time_unix_sec());
            digest.add_integer("network.counter", network.packet_counter(native_row));
        }
        for (std::size_t slot = 0; slot < plan.slot_count(); ++slot) {
            const auto &association = plan.association(network_id, slot);
            digest.add_integer("association.slot", slot);
            digest.add_integer("association.mapped", association.mapped());
            if (association.mapped()) {
                digest.add_integer("association.native-row", association.native_row);
            }
            else {
                digest.add_integer(
                    "association.absence",
                    static_cast<std::underlying_type_t<CoincidenceAbsenceReason>>(
                        association.absence_reason));
            }
        }
    }
    return digest.finish();
}

inline std::string native_cohort_pointing_plan_digest(
    const NativePointingPlan &plan) {
    NativeCohortDigestBuilder digest;
    digest.add("schema", "citlali-native-cohort-pointing-plan-v1");
    const auto &raw = plan.raw_telescope_trajectory_handle()->telescope_data();
    for (const auto &[key, values] : raw) {
        digest.add("raw.key", key);
        for (Eigen::Index index = 0; index < values.size(); ++index) {
            digest.add_double("raw.value", values(index));
        }
    }
    for (const auto network_id : plan.participant_network_ids()) {
        const auto &network = plan.network(network_id);
        digest.add_integer("network.id", network_id);
        for (Eigen::Index row = 0; row < network.row_count(); ++row) {
            const auto identity = network.identity(network.first_native_row() + row);
            digest.add_integer("network.row", identity.native_row());
            digest.add_double("network.time", identity.reconstructed_time_unix_sec());
            for (const auto &[key, values] : network.telescope_data()) {
                digest.add("network.tel.key", key);
                digest.add_double("network.tel.value", values(row));
            }
            for (const auto &[axis, values] : network.pointing_offsets_arcsec()) {
                digest.add("network.offset.axis", axis);
                digest.add_double("network.offset.value", values(row));
            }
        }
    }
    return digest.finish();
}

struct NativeCohortObservationBinding {
    std::size_t observation_index = 0;
    canonical_apt::ObservationIdentity raw_observation;
    PublishedAptScope artifact_scope;
    std::string detector_relation_digest;
    std::string raw_manifest_digest;
    std::string alignment_plan_digest;
    std::string pointing_plan_digest;

    std::string digest() const {
        NativeCohortDigestBuilder value;
        value.add("relation", detector_relation_digest);
        value.add("raw", raw_manifest_digest);
        value.add("alignment", alignment_plan_digest);
        value.add("pointing", pointing_plan_digest);
        value.add_integer("observation", observation_index);
        return value.finish();
    }

    void validate() const {
        if (artifact_scope.kind != PublishedAptKind::matched_observation ||
            artifact_scope.parent_content_revalidated ||
            !artifact_scope.matched_target || detector_relation_digest.empty() ||
            raw_manifest_digest.empty() || alignment_plan_digest.empty() ||
            pointing_plan_digest.empty() ||
            !(raw_observation == artifact_scope.matched_target->observation)) {
            throw std::logic_error(
                "native cohort observation binding is incomplete, stale, or not self-contained");
        }
    }
};

inline NativeCohortObservationBinding make_native_cohort_observation_binding(
    std::size_t observation_index, const AptDetectorRelation &relation,
    const std::shared_ptr<const NativeAlignmentPlan> &alignment,
    const std::shared_ptr<const NativePointingPlan> &pointing) {
    if (!alignment || !pointing || !pointing->bound_to(alignment)) {
        throw std::logic_error(
            "native cohort lineage requires coherent alignment and pointing plans");
    }
    NativeCohortObservationBinding binding{
        observation_index, relation.matched_target_scope().observation,
        relation.published_scope(), native_cohort_detector_relation_digest(relation),
        native_cohort_raw_manifest_digest(relation),
        native_cohort_alignment_plan_digest(*alignment),
        native_cohort_pointing_plan_digest(*pointing)};
    binding.validate();
    return binding;
}

struct NativeCohortParticipantRow {
    NativeSampleIdentity identity;
    TimestreamNativeRevision input_revision = 0;
    TimestreamNativeRevision output_revision = 0;
    CoincidenceCellState cell_state = CoincidenceCellState::mapped_valid;
};

struct NativeCohortOutputRow {
    Eigen::Index output_row = -1;
    std::size_t relational_common_slot = 0;
    std::vector<NativeCohortParticipantRow> participants;
    std::vector<NativeStrideSupport> participant_support;
};

struct NativeCohortMapContributionJoin {
    bool mapmaking_enabled = false;
    std::string method;
    std::string eligible_input_digest;
    std::vector<Eigen::Index> ordered_map_indices;
    std::string product_identity_digest;
    std::optional<std::string> jinc_processing_configuration_digest;
    std::optional<std::string> jinc_scan_trace_digest;
};

struct NativeCohortScanProvenance {
    std::string observation_binding_digest;
    NativeOperationIdentity operation{0, 0};
    TimestreamNativeRevision input_revision = 0;
    TimestreamNativeRevision output_revision = 0;
    std::string native_cell_action = "replaced-or-preserved-by-ptc-v1";
    std::vector<NativeCohortOutputRow> rows;
    NativeCohortMapContributionJoin map_join;

    void validate(const NativeCohortObservationBinding &binding,
                  std::size_t scan_count) const {
        binding.validate();
        if (observation_binding_digest != binding.digest() || operation.scan_index < 0 ||
            static_cast<std::size_t>(operation.scan_index) >= scan_count ||
            output_revision < input_revision || rows.empty() ||
            native_cell_action.empty()) {
            throw std::logic_error(
                "native cohort scan lineage has invalid operation or revision state");
        }
        std::optional<std::size_t> prior_slot;
        for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
            const auto &row = rows[row_index];
            if (row.output_row != static_cast<Eigen::Index>(row_index) ||
                row.participants.empty() ||
                row.participants.size() != row.participant_support.size() ||
                (prior_slot && row.relational_common_slot <= *prior_slot)) {
                throw std::logic_error(
                    "native cohort row lineage is incomplete or reuses grouping provenance");
            }
            prior_slot = row.relational_common_slot;
            for (std::size_t participant = 0;
                 participant < row.participants.size(); ++participant) {
                const auto &cell = row.participants[participant];
                const auto &support = row.participant_support[participant];
                if (cell.cell_state != CoincidenceCellState::mapped_valid ||
                    cell.input_revision != input_revision ||
                    cell.output_revision != output_revision ||
                    !(support.selected_anchor == cell.identity) ||
                    support.factor <= 0 || support.exact_support_rows.empty() ||
                    !(support.exact_support_rows.front() == cell.identity) ||
                    support.detector_columns.empty() ||
                    support.detector_columns.size() !=
                        support.ored_flag_support.size()) {
                    throw std::logic_error(
                        "native cohort row lineage lost an exact measured run anchor or support");
                }
            }
        }
        if (!map_join.mapmaking_enabled) {
            if (!map_join.method.empty() || !map_join.eligible_input_digest.empty() ||
                !map_join.ordered_map_indices.empty() ||
                !map_join.product_identity_digest.empty() ||
                map_join.jinc_processing_configuration_digest ||
                map_join.jinc_scan_trace_digest) {
                throw std::logic_error(
                    "disabled mapmaking cannot carry a product contribution join");
            }
            return;
        }
        if (map_join.method.empty() || map_join.eligible_input_digest.empty() ||
            map_join.ordered_map_indices.empty() ||
            map_join.product_identity_digest.empty()) {
            throw std::logic_error(
                "native cohort map contribution join is incomplete");
        }
        const bool jinc = map_join.method == "jinc";
        if (jinc != (map_join.jinc_processing_configuration_digest.has_value() &&
                     map_join.jinc_scan_trace_digest.has_value())) {
            throw std::logic_error(
                "native cohort JINC contribution join is incomplete or mislabeled");
        }
    }
};

struct NativeCohortProductProvenance {
    NativeCohortObservationBinding binding;
    std::vector<NativeCohortScanProvenance> scans;

    void validate_complete(std::size_t expected_scan_count) const {
        binding.validate();
        if (scans.size() != expected_scan_count) {
            throw std::logic_error(
                "native cohort lineage does not cover every observation scan");
        }
        for (std::size_t scan = 0; scan < scans.size(); ++scan) {
            scans[scan].validate(binding, expected_scan_count);
            if (scans[scan].operation.scan_index !=
                static_cast<std::int64_t>(scan)) {
                throw std::logic_error(
                    "native cohort lineage scan order is nondeterministic");
            }
        }
    }
};

class NativeCohortObservationLineage
    : public std::enable_shared_from_this<NativeCohortObservationLineage> {
public:
    enum class SlotPhase : std::uint8_t { empty, pending, committed };

    class Reservation {
    public:
        Reservation() = default;
        Reservation(const Reservation &) = delete;
        Reservation &operator=(const Reservation &) = delete;
        Reservation(Reservation &&other) noexcept
            : owner_{std::move(other.owner_)}, scan_{other.scan_},
              active_{other.active_} {
            other.active_ = false;
        }
        Reservation &operator=(Reservation &&other) noexcept {
            if (this != &other) {
                rollback();
                owner_ = std::move(other.owner_);
                scan_ = other.scan_;
                active_ = other.active_;
                other.active_ = false;
            }
            return *this;
        }
        ~Reservation() { rollback(); }

        void commit() noexcept {
            if (!active_ || !owner_) {
                return;
            }
            owner_->slots_[scan_]->phase.store(
                SlotPhase::committed, std::memory_order_release);
            active_ = false;
        }

    private:
        friend class NativeCohortObservationLineage;
        Reservation(std::shared_ptr<NativeCohortObservationLineage> owner,
                    std::size_t scan) noexcept
            : owner_{std::move(owner)}, scan_{scan}, active_{true} {}
        void rollback() noexcept {
            if (!active_ || !owner_) {
                return;
            }
            auto &slot = *owner_->slots_[scan_];
            slot.record.reset();
            slot.phase.store(SlotPhase::empty, std::memory_order_release);
            active_ = false;
        }
        std::shared_ptr<NativeCohortObservationLineage> owner_;
        std::size_t scan_ = 0;
        bool active_ = false;
    };

    static std::shared_ptr<NativeCohortObservationLineage> create(
        NativeCohortObservationBinding binding, std::size_t scan_count) {
        binding.validate();
        if (scan_count == 0) {
            throw std::invalid_argument(
                "native cohort observation lineage requires positive scan cardinality");
        }
        return std::shared_ptr<NativeCohortObservationLineage>(
            new NativeCohortObservationLineage{std::move(binding), scan_count});
    }

    const NativeCohortObservationBinding &binding() const noexcept {
        return binding_;
    }
    std::size_t scan_count() const noexcept { return slots_.size(); }

    Reservation reserve(NativeCohortScanProvenance record) {
        record.validate(binding_, scan_count());
        const auto scan = static_cast<std::size_t>(record.operation.scan_index);
        auto &slot = *slots_.at(scan);
        SlotPhase expected = SlotPhase::empty;
        if (!slot.phase.compare_exchange_strong(
                expected, SlotPhase::pending, std::memory_order_acq_rel)) {
            throw std::logic_error(
                "native cohort scan lineage is stale, duplicate, or already pending");
        }
        try {
            slot.record.emplace(std::move(record));
        }
        catch (...) {
            slot.record.reset();
            slot.phase.store(SlotPhase::empty, std::memory_order_release);
            throw;
        }
        return Reservation{shared_from_this(), scan};
    }

    NativeCohortProductProvenance snapshot_complete() const {
        NativeCohortProductProvenance result;
        result.binding = binding_;
        result.scans.reserve(slots_.size());
        for (std::size_t scan = 0; scan < slots_.size(); ++scan) {
            const auto &slot = *slots_[scan];
            if (slot.phase.load(std::memory_order_acquire) !=
                    SlotPhase::committed ||
                !slot.record) {
                throw std::logic_error(
                    "native cohort observation lineage is incomplete");
            }
            result.scans.push_back(*slot.record);
        }
        result.validate_complete(scan_count());
        return result;
    }

private:
    struct Slot {
        std::atomic<SlotPhase> phase{SlotPhase::empty};
        std::optional<NativeCohortScanProvenance> record;
    };

    NativeCohortObservationLineage(NativeCohortObservationBinding binding,
                                   std::size_t scan_count)
        : binding_{std::move(binding)} {
        slots_.reserve(scan_count);
        for (std::size_t scan = 0; scan < scan_count; ++scan) {
            slots_.push_back(std::make_unique<Slot>());
        }
    }

    NativeCohortObservationBinding binding_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

}  // namespace citlali::pipeline
