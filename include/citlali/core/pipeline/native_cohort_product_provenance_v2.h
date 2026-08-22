#pragma once

// Observation-owned SCI-ALIGN lineage for the compact-v2 consumer. This is a
// cold publication contract: it records exact admitted identities and
// operation support without changing any RTC, PTC, naive, or JINC arithmetic.

#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <citlali/core/pipeline/timestream_native_science_projection.h>
#include <citlali/core/utils/sha256.h>

#include <Eigen/Core>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

inline constexpr std::string_view native_cohort_product_provenance_v2_schema =
    "citlali-native-cohort-product-provenance-v2";

class NativeCohortDigestBuilderV2 {
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
        static_assert(std::is_integral_v<Integer> ||
                      std::is_enum_v<Integer>);
        if constexpr (std::is_enum_v<Integer>) {
            using Value = std::underlying_type_t<Integer>;
            add(label, std::to_string(static_cast<Value>(value)));
        }
        else {
            add(label, std::to_string(value));
        }
    }

    void add_double(std::string_view label, double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "native cohort provenance requires finite floating values");
        }
        std::ostringstream stream;
        stream << std::hexfloat << value;
        add(label, stream.str());
    }

    std::string finish() { return "sha256:" + digest_.finish(); }

private:
    citlali::utils::Sha256 digest_;
};

inline void native_cohort_add_component_identity_v2(
    NativeCohortDigestBuilderV2 &digest, std::string_view prefix,
    const canonical_apt_v2::ComponentIdentity &identity) {
    const std::string key{prefix};
    digest.add(key + ".schema", identity.schema);
    digest.add(key + ".occurrence", identity.occurrence);
    digest.add(key + ".semantic", identity.semantic_sha256);
    digest.add(key + ".envelope", identity.envelope_sha256);
}

inline void native_cohort_add_observation_identity_v2(
    NativeCohortDigestBuilderV2 &digest, std::string_view prefix,
    const canonical_apt_v2::ObservationIdentity &identity) {
    const std::string key{prefix};
    digest.add_integer(key + ".observation", identity.observation);
    digest.add_integer(key + ".subobservation", identity.subobservation);
    digest.add_integer(key + ".scan", identity.scan);
}

inline std::string native_cohort_detector_relation_digest_v2(
    const CanonicalAptDetectorRelationV2 &relation) {
    NativeCohortDigestBuilderV2 digest;
    digest.add("schema", "citlali-native-cohort-detector-relation-v2");
    native_cohort_add_component_identity_v2(
        digest, "bundle", relation.bundle_identity());
    native_cohort_add_component_identity_v2(
        digest, "relation", relation.relation_identity());
    native_cohort_add_component_identity_v2(
        digest, "target", relation.target_parent());
    native_cohort_add_component_identity_v2(
        digest, "baseline", relation.baseline_parent());
    native_cohort_add_observation_identity_v2(
        digest, "observation", relation.observation());
    digest.add_integer("raw.count", relation.raw_sources().size());
    for (const auto &source : relation.raw_sources()) {
        digest.add_integer("raw.uid", source.source_uid);
        digest.add_integer("raw.network", source.network);
        digest.add("raw.interface", source.interface_name);
        digest.add_integer("raw.channels", source.channel_count);
        digest.add("raw.sha256", source.content_sha256);
        digest.add_integer("raw.bytes", source.byte_count);
        native_cohort_add_observation_identity_v2(
            digest, "raw.observation", source.header_observation);
    }
    digest.add_integer("binding.count", relation.bindings().size());
    for (const auto &binding : relation.bindings()) {
        digest.add_integer("binding.column", binding.detector_column);
        digest.add_integer("binding.relation-uid", binding.relation_uid);
        digest.add_integer("binding.output-uid", binding.output_uid);
        native_cohort_add_component_identity_v2(
            digest, "binding.target", binding.target.artifact);
        digest.add_integer("binding.target-row", binding.target.local_uid);
        digest.add_integer("binding.target-input", binding.target_input_uid);
        digest.add_integer("binding.raw-source", binding.raw_source_uid);
        digest.add_integer("binding.array", binding.array);
        digest.add_integer("binding.network", binding.network);
        digest.add_integer("binding.channel", binding.channel);
        digest.add_integer("binding.source-rank", binding.source_rank);
        digest.add_integer("binding.application-rank",
                           binding.application_rank);
        digest.add_integer("binding.presentation-rank",
                           binding.presentation_rank);
        digest.add_integer("binding.disposition", binding.disposition);
        digest.add("binding.seed", binding.selected_seed
                       ? std::to_string(binding.selected_seed->local_uid)
                       : "null");
        if (binding.selected_seed) {
            native_cohort_add_component_identity_v2(
                digest, "binding.seed-artifact",
                binding.selected_seed->artifact);
        }
        digest.add("binding.flag", binding.flag
                       ? std::to_string(*binding.flag) : "null");
    }
    return digest.finish();
}

inline std::string native_cohort_raw_manifest_digest_v2(
    const CanonicalAptDetectorRelationV2 &relation) {
    NativeCohortDigestBuilderV2 digest;
    digest.add("schema", "citlali-native-cohort-raw-manifest-v2");
    native_cohort_add_component_identity_v2(
        digest, "bundle", relation.bundle_identity());
    native_cohort_add_observation_identity_v2(
        digest, "observation", relation.observation());
    digest.add_integer("raw.count", relation.raw_sources().size());
    for (const auto &source : relation.raw_sources()) {
        digest.add_integer("raw.uid", source.source_uid);
        digest.add_integer("raw.network", source.network);
        digest.add("raw.interface", source.interface_name);
        digest.add_integer("raw.channels", source.channel_count);
        digest.add("raw.sha256", source.content_sha256);
        digest.add_integer("raw.bytes", source.byte_count);
        native_cohort_add_observation_identity_v2(
            digest, "raw.observation", source.header_observation);
    }
    return digest.finish();
}

inline std::string native_cohort_alignment_plan_digest_v2(
    const NativeAlignmentPlan &plan) {
    NativeCohortDigestBuilderV2 digest;
    digest.add("schema", "citlali-native-cohort-alignment-plan-v2");
    digest.add_integer("scope.observation", plan.scope().observation);
    digest.add_integer("scope.subobservation", plan.scope().subobservation);
    digest.add_integer("scope.scan", plan.scope().scan);
    digest.add_integer("slot.count", plan.slot_count());
    for (Eigen::Index slot = 0;
         slot < plan.common_slot_reference_times_unix_sec().size(); ++slot) {
        digest.add_double("slot.reference-time",
                          plan.common_slot_reference_times_unix_sec()(slot));
    }
    for (const auto network_id : plan.participant_network_ids()) {
        const auto &network = plan.network(network_id);
        digest.add_integer("network.id", network_id);
        digest.add_integer("network.first-row", network.first_native_row());
        digest.add_integer("network.past-row",
                           network.past_last_native_row());
        for (TimestreamNativeRow row = network.first_native_row();
             row < network.past_last_native_row(); ++row) {
            digest.add_integer("network.row", row);
            digest.add_double(
                "network.time",
                network.identity(row).reconstructed_time_unix_sec());
            digest.add_integer("network.counter",
                               network.packet_counter(row));
        }
        for (std::size_t slot = 0; slot < plan.slot_count(); ++slot) {
            const auto &association = plan.association(network_id, slot);
            digest.add_integer("association.slot", slot);
            digest.add_integer("association.mapped", association.mapped());
            if (association.mapped()) {
                digest.add_integer("association.native-row",
                                   association.native_row);
            }
            else {
                digest.add_integer("association.absence",
                                   association.absence_reason);
            }
        }
    }
    return digest.finish();
}

inline std::string native_cohort_pointing_plan_digest_v2(
    const NativePointingPlan &plan) {
    NativeCohortDigestBuilderV2 digest;
    digest.add("schema", "citlali-native-cohort-pointing-plan-v2");
    const auto &raw = plan.raw_trajectory_handle()->telescope_data();
    for (const auto &[name, values] : raw) {
        digest.add("raw.series", name);
        for (Eigen::Index row = 0; row < values.size(); ++row) {
            digest.add_double("raw.value", values(row));
        }
    }
    for (const auto network_id : plan.participant_network_ids()) {
        const auto &network = plan.network(network_id);
        digest.add_integer("network.id", network_id);
        for (TimestreamNativeRow row = network.first_native_row();
             row < network.past_last_native_row(); ++row) {
            digest.add_integer("network.row", row);
            digest.add_double(
                "network.time",
                network.identity(row).reconstructed_time_unix_sec());
            const auto local = network.local_row(row);
            for (const auto &[name, values] : network.telescope_data()) {
                digest.add("network.series", name);
                digest.add_double("network.value", values(local));
            }
            for (const auto &[axis, values] :
                 network.pointing_offsets_arcsec()) {
                digest.add("network.offset-axis", axis);
                digest.add_double("network.offset-value", values(local));
            }
        }
    }
    return digest.finish();
}

struct NativeCohortObservationBindingV2 {
    std::size_t observation_index = 0;
    canonical_apt_v2::ObservationIdentity observation;
    canonical_apt_v2::ComponentIdentity bundle_identity;
    canonical_apt_v2::ComponentIdentity relation_identity;
    std::string detector_relation_digest;
    std::string raw_manifest_digest;
    std::string alignment_plan_digest;
    std::string pointing_plan_digest;

    std::string digest() const {
        NativeCohortDigestBuilderV2 value;
        value.add("schema", "citlali-native-cohort-observation-binding-v2");
        value.add_integer("observation-index", observation_index);
        native_cohort_add_observation_identity_v2(
            value, "observation", observation);
        native_cohort_add_component_identity_v2(
            value, "bundle", bundle_identity);
        native_cohort_add_component_identity_v2(
            value, "relation", relation_identity);
        value.add("detector-relation", detector_relation_digest);
        value.add("raw-manifest", raw_manifest_digest);
        value.add("alignment", alignment_plan_digest);
        value.add("pointing", pointing_plan_digest);
        return value.finish();
    }

    void validate() const {
        if (bundle_identity.schema !=
                canonical_apt_v2::matched_bundle_schema_v2 ||
            relation_identity.schema !=
                canonical_apt_v2::relation_table_schema_v2 ||
            bundle_identity.occurrence.empty() ||
            relation_identity.occurrence != bundle_identity.occurrence ||
            detector_relation_digest.empty() || raw_manifest_digest.empty() ||
            alignment_plan_digest.empty() || pointing_plan_digest.empty()) {
            throw std::logic_error(
                "native cohort compact-v2 observation binding is incomplete");
        }
    }
};

inline NativeCohortObservationBindingV2
make_native_cohort_observation_binding_v2(
    std::size_t observation_index,
    const CanonicalAptDetectorRelationV2 &relation,
    const std::shared_ptr<const NativeObservationCarriers> &carriers) {
    if (!carriers ||
        carriers->alignment_handle().get() !=
            carriers->pointing_handle()->alignment_plan_handle().get() ||
        relation.observation().observation != carriers->scope().observation ||
        relation.observation().subobservation !=
            carriers->scope().subobservation ||
        relation.observation().scan != carriers->scope().scan) {
        throw std::logic_error(
            "native cohort compact-v2 relation and carriers are stale or foreign");
    }
    NativeCohortObservationBindingV2 result{
        observation_index, relation.observation(),
        relation.bundle_identity(), relation.relation_identity(),
        native_cohort_detector_relation_digest_v2(relation),
        native_cohort_raw_manifest_digest_v2(relation),
        native_cohort_alignment_plan_digest_v2(
            *carriers->alignment_handle()),
        native_cohort_pointing_plan_digest_v2(
            *carriers->pointing_handle())};
    result.validate();
    return result;
}

struct NativeCohortRtcSupportV2 {
    std::size_t segment_ordinal = 0;
    Eigen::Index run_output_row = -1;
    int factor = 1;
    NativeSampleIdentity selected_anchor;
    std::vector<std::size_t> exact_common_slots;
    std::vector<NativeSampleIdentity> exact_native_support;
    bool final_short_support = false;
    std::vector<TimestreamDetectorColumn> detector_columns;
    std::vector<NativeDetectorFlagBits> ored_flag_support;
};

struct NativeCohortPtcGroupV2 {
    std::size_t segment_ordinal = 0;
    std::string effective_grouping;
    std::int64_t group_key = 0;
    std::size_t subgroup_index = 0;
    NativePtcGroupRole role = NativePtcGroupRole::pca_clean;
    std::vector<TimestreamDetectorColumn> detector_columns;
};

struct NativeCohortRevisionTransitionV2 {
    NativeSampleIdentity identity;
    TimestreamDetectorColumn detector_column = -1;
    TimestreamNativeRevision input_revision = 0;
    TimestreamNativeRevision output_revision = 0;
    NativeMeasuredDetectorLedger::RevisionAction action =
        NativeMeasuredDetectorLedger::RevisionAction::replaced_by_pca_result;
};

struct NativeCohortMapOccurrenceV2 {
    bool mapmaking_enabled = false;
    std::string method;
    std::string eligible_input_digest;
    std::string eligible_weight_digest;
    std::string product_occurrence;
    std::string product_identity_digest;
    std::optional<std::string>
        jinc_processing_configuration_digest;
    std::optional<std::string> jinc_scan_trace_digest;
    std::vector<Eigen::Index> ordered_map_indices;
};

struct NativeCohortScanProvenanceV2 {
    std::string observation_binding_digest;
    NativeScanChunkScope scope{NativeObservationScope{1, 0, 0}, 0, 0};
    NativeOperationIdentity operation{0, 0};
    std::vector<NativeCohortRtcSupportV2> rtc_support;
    std::vector<NativeCohortPtcGroupV2> ptc_groups;
    std::vector<NativeCohortRevisionTransitionV2> revisions;
    NativeCohortMapOccurrenceV2 map_occurrence;

    void validate(const NativeCohortObservationBindingV2 &binding,
                  std::size_t scan_count) const {
        binding.validate();
        if (observation_binding_digest != binding.digest() ||
            scope.observation_scope.observation !=
                binding.observation.observation ||
            scope.observation_scope.subobservation !=
                binding.observation.subobservation ||
            scope.observation_scope.scan != binding.observation.scan ||
            scope.scan_index != operation.scan_index ||
            operation.scan_index < 0 ||
            static_cast<std::size_t>(operation.scan_index) >= scan_count ||
            rtc_support.empty() || ptc_groups.empty() || revisions.empty()) {
            throw std::logic_error(
                "native cohort scan provenance is incomplete or foreign");
        }
        for (const auto &support : rtc_support) {
            if (support.run_output_row < 0 || support.factor <= 0 ||
                support.exact_common_slots.empty() ||
                support.exact_native_support.empty() ||
                !(support.selected_anchor ==
                  support.exact_native_support.front()) ||
                support.detector_columns.empty() ||
                support.detector_columns.size() !=
                    support.ored_flag_support.size()) {
                throw std::logic_error(
                    "native cohort RTC provenance lost exact support");
            }
        }
        for (const auto &group : ptc_groups) {
            if (group.effective_grouping.empty() ||
                group.detector_columns.empty()) {
                throw std::logic_error(
                    "native cohort PTC provenance is incomplete");
            }
        }
        for (const auto &revision : revisions) {
            if (revision.detector_column < 0 ||
                revision.output_revision != revision.input_revision + 1) {
                throw std::logic_error(
                    "native cohort revision transition is invalid");
            }
        }
        if (!map_occurrence.mapmaking_enabled) {
            if (!map_occurrence.method.empty() ||
                !map_occurrence.eligible_input_digest.empty() ||
                !map_occurrence.eligible_weight_digest.empty() ||
                !map_occurrence.product_occurrence.empty() ||
                !map_occurrence.product_identity_digest.empty() ||
                map_occurrence.jinc_processing_configuration_digest ||
                map_occurrence.jinc_scan_trace_digest ||
                !map_occurrence.ordered_map_indices.empty()) {
                throw std::logic_error(
                    "disabled native map occurrence carries product lineage");
            }
        }
        else if ((map_occurrence.method != "naive" &&
                  map_occurrence.method != "jinc") ||
                 map_occurrence.eligible_input_digest.empty() ||
                 map_occurrence.eligible_weight_digest.empty() ||
                 map_occurrence.product_occurrence.empty() ||
                 map_occurrence.product_identity_digest.empty() ||
                 map_occurrence.ordered_map_indices.empty()) {
            throw std::logic_error(
                "native cohort map occurrence is incomplete");
        }
        const bool jinc = map_occurrence.method == "jinc";
        if (jinc !=
                map_occurrence.jinc_processing_configuration_digest
                    .has_value() ||
            jinc != map_occurrence.jinc_scan_trace_digest.has_value()) {
            throw std::logic_error(
                "native cohort JINC product occurrence is incomplete or foreign");
        }
    }
};

struct NativeCohortProductProvenanceV2 {
    NativeCohortObservationBindingV2 binding;
    std::vector<NativeCohortScanProvenanceV2> scans;

    void validate_complete(std::size_t expected_scan_count) const {
        binding.validate();
        if (expected_scan_count == 0 || scans.size() != expected_scan_count) {
            throw std::logic_error(
                "native cohort provenance does not cover every scan");
        }
        for (std::size_t scan = 0; scan < scans.size(); ++scan) {
            scans[scan].validate(binding, expected_scan_count);
            if (scans[scan].operation.scan_index !=
                static_cast<std::int64_t>(scan)) {
                throw std::logic_error(
                    "native cohort scan publication order is nondeterministic");
            }
        }
    }
};

struct NativeCohortMapPublicationRequestV2 {
    bool mapmaking_enabled = false;
    std::string method;
    std::string product_occurrence;
    std::string product_identity_digest;
    std::string eligible_weight_digest;
    std::optional<std::string>
        jinc_processing_configuration_digest;
    std::optional<std::string> jinc_scan_trace_digest;
};

inline std::string native_cohort_eligible_input_digest_v2(
    const NativeCohortScanProvenanceV2 &record,
    const NativeScienceProjection &projection,
    const std::string &eligible_weight_digest) {
    NativeCohortDigestBuilderV2 digest;
    digest.add("schema", "citlali-native-cohort-eligible-input-v2");
    digest.add("observation-binding", record.observation_binding_digest);
    digest.add_integer("operation.sequence", record.operation.sequence);
    digest.add_integer("operation.scan", record.operation.scan_index);
    digest.add_integer("scope.chunk", record.scope.chunk_index);
    digest.add("map.weights", eligible_weight_digest);
    for (const auto &support : record.rtc_support) {
        digest.add_integer("rtc.segment", support.segment_ordinal);
        digest.add_integer("rtc.output-row", support.run_output_row);
        digest.add_integer("rtc.factor", support.factor);
        digest.add_integer("rtc.anchor.network",
                           support.selected_anchor.network_id());
        digest.add_integer("rtc.anchor.row",
                           support.selected_anchor.native_row());
        digest.add_double("rtc.anchor.time",
                          support.selected_anchor
                              .reconstructed_time_unix_sec());
        for (const auto slot : support.exact_common_slots) {
            digest.add_integer("rtc.common-slot", slot);
        }
        for (const auto &identity : support.exact_native_support) {
            digest.add_integer("rtc.support.network",
                               identity.network_id());
            digest.add_integer("rtc.support.row", identity.native_row());
            digest.add_double("rtc.support.time",
                              identity.reconstructed_time_unix_sec());
        }
        for (std::size_t detector = 0;
             detector < support.detector_columns.size(); ++detector) {
            digest.add_integer("rtc.detector",
                               support.detector_columns[detector]);
            digest.add_integer("rtc.flags",
                               support.ored_flag_support[detector]);
        }
    }
    for (const auto &group : record.ptc_groups) {
        digest.add_integer("ptc.segment", group.segment_ordinal);
        digest.add("ptc.grouping", group.effective_grouping);
        digest.add_integer("ptc.key", group.group_key);
        digest.add_integer("ptc.subgroup", group.subgroup_index);
        digest.add_integer("ptc.role", group.role);
        for (const auto detector : group.detector_columns) {
            digest.add_integer("ptc.detector", detector);
        }
    }
    for (const auto &revision : record.revisions) {
        digest.add_integer("revision.network",
                           revision.identity.network_id());
        digest.add_integer("revision.row",
                           revision.identity.native_row());
        digest.add_double("revision.time",
                          revision.identity
                              .reconstructed_time_unix_sec());
        digest.add_integer("revision.detector",
                           revision.detector_column);
        digest.add_integer("revision.input", revision.input_revision);
        digest.add_integer("revision.output", revision.output_revision);
        digest.add_integer("revision.action", revision.action);
    }
    for (const auto map_index : record.map_occurrence.ordered_map_indices) {
        digest.add_integer("map.index", map_index);
    }
    digest.add_integer("projection.rows", projection.row_count());
    digest.add_integer("projection.detectors", projection.detector_count());
    for (Eigen::Index row = 0; row < projection.row_count(); ++row) {
        for (Eigen::Index detector = 0;
             detector < projection.detector_count(); ++detector) {
            digest.add_double("projection.value",
                              projection.values()(row, detector));
            digest.add_integer("projection.flag",
                               projection.flags()(row, detector));
            digest.add_double("projection.latitude",
                              projection.latitudes_rad()(row, detector));
            digest.add_double("projection.longitude",
                              projection.longitudes_rad()(row, detector));
        }
    }
    return digest.finish();
}

inline NativeCohortScanProvenanceV2
make_native_cohort_scan_provenance_v2(
    const NativeCohortObservationBindingV2 &binding,
    const NativeMeasuredDetectorLedger &ledger,
    const NativeRtcDispatchResult &rtc,
    const NativePtcPreparedOperation &prepared,
    const NativeScienceProjection &projection,
    NativeCohortMapPublicationRequestV2 map_request) {
    const auto mapping = ledger.mapping_handle();
    if (!mapping || prepared.mapping_handle().get() != mapping.get() ||
        mapping->carriers_handle()->alignment_handle().get() !=
            mapping->carriers_handle()->pointing_handle()
                ->alignment_plan_handle().get() ||
        !ledger.last_operation() || !ledger.last_committed_operation() ||
        !(*ledger.last_operation() == prepared.operation()) ||
        !(*ledger.last_committed_operation() == prepared.operation()) ||
        !(projection.operation() == prepared.operation()) ||
        !(projection.scope() == mapping->scope()) ||
        rtc.runs.empty() || prepared.groups().empty()) {
        throw std::logic_error(
            "native cohort scan lineage requires one exact committed operation");
    }
    if (binding.detector_relation_digest !=
            native_cohort_detector_relation_digest_v2(
                *mapping->relation_handle()) ||
        binding.raw_manifest_digest !=
            native_cohort_raw_manifest_digest_v2(
                *mapping->relation_handle()) ||
        binding.alignment_plan_digest !=
            native_cohort_alignment_plan_digest_v2(
                *mapping->carriers_handle()->alignment_handle()) ||
        binding.pointing_plan_digest !=
            native_cohort_pointing_plan_digest_v2(
                *mapping->carriers_handle()->pointing_handle())) {
        throw std::logic_error(
            "native cohort scan lineage is stale for its observation binding");
    }

    NativeCohortScanProvenanceV2 result;
    result.observation_binding_digest = binding.digest();
    result.scope = mapping->scope();
    result.operation = prepared.operation();
    for (const auto &run : rtc.runs) {
        for (const auto &support : run.support) {
            result.rtc_support.push_back({
                support.segment_ordinal, support.run_output_row,
                support.factor, support.selected_anchor,
                support.exact_common_slots, support.exact_native_support,
                support.final_short_support, support.detector_columns,
                support.ored_flag_support});
        }
    }

    std::vector<std::pair<NativeDetectorSampleKey,
                          NativeCohortRevisionTransitionV2>> revisions;
    for (const auto &group : prepared.groups()) {
        result.ptc_groups.push_back({
            group.segment_ordinal(), group.effective_grouping(),
            group.group_key(), group.subgroup_index(), group.role(),
            group.detector_columns()});
        for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
            for (Eigen::Index local = 0;
                 local < group.detector_count(); ++local) {
                const auto &cell = group.cell(row, local);
                if (!cell.identity) {
                    throw std::logic_error(
                        "native cohort PTC lineage lost its measured identity");
                }
                const auto detector = group.detector_columns().at(
                    static_cast<std::size_t>(local));
                const NativeDetectorSampleKey key{
                    cell.identity->key(), detector};
                const auto current = ledger.record(key);
                if (!(current.identity == *cell.identity) ||
                    current.revision != cell.expected_revision + 1) {
                    throw std::logic_error(
                        "native cohort revision lineage is stale or partial");
                }
                const auto action = cell.state ==
                        CoincidenceCellState::mapped_invalid
                    ? NativeMeasuredDetectorLedger::RevisionAction::
                          preserved_pca_invalid
                    : group.role() == NativePtcGroupRole::pass_through
                        ? NativeMeasuredDetectorLedger::RevisionAction::
                              preserved_pass_through
                        : NativeMeasuredDetectorLedger::RevisionAction::
                              replaced_by_pca_result;
                revisions.push_back({
                    key,
                    {*cell.identity, detector, cell.expected_revision,
                     current.revision, action}});
            }
        }
    }
    std::sort(revisions.begin(), revisions.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
              });
    for (std::size_t index = 0; index < revisions.size(); ++index) {
        if (index > 0 &&
            !(revisions[index - 1].first < revisions[index].first)) {
            throw std::logic_error(
                "native cohort revision lineage repeats a detector sample");
        }
        result.revisions.push_back(std::move(revisions[index].second));
    }
    if (result.revisions.size() !=
        static_cast<std::size_t>(projection.row_count()) *
            static_cast<std::size_t>(projection.detector_count())) {
        throw std::logic_error(
            "native cohort revision lineage is incomplete");
    }

    if (map_request.mapmaking_enabled) {
        if ((map_request.method != "naive" &&
             map_request.method != "jinc") ||
            map_request.product_occurrence.empty() ||
            map_request.product_identity_digest.empty() ||
            map_request.eligible_weight_digest.empty()) {
            throw std::invalid_argument(
                "native cohort map publication request is incomplete");
        }
        const bool jinc = map_request.method == "jinc";
        if (jinc !=
                map_request.jinc_processing_configuration_digest
                    .has_value() ||
            jinc != map_request.jinc_scan_trace_digest.has_value()) {
            throw std::invalid_argument(
                "native cohort JINC map publication request is incomplete or foreign");
        }
        result.map_occurrence.mapmaking_enabled = true;
        result.map_occurrence.method = std::move(map_request.method);
        result.map_occurrence.product_occurrence =
            std::move(map_request.product_occurrence);
        result.map_occurrence.product_identity_digest =
            std::move(map_request.product_identity_digest);
        result.map_occurrence.eligible_weight_digest =
            std::move(map_request.eligible_weight_digest);
        result.map_occurrence.jinc_processing_configuration_digest =
            std::move(
                map_request.jinc_processing_configuration_digest);
        result.map_occurrence.jinc_scan_trace_digest =
            std::move(map_request.jinc_scan_trace_digest);
        result.map_occurrence.ordered_map_indices.reserve(
            projection.detectors().size());
        for (const auto &detector : projection.detectors()) {
            if (detector.detector_column != static_cast<Eigen::Index>(
                    result.map_occurrence.ordered_map_indices.size()) ||
                detector.map_index < 0) {
                throw std::logic_error(
                    "native cohort map occurrence has foreign detector ordering");
            }
            result.map_occurrence.ordered_map_indices.push_back(
                detector.map_index);
        }
        result.map_occurrence.eligible_input_digest =
            native_cohort_eligible_input_digest_v2(
                result, projection,
                result.map_occurrence.eligible_weight_digest);
    }
    else if (!map_request.method.empty() ||
             !map_request.product_occurrence.empty() ||
             !map_request.product_identity_digest.empty() ||
             !map_request.eligible_weight_digest.empty() ||
             map_request.jinc_processing_configuration_digest ||
             map_request.jinc_scan_trace_digest) {
        throw std::invalid_argument(
            "disabled native map publication request carries identity");
    }
    result.validate(
        binding,
        static_cast<std::size_t>(result.operation.scan_index + 1));
    return result;
}

class NativeCohortObservationLineageV2
    : public std::enable_shared_from_this<
          NativeCohortObservationLineageV2> {
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
            if (!active_ || !owner_) return;
            owner_->slots_[scan_]->phase.store(
                SlotPhase::committed, std::memory_order_release);
            active_ = false;
        }

        void rollback() noexcept {
            if (!active_ || !owner_) return;
            auto &slot = *owner_->slots_[scan_];
            slot.record.reset();
            slot.phase.store(SlotPhase::empty, std::memory_order_release);
            active_ = false;
        }

    private:
        friend class NativeCohortObservationLineageV2;
        Reservation(
            std::shared_ptr<NativeCohortObservationLineageV2> owner,
            std::size_t scan) noexcept
            : owner_{std::move(owner)}, scan_{scan}, active_{true} {}

        std::shared_ptr<NativeCohortObservationLineageV2> owner_;
        std::size_t scan_ = 0;
        bool active_ = false;
    };

    static std::shared_ptr<NativeCohortObservationLineageV2> create(
        NativeCohortObservationBindingV2 binding,
        std::size_t scan_count) {
        binding.validate();
        if (scan_count == 0) {
            throw std::invalid_argument(
                "native cohort lineage requires positive scan cardinality");
        }
        return std::shared_ptr<NativeCohortObservationLineageV2>(
            new NativeCohortObservationLineageV2{
                std::move(binding), scan_count});
    }

    const NativeCohortObservationBindingV2 &binding() const noexcept {
        return binding_;
    }
    std::size_t scan_count() const noexcept { return slots_.size(); }

    Reservation reserve(NativeCohortScanProvenanceV2 record) {
        record.validate(binding_, scan_count());
        const auto scan = static_cast<std::size_t>(
            record.operation.scan_index);
        auto &slot = *slots_.at(scan);
        SlotPhase expected = SlotPhase::empty;
        if (!slot.phase.compare_exchange_strong(
                expected, SlotPhase::pending,
                std::memory_order_acq_rel)) {
            throw std::logic_error(
                "native cohort scan lineage is stale, duplicate, or pending");
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

    NativeCohortProductProvenanceV2 snapshot_complete() const {
        NativeCohortProductProvenanceV2 result;
        result.binding = binding_;
        result.scans.reserve(slots_.size());
        for (const auto &slot_ptr : slots_) {
            const auto &slot = *slot_ptr;
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
        std::optional<NativeCohortScanProvenanceV2> record;
    };

    NativeCohortObservationLineageV2(
        NativeCohortObservationBindingV2 binding,
        std::size_t scan_count)
        : binding_{std::move(binding)} {
        slots_.reserve(scan_count);
        for (std::size_t scan = 0; scan < scan_count; ++scan) {
            slots_.push_back(std::make_unique<Slot>());
        }
    }

    NativeCohortObservationBindingV2 binding_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

}  // namespace citlali::pipeline
