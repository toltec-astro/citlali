#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/raw_timestream_config_serialization.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *raw_timestream_provenance_schema_version =
    "citlali-raw-timestream-provenance-v2";
inline constexpr const char *raw_timestream_provenance_filename =
    "raw_timestream_provenance.yaml";

template <class Value>
YAML::Node raw_optional_scalar_node(const std::optional<Value> &value) {
    YAML::Node node;
    node["available"] = value.has_value();
    if (value) {
        node["value"] = *value;
    }
    return node;
}

inline YAML::Node interface_sync_offset_config_node(
    const citlali::config::InterfaceSyncOffsetConfig &config) {
    YAML::Node node;
    node["unit"] = "s";
    for (std::size_t index = 0;
         index < citlali::config::toltec_interface_count; ++index) {
        node["offsets"]["toltec" + std::to_string(index)] =
            config.toltec_offset_sec[index];
    }
    node["offsets"]["hwpr"] = config.hwpr_offset_sec;
    return node;
}

inline const char *raw_downsample_request_kind_name(
    RawDownsampleRequestKind kind) {
    switch (kind) {
        case RawDownsampleRequestKind::disabled:
            return "disabled";
        case RawDownsampleRequestKind::explicit_factor:
            return "explicit_factor";
        case RawDownsampleRequestKind::target_frequency:
            return "target_frequency";
    }
    return "unknown";
}

inline YAML::Node raw_timestream_effective_resolutions_node(
    const RawTimestreamEffectiveResolutions &resolutions) {
    YAML::Node node;
    const auto &filtering = resolutions.filtering;
    auto filtering_node = node["filtering"];
    filtering_node["fir_requested"] = filtering.fir_requested;
    filtering_node["fixed_notch_requested"] =
        filtering.fixed_notch_requested;
    filtering_node["fixed_notch_effective"] =
        filtering.fixed_notch_effective;
    filtering_node["iir_highpass_requested"] =
        filtering.iir_highpass_requested;
    filtering_node["edge_guard_requested"] =
        filtering.edge_guard_requested;
    filtering_node["downsample_requested"] =
        filtering.downsample_requested;
    filtering_node["downsample_filter_dependency_satisfied"] =
        filtering.downsample_filter_dependency_satisfied;

    const auto &downsampling = resolutions.downsampling;
    auto downsampling_node = node["downsampling"];
    downsampling_node["kind"] =
        raw_downsample_request_kind_name(downsampling.kind);
    downsampling_node["requested_factor"] =
        downsampling.requested_factor;
    downsampling_node["requested_frequency_hz"] =
        downsampling.requested_frequency_hz;

    const auto &source = resolutions.source_protection;
    auto source_node = node["source_protection"];
    source_node["despike_requested"] = source.despike_requested;
    source_node["source_protection_requested"] =
        source.source_protection_requested;

    const auto &corrections = resolutions.corrections;
    auto corrections_node = node["corrections"];
    corrections_node["flux_calibration_requested"] =
        corrections.flux_calibration_requested;
    corrections_node["extinction_correction_requested"] =
        corrections.extinction_correction_requested;
    return node;
}

inline YAML::Node native_cohort_artifact_identity_node(
    const apt_observation::ArtifactIdentity &identity) {
    YAML::Node node;
    node["schema"] = identity.schema;
    node["occurrence"] = identity.occurrence;
    node["semantic_sha256"] = identity.semantic_sha256;
    node["envelope_sha256"] = identity.envelope_sha256;
    return node;
}

inline YAML::Node native_cohort_observation_identity_node(
    const canonical_apt::ObservationIdentity &identity) {
    YAML::Node node;
    node["observation"] = identity.observation;
    node["subobservation"] = identity.subobservation;
    node["scan"] = identity.scan;
    return node;
}

inline YAML::Node native_cohort_product_provenance_node(
    const NativeCohortProductProvenance &provenance) {
    provenance.validate_complete(provenance.scans.size());
    YAML::Node node;
    node["schema_version"] = native_cohort_product_provenance_schema_version;
    node["common_slot_semantics"] = native_cohort_common_slot_semantics;
    const auto &binding = provenance.binding;
    node["observation_index"] = binding.observation_index;
    node["raw_observation"] =
        native_cohort_observation_identity_node(binding.raw_observation);
    const auto &scope = binding.artifact_scope;
    node["apt"]["kind"] = "matched_observation";
    node["apt"]["artifact"] = native_cohort_artifact_identity_node(scope.artifact);
    node["apt"]["transport_scope"] = scope.transport.scope;
    node["apt"]["transport_sha256"] = scope.transport.sha256;
    node["apt"]["transport_byte_count"] = scope.transport.byte_count;
    node["apt"]["receipt_sha256"] = scope.receipt_sha256;
    node["apt"]["receipt_byte_count"] = scope.receipt_byte_count;
    node["apt"]["parent_content_revalidated"] =
        scope.parent_content_revalidated;
    node["apt"]["embedded_baseline"]["artifact"] =
        native_cohort_artifact_identity_node(scope.baseline_parent.artifact);
    node["apt"]["embedded_baseline"]["descriptor_sha256"] =
        scope.baseline_parent.descriptor_sha256;
    node["detector_relation_digest"] = binding.detector_relation_digest;
    node["raw_manifest_digest"] = binding.raw_manifest_digest;
    node["alignment_plan_digest"] = binding.alignment_plan_digest;
    node["pointing_plan_digest"] = binding.pointing_plan_digest;
    for (const auto &scan : provenance.scans) {
        YAML::Node scan_node;
        scan_node["operation"]["scan_index"] = scan.operation.scan_index;
        scan_node["operation"]["sequence"] = scan.operation.sequence;
        scan_node["input_revision"] = scan.input_revision;
        scan_node["output_revision"] = scan.output_revision;
        scan_node["cell_action"] = scan.native_cell_action;
        for (const auto &row : scan.rows) {
            YAML::Node row_node;
            row_node["output_row"] = row.output_row;
            row_node["common_slot"] = row.relational_common_slot;
            row_node["common_slot_semantics"] = native_cohort_common_slot_semantics;
            for (std::size_t participant = 0;
                 participant < row.participants.size(); ++participant) {
                const auto &cell = row.participants[participant];
                const auto &support = row.participant_support[participant];
                YAML::Node participant_node;
                participant_node["network"] = cell.identity.network_id();
                participant_node["native_row"] = cell.identity.native_row();
                participant_node["reconstructed_time_unix_sec"] =
                    native_cohort_hex_double(cell.identity.reconstructed_time_unix_sec());
                participant_node["cell_state"] = "mapped_valid";
                participant_node["input_revision"] = cell.input_revision;
                participant_node["output_revision"] = cell.output_revision;
                participant_node["run_ordinal"] = support.run_ordinal;
                participant_node["stride_factor"] = support.factor;
                participant_node["selected_anchor_native_row"] =
                    support.selected_anchor.native_row();
                participant_node["support_first_native_row"] =
                    support.first_support_native_row;
                participant_node["support_past_native_row"] =
                    support.past_last_support_native_row;
                participant_node["final_short_support"] =
                    support.final_short_support;
                for (const auto &sample : support.exact_support_rows) {
                    participant_node["exact_support_rows"].push_back(
                        sample.native_row());
                }
                for (std::size_t i = 0; i < support.detector_columns.size(); ++i) {
                    YAML::Node flag_node;
                    flag_node["detector_column"] = support.detector_columns[i];
                    flag_node["ored_flag_support"] = support.ored_flag_support[i];
                    participant_node["ored_flags"].push_back(flag_node);
                }
                row_node["participants"].push_back(participant_node);
            }
            scan_node["rows"].push_back(row_node);
        }
        scan_node["map_join"]["enabled"] = scan.map_join.mapmaking_enabled;
        scan_node["map_join"]["method"] = scan.map_join.method;
        scan_node["map_join"]["eligible_input_digest"] =
            scan.map_join.eligible_input_digest;
        scan_node["map_join"]["product_identity_digest"] =
            scan.map_join.product_identity_digest;
        for (const auto index : scan.map_join.ordered_map_indices) {
            scan_node["map_join"]["ordered_map_indices"].push_back(index);
        }
        if (scan.map_join.jinc_processing_configuration_digest) {
            scan_node["map_join"]["jinc_processing_configuration_digest"] =
                *scan.map_join.jinc_processing_configuration_digest;
            scan_node["map_join"]["jinc_scan_trace_digest"] =
                *scan.map_join.jinc_scan_trace_digest;
        }
        node["scans"].push_back(scan_node);
    }
    return node;
}

inline YAML::Node raw_timestream_observation_state_node(
    const std::optional<RawTimestreamObservationState> &observation) {
    YAML::Node node;
    node["available"] = observation.has_value();
    if (!observation) {
        return node;
    }

    auto value = node["value"];
    value["native_sample_rate_hz"] =
        raw_optional_scalar_node(observation->native_sample_rate_hz);
    value["effective_sample_rate_hz"] =
        raw_optional_scalar_node(observation->effective_sample_rate_hz);
    value["downsample_factor"] =
        raw_optional_scalar_node(observation->downsample_factor);
    value["filter_edge_guard_samples"] =
        raw_optional_scalar_node(observation->filter_edge_guard_samples);
    value["filter_outer_context_samples"] =
        raw_optional_scalar_node(
            observation->filter_outer_context_samples);
    value["filter_edge_guard_parity_deferred"] =
        observation->filter_edge_guard_parity_deferred;
    value["source_protection_active"] =
        raw_optional_scalar_node(observation->source_protection_active);
    value["extinction_active"] =
        raw_optional_scalar_node(observation->extinction_active);
    value["extinction_model"] =
        raw_optional_scalar_node(observation->extinction_model);
    value["native_cohort_lineage_required"] =
        static_cast<bool>(observation->native_cohort_lineage);
    return node;
}

inline YAML::Node raw_timestream_realized_state_node(
    const RawTimestreamRealizedState &realized) {
    YAML::Node node;
    node["execution_completed"] = realized.execution_completed;
    node["completed_scan_count"] =
        raw_optional_scalar_node(realized.completed_scan_count);
    node["flagged_sample_count"] =
        raw_optional_scalar_node(realized.flagged_sample_count);
    node["dynamic_notch_count"] =
        raw_optional_scalar_node(realized.dynamic_notch_count);
    node["required_timestream_write_count"] =
        raw_optional_scalar_node(
            realized.required_timestream_write_count);
    node["native_cohort_product_provenance_available"] =
        realized.native_cohort_provenance.has_value();
    if (realized.native_cohort_provenance) {
        node["native_cohort_product_provenance"] =
            native_cohort_product_provenance_node(
                *realized.native_cohort_provenance);
    }
    return node;
}

inline YAML::Node raw_timestream_provenance_node(
    const RawTimestreamExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = raw_timestream_provenance_schema_version;
    root["initialized"] = plan.initialized;
    auto requested = raw_timestream_request_node(plan.requested);
    requested["interface_sync_offset"] =
        interface_sync_offset_config_node(plan.interface_sync_requested);
    root["requested"] = requested;
    auto effective = raw_timestream_request_node(plan.effective);
    effective["interface_sync_offset"] =
        interface_sync_offset_config_node(plan.interface_sync_effective);
    root["effective"]["config"] = effective;
    root["effective"]["resolutions"] =
        raw_timestream_effective_resolutions_node(
            plan.effective_resolutions);
    root["observation"] =
        raw_timestream_observation_state_node(plan.observation);
    root["realized"] =
        raw_timestream_realized_state_node(plan.realized);
    return root;
}

inline std::filesystem::path raw_timestream_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / raw_timestream_provenance_filename;
}

inline void write_raw_timestream_provenance_file(
    const std::filesystem::path &reduction_dir,
    const RawTimestreamExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error(
            "cannot write uninitialized raw timestream provenance");
    }
    if (!plan.observation.has_value()) {
        throw std::logic_error(
            "cannot write raw timestream provenance before observation begins");
    }
    if (!plan.realized.execution_completed) {
        throw std::logic_error(
            "cannot write incomplete raw timestream provenance");
    }
    if (!plan.realized.completed_scan_count.has_value()
        || !plan.realized.required_timestream_write_count.has_value()) {
        throw std::logic_error(
            "cannot write raw timestream provenance without realized counts");
    }
    if (plan.observation->native_cohort_lineage &&
        !plan.realized.native_cohort_provenance) {
        throw std::logic_error(
            "cannot write native raw timestream provenance before complete lineage commits");
    }
    write_yaml_file_atomic(
        raw_timestream_provenance_path(reduction_dir),
        raw_timestream_provenance_node(plan));
}

}  // namespace citlali::pipeline
