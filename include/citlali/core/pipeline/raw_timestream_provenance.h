#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/raw_timestream_config_serialization.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>

#include <yaml-cpp/yaml.h>
#include <citlali_config/gitversion.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *raw_timestream_provenance_schema_version =
    "citlali-raw-timestream-provenance-v3";
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

inline const char *native_consumer_route_name(NativeConsumerRoute route) {
    switch (route) {
        case NativeConsumerRoute::legacy_inactive:
            return "legacy_inactive";
        case NativeConsumerRoute::native_required:
            return "native_required";
        case NativeConsumerRoute::beammap_raw_apt_producer:
            return "beammap_raw_apt_producer";
        case NativeConsumerRoute::beammap_calibration_table:
            return "beammap_calibration_table";
    }
    throw std::logic_error("unknown native consumer route");
}

inline YAML::Node native_component_identity_node_v2(
    const canonical_apt_v2::ComponentIdentity &identity) {
    YAML::Node node;
    node["schema"] = identity.schema;
    node["occurrence"] = identity.occurrence;
    node["semantic_sha256"] = identity.semantic_sha256;
    node["envelope_sha256"] = identity.envelope_sha256;
    return node;
}

inline YAML::Node native_sample_identity_node_v2(
    const NativeSampleIdentity &identity) {
    YAML::Node node;
    node["network_id"] = identity.network_id();
    node["native_row"] = identity.native_row();
    node["reconstructed_time_unix_sec"] =
        identity.reconstructed_time_unix_sec();
    return node;
}

inline const char *native_ptc_role_name(NativePtcGroupRole role) {
    return role == NativePtcGroupRole::pca_clean
        ? "pca_clean" : "pass_through";
}

inline const char *native_revision_action_name(
    NativeMeasuredDetectorLedger::RevisionAction action) {
    switch (action) {
        case NativeMeasuredDetectorLedger::RevisionAction::
                 replaced_by_pca_result:
            return "replaced_by_pca_result";
        case NativeMeasuredDetectorLedger::RevisionAction::
                 preserved_pca_invalid:
            return "preserved_pca_invalid";
        case NativeMeasuredDetectorLedger::RevisionAction::
                 preserved_pass_through:
            return "preserved_pass_through";
    }
    throw std::logic_error("unknown native revision action");
}

inline YAML::Node native_cohort_product_provenance_node_v2(
    const NativeCohortProductProvenanceV2 &provenance) {
    provenance.validate_complete(provenance.scans.size());
    YAML::Node node;
    node["schema_version"] =
        std::string{native_cohort_product_provenance_v2_schema};
    const auto &binding = provenance.binding;
    auto binding_node = node["observation_binding"];
    binding_node["observation_index"] = binding.observation_index;
    binding_node["observation"]["observation"] =
        binding.observation.observation;
    binding_node["observation"]["subobservation"] =
        binding.observation.subobservation;
    binding_node["observation"]["scan"] = binding.observation.scan;
    binding_node["bundle_identity"] =
        native_component_identity_node_v2(binding.bundle_identity);
    binding_node["relation_identity"] =
        native_component_identity_node_v2(binding.relation_identity);
    binding_node["detector_relation_digest"] =
        binding.detector_relation_digest;
    binding_node["raw_manifest_digest"] = binding.raw_manifest_digest;
    binding_node["alignment_plan_digest"] = binding.alignment_plan_digest;
    binding_node["pointing_plan_digest"] = binding.pointing_plan_digest;
    binding_node["binding_digest"] = binding.digest();

    for (const auto &scan : provenance.scans) {
        YAML::Node scan_node;
        scan_node["observation_binding_digest"] =
            scan.observation_binding_digest;
        scan_node["scan_index"] = scan.scope.scan_index;
        scan_node["chunk_index"] = scan.scope.chunk_index;
        scan_node["operation_sequence"] = scan.operation.sequence;
        for (const auto &support : scan.rtc_support) {
            YAML::Node support_node;
            support_node["segment_ordinal"] = support.segment_ordinal;
            support_node["run_output_row"] = support.run_output_row;
            support_node["factor"] = support.factor;
            support_node["selected_anchor"] =
                native_sample_identity_node_v2(support.selected_anchor);
            support_node["final_short_support"] =
                support.final_short_support;
            for (const auto slot : support.exact_common_slots) {
                support_node["exact_common_slots"].push_back(slot);
            }
            for (const auto &identity : support.exact_native_support) {
                support_node["exact_native_support"].push_back(
                    native_sample_identity_node_v2(identity));
            }
            for (std::size_t detector = 0;
                 detector < support.detector_columns.size(); ++detector) {
                YAML::Node detector_node;
                detector_node["detector_column"] =
                    support.detector_columns[detector];
                detector_node["ored_flag_support"] =
                    support.ored_flag_support[detector];
                support_node["detectors"].push_back(detector_node);
            }
            scan_node["rtc_support"].push_back(support_node);
        }
        for (const auto &group : scan.ptc_groups) {
            YAML::Node group_node;
            group_node["segment_ordinal"] = group.segment_ordinal;
            group_node["effective_grouping"] = group.effective_grouping;
            group_node["group_key"] = group.group_key;
            group_node["subgroup_index"] = group.subgroup_index;
            group_node["role"] = native_ptc_role_name(group.role);
            for (const auto detector : group.detector_columns) {
                group_node["detector_columns"].push_back(detector);
            }
            scan_node["ptc_groups"].push_back(group_node);
        }
        for (const auto &revision : scan.revisions) {
            YAML::Node revision_node;
            revision_node["identity"] =
                native_sample_identity_node_v2(revision.identity);
            revision_node["detector_column"] =
                revision.detector_column;
            revision_node["input_revision"] = revision.input_revision;
            revision_node["output_revision"] = revision.output_revision;
            revision_node["delivered_flag_bits"] =
                revision.delivered_flag_bits;
            revision_node["operation_exclusion_bits"] =
                revision.operation_exclusion_bits;
            revision_node["apt_flag"] =
                raw_optional_scalar_node(revision.apt_flag);
            revision_node["action"] =
                native_revision_action_name(revision.action);
            scan_node["revision_transitions"].push_back(revision_node);
        }
        auto map_node = scan_node["map_occurrence"];
        map_node["enabled"] = scan.map_occurrence.mapmaking_enabled;
        if (scan.map_occurrence.mapmaking_enabled) {
            map_node["method"] = scan.map_occurrence.method;
            map_node["eligible_input_digest"] =
                scan.map_occurrence.eligible_input_digest;
            map_node["eligible_weight_digest"] =
                scan.map_occurrence.eligible_weight_digest;
            map_node["product_occurrence"] =
                scan.map_occurrence.product_occurrence;
            map_node["product_identity_digest"] =
                scan.map_occurrence.product_identity_digest;
            if (scan.map_occurrence
                    .jinc_processing_configuration_digest) {
                map_node["jinc_processing_configuration_digest"] =
                    *scan.map_occurrence
                         .jinc_processing_configuration_digest;
            }
            if (scan.map_occurrence.jinc_scan_trace_digest) {
                map_node["jinc_scan_trace_digest"] =
                    *scan.map_occurrence.jinc_scan_trace_digest;
            }
            for (const auto map_index :
                 scan.map_occurrence.ordered_map_indices) {
                map_node["ordered_map_indices"].push_back(map_index);
            }
        }
        node["scans"].push_back(scan_node);
    }
    return node;
}

inline YAML::Node native_cohort_product_provenance_node_v3(
    const NativeCohortProductProvenanceV3 &provenance) {
    provenance.validate_complete(provenance.scans.size());
    YAML::Node node;
    node["schema_version"] =
        std::string{native_cohort_product_provenance_v3_schema};
    node["policy_schema_version"] =
        std::string{native_cohort_policy_schema_v3};
    node["serialization_policy"] =
        "authorities_causes_populations_and_identities_at_natural_scope";
    node["detector_sample_expansion"] = false;

    const auto &binding = provenance.binding;
    auto binding_node = node["observation_binding"];
    binding_node["observation_index"] = binding.observation_index;
    binding_node["observation"]["observation"] =
        binding.observation.observation;
    binding_node["observation"]["subobservation"] =
        binding.observation.subobservation;
    binding_node["observation"]["scan"] = binding.observation.scan;
    binding_node["bundle_identity"] =
        native_component_identity_node_v2(binding.bundle_identity);
    binding_node["relation_identity"] =
        native_component_identity_node_v2(binding.relation_identity);
    binding_node["detector_relation_digest"] =
        binding.detector_relation_digest;
    binding_node["raw_manifest_digest"] = binding.raw_manifest_digest;
    binding_node["alignment_plan_digest"] = binding.alignment_plan_digest;
    binding_node["pointing_plan_digest"] = binding.pointing_plan_digest;
    binding_node["binding_digest"] = binding.digest();

    const auto &detectors = provenance.detector_population;
    node["detector_population"]["detector_count"] =
        detectors.detector_count;
    node["detector_population"]["apt_eligible_detector_count"] =
        detectors.apt_eligible_detector_count;
    node["detector_population"]["apt_excluded_detector_count"] =
        detectors.apt_excluded_detector_count;
    for (const auto &exclusion : provenance.detector_exclusions) {
        YAML::Node item;
        item["detector_column"] = exclusion.detector_column;
        item["output_uid"] = exclusion.output_uid;
        item["network"] = exclusion.network;
        item["channel"] = exclusion.channel;
        item["apt_flag"] = raw_optional_scalar_node(exclusion.apt_flag);
        item["scope"] = "observation_detector";
        item["authority"] = exclusion.authority;
        item["cause"] = exclusion.cause;
        node["detector_exclusions"].push_back(item);
    }

    for (const auto &scan : provenance.scans) {
        YAML::Node item;
        item["observation_binding_digest"] =
            scan.observation_binding_digest;
        item["scan_index"] = scan.scope.scan_index;
        item["chunk_index"] = scan.scope.chunk_index;
        item["operation_sequence"] = scan.operation.sequence;
        item["rtc"]["run_count"] = scan.rtc.run_count;
        item["rtc"]["output_row_count"] = scan.rtc.output_row_count;
        item["rtc"]["exact_support_identity_count"] =
            scan.rtc.exact_support_identity_count;
        item["rtc"]["detector_support_count"] =
            scan.rtc.detector_support_count;
        item["rtc"]["flagged_detector_support_count"] =
            scan.rtc.flagged_detector_support_count;
        item["rtc"]["final_short_support_count"] =
            scan.rtc.final_short_support_count;
        item["ptc"]["requested_grouping"] =
            scan.ptc.requested_grouping;
        item["ptc"]["effective_grouping"] =
            scan.ptc.effective_grouping;
        item["ptc"]["segment_count"] = scan.ptc.segment_count;
        item["ptc"]["group_count"] = scan.ptc.group_count;
        item["ptc"]["pca_clean_group_count"] =
            scan.ptc.pca_clean_group_count;
        item["ptc"]["pass_through_group_count"] =
            scan.ptc.pass_through_group_count;
        item["ptc"]["detector_membership_count"] =
            scan.ptc.detector_membership_count;

        const auto &population = scan.population;
        auto population_node = item["population"];
        population_node["row_count"] = population.row_count;
        population_node["detector_count"] = population.detector_count;
        population_node["detector_sample_count"] =
            population.detector_sample_count;
        population_node["mapped_valid_sample_count"] =
            population.mapped_valid_sample_count;
        population_node["mapped_invalid_sample_count"] =
            population.mapped_invalid_sample_count;
        population_node["delivered_flagged_sample_count"] =
            population.delivered_flagged_sample_count;
        population_node["raw_input_flagged_sample_count"] =
            population.raw_input_flagged_sample_count;
        population_node["rtc_processing_flagged_sample_count"] =
            population.rtc_processing_flagged_sample_count;
        population_node["learned_rtc_excluded_sample_count"] =
            population.learned_rtc_excluded_sample_count;
        population_node["operation_excluded_sample_count"] =
            population.operation_excluded_sample_count;
        population_node["apt_excluded_sample_count"] =
            population.apt_excluded_sample_count;
        population_node["ptc_second_pass_excluded_sample_count"] =
            population.ptc_second_pass_excluded_sample_count;
        population_node["learned_ptc_excluded_sample_count"] =
            population.learned_ptc_excluded_sample_count;
        population_node["postclean_outlier_excluded_sample_count"] =
            population.postclean_outlier_excluded_sample_count;
        population_node["final_excluded_sample_count"] =
            population.final_excluded_sample_count;
        population_node["replaced_by_pca_sample_count"] =
            population.replaced_by_pca_sample_count;
        population_node["preserved_pca_invalid_sample_count"] =
            population.preserved_pca_invalid_sample_count;
        population_node["preserved_pass_through_sample_count"] =
            population.preserved_pass_through_sample_count;
        population_node["positive_weight_detector_count"] =
            population.positive_weight_detector_count;
        population_node["zero_weight_detector_count"] =
            population.zero_weight_detector_count;
        population_node["eligible_map_input_sample_count"] =
            population.eligible_map_input_sample_count;

        for (const auto &cause : scan.scoped_causes) {
            YAML::Node cause_node;
            cause_node["scope"] = cause.scope;
            cause_node["authority"] = cause.authority;
            cause_node["cause"] = cause.cause;
            cause_node["count_unit"] = cause.count_unit;
            cause_node["flag_bits"] =
                raw_optional_scalar_node(cause.flag_bits);
            cause_node["start_row"] =
                raw_optional_scalar_node(cause.start_row);
            cause_node["end_row"] =
                raw_optional_scalar_node(cause.end_row);
            cause_node["affected_count"] = cause.affected_count;
            for (const auto detector : cause.detector_columns) {
                cause_node["detector_columns"].push_back(detector);
            }
            item["scoped_causes"].push_back(cause_node);
        }

        auto map_node = item["map_occurrence"];
        map_node["enabled"] = scan.map_occurrence.mapmaking_enabled;
        if (scan.map_occurrence.mapmaking_enabled) {
            map_node["method"] = scan.map_occurrence.method;
            map_node["eligible_weight_digest"] =
                scan.map_occurrence.eligible_weight_digest;
            map_node["map_index_digest"] =
                scan.map_occurrence.map_index_digest;
            map_node["map_index_count"] =
                scan.map_occurrence.map_index_count;
            for (const auto detector :
                 scan.map_occurrence.zero_weight_detector_columns) {
                map_node["zero_weight_detector_columns"].push_back(
                    detector);
            }
            map_node["product_occurrence"] =
                scan.map_occurrence.product_occurrence;
            map_node["product_identity_digest"] =
                scan.map_occurrence.product_identity_digest;
            if (scan.map_occurrence
                    .jinc_processing_configuration_digest) {
                map_node["jinc_processing_configuration_digest"] =
                    *scan.map_occurrence
                         .jinc_processing_configuration_digest;
            }
            const auto &noise = scan.map_occurrence.noise_assignment;
            auto noise_node = map_node["noise_assignment"];
            noise_node["enabled"] = noise.enabled;
            if (noise.enabled) {
                noise_node["randomize_detectors"] =
                    noise.randomize_detectors;
                noise_node["realization_count"] =
                    noise.realization_count;
                noise_node["assignment_column_count"] =
                    noise.assignment_column_count;
                noise_node["assignment_count"] =
                    noise.assignment_count;
                noise_node["positive_sign_count"] =
                    noise.positive_sign_count;
                noise_node["negative_sign_count"] =
                    noise.negative_sign_count;
                noise_node["assignment_digest"] =
                    noise.assignment_digest;
                noise_node["support_authority"] =
                    noise.support_authority;
                noise_node["assignment_values_serialized"] = false;
            }
        }
        node["scans"].push_back(item);
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
    value["native_consumer_route"] =
        native_consumer_route_name(observation->native_consumer_route);
    value["native_cohort_lineage_prepared"] =
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
    node["native_cohort_provenance"]["available"] =
        realized.native_cohort_provenance.has_value();
    if (realized.native_cohort_provenance) {
        node["native_cohort_provenance"]["value"] =
            native_cohort_product_provenance_node_v3(
                *realized.native_cohort_provenance);
    }
    return node;
}

inline YAML::Node raw_timestream_provenance_node(
    const RawTimestreamExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = raw_timestream_provenance_schema_version;
    root["software_identity"]["citlali_revision"] =
        CITLALI_GIT_REVISION;
    root["software_identity"]["citlali_version"] =
        CITLALI_GIT_VERSION;
    root["software_identity"]["build_timestamp"] =
        CITLALI_BUILD_TIMESTAMP;
    root["canonical_run_identity"]["available"] =
        plan.canonical_run_identity.has_value();
    if (plan.canonical_run_identity) {
        const auto &identity = *plan.canonical_run_identity;
        root["canonical_run_identity"]["accepted_merged_config_sha256"] =
            identity.accepted_merged_config_sha256;
        root["canonical_run_identity"]
            ["effective_configuration_identity"] =
            identity.effective_configuration_identity;
        root["canonical_run_identity"]["runtime_effective_identity"] =
            identity.runtime_effective_identity;
        for (const auto &source : identity.config_sources) {
            YAML::Node item;
            item["precedence"] = source.precedence;
            item["path"] = source.path;
            item["size_bytes"] = source.size_bytes;
            item["sha256"] = source.sha256;
            root["canonical_run_identity"]["config_sources"].push_back(item);
        }
    }
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
    const bool native_canonical = plan.observation &&
        plan.observation->native_consumer_route ==
            NativeConsumerRoute::native_required;
    const bool native_validated = native_canonical &&
        plan.realized.execution_completed &&
        plan.realized.native_cohort_provenance.has_value() &&
        plan.canonical_run_identity &&
        plan.canonical_run_identity->complete();
    root["canonical_publication"]["required"] = native_canonical;
    root["canonical_publication"]["status"] = !native_canonical
        ? "not_applicable"
        : native_validated ? "validated_complete"
                           : "incomplete_not_publishable";
    root["canonical_publication"]["bounded_provenance_validated"] =
        native_validated;
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
    if (plan.observation->native_consumer_route ==
            NativeConsumerRoute::native_required &&
        (!plan.realized.native_cohort_provenance ||
         !plan.canonical_run_identity ||
         !plan.canonical_run_identity->complete())) {
        throw std::logic_error(
            "cannot publish native-required raw provenance without complete bounded identities");
    }
    write_yaml_file_atomic(
        raw_timestream_provenance_path(reduction_dir),
        raw_timestream_provenance_node(plan));
}

}  // namespace citlali::pipeline
