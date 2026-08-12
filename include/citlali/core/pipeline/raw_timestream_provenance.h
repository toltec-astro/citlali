#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/raw_timestream_config_serialization.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>
#include <citlali/core/utils/sha256.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace citlali::pipeline {

inline constexpr const char *raw_timestream_provenance_schema_version =
    "citlali-raw-timestream-provenance-v4";
inline constexpr const char *raw_timestream_provenance_filename =
    "raw_timestream_provenance.yaml";
inline constexpr const char *selected_calibration_apt_filename =
    "selected_calibration_apt.ecsv";

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
    value["tau225"] = raw_optional_scalar_node(observation->tau225);
    value["reference_spectral_index_alpha"] =
        raw_optional_scalar_node(
            observation->reference_spectral_index_alpha);
    value["reference_spectral_index_default_applied"] =
        raw_optional_scalar_node(
            observation->reference_spectral_index_default_applied);
    value["atmosphere_operator_id"] =
        raw_optional_scalar_node(observation->atmosphere_operator_id);
    value["atmosphere_operator_contract_sha256"] =
        raw_optional_scalar_node(
            observation->atmosphere_operator_contract_sha256);
    value["atmosphere_node_table_sha256"] =
        raw_optional_scalar_node(
            observation->atmosphere_node_table_sha256);
    value["passband_set_id"] =
        raw_optional_scalar_node(observation->passband_set_id);
    value["reference_profile_id"] =
        raw_optional_scalar_node(observation->reference_profile_id);
    value["calibration_quality_regime"] =
        raw_optional_scalar_node(
            observation->calibration_quality_regime);
    value["calibration_valid"] =
        raw_optional_scalar_node(observation->calibration_valid);
    value["calibration_validity_reason"] =
        raw_optional_scalar_node(
            observation->calibration_validity_reason);
    value["calibration_validity_detail"] =
        raw_optional_scalar_node(observation->calibration_validity_detail);
    value["calibration_product_schema"] =
        raw_optional_scalar_node(observation->calibration_product_schema);
    value["calibration_target_unit"] =
        raw_optional_scalar_node(observation->calibration_target_unit);
    value["calibration_photometry_policy"] =
        raw_optional_scalar_node(observation->calibration_photometry_policy);
    value["calibration_factor_composition"] =
        raw_optional_scalar_node(observation->calibration_factor_composition);
    value["calibration_factor_provenance"] =
        raw_optional_scalar_node(observation->calibration_factor_provenance);
    value["calibration_compatibility_fcf_semantics"] =
        raw_optional_scalar_node(observation->calibration_compatibility_fcf_semantics);
    value["calibration_weight_recipient_semantics"] =
        raw_optional_scalar_node(observation->calibration_weight_recipient_semantics);
    value["calibration_compact_covariance_state"] =
        raw_optional_scalar_node(observation->calibration_compact_covariance_state);
    value["observation_flxscale_correction_applied"] =
        raw_optional_scalar_node(
            observation->observation_flxscale_correction_applied);
    value["applied_observation_flxscale_correction"] =
        raw_optional_scalar_node(
            observation->applied_observation_flxscale_correction);
    value["observation_flxscale_correction_state"] =
        raw_optional_scalar_node(
            observation->observation_flxscale_correction_state);
    value["observation_flxscale_correction_source_identity"] =
        raw_optional_scalar_node(
            observation->observation_flxscale_correction_source_identity);
    value["observation_flxscale_correction_recipient_identity"] =
        raw_optional_scalar_node(
            observation->observation_flxscale_correction_recipient_identity);
    value["calibration_apt_artifact_sha256"] =
        raw_optional_scalar_node(observation->calibration_apt_artifact_sha256);
    value["calibration_acquisition_binding_sha256"] =
        raw_optional_scalar_node(observation->calibration_acquisition_binding_sha256);
    value["calibration_identity"] =
        raw_optional_scalar_node(observation->calibration_identity);
    value["calibration_package_identity"] =
        raw_optional_scalar_node(observation->calibration_package_identity);
    value["calibration_factor_state_sha256"] =
        raw_optional_scalar_node(observation->calibration_factor_state_sha256);
    value["calibration_raw_observation_identity"] =
        raw_optional_scalar_node(observation->calibration_raw_observation_identity);
    value["calibration_acquisition_binding_mode"] =
        raw_optional_scalar_node(observation->calibration_acquisition_binding_mode);
    value["calibration_acquisition_key_schema"] =
        raw_optional_scalar_node(observation->calibration_acquisition_key_schema);
    value["calibration_response_identity"] =
        raw_optional_scalar_node(observation->calibration_response_identity);
    value["calibration_conditional_variance_transfer"] =
        raw_optional_scalar_node(observation->calibration_conditional_variance_transfer);
    value["calibration_conditional_inverse_variance_transfer"] =
        raw_optional_scalar_node(observation->calibration_conditional_inverse_variance_transfer);
    value["calibration_precision_limitation"] =
        raw_optional_scalar_node(observation->calibration_precision_limitation);
    value["calibration_nuisance_states"] =
        raw_optional_scalar_node(observation->calibration_nuisance_states);
    value["calibration_minimum_total_multiplier"] =
        raw_optional_scalar_node(observation->calibration_minimum_total_multiplier);
    value["calibration_maximum_total_multiplier"] =
        raw_optional_scalar_node(observation->calibration_maximum_total_multiplier);
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
    node["reference_spectral_index_alpha"] =
        raw_optional_scalar_node(
            realized.reference_spectral_index_alpha);
    node["reference_spectral_index_default_applied"] =
        raw_optional_scalar_node(
            realized.reference_spectral_index_default_applied);
    node["tau225"] = raw_optional_scalar_node(realized.tau225);
    node["atmosphere_operator_id"] =
        raw_optional_scalar_node(realized.atmosphere_operator_id);
    node["atmosphere_operator_contract_sha256"] =
        raw_optional_scalar_node(
            realized.atmosphere_operator_contract_sha256);
    node["atmosphere_node_table_sha256"] =
        raw_optional_scalar_node(realized.atmosphere_node_table_sha256);
    node["passband_set_id"] =
        raw_optional_scalar_node(realized.passband_set_id);
    node["reference_profile_id"] =
        raw_optional_scalar_node(realized.reference_profile_id);
    node["calibration_quality_regime"] =
        raw_optional_scalar_node(realized.calibration_quality_regime);
    node["calibration_valid"] =
        raw_optional_scalar_node(realized.calibration_valid);
    node["calibration_validity_reason"] =
        raw_optional_scalar_node(realized.calibration_validity_reason);
    node["calibration_validity_detail"] =
        raw_optional_scalar_node(realized.calibration_validity_detail);
    node["calibration_product_schema"] =
        raw_optional_scalar_node(realized.calibration_product_schema);
    node["calibration_target_unit"] =
        raw_optional_scalar_node(realized.calibration_target_unit);
    node["calibration_photometry_policy"] =
        raw_optional_scalar_node(realized.calibration_photometry_policy);
    node["calibration_factor_composition"] =
        raw_optional_scalar_node(realized.calibration_factor_composition);
    node["calibration_factor_provenance"] =
        raw_optional_scalar_node(realized.calibration_factor_provenance);
    node["calibration_compatibility_fcf_semantics"] =
        raw_optional_scalar_node(realized.calibration_compatibility_fcf_semantics);
    node["calibration_weight_recipient_semantics"] =
        raw_optional_scalar_node(realized.calibration_weight_recipient_semantics);
    node["calibration_compact_covariance_state"] =
        raw_optional_scalar_node(realized.calibration_compact_covariance_state);
    node["observation_flxscale_correction_applied"] =
        raw_optional_scalar_node(
            realized.observation_flxscale_correction_applied);
    node["applied_observation_flxscale_correction"] =
        raw_optional_scalar_node(
            realized.applied_observation_flxscale_correction);
    node["observation_flxscale_correction_state"] =
        raw_optional_scalar_node(
            realized.observation_flxscale_correction_state);
    node["observation_flxscale_correction_source_identity"] =
        raw_optional_scalar_node(
            realized.observation_flxscale_correction_source_identity);
    node["observation_flxscale_correction_recipient_identity"] =
        raw_optional_scalar_node(
            realized.observation_flxscale_correction_recipient_identity);
    node["calibration_apt_artifact_sha256"] =
        raw_optional_scalar_node(realized.calibration_apt_artifact_sha256);
    node["calibration_acquisition_binding_sha256"] =
        raw_optional_scalar_node(realized.calibration_acquisition_binding_sha256);
    node["calibration_identity"] =
        raw_optional_scalar_node(realized.calibration_identity);
    node["calibration_package_identity"] =
        raw_optional_scalar_node(realized.calibration_package_identity);
    node["calibration_factor_state_sha256"] =
        raw_optional_scalar_node(realized.calibration_factor_state_sha256);
    node["calibration_raw_observation_identity"] =
        raw_optional_scalar_node(realized.calibration_raw_observation_identity);
    node["calibration_acquisition_binding_mode"] =
        raw_optional_scalar_node(realized.calibration_acquisition_binding_mode);
    node["calibration_acquisition_key_schema"] =
        raw_optional_scalar_node(realized.calibration_acquisition_key_schema);
    node["calibration_response_identity"] =
        raw_optional_scalar_node(realized.calibration_response_identity);
    node["calibration_conditional_variance_transfer"] =
        raw_optional_scalar_node(realized.calibration_conditional_variance_transfer);
    node["calibration_conditional_inverse_variance_transfer"] =
        raw_optional_scalar_node(realized.calibration_conditional_inverse_variance_transfer);
    node["calibration_precision_limitation"] =
        raw_optional_scalar_node(realized.calibration_precision_limitation);
    node["calibration_nuisance_states"] =
        raw_optional_scalar_node(realized.calibration_nuisance_states);
    node["calibration_minimum_total_multiplier"] =
        raw_optional_scalar_node(realized.calibration_minimum_total_multiplier);
    node["calibration_maximum_total_multiplier"] =
        raw_optional_scalar_node(realized.calibration_maximum_total_multiplier);
    return node;
}

inline YAML::Node calibration_reference_requested_node(
    const citlali::config::CalibrationConfig &config) {
    YAML::Node node;
    node["reference_spectral_index_alpha"] =
        raw_optional_scalar_node(
            config.reference.spectral_index_alpha);
    return node;
}

inline YAML::Node calibration_reference_effective_node(
    const CalibrationReferenceEffectiveState &state) {
    YAML::Node node;
    node["reference_spectral_index_alpha"] =
        state.spectral_index_alpha;
    node["reference_spectral_index_default_applied"] =
        state.default_applied;
    return node;
}

inline std::string calibration_package_identity(
    const timestream::CalibrationProduct &product) {
    return timestream::calibration_package_identity(product);
}

inline YAML::Node calibration_input_record_node(
    const timestream::CalibrationLineageInputRecord &record) {
    YAML::Node node;
    node["path"] = record.path;
    node["sha256"] = record.sha256;
    node["bytes"] = record.bytes;
    node["mtime_utc"] = record.mtime_utc;
    return node;
}

inline YAML::Node calibration_vector_identity_basis_node(
    const Eigen::VectorXd &values) {
    YAML::Node node;
    node["schema_version"] = "calibration-vector-hexfloat-v1";
    node["count"] = values.size();
    node["sha256"] = timestream::calibration_vector_identity(values);
    for (Eigen::Index index = 0; index < values.size(); ++index) {
        node["values"].push_back(
            timestream::calibration_hexfloat(values(index)));
    }
    return node;
}

inline YAML::Node canonical_calibration_lineage_node(
    const std::optional<RawTimestreamObservationState> &observation) {
    YAML::Node node;
    const bool available = observation.has_value() &&
        observation->canonical_calibration_product.has_value();
    node["available"] = available;
    if (!available) {
        return node;
    }
    const auto &product = *observation->canonical_calibration_product;
    const auto &lineage = product.package_lineage;
    auto value = node["value"];
    value["schema_version"] =
        "sci-cal-001-canonical-calibration-lineage-v1";
    value["package_identity"] = product.package_identity;
    value["calibration_identity"] = product.calibration_identity;
    value["component_identities"]["selected_apt_sha256"] =
        product.apt_artifact_sha256;
    value["component_identities"]["selected_apt_row_association_sha256"] =
        product.apt_row_association_sha256;
    value["component_identities"]["raw_acquisition_binding_sha256"] =
        product.acquisition_binding_sha256;
    value["component_identities"]["admitted_factor_state_sha256"] =
        product.factor_state_sha256;
    value["component_identities"]["tolapt_manifest_association_sha256"] =
        product.tolapt_manifest_association_sha256;

    auto apt = value["selected_apt"];
    apt["source_path"] = lineage.selected_apt_source_path;
    apt["source_sha256"] = lineage.selected_apt_sha256;
    apt["package_local_path"] = selected_calibration_apt_filename;
    apt["package_local_sha256"] = product.apt_artifact_sha256;
    apt["copy_semantics"] = "exact_byte_copy_digest_verified_required_output";
    apt["observation_identity"] = lineage.apt_observation_identity;
    apt["matched_observation_identity"] =
        lineage.apt_matched_observation_identity;
    apt["selected_source"] = lineage.apt_selected_source;
    apt["legacy_metadata_available"] = lineage.legacy_metadata_available;

    auto modern = apt["tolapt_manifest"];
    modern["available"] = lineage.modern_tolapt_manifest_available;
    if (lineage.modern_tolapt_manifest_available) {
        auto modern_value = modern["value"];
        modern_value["path"] = lineage.modern_tolapt_manifest_path;
        modern_value["sha256"] = lineage.modern_tolapt_manifest_sha256;
        modern_value["contract_version"] =
            lineage.modern_tolapt_contract_version;
        modern_value["run_id"] = lineage.modern_tolapt_run_id;
        modern_value["selected_output_key"] =
            lineage.modern_tolapt_output_key;
        modern_value["selected_output_path"] =
            lineage.modern_tolapt_output_path;
        modern_value["association_sha256"] =
            lineage.tolapt_manifest_association_sha256;
        modern_value["inputs"]["design_apt"] =
            calibration_input_record_node(
                lineage.modern_tolapt_design_input);
        modern_value["inputs"]["measured_apt"] =
            calibration_input_record_node(
                lineage.modern_tolapt_measured_input);
    }

    auto acquisition = value["raw_acquisition"];
    acquisition["raw_observation_identity"] =
        product.raw_observation_identity;
    acquisition["binding_mode"] = product.acquisition_binding_mode;
    acquisition["key_schema"] = product.acquisition_key_schema;
    acquisition["binding_sha256"] = product.acquisition_binding_sha256;
    auto artifacts = acquisition["artifacts"];
    for (const auto &artifact : lineage.raw_artifacts) {
        YAML::Node item;
        item["path"] = artifact.path;
        item["sha256"] = artifact.sha256;
        item["interface"] = artifact.interface;
        item["roach_index"] = artifact.network;
        for (const auto tone : artifact.absolute_tone_frequency_hz) {
            item["absolute_tone_frequency_hz"].push_back(tone);
        }
        artifacts.push_back(item);
    }

    auto joins = value["stable_joins"];
    joins["ordered_row_association_sha256"] =
        lineage.apt_row_association_sha256;
    joins["apt_row_order_authoritative"] = false;
    for (const auto &row : lineage.ordered_rows) {
        YAML::Node item;
        item["ordered_detector_index"] = row.ordered_detector_index;
        item["selected_apt_source_row_index"] =
            row.selected_source_row_index;
        item["raw_network"] = row.network;
        item["raw_network_local_tone"] = row.network_local_tone;
        item["absolute_tone_frequency_hz"] =
            row.absolute_tone_frequency_hz;
        item["uid"] = row.uid;
        item["eligible"] = row.eligible;
        item["validity_basis"] = row.validity_basis;
        item["stable_association"] = row.stable_association;
        for (const auto &field : row.retained_fields) {
            YAML::Node retained;
            retained["name"] = field.name;
            retained["ecsv_datatype"] = field.ecsv_datatype;
            retained["value"] = field.value;
            item["retained_fields"].push_back(retained);
        }
        joins["ordered_detector_apt_rows"].push_back(item);
    }

    auto factors = value["factor_operator_state"];
    factors["target_unit"] = product.target_unit;
    factors["photometry_policy"] = std::string{product.photometry_policy};
    factors["factor_composition"] = std::string{product.factor_composition};
    factors["factor_provenance"] = std::string{product.factor_provenance};
    factors["factor_state_sha256"] = product.factor_state_sha256;
    auto factor_basis = factors["identity_basis"];
    factor_basis["schema_version"] =
        "sci-cal-001-admitted-factor-identity-basis-v1";
    factor_basis["target_unit_factor"] =
        calibration_vector_identity_basis_node(
            product.identity_target_unit_factor);
    factor_basis["detector_flxscale"] =
        calibration_vector_identity_basis_node(product.detector_flxscale);
    factor_basis["minimum_extinction_correction"] =
        calibration_vector_identity_basis_node(
            product.minimum_extinction_correction);
    factor_basis["maximum_extinction_correction"] =
        calibration_vector_identity_basis_node(
            product.maximum_extinction_correction);
    auto extinction = factor_basis["applied_sample_extinction_state"];
    extinction["schema_version"] =
        "sci-cal-001-applied-extinction-state-basis-v1";
    extinction["available"] =
        product.applied_sample_extinction_state.available;
    extinction["active"] = product.applied_sample_extinction_state.active;
    extinction["sha256"] =
        product.applied_sample_extinction_state_sha256;
    if (product.applied_sample_extinction_state.active) {
        extinction["sample_elevation_rad"] =
            calibration_vector_identity_basis_node(
                product.applied_sample_extinction_state.sample_elevation_rad);
        for (const auto &[array_id, values] :
             product.applied_sample_extinction_state.los_tau_by_array) {
            YAML::Node item;
            item["array_index"] = array_id;
            item["los_tau"] =
                calibration_vector_identity_basis_node(values);
            extinction["los_tau_by_array"].push_back(item);
        }
    }
    factors["observation_flxscale_correction_applied"] =
        product.observation_flxscale_correction_applied;
    factors["applied_observation_flxscale_correction"] =
        product.applied_observation_flxscale_correction;
    factors["observation_flxscale_correction_state"] =
        product.observation_flxscale_correction_state;
    factors["observation_flxscale_correction_source_identity"] =
        product.observation_flxscale_correction_source_identity;
    factors["observation_flxscale_correction_recipient_identity"] =
        product.observation_flxscale_correction_recipient_identity;
    factors["atmosphere_operator_id"] = product.atmosphere_operator_id;
    factors["atmosphere_operator_contract_sha256"] =
        product.atmosphere_operator_contract_sha256;
    factors["atmosphere_node_table_sha256"] =
        product.atmosphere_node_table_sha256;
    factors["passband_set_id"] = product.passband_set_id;
    factors["reference_profile_id"] = product.reference_profile_id;
    factors["reference_spectral_index_alpha"] =
        product.reference_spectral_index_alpha;
    factors["reference_spectral_index_default_applied"] =
        product.reference_spectral_index_default_applied;
    factors["tau225"] = product.tau225;
    factors["conditional_variance_transfer"] =
        std::string{product.conditional_variance_transfer};
    factors["conditional_inverse_variance_transfer"] =
        std::string{product.conditional_inverse_variance_transfer};
    value["response_basis"]["provenance"] = product.response_identity;
    value["response_basis"]["semantics"] =
        "declared_realized_state_only_no_empirical_response_validation";
    value["precision_limitation"] = std::string{product.precision_limitation};
    value["nuisance_states"] =
        timestream::calibration_nuisance_state_summary(product);
    return node;
}

inline YAML::Node raw_timestream_provenance_node(
    const RawTimestreamExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = raw_timestream_provenance_schema_version;
    root["initialized"] = plan.initialized;
    auto requested = raw_timestream_request_node(plan.requested);
    requested["calibration"] =
        calibration_reference_requested_node(
            plan.calibration_requested);
    requested["interface_sync_offset"] =
        interface_sync_offset_config_node(plan.interface_sync_requested);
    root["requested"] = requested;
    auto effective = raw_timestream_request_node(plan.effective);
    effective["calibration"] =
        calibration_reference_effective_node(
            plan.calibration_effective);
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
    root["calibration_lineage"] =
        canonical_calibration_lineage_node(plan.observation);
    return root;
}

inline std::filesystem::path raw_timestream_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / raw_timestream_provenance_filename;
}

inline std::filesystem::path selected_calibration_apt_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / selected_calibration_apt_filename;
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
    const auto yaml_path = raw_timestream_provenance_path(reduction_dir);
    const bool calibration_required =
        plan.effective.flux_calibration_enabled;
    const auto *product = plan.observation &&
            plan.observation->canonical_calibration_product
        ? &*plan.observation->canonical_calibration_product
        : nullptr;
    if (calibration_required &&
        (product == nullptr || !product->valid())) {
        throw std::logic_error(
            "calibrated raw timestream provenance requires an admitted canonical calibration product");
    }
    if (product == nullptr) {
        write_yaml_file_atomic(yaml_path, raw_timestream_provenance_node(plan));
        return;
    }

    const auto &lineage = product->package_lineage;
    if (product->calibration_identity.empty() ||
        product->package_identity.empty() ||
        !product->applied_identity_finalized ||
        product->factor_state_sha256.empty() ||
        lineage.selected_apt_source_path.empty() ||
        lineage.selected_apt_sha256.empty() ||
        lineage.selected_apt_sha256 != product->apt_artifact_sha256) {
        throw std::logic_error(
            "canonical calibration lineage is incomplete or internally inconsistent");
    }
    const std::filesystem::path source_path{
        lineage.selected_apt_source_path};
    if (!std::filesystem::is_regular_file(source_path)) {
        throw std::ios_base::failure(
            "selected calibration APT source is unavailable: " +
            source_path.string());
    }
    if (citlali::utils::sha256_file(source_path) !=
        product->apt_artifact_sha256) {
        throw std::runtime_error(
            "selected calibration APT source digest changed before publication");
    }

    const auto destination_path =
        selected_calibration_apt_path(reduction_dir);
    auto temporary_path = destination_path;
    temporary_path += ".tmp";
    bool destination_created = false;
    try {
        std::error_code equivalent_error;
        const bool source_is_destination =
            std::filesystem::exists(destination_path) &&
            std::filesystem::equivalent(
                source_path, destination_path, equivalent_error) &&
            !equivalent_error;
        if (std::filesystem::exists(destination_path)) {
            if (citlali::utils::sha256_file(destination_path) !=
                product->apt_artifact_sha256) {
                throw std::runtime_error(
                    "existing package-local selected calibration APT has a conflicting digest");
            }
        }
        else if (!source_is_destination) {
            std::error_code ignored;
            std::filesystem::remove(temporary_path, ignored);
            std::filesystem::copy_file(
                source_path, temporary_path,
                std::filesystem::copy_options::overwrite_existing);
            if (citlali::utils::sha256_file(temporary_path) !=
                product->apt_artifact_sha256) {
                throw std::runtime_error(
                    "staged package-local selected calibration APT digest mismatch");
            }
            std::filesystem::rename(temporary_path, destination_path);
            destination_created = true;
        }
        if (!std::filesystem::is_regular_file(destination_path) ||
            citlali::utils::sha256_file(destination_path) !=
                product->apt_artifact_sha256) {
            throw std::runtime_error(
                "package-local selected calibration APT is missing or stale");
        }
        write_yaml_file_atomic(yaml_path, raw_timestream_provenance_node(plan));
    }
    catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        if (destination_created) {
            std::filesystem::remove(destination_path, ignored);
        }
        throw;
    }
}

}  // namespace citlali::pipeline
