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
    value["calibration_apt_artifact_sha256"] =
        raw_optional_scalar_node(observation->calibration_apt_artifact_sha256);
    value["calibration_acquisition_binding_sha256"] =
        raw_optional_scalar_node(observation->calibration_acquisition_binding_sha256);
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
    node["calibration_apt_artifact_sha256"] =
        raw_optional_scalar_node(realized.calibration_apt_artifact_sha256);
    node["calibration_acquisition_binding_sha256"] =
        raw_optional_scalar_node(realized.calibration_acquisition_binding_sha256);
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
    write_yaml_file_atomic(
        raw_timestream_provenance_path(reduction_dir),
        raw_timestream_provenance_node(plan));
}

}  // namespace citlali::pipeline
