#pragma once

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/config/interface_sync_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_timestream_resolution.h>
#include <citlali/core/timestream/calibration_product.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

struct RawTimestreamObservationState {
    std::optional<std::string> reduced_observation_identity;
    std::optional<double> native_sample_rate_hz;
    std::optional<double> effective_sample_rate_hz;
    std::optional<int> downsample_factor;
    std::optional<int> filter_edge_guard_samples;
    std::optional<int> filter_outer_context_samples;
    bool filter_edge_guard_parity_deferred = false;
    std::optional<bool> source_protection_active;
    std::optional<bool> extinction_active;
    std::optional<std::string> extinction_model;
    std::optional<double> tau225;
    std::optional<double> reference_spectral_index_alpha;
    std::optional<bool> reference_spectral_index_default_applied;
    std::optional<std::string> atmosphere_operator_id;
    std::optional<std::string> atmosphere_operator_contract_sha256;
    std::optional<std::string> atmosphere_node_table_sha256;
    std::optional<std::string> passband_set_id;
    std::optional<std::string> reference_profile_id;
    std::optional<std::string> calibration_quality_regime;
    std::optional<bool> calibration_valid;
    std::optional<std::string> calibration_validity_reason;
    std::optional<std::string> calibration_validity_detail;
    std::optional<std::string> calibration_product_schema;
    std::optional<std::string> calibration_target_unit;
    std::optional<std::string> calibration_photometry_policy;
    std::optional<std::string> calibration_factor_composition;
    std::optional<std::string> calibration_factor_provenance;
    std::optional<std::string> calibration_compatibility_fcf_semantics;
    std::optional<std::string> calibration_weight_recipient_semantics;
    std::optional<std::string> calibration_compact_covariance_state;
    std::optional<bool> observation_flxscale_correction_applied;
    std::optional<double> applied_observation_flxscale_correction;
    std::optional<std::string> observation_flxscale_correction_state;
    std::optional<std::string>
        observation_flxscale_correction_source_identity;
    std::optional<std::string>
        observation_flxscale_correction_recipient_identity;
    std::optional<std::string> calibration_apt_artifact_sha256;
    std::optional<std::string> calibration_acquisition_binding_sha256;
    std::optional<std::string> calibration_identity;
    std::optional<std::string> calibration_package_identity;
    std::optional<std::string> calibration_factor_state_sha256;
    std::optional<std::string> calibration_raw_observation_identity;
    std::optional<std::string> calibration_acquisition_binding_mode;
    std::optional<std::string> calibration_acquisition_key_schema;
    std::optional<std::string> calibration_response_identity;
    std::optional<std::string> calibration_conditional_variance_transfer;
    std::optional<std::string> calibration_conditional_inverse_variance_transfer;
    std::optional<std::string> calibration_precision_limitation;
    std::optional<std::string> calibration_nuisance_states;
    std::optional<double> calibration_minimum_total_multiplier;
    std::optional<double> calibration_maximum_total_multiplier;
    std::optional<timestream::CalibrationProduct>
        canonical_calibration_product;
};

struct CalibrationReferenceEffectiveState {
    double spectral_index_alpha = 0.0;
    bool default_applied = true;
};

struct RawTimestreamRealizedState {
    bool execution_completed = false;
    std::optional<std::size_t> completed_scan_count;
    std::optional<std::size_t> flagged_sample_count;
    std::optional<std::size_t> dynamic_notch_count;
    std::optional<std::size_t> required_timestream_write_count;
    std::optional<std::string> reduced_observation_identity;
    std::optional<double> reference_spectral_index_alpha;
    std::optional<bool> reference_spectral_index_default_applied;
    std::optional<double> tau225;
    std::optional<std::string> atmosphere_operator_id;
    std::optional<std::string> atmosphere_operator_contract_sha256;
    std::optional<std::string> atmosphere_node_table_sha256;
    std::optional<std::string> passband_set_id;
    std::optional<std::string> reference_profile_id;
    std::optional<std::string> calibration_quality_regime;
    std::optional<bool> calibration_valid;
    std::optional<std::string> calibration_validity_reason;
    std::optional<std::string> calibration_validity_detail;
    std::optional<std::string> calibration_product_schema;
    std::optional<std::string> calibration_target_unit;
    std::optional<std::string> calibration_photometry_policy;
    std::optional<std::string> calibration_factor_composition;
    std::optional<std::string> calibration_factor_provenance;
    std::optional<std::string> calibration_compatibility_fcf_semantics;
    std::optional<std::string> calibration_weight_recipient_semantics;
    std::optional<std::string> calibration_compact_covariance_state;
    std::optional<bool> observation_flxscale_correction_applied;
    std::optional<double> applied_observation_flxscale_correction;
    std::optional<std::string> observation_flxscale_correction_state;
    std::optional<std::string>
        observation_flxscale_correction_source_identity;
    std::optional<std::string>
        observation_flxscale_correction_recipient_identity;
    std::optional<std::string> calibration_apt_artifact_sha256;
    std::optional<std::string> calibration_acquisition_binding_sha256;
    std::optional<std::string> calibration_identity;
    std::optional<std::string> calibration_package_identity;
    std::optional<std::string> calibration_factor_state_sha256;
    std::optional<std::string> calibration_raw_observation_identity;
    std::optional<std::string> calibration_acquisition_binding_mode;
    std::optional<std::string> calibration_acquisition_key_schema;
    std::optional<std::string> calibration_response_identity;
    std::optional<std::string> calibration_conditional_variance_transfer;
    std::optional<std::string> calibration_conditional_inverse_variance_transfer;
    std::optional<std::string> calibration_precision_limitation;
    std::optional<std::string> calibration_nuisance_states;
    std::optional<double> calibration_minimum_total_multiplier;
    std::optional<double> calibration_maximum_total_multiplier;
};

inline bool raw_calibration_snapshot_matches(
    const RawTimestreamObservationState &observation,
    const RawTimestreamRealizedState &realized) {
    return observation.reduced_observation_identity ==
               realized.reduced_observation_identity &&
        observation.reference_spectral_index_alpha ==
               realized.reference_spectral_index_alpha &&
        observation.reference_spectral_index_default_applied ==
               realized.reference_spectral_index_default_applied &&
        observation.tau225 == realized.tau225 &&
        observation.atmosphere_operator_id ==
               realized.atmosphere_operator_id &&
        observation.atmosphere_operator_contract_sha256 ==
               realized.atmosphere_operator_contract_sha256 &&
        observation.atmosphere_node_table_sha256 ==
               realized.atmosphere_node_table_sha256 &&
        observation.passband_set_id == realized.passband_set_id &&
        observation.reference_profile_id == realized.reference_profile_id &&
        observation.calibration_quality_regime ==
               realized.calibration_quality_regime &&
        observation.calibration_valid == realized.calibration_valid &&
        observation.calibration_validity_reason ==
               realized.calibration_validity_reason &&
        observation.calibration_validity_detail ==
               realized.calibration_validity_detail &&
        observation.calibration_product_schema ==
               realized.calibration_product_schema &&
        observation.calibration_target_unit ==
               realized.calibration_target_unit &&
        observation.calibration_photometry_policy ==
               realized.calibration_photometry_policy &&
        observation.calibration_factor_composition ==
               realized.calibration_factor_composition &&
        observation.calibration_factor_provenance ==
               realized.calibration_factor_provenance &&
        observation.calibration_compatibility_fcf_semantics ==
               realized.calibration_compatibility_fcf_semantics &&
        observation.calibration_weight_recipient_semantics ==
               realized.calibration_weight_recipient_semantics &&
        observation.calibration_compact_covariance_state ==
               realized.calibration_compact_covariance_state &&
        observation.observation_flxscale_correction_applied ==
               realized.observation_flxscale_correction_applied &&
        observation.applied_observation_flxscale_correction ==
               realized.applied_observation_flxscale_correction &&
        observation.observation_flxscale_correction_state ==
               realized.observation_flxscale_correction_state &&
        observation.observation_flxscale_correction_source_identity ==
               realized.observation_flxscale_correction_source_identity &&
        observation.observation_flxscale_correction_recipient_identity ==
               realized.observation_flxscale_correction_recipient_identity &&
        observation.calibration_apt_artifact_sha256 ==
               realized.calibration_apt_artifact_sha256 &&
        observation.calibration_acquisition_binding_sha256 ==
               realized.calibration_acquisition_binding_sha256 &&
        observation.calibration_identity == realized.calibration_identity &&
        observation.calibration_package_identity ==
               realized.calibration_package_identity &&
        observation.calibration_factor_state_sha256 ==
               realized.calibration_factor_state_sha256 &&
        observation.calibration_raw_observation_identity ==
               realized.calibration_raw_observation_identity &&
        observation.calibration_acquisition_binding_mode ==
               realized.calibration_acquisition_binding_mode &&
        observation.calibration_acquisition_key_schema ==
               realized.calibration_acquisition_key_schema &&
        observation.calibration_response_identity ==
               realized.calibration_response_identity &&
        observation.calibration_conditional_variance_transfer ==
               realized.calibration_conditional_variance_transfer &&
        observation.calibration_conditional_inverse_variance_transfer ==
               realized.calibration_conditional_inverse_variance_transfer &&
        observation.calibration_precision_limitation ==
               realized.calibration_precision_limitation &&
        observation.calibration_nuisance_states ==
               realized.calibration_nuisance_states &&
        observation.calibration_minimum_total_multiplier ==
               realized.calibration_minimum_total_multiplier &&
        observation.calibration_maximum_total_multiplier ==
               realized.calibration_maximum_total_multiplier;
}

struct RawTimestreamExecutionPlan {
    bool initialized = false;
    citlali::config::RawTimeChunkConfig requested;
    citlali::config::RawTimeChunkConfig effective;
    citlali::config::CalibrationConfig calibration_requested;
    CalibrationReferenceEffectiveState calibration_effective;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_requested;
    citlali::config::InterfaceSyncOffsetConfig interface_sync_effective;
    RawTimestreamEffectiveResolutions effective_resolutions;
    std::optional<RawTimestreamObservationState> observation;
    RawTimestreamRealizedState realized;

    void reset_from_request(
        const citlali::config::RawTimeChunkConfig &request,
        const citlali::config::InterfaceSyncOffsetConfig
            &interface_sync_request = {},
        const citlali::config::CalibrationConfig
            &calibration_request = {}) {
        initialized = true;
        requested = request;
        effective = request;
        calibration_requested = calibration_request;
        calibration_effective.spectral_index_alpha =
            calibration_request.reference.spectral_index_alpha.value_or(0.0);
        calibration_effective.default_applied =
            !calibration_request.reference.spectral_index_alpha.has_value();
        interface_sync_requested = interface_sync_request;
        interface_sync_effective = interface_sync_request;
        effective_resolutions =
            resolve_raw_timestream_effective_request(request);
        observation.reset();
        realized = {};
    }

    RawTimestreamObservationState &begin_observation() {
        if (!initialized) {
            throw std::logic_error(
                "raw timestream plan is not initialized");
        }
        observation.emplace();
        realized = {};
        return *observation;
    }
};

}  // namespace citlali::pipeline
#include <citlali/core/config/calibration_config.h>
