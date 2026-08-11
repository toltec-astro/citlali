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
