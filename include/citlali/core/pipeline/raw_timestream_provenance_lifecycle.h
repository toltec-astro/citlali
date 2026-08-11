#pragma once

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/raw_timestream_provenance.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_output_context.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline std::size_t raw_realized_count(Eigen::Index count,
                                      const char *field) {
    if (count < 0) {
        throw std::logic_error(std::string(field) + " cannot be negative");
    }
    return static_cast<std::size_t>(count);
}

inline std::size_t raw_required_timestream_write_count(
    const TimestreamOutputExpectations &expectations) {
    const std::array<Eigen::Index, 4> counts{
        expectations.rtc, expectations.ptc,
        expectations.rtcdiag, expectations.ptcdiag};
    std::size_t total = 0;
    for (const auto count : counts) {
        const auto value =
            raw_realized_count(count, "required timestream write count");
        if (value > std::numeric_limits<std::size_t>::max() - total) {
            throw std::overflow_error(
                "required timestream write count overflow");
        }
        total += value;
    }
    return total;
}

inline void complete_raw_timestream_observation(
    RawTimestreamExecutionPlan &plan, std::size_t completed_scan_count,
    std::size_t required_timestream_write_count) {
    if (!plan.initialized) {
        throw std::logic_error(
            "cannot complete uninitialized raw timestream plan");
    }
    if (!plan.observation.has_value()) {
        throw std::logic_error(
            "cannot complete raw timestream plan before observation begins");
    }
    plan.realized.completed_scan_count = completed_scan_count;
    plan.realized.required_timestream_write_count =
        required_timestream_write_count;
    plan.realized.reference_spectral_index_alpha =
        plan.observation->reference_spectral_index_alpha;
    plan.realized.reference_spectral_index_default_applied =
        plan.observation->reference_spectral_index_default_applied;
    plan.realized.tau225 = plan.observation->tau225;
    plan.realized.atmosphere_operator_id =
        plan.observation->atmosphere_operator_id;
    plan.realized.atmosphere_operator_contract_sha256 =
        plan.observation->atmosphere_operator_contract_sha256;
    plan.realized.atmosphere_node_table_sha256 =
        plan.observation->atmosphere_node_table_sha256;
    plan.realized.passband_set_id =
        plan.observation->passband_set_id;
    plan.realized.reference_profile_id =
        plan.observation->reference_profile_id;
    plan.realized.calibration_quality_regime =
        plan.observation->calibration_quality_regime;
    plan.realized.calibration_valid =
        plan.observation->calibration_valid;
    plan.realized.calibration_validity_reason =
        plan.observation->calibration_validity_reason;
    plan.realized.calibration_validity_detail =
        plan.observation->calibration_validity_detail;
    plan.realized.calibration_product_schema =
        plan.observation->calibration_product_schema;
    plan.realized.calibration_target_unit =
        plan.observation->calibration_target_unit;
    plan.realized.calibration_photometry_policy =
        plan.observation->calibration_photometry_policy;
    plan.realized.calibration_factor_composition =
        plan.observation->calibration_factor_composition;
    plan.realized.calibration_factor_provenance =
        plan.observation->calibration_factor_provenance;
    plan.realized.calibration_compatibility_fcf_semantics =
        plan.observation->calibration_compatibility_fcf_semantics;
    plan.realized.calibration_weight_recipient_semantics =
        plan.observation->calibration_weight_recipient_semantics;
    plan.realized.calibration_compact_covariance_state =
        plan.observation->calibration_compact_covariance_state;
    plan.realized.calibration_apt_artifact_sha256 =
        plan.observation->calibration_apt_artifact_sha256;
    plan.realized.calibration_acquisition_binding_sha256 =
        plan.observation->calibration_acquisition_binding_sha256;
    plan.realized.calibration_identity =
        plan.observation->calibration_identity;
    plan.realized.calibration_package_identity =
        plan.observation->calibration_package_identity;
    plan.realized.calibration_factor_state_sha256 =
        plan.observation->calibration_factor_state_sha256;
    plan.realized.calibration_raw_observation_identity =
        plan.observation->calibration_raw_observation_identity;
    plan.realized.calibration_acquisition_binding_mode =
        plan.observation->calibration_acquisition_binding_mode;
    plan.realized.calibration_acquisition_key_schema =
        plan.observation->calibration_acquisition_key_schema;
    plan.realized.calibration_response_identity =
        plan.observation->calibration_response_identity;
    plan.realized.calibration_conditional_variance_transfer =
        plan.observation->calibration_conditional_variance_transfer;
    plan.realized.calibration_conditional_inverse_variance_transfer =
        plan.observation->calibration_conditional_inverse_variance_transfer;
    plan.realized.calibration_precision_limitation =
        plan.observation->calibration_precision_limitation;
    plan.realized.calibration_nuisance_states =
        plan.observation->calibration_nuisance_states;
    plan.realized.calibration_minimum_total_multiplier =
        plan.observation->calibration_minimum_total_multiplier;
    plan.realized.calibration_maximum_total_multiplier =
        plan.observation->calibration_maximum_total_multiplier;
    plan.realized.execution_completed = true;
}

template <bool IsBeammap, class Engine>
TimestreamOutputExpectations raw_observation_output_expectations(
    const Engine &engine) {
    if (!timestream_processing_enabled(engine)) {
        return {};
    }
    if constexpr (IsBeammap) {
        const auto flags =
            beammap_timestream_output_flags(engine, true);
        return beammap_timestream_output_expectations(engine, flags);
    }
    else {
        const auto flags = standard_timestream_output_flags(engine);
        return standard_timestream_output_expectations(engine, flags);
    }
}

template <bool IsBeammap, class Engine>
std::optional<std::filesystem::path>
publish_completed_raw_timestream_provenance(Engine &engine) {
    if constexpr (has_raw_timestream_plan_v<Engine>) {
        auto &plan = raw_timestream_plan(engine);
        const auto expectations =
            raw_observation_output_expectations<IsBeammap>(engine);
        const Eigen::Index scan_count =
            timestream_processing_enabled(engine)
                ? engine.telescope.scan_indices.cols()
                : 0;
        complete_raw_timestream_observation(
            plan, raw_realized_count(scan_count, "completed scan count"),
            raw_required_timestream_write_count(expectations));
        const auto path = raw_timestream_provenance_path(
            engine.output_paths.obsnum_dir_name);
        write_raw_timestream_provenance_file(
            engine.output_paths.obsnum_dir_name, plan);
        return path;
    }
    return std::nullopt;
}

}  // namespace citlali::pipeline
