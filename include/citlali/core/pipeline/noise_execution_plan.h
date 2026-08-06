#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>
#include <citlali/core/utils/sha256.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace citlali::pipeline {

inline constexpr std::uint32_t noise_random_seed = 5489U;
inline constexpr const char *noise_random_engine_name =
    "boost::random::mt19937";
inline constexpr const char *noise_seed_policy_name =
    "fixed_internal_default";
inline constexpr const char *noise_generator_scope_name =
    "reduction_pipeline_invocation";
inline constexpr const char *noise_package_id =
    "citlali-noise-products";
inline constexpr const char *noise_provenance_join_id =
    "noise_products_provenance.yaml";
inline constexpr const char *noise_product_contract_version =
    "SCI-NOI-002-v1";
inline constexpr const char *noise_ensemble_identity =
    "source_imprinted_current";
inline constexpr const char *noise_estimator_identity =
    "conditional_finite_stack_scatter";
inline constexpr const char *noise_estimator_target =
    "completed_source_imprinted_current_stack";
inline constexpr const char *noise_centering_identity =
    "empirical_completed_stack_mean";
inline constexpr const char *noise_normalization_identity =
    "centered_sum_of_squares_divided_by_completed_R";
inline constexpr const char *noise_support_identity =
    "product_specific_common_map_support_and_response";
inline constexpr const char *noise_missingness_identity =
    "nonfinite_or_invalid_values_unavailable";
inline constexpr const char *noise_dependence_status =
    "joint_design_class_recorded_assignment_census_not_persisted";
inline constexpr const char *noise_scale_diagnostic_identity =
    "global_nonprecision_scale_diagnostic";
inline constexpr const char *noise_scale_diagnostic_formula =
    "alpha=1/median_D(q_p*V_p)";
inline constexpr const char *noise_filter_parity_gate_status =
    "scope_blocked_not_applicable_pending_FLT";
inline constexpr const char *noise_filter_parity_gate_reason =
    "signal filter_maps uses signal-background affine edge handling and kernel response while realization filter_noise paths are zero-centered; strict operator-edge parity is not established and FLT mathematics is excluded";

inline constexpr const char *noise_conditional_stack_scatter_product_id =
    "conditional_finite_stack_scatter";
inline constexpr const char *noise_formal_coefficient_product_id =
    "formal_nonprecision_coefficient_snapshot";
inline constexpr const char *noise_scaled_coefficient_product_id =
    "global_nonprecision_scaled_coefficient";
inline constexpr const char *noise_coefficient_standardized_signal_product_id =
    "coefficient_standardized_signal";
inline constexpr const char *noise_filtered_pixel_stack_scatter_product_id =
    "filtered_pixel_stack_scatter";
inline constexpr const char *noise_stack_scatter_ratio_product_id =
    "conditional_stack_scatter_ratio";
inline constexpr const char *noise_realization_product_id =
    "source_imprinted_current_realization";
inline constexpr const char *noise_pooled_stack_scale_product_id =
    "pooled_stack_scale_diagnostic";
inline constexpr const char *noise_source_finder_score_product_id =
    "source_finder_engineering_score";
inline constexpr const char *noise_fitted_amplitude_rms_ratio_product_id =
    "fitted_amplitude_over_full_map_rms_ratio";
inline constexpr const char *noise_fixed_projection_scatter_product_id =
    "fixed_projection_stack_scatter";

inline const char *noise_joint_assignment_design_identity(
    bool randomize_dets) {
    return randomize_dets
        ? "sequential_mt19937_per_scan_per_detector_sign_assignments"
        : "sequential_mt19937_per_scan_detector_shared_sign_assignments";
}

inline std::string noise_product_semantic_digest(
    std::string_view product_identity) {
    const std::string canonical =
        std::string{noise_package_id} + "|" +
        noise_product_contract_version + "|" +
        std::string{product_identity};
    return "sha256:" + citlali::utils::sha256(canonical);
}

struct NoiseEffectiveResolutionRecord {
    bool mapmaking_enabled = false;
    bool requested_enabled = false;
    bool effective_enabled = false;
    bool disabled_by_mapmaking = false;
    int requested_n_noise_maps = 0;
    int effective_n_noise_maps = 0;
    bool count_zeroed_while_disabled = false;
    std::uint32_t random_seed = noise_random_seed;
    std::string random_engine = noise_random_engine_name;
    std::string seed_policy = noise_seed_policy_name;
    std::string generator_scope = noise_generator_scope_name;
    std::string joint_assignment_design;
};

struct NoiseRealizedState {
    bool reduction_completed = false;
    bool generation_executed = false;
    std::optional<std::size_t> noise_maps_per_scientific_map;
    std::optional<std::size_t> observation_scientific_map_count;
    std::optional<std::size_t> observation_noise_realization_count;
    std::optional<std::size_t> coadd_scientific_map_count;
    std::optional<std::size_t> coadd_noise_realization_count;
    std::optional<std::size_t> total_noise_realization_count;
    std::optional<std::size_t> empirical_product_map_count;
    std::optional<std::size_t> realization_image_write_count;
    bool actual_completion_valid = false;
    bool completed_count_matches_effective = false;
    bool uncertainty_use_valid = false;
    std::string completion_basis;
    bool outputs_completed = false;
};

struct NoiseExecutionPlan {
    bool initialized = false;
    citlali::config::NoiseConfig requested;
    citlali::config::NoiseConfig effective;
    NoiseEffectiveResolutionRecord effective_resolution;
    NoiseRealizedState realized;

    void reset_from_request(
        const citlali::config::NoiseConfig &request,
        bool mapmaking_enabled) {
        if (mapmaking_enabled && request.enabled &&
            request.n_noise_maps <= 0) {
            throw std::invalid_argument(
                "enabled noise execution requires a positive effective realization count");
        }
        initialized = true;
        requested = request;
        effective = request;
        if (!mapmaking_enabled || !request.enabled) {
            effective.enabled = false;
            effective.n_noise_maps = 0;
        }
        effective_resolution = {};
        effective_resolution.mapmaking_enabled = mapmaking_enabled;
        effective_resolution.requested_enabled = request.enabled;
        effective_resolution.effective_enabled = effective.enabled;
        effective_resolution.disabled_by_mapmaking =
            request.enabled && !mapmaking_enabled;
        effective_resolution.requested_n_noise_maps = request.n_noise_maps;
        effective_resolution.effective_n_noise_maps = effective.n_noise_maps;
        effective_resolution.count_zeroed_while_disabled =
            request.n_noise_maps != effective.n_noise_maps;
        effective_resolution.random_seed = noise_random_seed;
        effective_resolution.random_engine = noise_random_engine_name;
        effective_resolution.seed_policy = noise_seed_policy_name;
        effective_resolution.generator_scope = noise_generator_scope_name;
        effective_resolution.joint_assignment_design =
            noise_joint_assignment_design_identity(effective.randomize_dets);
        realized = {};
    }
};

inline std::size_t checked_noise_count_product(
    std::size_t lhs, std::size_t rhs, const char *label) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error(
            std::string{"noise cardinality overflow: "} + label);
    }
    return lhs * rhs;
}

inline void record_noise_run_completed(
    NoiseExecutionPlan &plan,
    const MapmakingExecutionPlan &mapmaking_plan,
    bool filtered_maps_enabled) {
    if (!plan.initialized) {
        throw std::logic_error("noise plan is not initialized");
    }
    if (!mapmaking_plan.initialized ||
        !mapmaking_plan.realized.reduction_completed) {
        throw std::logic_error(
            "noise completion requires completed mapmaking provenance");
    }
    if (plan.effective.enabled &&
        !mapmaking_plan.realized.mapmaking_executed) {
        throw std::logic_error(
            "enabled noise maps require completed mapmaking");
    }

    plan.realized = {};
    if (!plan.effective.enabled) {
        plan.realized.reduction_completed = true;
        plan.realized.noise_maps_per_scientific_map = 0U;
        plan.realized.observation_scientific_map_count = 0U;
        plan.realized.observation_noise_realization_count = 0U;
        plan.realized.coadd_scientific_map_count = 0U;
        plan.realized.coadd_noise_realization_count = 0U;
        plan.realized.total_noise_realization_count = 0U;
        plan.realized.empirical_product_map_count = 0U;
        plan.realized.realization_image_write_count = 0U;
        plan.realized.actual_completion_valid = true;
        plan.realized.completed_count_matches_effective = true;
        plan.realized.uncertainty_use_valid = false;
        plan.realized.completion_basis = "effective_disabled_zero_work";
        plan.realized.outputs_completed = true;
        return;
    }
    if (plan.effective.n_noise_maps <= 0) {
        throw std::logic_error(
            "enabled effective noise-map count is not positive");
    }

    std::size_t observation_map_count = 0;
    for (const auto &observation : mapmaking_plan.observations) {
        if (!observation.outputs_completed) {
            throw std::logic_error(
                "noise completion found incomplete observation maps");
        }
        observation_map_count += observation.map_count;
    }
    const bool coadd_available = mapmaking_plan.coadd.has_value();
    const bool observation_noise_generated = true;
    const std::size_t coadd_map_count = coadd_available
        ? mapmaking_plan.coadd->map_count
        : std::size_t{0};
    const auto noise_count = static_cast<std::size_t>(
        plan.effective.n_noise_maps);
    const std::size_t observation_realization_count =
        observation_noise_generated
        ? checked_noise_count_product(
              observation_map_count, noise_count, "observation realizations")
        : std::size_t{0};
    const std::size_t coadd_realization_count =
        checked_noise_count_product(
            coadd_map_count, noise_count, "coadd realizations");
    const std::size_t total_realization_count =
        observation_realization_count + coadd_realization_count;
    const std::size_t observation_output_stage_count =
        filtered_maps_enabled && !coadd_available ? std::size_t{2}
                                                  : std::size_t{1};
    const std::size_t coadd_output_stage_count =
        coadd_available
        ? (filtered_maps_enabled ? std::size_t{2} : std::size_t{1})
        : std::size_t{0};
    const std::size_t observation_product_map_count =
        plan.effective.products_enabled
        ? checked_noise_count_product(
              observation_noise_generated ? observation_map_count
                                          : std::size_t{0},
              observation_output_stage_count,
              "observation empirical product maps")
        : std::size_t{0};
    const std::size_t coadd_product_map_count =
        plan.effective.products_enabled
        ? checked_noise_count_product(
              coadd_map_count, coadd_output_stage_count,
              "coadd empirical product maps")
        : std::size_t{0};
    const std::size_t product_map_count =
        observation_product_map_count + coadd_product_map_count;
    const std::size_t observation_realization_image_write_count =
        plan.effective.write_realizations
        ? checked_noise_count_product(
              observation_realization_count,
              observation_output_stage_count,
              "observation realization image writes")
        : std::size_t{0};
    const std::size_t coadd_realization_image_write_count =
        plan.effective.write_realizations
        ? checked_noise_count_product(
              coadd_realization_count, coadd_output_stage_count,
              "coadd realization image writes")
        : std::size_t{0};
    const std::size_t realization_image_write_count =
        observation_realization_image_write_count +
        coadd_realization_image_write_count;

    plan.realized.reduction_completed = true;
    plan.realized.generation_executed = noise_count > 0;
    plan.realized.noise_maps_per_scientific_map = noise_count;
    plan.realized.observation_scientific_map_count =
        observation_noise_generated ? observation_map_count : 0;
    plan.realized.observation_noise_realization_count =
        observation_realization_count;
    plan.realized.coadd_scientific_map_count = coadd_map_count;
    plan.realized.coadd_noise_realization_count =
        coadd_realization_count;
    plan.realized.total_noise_realization_count =
        total_realization_count;
    plan.realized.empirical_product_map_count = product_map_count;
    plan.realized.realization_image_write_count =
        realization_image_write_count;
    plan.realized.actual_completion_valid = true;
    plan.realized.completed_count_matches_effective = true;
    plan.realized.uncertainty_use_valid = noise_count >= 2;
    plan.realized.completion_basis =
        "successful_pipeline_return_under_effective_plan";
    plan.realized.outputs_completed = true;
}

}  // namespace citlali::pipeline
