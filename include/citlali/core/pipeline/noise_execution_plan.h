#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline constexpr std::uint32_t noise_random_seed = 5489U;
inline constexpr const char *noise_random_engine_name =
    "boost::random::mt19937";
inline constexpr const char *noise_seed_policy_name =
    "fixed_internal_default";
inline constexpr const char *noise_generator_scope_name =
    "reduction_pipeline_invocation";

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
        initialized = true;
        requested = request;
        effective = request;
        if (!mapmaking_enabled || !request.enabled) {
            effective.enabled = false;
            effective.n_noise_maps = 0;
        }
        effective_resolution = NoiseEffectiveResolutionRecord{
            mapmaking_enabled,
            request.enabled,
            effective.enabled,
            request.enabled && !mapmaking_enabled,
            request.n_noise_maps,
            effective.n_noise_maps,
            request.n_noise_maps != effective.n_noise_maps,
            noise_random_seed,
            noise_random_engine_name,
            noise_seed_policy_name,
            noise_generator_scope_name,
        };
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
        return;
    }
    if (plan.effective.n_noise_maps < 0) {
        throw std::logic_error("effective noise-map count is negative");
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
    plan.realized.outputs_completed = true;
}

}  // namespace citlali::pipeline
