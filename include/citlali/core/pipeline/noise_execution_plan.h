#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>
#include <citlali/core/pipeline/noise_realization_identity.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {

struct NoiseEffectiveResolutionRecord {
    bool mapmaking_enabled = false;
    bool requested_enabled = false;
    bool effective_enabled = false;
    bool disabled_by_mapmaking = false;
    int requested_n_noise_maps = 0;
    int effective_n_noise_maps = 0;
    bool count_zeroed_while_disabled = false;
    bool outputs_suppressed_while_disabled = false;
    std::uint32_t random_seed = noise_random_seed;
    std::string random_engine = noise_random_engine_name;
    std::string seed_policy = noise_seed_policy_name;
    std::string generator_scope = noise_generator_scope_name;
};

struct NoiseRealizedState {
    bool reduction_completed = false;
    bool generation_executed = false;
    bool zero_work = false;
    bool outputs_promised = false;
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
    std::vector<NoiseAssignmentRecord> assignments;
    NoiseRealizedState realized;

    void reset_from_request(
        const citlali::config::NoiseConfig &request,
        bool mapmaking_enabled) {
        if (request.n_noise_maps < 0 ||
            (request.enabled && request.n_noise_maps < 1)) {
            throw std::logic_error(
                "noise-map count is invalid for the requested state");
        }
        initialized = true;
        requested = request;
        effective = request;
        if (!mapmaking_enabled || !request.enabled) {
            effective.enabled = false;
            effective.n_noise_maps = 0;
            effective.write_realizations = false;
            effective.products_enabled = false;
            effective.apply_empirical_weights = false;
        }
        effective_resolution = NoiseEffectiveResolutionRecord{
            mapmaking_enabled,
            request.enabled,
            effective.enabled,
            request.enabled && !mapmaking_enabled,
            request.n_noise_maps,
            effective.n_noise_maps,
            request.n_noise_maps != effective.n_noise_maps,
            !effective.enabled &&
                (request.enabled || request.write_realizations ||
                 request.products_enabled ||
                 request.apply_empirical_weights),
            noise_random_seed,
            noise_random_engine_name,
            noise_seed_policy_name,
            noise_generator_scope_name,
        };
        assignments.clear();
        realized = {};
    }
};

inline void record_noise_assignment_completed(
    NoiseExecutionPlan &plan, const NoiseAssignmentContext &context) {
    if (!plan.initialized || !plan.effective.enabled) {
        throw std::logic_error(
            "cannot record an assignment for disabled noise maps");
    }
    if (context.n_realizations != plan.effective.n_noise_maps ||
        context.ensemble_mode !=
            noise_ensemble_mode_source_imprinted_current) {
        throw std::logic_error(
            "noise assignment differs from the effective realization plan");
    }
    const auto record = make_noise_assignment_record(context);
    const auto same_named_pass = [&](const auto &existing) {
        return existing.observation_id == record.observation_id &&
               existing.conditioning_iteration ==
                   record.conditioning_iteration &&
               existing.pass_id == record.pass_id;
    };
    const auto found = std::find_if(
        plan.assignments.begin(), plan.assignments.end(), same_named_pass);
    if (found != plan.assignments.end()) {
        if (found->reconstruction_digest != record.reconstruction_digest) {
            throw std::logic_error(
                "noise assignment changed within one named pass");
        }
        return;
    }
    plan.assignments.push_back(record);
    std::sort(
        plan.assignments.begin(), plan.assignments.end(),
        [](const auto &left, const auto &right) {
            return noise_assignment_record_sort_key(left) <
                   noise_assignment_record_sort_key(right);
        });
}

inline NoiseRealizationProductIdentity noise_realization_product_identity(
    const NoiseExecutionPlan &plan, const std::string &observation_id,
    bool coadd, std::size_t realization_id) {
    if (!plan.initialized || !plan.effective.enabled) {
        throw std::logic_error(
            "noise realization product identity requires enabled provenance");
    }
    return noise_realization_product_identity(
        plan.assignments, observation_id, coadd, realization_id);
}

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
        plan.realized.zero_work = true;
        plan.realized.outputs_promised = false;
        plan.realized.noise_maps_per_scientific_map = std::size_t{0};
        plan.realized.observation_scientific_map_count = std::size_t{0};
        plan.realized.observation_noise_realization_count = std::size_t{0};
        plan.realized.coadd_scientific_map_count = std::size_t{0};
        plan.realized.coadd_noise_realization_count = std::size_t{0};
        plan.realized.total_noise_realization_count = std::size_t{0};
        plan.realized.empirical_product_map_count = std::size_t{0};
        plan.realized.realization_image_write_count = std::size_t{0};
        return;
    }
    if (plan.effective.n_noise_maps < 1) {
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
        const bool assignment_completed = std::any_of(
            plan.assignments.begin(), plan.assignments.end(),
            [&](const auto &assignment) {
                const auto expected_count = static_cast<std::size_t>(
                    plan.effective.n_noise_maps);
                if (assignment.observation_id != observation.obsnum ||
                    !assignment.compact() ||
                    assignment.completed_realization_ids.size() !=
                        expected_count ||
                    assignment.reconstruction_digest !=
                        noise_assignment_record_reconstruction_digest(
                            assignment)) {
                    return false;
                }
                for (std::size_t realization = 0;
                     realization < expected_count; ++realization) {
                    if (assignment.completed_realization_ids[realization] !=
                        realization) {
                        return false;
                    }
                }
                return true;
            });
        if (!assignment_completed) {
            throw std::logic_error(
                "noise completion lacks an observation assignment record");
        }
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
    plan.realized.zero_work = false;
    plan.realized.outputs_promised = true;
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
