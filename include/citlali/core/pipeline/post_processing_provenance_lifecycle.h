#pragma once

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>
#include <citlali/core/pipeline/post_processing_execution_plan.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <cstddef>
#include <stdexcept>

namespace citlali::pipeline {

enum class PostProcessingMapContext {
    observation,
    coadd,
};

inline PostProcessingMapContextRealizedState &post_processing_map_context(
    PostProcessingExecutionPlan &plan, PostProcessingMapContext context) {
    return context == PostProcessingMapContext::observation
        ? plan.realized.observation
        : plan.realized.coadd;
}

inline void require_post_processing_plan(
    const PostProcessingExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error("post-processing plan is not initialized");
    }
    if (plan.realized.reduction_completed) {
        throw std::logic_error(
            "post-processing reduction is already completed");
    }
}

inline void record_post_processing_filter_completed(
    PostProcessingExecutionPlan &plan, PostProcessingMapContext context,
    std::size_t map_count) {
    require_post_processing_plan(plan);
    if (!plan.effective.map_filtering.enabled || map_count == 0) {
        throw std::logic_error(
            "completed map filtering requires enabled policy and maps");
    }
    auto &state = post_processing_map_context(plan, context);
    ++state.filter_context_count;
    state.filtered_map_count += map_count;
}

template <class Engine>
void record_post_processing_filter_completed_if_available(
    Engine &engine, PostProcessingMapContext context,
    std::size_t map_count) {
    if constexpr (has_post_processing_plan_v<Engine>) {
        auto &plan = post_processing_plan(engine);
        if (plan.initialized) {
            record_post_processing_filter_completed(
                plan, context, map_count);
        }
    }
}

template <class Engine>
void record_post_processing_filter_completed_if_available(
    Engine &engine, PostProcessingMapContext context) {
    if constexpr (has_post_processing_plan_v<Engine>) {
        auto &plan = post_processing_plan(engine);
        if (plan.initialized) {
            record_post_processing_filter_completed(
                plan, context,
                static_cast<std::size_t>(engine.map_indices.n_maps));
        }
    }
}

inline void record_post_processing_catalog_fits_completed(
    PostProcessingExecutionPlan &plan, PostProcessingMapContext context,
    std::size_t attempt_count, std::size_t valid_count) {
    require_post_processing_plan(plan);
    if (!plan.effective.source_finding.enabled) {
        throw std::logic_error(
            "completed source finding requires enabled policy");
    }
    if (valid_count > attempt_count) {
        throw std::logic_error(
            "valid source fits exceed attempted source fits");
    }
    auto &state = post_processing_map_context(plan, context);
    if (state.source_finding_context_count >= state.filter_context_count) {
        throw std::logic_error(
            "source finding completed without a new filtered context");
    }
    ++state.source_finding_context_count;
    state.detected_source_count += attempt_count;
    ++state.catalog_fits.context_count;
    state.catalog_fits.attempt_count += attempt_count;
    state.catalog_fits.valid_count += valid_count;
}

template <class Engine>
void record_post_processing_catalog_fits_completed_if_available(
    Engine &engine, PostProcessingMapContext context,
    std::size_t attempt_count, std::size_t valid_count) {
    if constexpr (has_post_processing_plan_v<Engine>) {
        auto &plan = post_processing_plan(engine);
        if (plan.initialized) {
            record_post_processing_catalog_fits_completed(
                plan, context, attempt_count, valid_count);
        }
    }
}

inline void record_post_processing_source_table_written(
    PostProcessingExecutionPlan &plan, PostProcessingMapContext context,
    std::size_t row_count) {
    require_post_processing_plan(plan);
    auto &state = post_processing_map_context(plan, context);
    if (state.source_table_write_count >=
        state.source_finding_context_count) {
        throw std::logic_error(
            "source table written without a new source-finding context");
    }
    ++state.source_table_write_count;
    state.source_table_row_count += row_count;
}

inline void add_post_processing_fit_cardinality(
    PostProcessingFitCardinality &cardinality, std::size_t attempt_count,
    std::size_t valid_count) {
    if (valid_count > attempt_count) {
        throw std::logic_error("valid fits exceed attempted fits");
    }
    ++cardinality.context_count;
    cardinality.attempt_count += attempt_count;
    cardinality.valid_count += valid_count;
}

inline void record_post_processing_pointing_fits_completed(
    PostProcessingExecutionPlan &plan, bool filtered,
    std::size_t attempt_count, std::size_t valid_count) {
    require_post_processing_plan(plan);
    add_post_processing_fit_cardinality(
        filtered ? plan.realized.pointing_filtered_fits
                 : plan.realized.pointing_raw_fits,
        attempt_count, valid_count);
}

inline void record_post_processing_beammap_fits_completed(
    PostProcessingExecutionPlan &plan, std::size_t attempt_count,
    std::size_t valid_count) {
    require_post_processing_plan(plan);
    add_post_processing_fit_cardinality(
        plan.realized.beammap_fits, attempt_count, valid_count);
}

inline void require_map_context_cardinality(
    const PostProcessingMapContextRealizedState &state,
    std::size_t expected_contexts, std::size_t expected_maps,
    bool source_finding_enabled) {
    if (state.filter_context_count != expected_contexts ||
        state.filtered_map_count != expected_maps) {
        throw std::logic_error(
            "post-processing filter cardinality is incomplete");
    }
    const std::size_t expected_source_contexts =
        source_finding_enabled ? expected_contexts : 0;
    if (state.source_finding_context_count != expected_source_contexts ||
        state.catalog_fits.context_count != expected_source_contexts ||
        state.source_table_write_count != expected_source_contexts) {
        throw std::logic_error(
            "post-processing source product cardinality is incomplete");
    }
    if (state.catalog_fits.attempt_count != state.detected_source_count ||
        state.catalog_fits.valid_count > state.catalog_fits.attempt_count ||
        state.source_table_row_count != state.detected_source_count) {
        throw std::logic_error(
            "post-processing source row cardinality is inconsistent");
    }
}

inline void record_post_processing_run_completed(
    PostProcessingExecutionPlan &plan,
    const MapmakingExecutionPlan &mapmaking_plan) {
    require_post_processing_plan(plan);
    if (!mapmaking_plan.initialized ||
        !mapmaking_plan.realized.reduction_completed) {
        throw std::logic_error(
            "post-processing completion requires completed mapmaking");
    }

    std::size_t expected_observation_contexts = 0;
    std::size_t expected_observation_maps = 0;
    std::size_t expected_coadd_contexts = 0;
    std::size_t expected_coadd_maps = 0;
    if (plan.effective.map_filtering.enabled) {
        if (plan.effective_resolution.coadd_enabled) {
            expected_coadd_contexts = 1;
            if (!mapmaking_plan.coadd.has_value()) {
                throw std::logic_error(
                    "filtered coadd provenance requires a mapmaking coadd");
            }
            expected_coadd_maps = mapmaking_plan.coadd->map_count;
        }
        else {
            expected_observation_contexts =
                mapmaking_plan.observations.size();
            for (const auto &observation : mapmaking_plan.observations) {
                expected_observation_maps += observation.map_count;
            }
        }
    }

    require_map_context_cardinality(
        plan.realized.observation, expected_observation_contexts,
        expected_observation_maps, plan.effective.source_finding.enabled);
    require_map_context_cardinality(
        plan.realized.coadd, expected_coadd_contexts, expected_coadd_maps,
        plan.effective.source_finding.enabled);

    const auto reduction_type = plan.effective_resolution.reduction_type;
    if (citlali::config::is_pointing_reduction_type(reduction_type)) {
        const auto expected_raw = mapmaking_plan.observations.size();
        const auto expected_filtered =
            plan.effective.map_filtering.enabled &&
                    !plan.effective_resolution.coadd_enabled
                ? expected_raw
                : 0;
        if (plan.realized.pointing_raw_fits.context_count != expected_raw ||
            plan.realized.pointing_filtered_fits.context_count !=
                expected_filtered) {
            throw std::logic_error(
                "post-processing pointing fit cardinality is incomplete");
        }
    }
    else if (plan.realized.pointing_raw_fits.context_count != 0 ||
             plan.realized.pointing_filtered_fits.context_count != 0) {
        throw std::logic_error(
            "non-pointing reduction recorded pointing fits");
    }

    if (citlali::config::is_beammap_reduction_type(reduction_type) &&
        plan.effective_resolution.mapmaking_enabled) {
        if (plan.realized.beammap_fits.context_count == 0) {
            throw std::logic_error(
                "beammap reduction recorded no fitting contexts");
        }
    }
    else if (plan.realized.beammap_fits.context_count != 0) {
        throw std::logic_error(
            "non-beammap reduction recorded beammap fits");
    }

    plan.realized.outputs_completed = true;
    plan.realized.reduction_completed = true;
}

template <class Engine>
void begin_post_processing_iteration_if_available(Engine &engine) {
    if constexpr (has_post_processing_plan_v<Engine>) {
        auto &plan = post_processing_plan(engine);
        if (plan.initialized) {
            plan.begin_iteration();
        }
    }
}

}  // namespace citlali::pipeline
