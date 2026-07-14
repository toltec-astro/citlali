#pragma once

#include <citlali/core/config/pointing_config.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

enum class PointingFitStage {
    raw_observation,
    filtered_observation,
};

inline const char *pointing_fit_stage_name(PointingFitStage stage) {
    return stage == PointingFitStage::raw_observation
               ? "raw observation"
               : "filtered observation";
}

struct PointingRequestPresence {
    bool source_strategy = false;
    bool fit_gaussian = false;
    bool fruitloops_center_mode = false;
    bool header_max_radius_arcsec = false;
    bool header_require_coverage = false;
};

struct PointingEffectiveResolutionRecord {
    bool mapmaking_enabled = false;
    bool map_filter_enabled = false;
    bool coadd_enabled = false;
    bool fit_output_path_available = false;
    PointingRequestPresence explicit_request;
    bool fit_disabled_by_mapmaking = false;
    bool fit_disabled_by_output_policy = false;
    double default_header_max_radius_arcsec = 0.0;
    bool header_max_radius_defaulted = false;
};

struct PointingFitResultState {
    std::size_t attempt_count = 0;
    std::size_t valid_count = 0;
    bool recorded = false;
};

struct PointingObservationState {
    std::size_t observation_index = 0;
    std::string obsnum;
    std::size_t map_count = 0;
    PointingFitResultState raw_fit;
    PointingFitResultState filtered_fit;
    bool outputs_completed = false;
};

struct PointingRealizedState {
    bool reduction_completed = false;
    bool pointing_executed = false;
    std::size_t completed_observation_count = 0;
    std::size_t scientific_map_count = 0;
    std::size_t raw_fit_attempt_count = 0;
    std::size_t raw_valid_fit_count = 0;
    std::size_t filtered_fit_attempt_count = 0;
    std::size_t filtered_valid_fit_count = 0;
    bool outputs_completed = false;
};

struct PointingExecutionPlan {
    bool initialized = false;
    citlali::config::PointingConfig requested;
    citlali::config::PointingConfig effective;
    PointingEffectiveResolutionRecord effective_resolution;
    std::vector<PointingObservationState> observations;
    PointingRealizedState realized;

    void reset_from_request(
        const citlali::config::PointingConfig &request,
        const PointingRequestPresence &presence,
        bool mapmaking_enabled, bool map_filter_enabled,
        bool coadd_enabled,
        double default_header_max_radius_arcsec) {
        initialized = true;
        requested = request;
        effective = request;
        const double resolved_default_header_max_radius_arcsec =
            citlali::config::is_standard_pointing_source_strategy(
                request.source_strategy)
                ? default_header_max_radius_arcsec
                : 0.0;
        if (!presence.header_max_radius_arcsec) {
            effective.header_max_radius_arcsec =
                resolved_default_header_max_radius_arcsec;
        }
        // Pointing fits consume normalized observation maps before optional
        // filtering or coaddition, so only mapmaking controls availability.
        const bool fit_output_path_available = mapmaking_enabled;
        effective.fit_gaussian =
            request.fit_gaussian && fit_output_path_available;
        effective_resolution = PointingEffectiveResolutionRecord{
            mapmaking_enabled,
            map_filter_enabled,
            coadd_enabled,
            fit_output_path_available,
            presence,
            request.fit_gaussian && !mapmaking_enabled,
            false,
            resolved_default_header_max_radius_arcsec,
            !presence.header_max_radius_arcsec,
        };
        observations.clear();
        realized = {};
    }

    void begin_iteration() {
        if (!initialized) {
            throw std::logic_error("pointing plan is not initialized");
        }
        observations.clear();
        realized = {};
    }

    PointingObservationState &begin_observation(
        std::size_t observation_index, std::string obsnum,
        std::size_t map_count) {
        if (!initialized || !effective_resolution.mapmaking_enabled) {
            throw std::logic_error(
                "pointing observation requires enabled mapmaking");
        }
        if (map_count == 0) {
            throw std::logic_error(
                "pointing observation map count must be positive");
        }
        observations.push_back(PointingObservationState{
            observation_index, std::move(obsnum), map_count});
        return observations.back();
    }
};

inline PointingFitResultState &pointing_fit_result_state(
    PointingObservationState &observation, PointingFitStage stage) {
    return stage == PointingFitStage::raw_observation
               ? observation.raw_fit
               : observation.filtered_fit;
}

inline bool pointing_fit_stage_expected(
    const PointingExecutionPlan &plan, PointingFitStage stage) {
    if (stage == PointingFitStage::raw_observation) {
        return plan.effective_resolution.mapmaking_enabled;
    }
    return plan.effective_resolution.mapmaking_enabled &&
           plan.effective_resolution.map_filter_enabled &&
           !plan.effective_resolution.coadd_enabled;
}

inline void record_pointing_fit_results(
    PointingExecutionPlan &plan, PointingFitStage stage,
    std::size_t fit_attempt_count, std::size_t valid_fit_count) {
    if (!plan.initialized || plan.observations.empty()) {
        throw std::logic_error(
            "cannot record pointing fits before an observation begins");
    }
    auto &observation = plan.observations.back();
    auto &fit = pointing_fit_result_state(observation, stage);
    if (!pointing_fit_stage_expected(plan, stage)) {
        throw std::logic_error(
            std::string{"pointing fit stage is not expected: "} +
            pointing_fit_stage_name(stage));
    }
    if (fit.recorded) {
        throw std::logic_error(
            std::string{"pointing fit results already recorded: "} +
            pointing_fit_stage_name(stage));
    }
    const std::size_t expected_attempts =
        plan.effective.fit_gaussian ? observation.map_count : 0;
    if (fit_attempt_count != expected_attempts ||
        valid_fit_count > fit_attempt_count) {
        throw std::logic_error(
            "pointing fit cardinality is inconsistent");
    }
    fit.attempt_count = fit_attempt_count;
    fit.valid_count = valid_fit_count;
    fit.recorded = true;
}

inline void complete_pointing_observation(
    PointingExecutionPlan &plan) {
    if (!plan.initialized || plan.observations.empty()) {
        throw std::logic_error(
            "cannot complete pointing observation before it begins");
    }
    auto &observation = plan.observations.back();
    for (const auto stage : {
             PointingFitStage::raw_observation,
             PointingFitStage::filtered_observation}) {
        auto &fit = pointing_fit_result_state(observation, stage);
        const bool expected = pointing_fit_stage_expected(plan, stage);
        if (!expected) {
            if (fit.recorded) {
                throw std::logic_error(
                    std::string{
                        "unexpected pointing fit stage was recorded: "} +
                    pointing_fit_stage_name(stage));
            }
            continue;
        }
        if (!fit.recorded) {
            if (plan.effective.fit_gaussian) {
                throw std::logic_error(
                    std::string{"pointing observation has no fit summary: "} +
                    pointing_fit_stage_name(stage));
            }
            record_pointing_fit_results(plan, stage, 0, 0);
        }
    }
    if (observation.outputs_completed) {
        throw std::logic_error(
            "pointing observation outputs already completed");
    }
    observation.outputs_completed = true;
}

inline void record_pointing_run_completed(
    PointingExecutionPlan &plan,
    const MapmakingExecutionPlan &mapmaking_plan) {
    if (!plan.initialized) {
        throw std::logic_error("pointing plan is not initialized");
    }
    if (!mapmaking_plan.initialized ||
        !mapmaking_plan.realized.reduction_completed) {
        throw std::logic_error(
            "pointing completion requires completed mapmaking provenance");
    }
    if (plan.observations.size() != mapmaking_plan.observations.size()) {
        throw std::logic_error(
            "pointing and mapmaking observation counts differ");
    }

    PointingRealizedState realized;
    realized.pointing_executed =
        plan.effective_resolution.mapmaking_enabled;
    for (std::size_t index = 0; index < plan.observations.size(); ++index) {
        const auto &pointing = plan.observations[index];
        const auto &mapmaking = mapmaking_plan.observations[index];
        if (!pointing.outputs_completed ||
            !mapmaking.outputs_completed ||
            pointing.observation_index != mapmaking.observation_index ||
            pointing.obsnum != mapmaking.obsnum ||
            pointing.map_count != mapmaking.map_count) {
            throw std::logic_error(
                "pointing and mapmaking observation state differs");
        }
        ++realized.completed_observation_count;
        realized.scientific_map_count += pointing.map_count;
        realized.raw_fit_attempt_count +=
            pointing.raw_fit.attempt_count;
        realized.raw_valid_fit_count += pointing.raw_fit.valid_count;
        realized.filtered_fit_attempt_count +=
            pointing.filtered_fit.attempt_count;
        realized.filtered_valid_fit_count +=
            pointing.filtered_fit.valid_count;
    }
    if (plan.effective_resolution.mapmaking_enabled &&
        plan.observations.empty()) {
        throw std::logic_error(
            "enabled pointing completed without observation products");
    }
    if (!plan.effective_resolution.mapmaking_enabled &&
        !plan.observations.empty()) {
        throw std::logic_error(
            "disabled pointing recorded observation products");
    }
    realized.outputs_completed = true;
    realized.reduction_completed = true;
    plan.realized = realized;
}

}  // namespace citlali::pipeline
