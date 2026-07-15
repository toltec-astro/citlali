#pragma once

#include <citlali/core/pipeline/filtered_map_outputs.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/pointing_execution_plan.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <auto FilteredObsMap, class Engine, class Logger>
void filter_observation_maps(Engine &engine,
                             StageProfileCollector &stage_profile,
                             const Logger &logger) {
    filter_maps<FilteredObsMap>(
        engine, engine.omb, stage_profile, logger, "filtering obs maps");
    record_post_processing_filter_completed_if_available(
        engine, PostProcessingMapContext::observation);
}

template <class Engine, class Logger>
void calculate_filtered_observation_noise_products_if_needed(
    Engine &engine, StageProfileCollector &stage_profile,
    const Logger &logger) {
    calculate_filtered_map_noise_products_if_needed(
        engine, engine.omb, stage_profile, logger,
        "calculating filtered obs empirical noise products");
}

template <class Engine, class Logger>
void calculate_filtered_observation_map_diagnostics(Engine &engine,
                                                    StageProfileCollector &stage_profile,
                                                    const Logger &logger) {
    calculate_filtered_map_diagnostics(
        engine.omb, stage_profile, logger,
        "calculating filtered obs map psds",
        "calculating filtered obs map histograms");
}

template <bool FitMaps, class Engine, class Logger>
void fit_filtered_observation_maps_if_requested(Engine &engine,
                                                StageProfileCollector &stage_profile,
                                                const Logger &logger) {
    (void)stage_profile;
    if constexpr (FitMaps) {
        const auto profile_scope =
            profile_stage(stage_profile, "filtered_observation.fit_maps", logger);
        engine.fit_maps(PointingFitStage::filtered_observation);
    }
}

template <auto FilteredObsMap, bool FitMaps, class Engine, class Logger>
void find_and_fit_filtered_observation_maps_if_needed(
    Engine &engine, StageProfileCollector &stage_profile,
    const Logger &logger) {
    find_filtered_map_sources_if_needed<FilteredObsMap>(
        engine, engine.omb, stage_profile, logger,
        "finding filtered obs map sources",
        PostProcessingMapContext::observation);

    fit_filtered_observation_maps_if_requested<FitMaps>(
        engine, stage_profile, logger);
}

template <auto FilteredObsMap, class Engine, class Logger>
void output_filtered_observation_maps_if_needed(Engine &engine,
                                                StageProfileCollector &stage_profile,
                                                const Logger &logger) {
    output_filtered_maps_if_needed<FilteredObsMap>(
        engine, stage_profile, logger, "outputting filtered obs files",
        "filtered obs files already written during Wiener filtering; "
        "skipping post-filter output stage");
}

template <auto FilteredObsMap, bool FitMaps, class TodProc, class Logger>
void write_filtered_observation_outputs(TodProc &todproc,
                                        StageProfileCollector &stage_profile,
                                        const Logger &logger) {
    auto &engine = todproc.engine();
    (void)stage_profile;
    const auto profile_scope =
        profile_stage(stage_profile, "filtered_observation.outputs", logger);

    filter_observation_maps<FilteredObsMap>(engine, stage_profile, logger);
    calculate_filtered_observation_noise_products_if_needed(
        engine, stage_profile, logger);
    calculate_filtered_observation_map_diagnostics(
        engine, stage_profile, logger);
    find_and_fit_filtered_observation_maps_if_needed<FilteredObsMap,
                                                     FitMaps>(
        engine, stage_profile, logger);
    output_filtered_observation_maps_if_needed<FilteredObsMap>(
        engine, stage_profile, logger);
}

template <auto FilteredObsMap, bool FitMaps, class TodProc, class Logger>
void write_filtered_observation_outputs_if_needed(TodProc &todproc,
                                                  StageProfileCollector &stage_profile,
                                                  const Logger &logger) {
    auto &engine = todproc.engine();

    if (should_write_filtered_outputs(engine)) {
        write_filtered_observation_outputs<FilteredObsMap, FitMaps>(
            todproc, stage_profile, logger);
    }
}

}  // namespace citlali::pipeline
