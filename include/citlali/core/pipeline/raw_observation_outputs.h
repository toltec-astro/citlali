#pragma once

#include <citlali/core/pipeline/raw_map_outputs.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class Engine>
bool should_output_raw_observation_maps(const Engine &engine) {
    return mapmaking_outputs_enabled(engine);
}

template <class Engine, class Logger>
void calculate_raw_observation_noise_products_if_needed(
    Engine &engine, StageProfileCollector &stage_profile,
    const Logger &logger) {
    calculate_unfiltered_map_noise_products_if_needed(
        engine, engine.omb, stage_profile, logger, true,
        "calculating raw obs empirical noise products");
}

template <auto RawObsMap, class Engine, class Logger>
void output_raw_observation_maps_if_needed(Engine &engine,
                                           StageProfileCollector &stage_profile,
                                           const Logger &logger) {
    const auto should_output = should_output_raw_observation_maps(engine);
    if (should_output) {
        engine.create_obs_map_files();
    }
    output_unfiltered_maps_if_needed<RawObsMap>(
        engine, stage_profile, logger, should_output,
        "outputting raw obs files",
        "mapmaking disabled; skipping raw obs map output");
}

template <auto RawObsMap, class TodProc, class Logger>
void write_raw_observation_outputs(TodProc &todproc,
                                   StageProfileCollector &stage_profile,
                                   const Logger &logger) {
    auto &engine = todproc.engine();
    (void)stage_profile;
    const auto profile_scope =
        profile_stage("raw_observation.outputs", logger);

    calculate_raw_observation_noise_products_if_needed(
        engine, stage_profile, logger);
    output_raw_observation_maps_if_needed<RawObsMap>(
        engine, stage_profile, logger);
}

}  // namespace citlali::pipeline
