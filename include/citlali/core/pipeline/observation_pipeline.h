#pragma once

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class Engine>
bool should_run_observation_tod(const Engine &engine) {
    return timestream_processing_enabled(engine);
}

template <class Engine, class Logger>
void setup_observation_pipeline(Engine &engine, const Logger &logger) {
    logger->info("pipeline setup");
    const auto profile_scope = profile_stage("observation.setup", logger);
    engine.setup();
}

template <class Engine, class KidsProc, class RawObs, class Logger>
void run_observation_tod_pipeline(Engine &engine, KidsProc &kidsproc,
    const RawObs &rawobs,
    const Logger &logger) {
    logger->info("running pipeline");
    const auto profile_scope = profile_stage("observation.tod_pipeline", logger);
    engine.pipeline(kidsproc, rawobs);
}

template <class Engine, class KidsProc, class RawObs, class Logger>
void run_observation_tod_pipeline_if_needed(Engine &engine,
                                            KidsProc &kidsproc,
                                            const RawObs &rawobs,
                                            const Logger &logger) {
    if (should_run_observation_tod(engine)) {
        run_observation_tod_pipeline(engine, kidsproc, rawobs, logger);
    }
}

template <class Engine, class KidsProc, class RawObs, class Logger>
void setup_and_run_observation_pipeline(Engine &engine, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        const Logger &logger) {
    setup_observation_pipeline(engine, logger);
    run_observation_tod_pipeline_if_needed(
        engine, kidsproc, rawobs, logger);
}

}  // namespace citlali::pipeline
