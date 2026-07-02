#pragma once

namespace citlali::pipeline {

template <class Engine>
bool should_run_observation_tod(const Engine &engine) {
    return engine.run_tod;
}

template <class Engine, class Logger>
void setup_observation_pipeline(Engine &engine, const Logger &logger) {
    logger->info("pipeline setup");
    engine.setup();
}

template <class Engine, class KidsProc, class RawObs, class Logger>
void run_observation_tod_pipeline(Engine &engine, KidsProc &kidsproc,
                                  const RawObs &rawobs,
                                  const Logger &logger) {
    logger->info("running pipeline");
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
