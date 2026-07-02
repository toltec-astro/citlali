#pragma once

namespace citlali::pipeline {

template <class Engine>
bool should_calculate_raw_observation_noise_products(const Engine &engine) {
    return engine.run_mapmaking &&
           engine.run_noise_products &&
           engine.run_noise;
}

template <class Engine, class Logger>
void calculate_raw_observation_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    if (should_calculate_raw_observation_noise_products(engine)) {
        logger->info("calculating raw obs empirical noise products");
        engine.omb.calc_noise_products(engine.apply_empirical_noise_weights);
    }
}

template <auto RawObsMap, class Engine, class Logger>
void output_raw_observation_maps_if_needed(Engine &engine,
                                           const Logger &logger) {
    if (engine.run_mapmaking) {
        engine.create_obs_map_files();
        logger->info("outputting raw obs files");
        engine.template output<RawObsMap>();
    }
    else {
        logger->info("mapmaking disabled; skipping raw obs map output");
    }
}

template <auto RawObsMap, class TodProc, class Logger>
void write_raw_observation_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    calculate_raw_observation_noise_products_if_needed(engine, logger);
    output_raw_observation_maps_if_needed<RawObsMap>(engine, logger);
}

}  // namespace citlali::pipeline
