#pragma once

namespace citlali::pipeline {

template <class Engine, class Logger>
void calculate_raw_observation_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    if (engine.run_mapmaking &&
        engine.run_noise_products &&
        engine.run_noise) {
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

template <auto FilteredObsMap, class Engine, class Logger>
void filter_observation_maps(Engine &engine, const Logger &logger) {
    logger->info("filtering obs maps");
    engine.template run_wiener_filter<FilteredObsMap>(engine.omb);
}

template <class Engine, class Logger>
void calculate_filtered_observation_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    if (engine.run_noise_products &&
        engine.run_noise &&
        !engine.write_filtered_maps_partial) {
        logger->info("calculating filtered obs empirical noise products");
        engine.omb.calc_noise_products(
            engine.apply_empirical_noise_weights ||
            engine.wiener_filter.normalize_error);
    }
}

}  // namespace citlali::pipeline
