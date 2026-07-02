#pragma once

#include <citlali/core/pipeline/map_noise_products.h>
#include <citlali/core/pipeline/map_output.h>
#include <citlali/core/pipeline/noise_weight_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool should_calculate_unfiltered_map_noise_products(
    const Engine &engine, bool require_mapmaking) {
    return (!require_mapmaking || engine.run_mapmaking) &&
           engine.run_noise_products &&
           engine.run_noise;
}

template <class Engine>
bool unfiltered_map_noise_products_apply_empirical_weights(
    const Engine &engine) {
    return unfiltered_noise_products_apply_empirical_weights(engine);
}

template <class Engine, class MapBuffer, class Logger>
void calculate_unfiltered_map_noise_products_if_needed(
    Engine &engine, MapBuffer &map_buffer, const Logger &logger,
    bool require_mapmaking, const char *log_message) {
    if (should_calculate_unfiltered_map_noise_products(
            engine, require_mapmaking)) {
        calculate_map_noise_products_with_log(
            map_buffer,
            unfiltered_map_noise_products_apply_empirical_weights(engine),
            logger, log_message);
    }
}

template <auto RawMap, class Engine, class Logger>
void output_unfiltered_maps_with_log(Engine &engine, const Logger &logger,
                                     const char *log_message) {
    output_map_with_log<RawMap>(engine, logger, log_message);
}

template <auto RawMap, class Engine, class Logger>
void output_unfiltered_maps_if_needed(
    Engine &engine, const Logger &logger, bool should_output,
    const char *output_log_message, const char *skip_log_message) {
    output_map_if_needed<RawMap>(
        engine, logger, should_output, output_log_message, skip_log_message);
}

}  // namespace citlali::pipeline
