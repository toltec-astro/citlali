#pragma once

#include <citlali/core/pipeline/map_noise_products.h>
#include <citlali/core/pipeline/map_output.h>
#include <citlali/core/pipeline/noise_weight_policy.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class Engine>
bool should_calculate_unfiltered_map_noise_products(
    const Engine &engine, bool require_mapmaking) {
    return (!require_mapmaking || mapmaking_outputs_enabled(engine)) &&
           noise_product_outputs_enabled(engine) &&
           noise_maps_enabled(engine);
}

template <class Engine>
bool unfiltered_map_noise_products_apply_empirical_weights(
    const Engine &engine) {
    return unfiltered_noise_products_apply_empirical_weights(engine);
}

template <class Engine, class MapBuffer, class Logger>
void calculate_unfiltered_map_noise_products_if_needed(
    Engine &engine, MapBuffer &map_buffer,
    StageProfileCollector &stage_profile, const Logger &logger,
    bool require_mapmaking, const char *log_message) {
    calculate_map_noise_products_if_needed(
        map_buffer,
        should_calculate_unfiltered_map_noise_products(
            engine, require_mapmaking),
        unfiltered_map_noise_products_apply_empirical_weights(engine),
        stage_profile, logger, log_message);
}

template <auto RawMap, class Engine, class Logger>
void output_unfiltered_maps_with_log(
    Engine &engine, StageProfileCollector &stage_profile,
    const Logger &logger,
                                     const char *log_message) {
    output_map_with_log<RawMap>(engine, stage_profile, logger, log_message);
}

template <auto RawMap, class Engine, class Logger>
void output_unfiltered_maps_if_needed(
    Engine &engine, StageProfileCollector &stage_profile,
    const Logger &logger, bool should_output,
    const char *output_log_message, const char *skip_log_message) {
    output_map_if_needed<RawMap>(
        engine, stage_profile, logger, should_output, output_log_message,
        skip_log_message);
}

}  // namespace citlali::pipeline
