#pragma once

#include <citlali/core/pipeline/map_diagnostics.h>
#include <citlali/core/pipeline/map_filtering.h>
#include <citlali/core/pipeline/map_noise_products.h>
#include <citlali/core/pipeline/map_output.h>
#include <citlali/core/pipeline/map_source_finding.h>
#include <citlali/core/pipeline/noise_weight_policy.h>
#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool should_calculate_filtered_map_noise_products(const Engine &engine) {
    return should_calculate_filtered_noise_products(engine);
}

template <class Engine>
bool should_find_filtered_map_sources(const Engine &engine) {
    return should_find_filtered_sources(engine);
}

template <class Engine>
bool filtered_map_written_during_filtering(const Engine &engine) {
    return filtered_maps_written_during_filtering(engine);
}

template <class Engine>
bool filtered_map_noise_products_apply_empirical_weights(
    const Engine &engine) {
    return filtered_noise_products_apply_empirical_weights(engine);
}

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void filter_maps(Engine &engine, MapBuffer &map_buffer,
                 const Logger &logger, const char *log_message) {
    run_wiener_filter_with_log<FilteredMap>(
        engine, map_buffer, logger, log_message);
}

template <class Engine, class MapBuffer, class Logger>
void calculate_filtered_map_noise_products_if_needed(
    Engine &engine, MapBuffer &map_buffer, const Logger &logger,
    const char *log_message) {
    if (should_calculate_filtered_map_noise_products(engine)) {
        calculate_map_noise_products_with_log(
            map_buffer,
            filtered_map_noise_products_apply_empirical_weights(engine),
            logger, log_message);
    }
}

template <class MapBuffer, class Logger>
void calculate_filtered_map_diagnostics(
    MapBuffer &map_buffer, const Logger &logger, const char *psd_log_message,
    const char *histogram_log_message) {
    calculate_map_diagnostics(
        map_buffer, logger, psd_log_message, histogram_log_message);
}

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void find_filtered_map_sources_if_needed(
    Engine &engine, MapBuffer &map_buffer, const Logger &logger,
    const char *log_message) {
    if (should_find_filtered_map_sources(engine)) {
        find_map_sources_with_log<FilteredMap>(
            engine, map_buffer, logger, log_message);
    }
}

template <auto FilteredMap, class Engine, class Logger>
void output_filtered_maps_if_needed(
    Engine &engine, const Logger &logger, const char *output_log_message,
    const char *skip_log_message) {
    output_map_if_needed<FilteredMap>(
        engine, logger, !filtered_map_written_during_filtering(engine),
        output_log_message, skip_log_message);
}

}  // namespace citlali::pipeline
