#pragma once

#include <citlali/core/pipeline/filtered_map_outputs.h>
#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool should_calculate_filtered_observation_noise_products(
    const Engine &engine) {
    return should_calculate_filtered_map_noise_products(engine);
}

template <class Engine>
bool should_find_filtered_observation_sources(const Engine &engine) {
    return should_find_filtered_map_sources(engine);
}

template <class Engine>
bool filtered_observation_maps_written_during_filtering(
    const Engine &engine) {
    return filtered_map_written_during_filtering(engine);
}

template <class Engine>
bool filtered_observation_noise_products_apply_empirical_weights(
    const Engine &engine) {
    return filtered_map_noise_products_apply_empirical_weights(engine);
}

template <auto FilteredObsMap, class Engine, class Logger>
void filter_observation_maps(Engine &engine, const Logger &logger) {
    filter_maps<FilteredObsMap>(
        engine, engine.omb, logger, "filtering obs maps");
}

template <class Engine, class Logger>
void calculate_filtered_observation_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    calculate_filtered_map_noise_products_if_needed(
        engine, engine.omb, logger,
        "calculating filtered obs empirical noise products");
}

template <class Engine, class Logger>
void calculate_filtered_observation_map_diagnostics(Engine &engine,
                                                    const Logger &logger) {
    calculate_filtered_map_diagnostics(
        engine.omb, logger, "calculating filtered obs map psds",
        "calculating filtered obs map histograms");
}

template <bool FitMaps, class Engine>
void fit_filtered_observation_maps_if_requested(Engine &engine) {
    if constexpr (FitMaps) {
        engine.fit_maps();
    }
}

template <auto FilteredObsMap, bool FitMaps, class Engine, class Logger>
void find_and_fit_filtered_observation_maps_if_needed(
    Engine &engine, const Logger &logger) {
    find_filtered_map_sources_if_needed<FilteredObsMap>(
        engine, engine.omb, logger, "finding filtered obs map sources");

    fit_filtered_observation_maps_if_requested<FitMaps>(engine);
}

template <auto FilteredObsMap, class Engine, class Logger>
void output_filtered_observation_maps_if_needed(Engine &engine,
                                                const Logger &logger) {
    output_filtered_maps_if_needed<FilteredObsMap>(
        engine, logger, "outputting filtered obs files",
        "filtered obs files already written during Wiener filtering; "
        "skipping post-filter output stage");
}

template <auto FilteredObsMap, bool FitMaps, class TodProc, class Logger>
void write_filtered_observation_outputs(TodProc &todproc,
                                        const Logger &logger) {
    auto &engine = todproc.engine();

    filter_observation_maps<FilteredObsMap>(engine, logger);
    calculate_filtered_observation_noise_products_if_needed(engine, logger);
    calculate_filtered_observation_map_diagnostics(engine, logger);
    find_and_fit_filtered_observation_maps_if_needed<FilteredObsMap,
                                                     FitMaps>(engine, logger);
    output_filtered_observation_maps_if_needed<FilteredObsMap>(engine, logger);
}

template <auto FilteredObsMap, bool FitMaps, class TodProc, class Logger>
void write_filtered_observation_outputs_if_needed(TodProc &todproc,
                                                  const Logger &logger) {
    auto &engine = todproc.engine();

    if (should_write_filtered_outputs(engine)) {
        write_filtered_observation_outputs<FilteredObsMap, FitMaps>(
            todproc, logger);
    }
}

}  // namespace citlali::pipeline
