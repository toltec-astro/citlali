#pragma once

#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool should_calculate_filtered_observation_noise_products(
    const Engine &engine) {
    return engine.run_noise_products &&
           engine.run_noise &&
           !engine.write_filtered_maps_partial;
}

template <class Engine>
bool should_find_filtered_observation_sources(const Engine &engine) {
    return engine.run_source_finder;
}

template <class Engine>
bool filtered_observation_maps_written_during_filtering(
    const Engine &engine) {
    return engine.write_filtered_maps_partial;
}

template <class Engine>
bool filtered_observation_noise_products_apply_empirical_weights(
    const Engine &engine) {
    return engine.apply_empirical_noise_weights ||
           engine.wiener_filter.normalize_error;
}

template <auto FilteredObsMap, class Engine, class Logger>
void filter_observation_maps(Engine &engine, const Logger &logger) {
    logger->info("filtering obs maps");
    engine.template run_wiener_filter<FilteredObsMap>(engine.omb);
}

template <class Engine, class Logger>
void calculate_filtered_observation_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    if (should_calculate_filtered_observation_noise_products(engine)) {
        logger->info("calculating filtered obs empirical noise products");
        engine.omb.calc_noise_products(
            filtered_observation_noise_products_apply_empirical_weights(
                engine));
    }
}

template <class Engine, class Logger>
void calculate_filtered_observation_map_diagnostics(Engine &engine,
                                                    const Logger &logger) {
    logger->info("calculating filtered obs map psds");
    engine.omb.calc_map_psd();
    logger->info("calculating filtered obs map histograms");
    engine.omb.calc_map_hist();

    engine.omb.calc_median_err();
    engine.omb.calc_median_rms();
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
    if (should_find_filtered_observation_sources(engine)) {
        logger->info("finding filtered obs map sources");
        engine.template find_sources<FilteredObsMap>(engine.omb);
    }

    fit_filtered_observation_maps_if_requested<FitMaps>(engine);
}

template <auto FilteredObsMap, class Engine, class Logger>
void output_filtered_observation_maps_if_needed(Engine &engine,
                                                const Logger &logger) {
    if (filtered_observation_maps_written_during_filtering(engine)) {
        logger->info(
            "filtered obs files already written during Wiener filtering; "
            "skipping post-filter output stage");
    }
    else {
        logger->info("outputting filtered obs files");
        engine.template output<FilteredObsMap>();
    }
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
