#pragma once

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/raw_observation_outputs.h>

namespace citlali::pipeline {

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

template <auto FilteredObsMap, bool FitMaps, class Engine, class Logger>
void find_and_fit_filtered_observation_maps_if_needed(
    Engine &engine, const Logger &logger) {
    if (engine.run_source_finder) {
        logger->info("finding filtered obs map sources");
        engine.template find_sources<FilteredObsMap>(engine.omb);
    }

    if constexpr (FitMaps) {
        engine.fit_maps();
    }
}

template <auto FilteredObsMap, class Engine, class Logger>
void output_filtered_observation_maps_if_needed(Engine &engine,
                                                const Logger &logger) {
    if (engine.write_filtered_maps_partial) {
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

template <class TodProc, class Logger>
void coadd_observation(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("coadding");
    if (!engine.rtcproc.run_polarization) {
        todproc.coadd();
    }
}

template <auto RawObsMap, auto FilteredObsMap, bool FitMaps, class TodProc,
          class Logger>
void write_observation_outputs_and_accumulate(TodProc &todproc,
                                              const Logger &logger) {
    auto &engine = todproc.engine();

    write_raw_observation_outputs<RawObsMap>(todproc, logger);

    if (engine.run_coadd) {
        coadd_observation(todproc, logger);
    }
    else {
        write_filtered_observation_outputs_if_needed<FilteredObsMap, FitMaps>(
            todproc, logger);
    }
}

}  // namespace citlali::pipeline
