#pragma once

#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool should_calculate_filtered_coadd_noise_products(
    const Engine &engine) {
    return engine.run_noise_products &&
           engine.run_noise &&
           !engine.write_filtered_maps_partial;
}

template <class Engine>
bool should_find_filtered_coadd_sources(const Engine &engine) {
    return engine.run_source_finder;
}

template <auto FilteredCoaddMap, class Engine, class Logger>
void filter_coadd_maps(Engine &engine, const Logger &logger) {
    logger->info("filtering coadded maps");
    engine.template run_wiener_filter<FilteredCoaddMap>(engine.cmb);
}

template <class Engine, class Logger>
void calculate_filtered_coadd_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    if (should_calculate_filtered_coadd_noise_products(engine)) {
        logger->info("calculating filtered coadd empirical noise products");
        engine.cmb.calc_noise_products(
            engine.apply_empirical_noise_weights ||
            engine.wiener_filter.normalize_error);
    }
}

template <class Engine, class Logger>
void calculate_filtered_coadd_map_diagnostics(Engine &engine,
                                              const Logger &logger) {
    logger->info("calculating filtered coadded map psds");
    engine.cmb.calc_map_psd();
    logger->info("calculating filtered coadded map histograms");
    engine.cmb.calc_map_hist();

    engine.cmb.calc_median_err();
    engine.cmb.calc_median_rms();
}

template <auto FilteredCoaddMap, class Engine, class Logger>
void find_filtered_coadd_sources_if_needed(Engine &engine,
                                           const Logger &logger) {
    if (engine.run_source_finder) {
        logger->info("finding filtered coadded map sources");
        engine.template find_sources<FilteredCoaddMap>(engine.cmb);
    }
}

template <auto FilteredCoaddMap, class Engine, class Logger>
void output_filtered_coadd_maps_if_needed(Engine &engine,
                                          const Logger &logger) {
    if (engine.write_filtered_maps_partial) {
        logger->info(
            "filtered coadded files already written during Wiener filtering; "
            "skipping post-filter output stage");
    }
    else {
        logger->info("outputting filtered coadded files");
        engine.template output<FilteredCoaddMap>();
    }
}

template <auto FilteredCoaddMap, class TodProc, class Logger>
void write_filtered_coadd_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    filter_coadd_maps<FilteredCoaddMap>(engine, logger);
    calculate_filtered_coadd_noise_products_if_needed(engine, logger);
    calculate_filtered_coadd_map_diagnostics(engine, logger);
    find_filtered_coadd_sources_if_needed<FilteredCoaddMap>(engine, logger);
    output_filtered_coadd_maps_if_needed<FilteredCoaddMap>(engine, logger);
}

template <auto FilteredCoaddMap, class TodProc, class Logger>
void write_filtered_coadd_outputs_if_needed(TodProc &todproc,
                                            const Logger &logger) {
    auto &engine = todproc.engine();

    if (should_write_filtered_outputs(engine)) {
        write_filtered_coadd_outputs<FilteredCoaddMap>(todproc, logger);
    }
}

}  // namespace citlali::pipeline
