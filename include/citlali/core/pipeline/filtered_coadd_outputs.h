#pragma once

#include <citlali/core/pipeline/filtered_map_outputs.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <auto FilteredCoaddMap, class Engine, class Logger>
void filter_coadd_maps(Engine &engine, const Logger &logger) {
    filter_maps<FilteredCoaddMap>(
        engine, engine.cmb, logger, "filtering coadded maps");
    record_post_processing_filter_completed_if_available(
        engine, PostProcessingMapContext::coadd);
}

template <class Engine, class Logger>
void calculate_filtered_coadd_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    calculate_filtered_map_noise_products_if_needed(
        engine, engine.cmb, logger,
        "calculating filtered coadd empirical noise products");
}

template <class Engine, class Logger>
void calculate_filtered_coadd_map_diagnostics(Engine &engine,
                                              const Logger &logger) {
    calculate_filtered_map_diagnostics(
        engine.cmb, logger, "calculating filtered coadded map psds",
        "calculating filtered coadded map histograms");
}

template <auto FilteredCoaddMap, class Engine, class Logger>
void find_filtered_coadd_sources_if_needed(Engine &engine,
                                           const Logger &logger) {
    find_filtered_map_sources_if_needed<FilteredCoaddMap>(
        engine, engine.cmb, logger,
        "finding filtered coadded map sources",
        PostProcessingMapContext::coadd);
}

template <auto FilteredCoaddMap, class Engine, class Logger>
void output_filtered_coadd_maps_if_needed(Engine &engine,
                                          const Logger &logger) {
    output_filtered_maps_if_needed<FilteredCoaddMap>(
        engine, logger, "outputting filtered coadded files",
        "filtered coadded files already written during Wiener filtering; "
        "skipping post-filter output stage");
}

template <auto FilteredCoaddMap, class TodProc, class Logger>
void write_filtered_coadd_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();
    const auto profile_scope =
        profile_stage("filtered_coadd.outputs", logger);

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
