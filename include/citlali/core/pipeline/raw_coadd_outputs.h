#pragma once

#include <citlali/core/pipeline/map_diagnostics.h>
#include <citlali/core/pipeline/raw_map_outputs.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class Engine>
bool should_normalize_polarized_raw_coadd_maps(const Engine &engine) {
    return engine.rtcproc.run_polarization;
}

template <class TodProc, class Logger>
void prepare_raw_coadd_map_files(TodProc &todproc,
                                 const Logger &logger) {
    logger->debug("creating cmb filenames");
    todproc.create_coadded_map_files();
}

template <class TodProc, class Logger>
void prepare_raw_coadd_maps(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();
    const auto profile_scope = profile_stage("raw_coadd.prepare", logger);

    prepare_raw_coadd_map_files(todproc, logger);
    logger->info("normalizing coadded maps");
    if (should_normalize_polarized_raw_coadd_maps(engine)) {
        engine.cmb.normalize_polarized_maps();
    }
    else {
        engine.cmb.normalize_maps();
    }
}

template <class Engine, class Logger>
void calculate_raw_coadd_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    calculate_unfiltered_map_noise_products_if_needed(
        engine, engine.cmb, logger, false,
        "calculating raw coadd empirical noise products");
}

template <class Engine, class Logger>
void calculate_raw_coadd_map_diagnostics(Engine &engine,
                                         const Logger &logger) {
    calculate_map_diagnostics(
        engine.cmb, logger, "calculating coadded map psd",
        "calculating coadded map histogram");
}

template <auto RawCoaddMap, class Engine, class Logger>
void output_raw_coadd_maps(Engine &engine, const Logger &logger) {
    output_unfiltered_maps_with_log<RawCoaddMap>(
        engine, logger, "outputting raw coadded files");
}

template <auto RawCoaddMap, class TodProc, class Logger>
void write_raw_coadd_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();
    const auto profile_scope = profile_stage("raw_coadd.outputs", logger);

    prepare_raw_coadd_maps(todproc, logger);
    calculate_raw_coadd_noise_products_if_needed(engine, logger);
    calculate_raw_coadd_map_diagnostics(engine, logger);
    output_raw_coadd_maps<RawCoaddMap>(engine, logger);
}

}  // namespace citlali::pipeline
