#pragma once

namespace citlali::pipeline {

template <class TodProc, class Logger>
void prepare_raw_coadd_maps(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->debug("creating cmb filenames");
    todproc.create_coadded_map_files();

    logger->info("normalizing coadded maps");
    if (engine.rtcproc.run_polarization) {
        engine.cmb.normalize_polarized_maps();
    }
    else {
        engine.cmb.normalize_maps();
    }
}

template <class Engine, class Logger>
void calculate_raw_coadd_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    if (engine.run_noise_products && engine.run_noise) {
        logger->info("calculating raw coadd empirical noise products");
        engine.cmb.calc_noise_products(engine.apply_empirical_noise_weights);
    }
}

template <class Engine, class Logger>
void calculate_raw_coadd_map_diagnostics(Engine &engine,
                                         const Logger &logger) {
    logger->info("calculating coadded map psd");
    engine.cmb.calc_map_psd();
    logger->info("calculating coadded map histogram");
    engine.cmb.calc_map_hist();

    engine.cmb.calc_median_err();
    engine.cmb.calc_median_rms();
}

template <auto RawCoaddMap, class Engine, class Logger>
void output_raw_coadd_maps(Engine &engine, const Logger &logger) {
    logger->info("outputting raw coadded files");
    engine.template output<RawCoaddMap>();
}

template <auto RawCoaddMap, class TodProc, class Logger>
void write_raw_coadd_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    prepare_raw_coadd_maps(todproc, logger);
    calculate_raw_coadd_noise_products_if_needed(engine, logger);
    calculate_raw_coadd_map_diagnostics(engine, logger);
    output_raw_coadd_maps<RawCoaddMap>(engine, logger);
}

}  // namespace citlali::pipeline
