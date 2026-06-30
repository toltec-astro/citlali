#pragma once

namespace citlali::pipeline {

template <class Engine, class KidsProc, class RawObs, class Logger>
void setup_and_run_observation_pipeline(Engine &engine, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        const Logger &logger) {
    logger->info("pipeline setup");
    engine.setup();

    if (engine.run_tod) {
        logger->info("running pipeline");
        engine.pipeline(kidsproc, rawobs);
    }
}

template <class TodProc, class Logger>
void prepare_coadd_iteration_buffers(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("allocating cmb");
    todproc.allocate_cmb();
    if (engine.run_noise) {
        logger->info("allocating nmb");
        todproc.allocate_nmb(engine.cmb);
    }

    engine.cmb.obsnums.clear();
    engine.cmb.exposure_time = 0;
}

template <class TodProc, class MapExtent, class MapCoord, class Logger>
void allocate_observation_map_buffers(TodProc &todproc,
                                      const MapExtent &map_extent,
                                      const MapCoord &map_coord,
                                      const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("calculating number of maps");
    todproc.calc_map_num();
    logger->info("allocating obs map buffer");
    todproc.allocate_omb(map_extent, map_coord);
    engine.configure_map_pixel_contribution_targets(engine.omb, "raw_obs");

    if (engine.run_noise &&
        (!engine.run_coadd || engine.map_method == "jinc")) {
        logger->info("allocating obs noise maps");
        todproc.allocate_nmb(engine.omb);
    }
}

template <class RawObsMap, class TodProc, class Logger>
void write_raw_observation_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    if (engine.run_mapmaking &&
        engine.run_noise_products &&
        engine.run_noise) {
        logger->info("calculating raw obs empirical noise products");
        engine.omb.calc_noise_products(engine.apply_empirical_noise_weights);
    }

    if (engine.run_mapmaking) {
        engine.create_obs_map_files();
    }

    if (engine.run_mapmaking) {
        logger->info("outputting raw obs files");
        engine.template output<RawObsMap>();
    }
    else {
        logger->info("mapmaking disabled; skipping raw obs map output");
    }
}

}  // namespace citlali::pipeline
