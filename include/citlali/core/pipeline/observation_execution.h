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
                                      MapExtent &map_extent,
                                      MapCoord &map_coord,
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

template <auto RawObsMap, class TodProc, class Logger>
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

template <auto FilteredObsMap, bool FitMaps, class TodProc, class Logger>
void write_filtered_observation_outputs(TodProc &todproc,
                                        const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("filtering obs maps");
    engine.template run_wiener_filter<FilteredObsMap>(engine.omb);

    if (engine.run_noise_products &&
        engine.run_noise &&
        !engine.write_filtered_maps_partial) {
        logger->info("calculating filtered obs empirical noise products");
        engine.omb.calc_noise_products(
            engine.apply_empirical_noise_weights ||
            engine.wiener_filter.normalize_error);
    }

    logger->info("calculating filtered obs map psds");
    engine.omb.calc_map_psd();
    logger->info("calculating filtered obs map histograms");
    engine.omb.calc_map_hist();

    engine.omb.calc_median_err();
    engine.omb.calc_median_rms();

    if (engine.run_source_finder) {
        logger->info("finding filtered obs map sources");
        engine.template find_sources<FilteredObsMap>(engine.omb);
    }

    if constexpr (FitMaps) {
        engine.fit_maps();
    }

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

}  // namespace citlali::pipeline
