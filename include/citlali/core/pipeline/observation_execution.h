#pragma once

#include <citlali/core/pipeline/fruit_loop_paths.h>
#include <citlali/core/pipeline/observation_preflight.h>

#include <cstddef>

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

template <class TodProc, class Logger>
void prepare_iteration_observation_buffers(TodProc &todproc,
                                           const Logger &logger) {
    auto &engine = todproc.engine();

    engine.date_obs.clear();
    if (engine.run_coadd) {
        prepare_coadd_iteration_buffers(todproc, logger);
    }
}

template <class TodProc, class MapExtents, class MapCoords, class Logger>
void calculate_initial_observation_map_dimensions(TodProc &todproc,
                                                 MapExtents &map_extents,
                                                 MapCoords &map_coords,
                                                 const Logger &logger) {
    auto &engine = todproc.engine();

    if (!engine.run_mapmaking) {
        return;
    }

    logger->info("calculating number of maps");
    todproc.calc_map_num();
    logger->info("calculating obs map dimensions");
    todproc.calc_omb_size(map_extents, map_coords);
}

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class Logger>
bool prepare_initial_observation_setup(TodProc &todproc, const RawObs &rawobs,
                                       const RawObsKidsMeta &rawobs_kids_meta,
                                       MapExtents &map_extents,
                                       MapCoords &map_coords,
                                       const Logger &logger) {
    auto &engine = todproc.engine();

    configure_observation_calibration<IsBeammap>(todproc, rawobs, logger);
    if (!apply_flxscale_correction(engine, rawobs, logger)) {
        return false;
    }

    check_observation_inputs(todproc, rawobs, logger);
    update_sample_rate_from_rawobs_meta(engine, rawobs_kids_meta, logger);
    load_and_align_telescope_data(todproc, rawobs, logger);
    calculate_telescope_pointing(todproc, logger);
    calculate_scan_indices(engine, logger);
    calculate_initial_observation_map_dimensions(
        todproc, map_extents, map_coords, logger);
    return true;
}

template <class TodProc, class MapCoords, class Logger>
void calculate_initial_coadd_map_dimensions(TodProc &todproc,
                                            MapCoords &map_coords,
                                            const Logger &logger) {
    auto &engine = todproc.engine();

    if (!engine.run_coadd) {
        return;
    }

    logger->info("calculating cmb dimensions");
    todproc.calc_cmb_size(map_coords);
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

template <class TodProc, class MapExtents, class MapCoords, class Logger>
void allocate_observation_map_buffers_if_needed(
    TodProc &todproc, MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, const Logger &logger) {
    auto &engine = todproc.engine();

    if (!engine.run_mapmaking) {
        return;
    }

    allocate_observation_map_buffers(
        todproc, map_extents[observation_index], map_coords[observation_index],
        logger);
}

template <class TodProc, class Logger>
void coadd_observation(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("coadding");
    if (!engine.rtcproc.run_polarization) {
        todproc.coadd();
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

template <auto RawCoaddMap, class TodProc, class Logger>
void write_raw_coadd_outputs(TodProc &todproc, const Logger &logger) {
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

    if (engine.run_noise_products && engine.run_noise) {
        logger->info("calculating raw coadd empirical noise products");
        engine.cmb.calc_noise_products(engine.apply_empirical_noise_weights);
    }

    logger->info("calculating coadded map psd");
    engine.cmb.calc_map_psd();
    logger->info("calculating coadded map histogram");
    engine.cmb.calc_map_hist();

    engine.cmb.calc_median_err();
    engine.cmb.calc_median_rms();

    logger->info("outputting raw coadded files");
    engine.template output<RawCoaddMap>();
}

template <auto FilteredCoaddMap, class TodProc, class Logger>
void write_filtered_coadd_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("filtering coadded maps");
    engine.template run_wiener_filter<FilteredCoaddMap>(engine.cmb);

    if (engine.run_noise_products &&
        engine.run_noise &&
        !engine.write_filtered_maps_partial) {
        logger->info("calculating filtered coadd empirical noise products");
        engine.cmb.calc_noise_products(
            engine.apply_empirical_noise_weights ||
            engine.wiener_filter.normalize_error);
    }

    logger->info("calculating filtered coadded map psds");
    engine.cmb.calc_map_psd();
    logger->info("calculating filtered coadded map histograms");
    engine.cmb.calc_map_hist();

    engine.cmb.calc_median_err();
    engine.cmb.calc_median_rms();

    if (engine.run_source_finder) {
        logger->info("finding filtered coadded map sources");
        engine.template find_sources<FilteredCoaddMap>(engine.cmb);
    }

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

template <auto RawCoaddMap, auto FilteredCoaddMap, class TodProc,
          class Logger>
void write_iteration_coadd_outputs_if_needed(TodProc &todproc,
                                             const Logger &logger) {
    auto &engine = todproc.engine();

    if (!engine.run_coadd) {
        return;
    }

    write_raw_coadd_outputs<RawCoaddMap>(todproc, logger);
    if (engine.run_map_filter) {
        write_filtered_coadd_outputs<FilteredCoaddMap>(todproc, logger);
    }
}

template <class Engine>
void load_initial_fruit_loop_model_if_requested(Engine &engine) {
    if (engine.ptcproc.run_fruit_loops && engine.fruit_iter == 0) {
        if (engine.ptcproc.fruit_loops_path != "null") {
            const auto fruit_dir = fruit_loop_map_dir(
                engine.ptcproc.fruit_loops_path,
                engine.ptcproc.fruit_loops_type,
                engine.omb.obsnums.back());

            engine.ptcproc.tod_mb.cov_cut = engine.omb.cov_cut;
            engine.ptcproc.load_mb(fruit_dir, fruit_dir, engine.calib,
                                   engine.map_grouping,
                                   engine.telescope.pixel_axes,
                                   engine.omb.pixel_size_rad);
        }
    }
}

template <class Engine, class Logger>
void load_previous_fruit_loop_model_if_needed(Engine &engine,
                                              const Logger &logger) {
    if (engine.fruit_iter > 0) {
        auto fruit_dir = std::string{};
        if (engine.ptcproc.save_all_iters) {
            fruit_dir = previous_fruit_loop_map_dir(
                engine.output_dir, engine.redu_dir_num,
                engine.ptcproc.fruit_loops_type,
                engine.omb.obsnums.back());
        }
        else {
            logger->info(
                "loading previous iter maps for fruit loops iteration {}",
                engine.fruit_iter);
            fruit_dir = fruit_loop_map_dir(engine.redu_dir_name,
                                           engine.ptcproc.fruit_loops_type,
                                           engine.omb.obsnums.back());
        }

        engine.ptcproc.tod_mb.cov_cut = engine.omb.cov_cut;

        logger->info("reading in {} for fruit loops iteration {}", fruit_dir,
                     engine.fruit_iter);
        engine.ptcproc.load_mb(fruit_dir, fruit_dir, engine.calib,
                               engine.map_grouping,
                               engine.telescope.pixel_axes,
                               engine.omb.pixel_size_rad);
    }
}

template <bool IsBeammap, class Engine, class Logger>
void load_observation_fruit_loop_models_if_needed(Engine &engine,
                                                  const Logger &logger) {
    if constexpr (!IsBeammap) {
        load_initial_fruit_loop_model_if_requested(engine);
        load_previous_fruit_loop_model_if_needed(engine, logger);
    }
}

}  // namespace citlali::pipeline
