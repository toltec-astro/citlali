#pragma once

#include <citlali/core/pipeline/fruit_loop_paths.h>
#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/observation_preflight.h>
#include <citlali/core/pipeline/output_layout.h>

#include <cstddef>
#include <utility>

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

template <class TodProc, class ConfigFilepaths, class Logger>
void begin_reduction_iteration(TodProc &todproc,
                               const ConfigFilepaths &config_filepaths,
                               const Logger &logger) {
    auto &engine = todproc.engine();

    begin_fruit_loop_iteration(engine, logger);
    prepare_iteration_output_layout_if_needed(todproc, config_filepaths,
                                              logger);
    prepare_iteration_observation_buffers(todproc, logger);
}

template <class Engine, class Logger>
void initialize_reduction_iterations(Engine &engine,
                                     bool &fruit_loops_converged,
                                     const Logger &logger) {
    engine.fruit_iter = 0;
    fruit_loops_converged = false;
    configure_fruit_loop_iteration_policy(engine, logger);
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

template <bool IsBeammap, class KidsDataProc, class TodProc,
          class CitlaliConfig, class RawObs, class MapExtents,
          class MapCoords, class Logger>
bool prepare_initial_observation(
    TodProc &todproc, CitlaliConfig &citlali_config, const RawObs &rawobs,
    MapExtents &map_extents, MapCoords &map_coords, const Logger &logger) {
    auto kidsproc = make_kids_data_proc<KidsDataProc>(citlali_config);
    auto rawobs_kids_meta = load_rawobs_kids_meta(kidsproc, rawobs, logger);

    return prepare_initial_observation_setup<IsBeammap>(
        todproc, rawobs, rawobs_kids_meta, map_extents, map_coords, logger);
}

template <bool IsBeammap, class KidsDataProc, class TodProc,
          class IOCoordinator, class CitlaliConfig, class MapExtents,
          class MapCoords, class Logger>
bool prepare_initial_observations(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    MapExtents &map_extents, MapCoords &map_coords, const Logger &logger) {
    logger->info("starting initial loop through input obs");
    std::size_t observation_index = 0;
    for (const auto &rawobs : co.inputs()) {
        logger->info("starting setup of observation {}/{}",
                     observation_index + 1, co.n_inputs());
        if (!prepare_initial_observation<IsBeammap, KidsDataProc>(
                todproc, citlali_config, rawobs, map_extents, map_coords,
                logger)) {
            return false;
        }
        ++observation_index;
    }
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

template <bool IsBeammap, class KidsDataProc, class TodProc,
          class IOCoordinator, class CitlaliConfig, class MapExtents,
          class MapCoords, class Logger>
bool prepare_initial_reduction_geometry(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    MapExtents &map_extents, MapCoords &map_coords, const Logger &logger) {
    if (!prepare_initial_observations<IsBeammap, KidsDataProc>(
            todproc, co, citlali_config, map_extents, map_coords, logger)) {
        return false;
    }

    calculate_initial_coadd_map_dimensions(todproc, map_coords, logger);
    return true;
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

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class DateObs, class Logger>
bool prepare_reduction_observation_inputs(
    TodProc &todproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool has_multiple_inputs,
    MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, DateObs &&date_obs, const Logger &logger) {
    auto &engine = todproc.engine();

    if (!configure_reduction_observation_calibration_if_needed<IsBeammap>(
            todproc, rawobs, rawobs_kids_meta, has_multiple_inputs, logger)) {
        return false;
    }

    if (!configure_effective_sample_rate(engine, logger)) {
        return false;
    }

    load_raw_detector_diagnostics(todproc, rawobs, logger);
    prepare_observation_output_layout_from_rawobs_meta(
        engine, rawobs_kids_meta, logger);
    load_hwpr_data_if_requested(engine, rawobs, logger);
    calculate_flux_calibration(engine, logger);
    load_and_point_telescope_data_if_needed(
        todproc, rawobs, has_multiple_inputs, logger);
    append_observation_date(engine, std::forward<DateObs>(date_obs));
    record_timing_gaps_if_needed(engine, logger);
    calculate_scan_indices_if_needed(engine, has_multiple_inputs, logger);
    allocate_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, observation_index, logger);
    update_observation_exposure_time(engine);
    return true;
}

template <class TodProc, class Logger>
void coadd_observation(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("coadding");
    if (!engine.rtcproc.run_polarization) {
        todproc.coadd();
    }
}

template <class Engine, class Logger>
void calculate_raw_observation_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    if (engine.run_mapmaking &&
        engine.run_noise_products &&
        engine.run_noise) {
        logger->info("calculating raw obs empirical noise products");
        engine.omb.calc_noise_products(engine.apply_empirical_noise_weights);
    }
}

template <auto RawObsMap, class Engine, class Logger>
void output_raw_observation_maps_if_needed(Engine &engine,
                                           const Logger &logger) {
    if (engine.run_mapmaking) {
        engine.create_obs_map_files();
        logger->info("outputting raw obs files");
        engine.template output<RawObsMap>();
    }
    else {
        logger->info("mapmaking disabled; skipping raw obs map output");
    }
}

template <auto RawObsMap, class TodProc, class Logger>
void write_raw_observation_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    calculate_raw_observation_noise_products_if_needed(engine, logger);
    output_raw_observation_maps_if_needed<RawObsMap>(engine, logger);
}

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

template <auto RawObsMap, auto FilteredObsMap, bool FitMaps, class TodProc,
          class Logger>
void write_observation_outputs_and_accumulate(TodProc &todproc,
                                              const Logger &logger) {
    auto &engine = todproc.engine();

    write_raw_observation_outputs<RawObsMap>(todproc, logger);

    if (engine.run_coadd) {
        coadd_observation(todproc, logger);
    }
    else if (engine.run_map_filter) {
        write_filtered_observation_outputs<FilteredObsMap, FitMaps>(
            todproc, logger);
    }
}

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

template <auto FilteredCoaddMap, class Engine, class Logger>
void filter_coadd_maps(Engine &engine, const Logger &logger) {
    logger->info("filtering coadded maps");
    engine.template run_wiener_filter<FilteredCoaddMap>(engine.cmb);
}

template <class Engine, class Logger>
void calculate_filtered_coadd_noise_products_if_needed(
    Engine &engine, const Logger &logger) {
    if (engine.run_noise_products &&
        engine.run_noise &&
        !engine.write_filtered_maps_partial) {
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

template <auto RawCoaddMap, auto FilteredCoaddMap, class TodProc,
          class Logger>
void finish_reduction_iteration(TodProc &todproc, const Logger &logger) {
    write_iteration_coadd_outputs_if_needed<RawCoaddMap, FilteredCoaddMap>(
        todproc, logger);
    finalize_iteration_outputs(todproc, logger);
}

template <class Engine>
bool should_load_initial_fruit_loop_model(const Engine &engine) {
    return engine.ptcproc.run_fruit_loops &&
           engine.fruit_iter == 0 &&
           engine.ptcproc.fruit_loops_path != "null";
}

template <class Engine>
std::string initial_fruit_loop_model_dir(const Engine &engine) {
    return fruit_loop_map_dir(engine.ptcproc.fruit_loops_path,
                              engine.ptcproc.fruit_loops_type,
                              engine.omb.obsnums.back());
}

template <class Engine>
void load_fruit_loop_model(Engine &engine, const std::string &fruit_dir) {
    engine.ptcproc.tod_mb.cov_cut = engine.omb.cov_cut;
    engine.ptcproc.load_mb(fruit_dir, fruit_dir, engine.calib,
                           engine.map_grouping,
                           engine.telescope.pixel_axes,
                           engine.omb.pixel_size_rad);
}

template <class Engine>
void load_initial_fruit_loop_model_if_requested(Engine &engine) {
    if (should_load_initial_fruit_loop_model(engine)) {
        const auto fruit_dir = initial_fruit_loop_model_dir(engine);
        load_fruit_loop_model(engine, fruit_dir);
    }
}

template <class Engine>
bool should_load_previous_fruit_loop_model(const Engine &engine) {
    return engine.fruit_iter > 0;
}

template <class Engine, class Logger>
void load_previous_fruit_loop_model_if_needed(Engine &engine,
                                              const Logger &logger) {
    if (should_load_previous_fruit_loop_model(engine)) {
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

        logger->info("reading in {} for fruit loops iteration {}", fruit_dir,
                     engine.fruit_iter);
        load_fruit_loop_model(engine, fruit_dir);
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

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class Logger>
void run_reduction_observation_pipeline(TodProc &todproc, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        const Logger &logger) {
    auto &engine = todproc.engine();

    load_observation_fruit_loop_models_if_needed<IsBeammap>(engine, logger);
    setup_and_run_observation_pipeline(engine, kidsproc, rawobs, logger);
    write_observation_outputs_and_accumulate<RawObsMap, FilteredObsMap,
                                             FitMaps>(todproc, logger);
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class DateObs, class Logger>
bool run_reduction_observation(
    TodProc &todproc, KidsProc &kidsproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool has_multiple_inputs,
    MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, DateObs &&date_obs, const Logger &logger) {
    if (!prepare_reduction_observation_inputs<IsBeammap>(
            todproc, rawobs, rawobs_kids_meta, has_multiple_inputs,
            map_extents, map_coords, observation_index,
            std::forward<DateObs>(date_obs), logger)) {
        return false;
    }

    run_reduction_observation_pipeline<IsBeammap, RawObsMap, FilteredObsMap,
                                       FitMaps>(
        todproc, kidsproc, rawobs, logger);
    return true;
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class MapExtents, class MapCoords,
          class DateObsFactory, class Logger>
bool run_reduction_observation_at_index(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, DateObsFactory &&date_obs_factory,
    const Logger &logger) {
    logger->info("starting reduction of observation {}/{}",
                 observation_index + 1, co.n_inputs());
    auto kidsproc = make_kids_data_proc<KidsDataProc>(citlali_config);
    const auto &rawobs = co.inputs()[observation_index];
    auto rawobs_kids_meta = load_rawobs_kids_meta(kidsproc, rawobs, logger);

    return run_reduction_observation<IsBeammap, RawObsMap, FilteredObsMap,
                                     FitMaps>(
        todproc, kidsproc, rawobs, rawobs_kids_meta, co.n_inputs() > 1,
        map_extents, map_coords, observation_index,
        date_obs_factory(todproc.engine()), logger);
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class MapExtents, class MapCoords,
          class DateObsFactory, class Logger>
bool run_reduction_iteration_observations(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    MapExtents &map_extents, MapCoords &map_coords,
    DateObsFactory &&date_obs_factory, const Logger &logger) {
    for (std::size_t observation_index = 0; observation_index < co.n_inputs();
         ++observation_index) {
        if (!run_reduction_observation_at_index<
                IsBeammap, RawObsMap, FilteredObsMap, FitMaps,
                KidsDataProc>(
                todproc, co, citlali_config, map_extents, map_coords,
                observation_index, date_obs_factory, logger)) {
            return false;
        }
    }
    return true;
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class ConfigFilepaths, class MapExtents,
          class MapCoords, class DateObsFactory, class Logger>
bool run_reduction_iteration(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    const ConfigFilepaths &config_filepaths, MapExtents &map_extents,
    MapCoords &map_coords, DateObsFactory &&date_obs_factory,
    const Logger &logger) {
    begin_reduction_iteration(todproc, config_filepaths, logger);

    if (!run_reduction_iteration_observations<
            IsBeammap, RawObsMap, FilteredObsMap, FitMaps, KidsDataProc>(
            todproc, co, citlali_config, map_extents, map_coords,
            date_obs_factory, logger)) {
        return false;
    }

    finish_reduction_iteration<RawCoaddMap, FilteredCoaddMap>(todproc,
                                                              logger);
    return true;
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class ConfigFilepaths, class MapExtents,
          class MapCoords, class DateObsFactory, class Logger>
bool run_reduction_iterations(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    const ConfigFilepaths &config_filepaths, MapExtents &map_extents,
    MapCoords &map_coords, DateObsFactory &&date_obs_factory,
    const Logger &logger) {
    bool fruit_loops_converged = false;
    auto &engine = todproc.engine();
    initialize_reduction_iterations(engine, fruit_loops_converged, logger);

    while (fruit_loop_iteration_pending(engine, fruit_loops_converged)) {
        if (!run_reduction_iteration<
                IsBeammap, RawObsMap, FilteredObsMap, RawCoaddMap,
                FilteredCoaddMap, FitMaps, KidsDataProc>(
                todproc, co, citlali_config, config_filepaths, map_extents,
                map_coords, date_obs_factory, logger)) {
            return false;
        }
    }
    return true;
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class ConfigFilepaths, class MapExtents,
          class MapCoords, class DateObsFactory, class Logger>
bool run_reduction_pipeline(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    const ConfigFilepaths &config_filepaths, MapExtents &map_extents,
    MapCoords &map_coords, DateObsFactory &&date_obs_factory,
    const Logger &logger) {
    if (!prepare_initial_reduction_geometry<IsBeammap, KidsDataProc>(
            todproc, co, citlali_config, map_extents, map_coords, logger)) {
        return false;
    }

    return run_reduction_iterations<
        IsBeammap, RawObsMap, FilteredObsMap, RawCoaddMap, FilteredCoaddMap,
        FitMaps, KidsDataProc>(
        todproc, co, citlali_config, config_filepaths, map_extents,
        map_coords, date_obs_factory, logger);
}

}  // namespace citlali::pipeline
