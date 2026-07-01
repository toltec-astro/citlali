#pragma once

#include <citlali/core/pipeline/coadd_outputs.h>
#include <citlali/core/pipeline/fruit_loop_map_loading.h>
#include <citlali/core/pipeline/initial_reduction_geometry.h>
#include <citlali/core/pipeline/iteration_buffers.h>
#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/observation_outputs.h>
#include <citlali/core/pipeline/observation_preflight.h>
#include <citlali/core/pipeline/observation_pipeline.h>
#include <citlali/core/pipeline/output_layout.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_iteration_setup.h>

#include <cstddef>
#include <utility>

namespace citlali::pipeline {

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

template <auto RawCoaddMap, auto FilteredCoaddMap, class TodProc,
          class Logger>
void finish_reduction_iteration(TodProc &todproc, const Logger &logger) {
    write_iteration_coadd_outputs_if_needed<RawCoaddMap, FilteredCoaddMap>(
        todproc, logger);
    finalize_iteration_outputs(todproc, logger);
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class Logger>
void run_reduction_observation_pipeline(TodProc &todproc, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        const Logger &logger) {
    auto &engine = todproc.engine();

    load_observation_fruit_loop_maps_if_needed<IsBeammap>(engine, logger);
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
