#pragma once

#include <citlali/core/pipeline/coadd_outputs.h>
#include <citlali/core/pipeline/fruit_loop_map_loading.h>
#include <citlali/core/pipeline/initial_reduction_geometry.h>
#include <citlali/core/pipeline/iteration_buffers.h>
#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/observation_buffers.h>
#include <citlali/core/pipeline/observation_outputs.h>
#include <citlali/core/pipeline/observation_preflight.h>
#include <citlali/core/pipeline/observation_pipeline.h>
#include <citlali/core/pipeline/reduction_iteration_outputs.h>
#include <citlali/core/pipeline/reduction_observation_inputs.h>
#include <citlali/core/pipeline/reduction_observation_pipeline.h>
#include <citlali/core/pipeline/output_layout.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_iteration_setup.h>

#include <cstddef>
#include <utility>

namespace citlali::pipeline {

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
