#pragma once

#include <citlali/core/cli/reduction_date_obs.h>
#include <citlali/core/cli/reduction_runtime.h>
#include <citlali/core/cli/run_logging.h>
#include <citlali/core/cli/runtime_setup.h>
#include <citlali/core/pipeline/map_geometry.h>
#include <citlali/core/pipeline/reduction_pipeline.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <ostream>
#include <type_traits>

namespace citlali::cli {

template <class TodProc, class BeammapTodProc>
inline constexpr bool is_beammap_tod_processor_v =
    std::is_same_v<TodProc, BeammapTodProc>;

template <class TodProc, class PointingTodProc>
inline constexpr bool fits_maps_for_tod_processor_v =
    std::is_same_v<TodProc, PointingTodProc>;

template <class TodProc, class Config, class Logger>
bool prepare_cli_reduction_runtime_or_report_errors(
    TodProc &todproc, Config &config, const Logger &logger,
    std::ostream &os) {
    return prepare_reduction_runtime_or_report_errors(
        todproc, config, logger,
        []() { spdlog::set_level(spdlog::level::debug); },
        [&](const auto &engine) {
            configure_citlali_runtime_threads(engine, logger);
        },
        os);
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class Config, class ConfigFilepaths, class MapGeometry,
          class Logger>
bool run_cli_reduction_pipeline(TodProc &todproc, const IOCoordinator &co,
                                Config &config,
                                const ConfigFilepaths &config_filepaths,
                                MapGeometry &map_geometry,
                                const Logger &logger) {
    return citlali::pipeline::run_reduction_pipeline<
        IsBeammap, RawObsMap, FilteredObsMap, RawCoaddMap, FilteredCoaddMap,
        FitMaps, KidsDataProc>(
        todproc, co, config, config_filepaths, map_geometry.extents,
        map_geometry.coords,
        [](auto &engine) {
            return date_obs_from_engine_telescope_time(engine);
        },
        logger);
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class Config, class ConfigFilepaths, class MapGeometry,
          class Logger>
bool prepare_and_run_cli_reduction_pipeline(
    TodProc &todproc, const IOCoordinator &co, Config &config,
    const ConfigFilepaths &config_filepaths, MapGeometry &map_geometry,
    const Logger &logger, std::ostream &os) {
    if (!prepare_cli_reduction_runtime_or_report_errors(
            todproc, config, logger, os)) {
        return false;
    }

    return run_cli_reduction_pipeline<
        IsBeammap, RawObsMap, FilteredObsMap, RawCoaddMap, FilteredCoaddMap,
        FitMaps, KidsDataProc>(
        todproc, co, config, config_filepaths, map_geometry, logger);
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class Config, class ConfigFilepaths, class Logger>
bool prepare_and_run_cli_reduction_pipeline(
    TodProc &todproc, const IOCoordinator &co, Config &config,
    const ConfigFilepaths &config_filepaths, const Logger &logger,
    std::ostream &os) {
    auto map_geometry =
        citlali::pipeline::make_reduction_map_geometry<TodProc>();
    return prepare_and_run_cli_reduction_pipeline<
        IsBeammap, RawObsMap, FilteredObsMap, RawCoaddMap, FilteredCoaddMap,
        FitMaps, KidsDataProc>(
        todproc, co, config, config_filepaths, map_geometry, logger, os);
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class Config, class ConfigFilepaths, class Logger>
int run_cli_reduction_processor(
    TodProc &todproc, const IOCoordinator &co, Config &config,
    const ConfigFilepaths &config_filepaths, const Logger &logger,
    std::ostream &os) {
    if (!prepare_and_run_cli_reduction_pipeline<
            IsBeammap, RawObsMap, FilteredObsMap, RawCoaddMap,
            FilteredCoaddMap, FitMaps, KidsDataProc>(
            todproc, co, config, config_filepaths, logger, os)) {
        return EXIT_FAILURE;
    }

    log_reduction_complete(logger);
    return EXIT_SUCCESS;
}

template <class TodProc, class BeammapTodProc, class PointingTodProc,
          auto RawObsMap, auto FilteredObsMap, auto RawCoaddMap,
          auto FilteredCoaddMap, class KidsDataProc, class IOCoordinator,
          class Config, class ConfigFilepaths, class Logger>
int run_cli_reduction_processor_for_mode(
    TodProc &todproc, const IOCoordinator &co, Config &config,
    const ConfigFilepaths &config_filepaths, const Logger &logger,
    std::ostream &os) {
    return run_cli_reduction_processor<
        is_beammap_tod_processor_v<TodProc, BeammapTodProc>, RawObsMap,
        FilteredObsMap, RawCoaddMap, FilteredCoaddMap,
        fits_maps_for_tod_processor_v<TodProc, PointingTodProc>,
        KidsDataProc>(
        todproc, co, config, config_filepaths, logger, os);
}

}  // namespace citlali::cli
