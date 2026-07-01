#pragma once

#include <citlali/core/cli/reduction_date_obs.h>
#include <citlali/core/cli/reduction_runtime.h>
#include <citlali/core/cli/runtime_setup.h>
#include <citlali/core/pipeline/reduction_pipeline.h>
#include <spdlog/spdlog.h>

#include <ostream>

namespace citlali::cli {

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

}  // namespace citlali::cli
