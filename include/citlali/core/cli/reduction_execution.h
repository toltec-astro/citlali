#pragma once

#include <citlali/core/cli/reduction_date_obs.h>
#include <citlali/core/cli/reduction_runtime.h>
#include <citlali/core/cli/run_logging.h>
#include <citlali/core/cli/runtime_setup.h>
#include <citlali/core/cli/tod_processor_selection.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/pipeline/beammap_provenance.h>
#include <citlali/core/pipeline/beammap_provenance_lifecycle.h>
#include <citlali/core/pipeline/config_source_manifest.h>
#include <citlali/core/pipeline/kids_external_provenance.h>
#include <citlali/core/pipeline/map_geometry.h>
#include <citlali/core/pipeline/mapmaking_provenance.h>
#include <citlali/core/pipeline/coadd_provenance.h>
#include <citlali/core/pipeline/noise_provenance.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/pointing_provenance.h>
#include <citlali/core/pipeline/post_processing_provenance.h>
#include <citlali/core/pipeline/post_processing_provenance_lifecycle.h>
#include <citlali/core/pipeline/processed_timestream_provenance.h>
#include <citlali/core/pipeline/reduction_pipeline.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <ostream>
#include <type_traits>
#include <utility>
#include <variant>

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
        [&](auto &engine) {
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

    auto &engine = todproc.engine();
    citlali::pipeline::write_config_source_manifest(
        engine.output_paths.redu_dir_name, config_filepaths, config.to_str());
    logger->info(
        "config source manifest: {}",
        citlali::pipeline::config_source_manifest_path(
            engine.output_paths.redu_dir_name)
            .string());

    if constexpr (citlali::pipeline::has_kids_external_plan_v<
                      decltype(engine)>) {
        citlali::pipeline::write_kids_external_provenance_file(
            engine.output_paths.redu_dir_name,
            citlali::pipeline::kids_external_plan(engine));
        logger->info(
            "KIDs external provenance sidecar: {}",
            citlali::pipeline::kids_external_provenance_path(
                engine.output_paths.redu_dir_name)
                .string());
    }

    const auto &plan =
        citlali::pipeline::processed_timestream_plan(engine);
    citlali::pipeline::write_processed_timestream_provenance_file(
        engine.output_paths.redu_dir_name, plan);
    logger->info(
        "processed timestream provenance sidecar: {}",
        citlali::pipeline::processed_timestream_provenance_path(
            engine.output_paths.redu_dir_name)
            .string());

    auto &mapmaking_plan =
        citlali::pipeline::mapmaking_plan(engine);
    citlali::pipeline::record_mapmaking_run_completed(mapmaking_plan);
    citlali::pipeline::write_mapmaking_provenance_file(
        engine.output_paths.redu_dir_name, mapmaking_plan);
    logger->info(
        "mapmaking provenance sidecar: {}",
        citlali::pipeline::mapmaking_provenance_path(
            engine.output_paths.redu_dir_name)
            .string());

    auto &coadd_plan = citlali::pipeline::coadd_plan(engine);
    citlali::pipeline::record_coadd_run_completed(
        coadd_plan, mapmaking_plan);
    citlali::pipeline::write_coadd_provenance_file(
        engine.output_paths.redu_dir_name, coadd_plan);
    logger->info(
        "coadd provenance sidecar: {}",
        citlali::pipeline::coadd_provenance_path(
            engine.output_paths.redu_dir_name)
            .string());

    auto &noise_plan = citlali::pipeline::noise_plan(engine);
    citlali::pipeline::record_noise_run_completed(
        noise_plan, mapmaking_plan,
        citlali::pipeline::map_filter_outputs_enabled(engine));
    citlali::pipeline::write_noise_provenance_file(
        engine.output_paths.redu_dir_name, noise_plan);
    logger->info(
        "noise-products provenance sidecar: {}",
        citlali::pipeline::noise_provenance_path(
            engine.output_paths.redu_dir_name)
            .string());

    auto &post_processing_plan =
        citlali::pipeline::post_processing_plan(engine);
    citlali::pipeline::record_post_processing_run_completed(
        post_processing_plan, mapmaking_plan);
    if constexpr (IsBeammap) {
        citlali::pipeline::record_beammap_run_completed(
            citlali::pipeline::beammap_plan(engine), mapmaking_plan,
            post_processing_plan);
    }
    citlali::pipeline::write_post_processing_provenance_file(
        engine.output_paths.redu_dir_name, post_processing_plan);
    logger->info(
        "post-processing provenance sidecar: {}",
        citlali::pipeline::post_processing_provenance_path(
            engine.output_paths.redu_dir_name)
            .string());

    if constexpr (IsBeammap) {
        auto &beammap_plan = citlali::pipeline::beammap_plan(engine);
        citlali::pipeline::write_beammap_provenance_file(
            engine.output_paths.redu_dir_name, beammap_plan);
        logger->info(
            "beammap provenance sidecar: {}",
            citlali::pipeline::beammap_provenance_path(
                engine.output_paths.redu_dir_name)
                .string());
    }

    if constexpr (FitMaps) {
        auto &pointing_plan =
            citlali::pipeline::pointing_plan(engine);
        citlali::pipeline::record_pointing_run_completed(
            pointing_plan, mapmaking_plan);
        citlali::pipeline::write_pointing_provenance_file(
            engine.output_paths.redu_dir_name, pointing_plan);
        logger->info(
            "pointing provenance sidecar: {}",
            citlali::pipeline::pointing_provenance_path(
                engine.output_paths.redu_dir_name)
                .string());
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

template <class TodProc, class BeammapTodProc, class PointingTodProc,
          class KidsDataProc, class IOCoordinator, class Config,
          class ConfigFilepaths, class Logger>
int run_standard_cli_reduction_processor(
    TodProc &todproc, const IOCoordinator &co, Config &config,
    const ConfigFilepaths &config_filepaths, const Logger &logger,
    std::ostream &os) {
    return run_cli_reduction_processor_for_mode<
        TodProc, BeammapTodProc, PointingTodProc, mapmaking::RawObs,
        mapmaking::FilteredObs, mapmaking::RawCoadd,
        mapmaking::FilteredCoadd, KidsDataProc>(
        todproc, co, config, config_filepaths, logger, os);
}

template <class TodProcVariant, class RunProcessor>
int visit_tod_processor_or_failure(TodProcVariant &todproc,
                                   RunProcessor &&run_processor) {
    return std::visit(
        [&](auto &selected_todproc) {
            using todproc_t = std::decay_t<decltype(selected_todproc)>;
            if constexpr (is_empty_tod_processor_v<todproc_t>) {
                return EXIT_FAILURE;
            }
            else {
                return run_processor(selected_todproc);
            }
        },
        todproc);
}

template <class BeammapTodProc, class PointingTodProc, class KidsDataProc,
          class TodProcVariant, class IOCoordinator, class Config,
          class ConfigFilepaths, class Logger>
int run_standard_cli_reduction_variant(
    TodProcVariant &todproc, const IOCoordinator &co, Config &config,
    const ConfigFilepaths &config_filepaths, const Logger &logger,
    std::ostream &os) {
    return visit_tod_processor_or_failure(
        todproc,
        [&](auto &selected_todproc) {
            using todproc_t = std::decay_t<decltype(selected_todproc)>;
            return run_standard_cli_reduction_processor<
                todproc_t, BeammapTodProc, PointingTodProc, KidsDataProc>(
                selected_todproc, co, config, config_filepaths, logger, os);
        });
}

}  // namespace citlali::cli
