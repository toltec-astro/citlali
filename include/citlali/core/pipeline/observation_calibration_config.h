#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/array_properties_table.h>
#include <citlali/core/pipeline/pointing_offsets_config_read.h>
#include <citlali/core/pipeline/rawobs_observation_output_layout.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <cstddef>
#include <utility>

namespace citlali::pipeline {

template <class Engine, class RawObs, class Logger>
void load_astrometry_config(Engine &engine, const RawObs &rawobs,
                            const Logger &logger) {
    logger->debug("getting astrometry config");
    engine.get_astrometry_config(rawobs.astrometry_calib_info().config());
}

template <class Engine, class RawObs, class RawObsKidsMeta, class Logger>
void load_astrometry_config_with_context(
    Engine &engine, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta,
    std::size_t observation_index, const Logger &logger) {
    if constexpr (!has_astrometry_plan_v<Engine>) {
        load_astrometry_config(engine, rawobs, logger);
    }
    else {
        logger->debug("getting astrometry config");
        auto request = read_astrometry_config(
            rawobs.astrometry_calib_info().config(), logger);
        require_valid_astrometry_config(request, logger);
        auto &plan = astrometry_plan(engine);
        record_astrometry_request(
            plan, observation_index,
            obsnum_from_rawobs_meta(rawobs_kids_meta, logger), request);
        install_astrometry_config(
            std::move(request), astrometry_config(engine),
            engine.pointing_offsets);
        record_astrometry_installed(plan);
    }
}

template <class Engine, class RawObs>
void load_photometry_config(Engine &engine, const RawObs &rawobs) {
    engine.get_photometry_config(rawobs.photometry_calib_info().config());
}

template <class Engine>
bool should_make_apt_from_raw_files(const Engine &engine) {
    const auto grouping = mapmaking_config(engine).grouping;
    return citlali::config::is_detector_map_grouping(grouping) ||
           citlali::config::is_automatic_map_grouping(grouping);
}

template <class TodProc, class RawObs, class Logger>
void make_apt_from_raw_files(TodProc &todproc, const RawObs &rawobs,
                             const Logger &logger) {
    logger->info("making apt file from raw nc files");
    todproc.get_apt_from_files(rawobs);
}

template <bool IsBeammap, class TodProc, class RawObs, class Logger>
void finish_observation_calibration(TodProc &todproc, const RawObs &rawobs,
                                    const Logger &logger) {
    auto &engine = todproc.engine();

    if constexpr (IsBeammap) {
        load_photometry_config(engine, rawobs);
        if (should_make_apt_from_raw_files(engine)) {
            make_apt_from_raw_files(todproc, rawobs, logger);
            return;
        }
    }

    load_array_properties_table(
        engine, rawobs, logger,
        IsBeammap ? AptDetectorRelationRetention::discard
                  : AptDetectorRelationRetention::retain);
}

template <bool IsBeammap, class TodProc, class RawObs, class Logger>
void configure_observation_calibration(TodProc &todproc, const RawObs &rawobs,
                                       const Logger &logger) {
    load_astrometry_config(todproc.engine(), rawobs, logger);
    finish_observation_calibration<IsBeammap>(todproc, rawobs, logger);
}

template <bool IsBeammap, class TodProc, class RawObs,
          class RawObsKidsMeta, class Logger>
void configure_observation_calibration_with_context(
    TodProc &todproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta,
    std::size_t observation_index, const Logger &logger) {
    load_astrometry_config_with_context(
        todproc.engine(), rawobs, rawobs_kids_meta, observation_index,
        logger);
    finish_observation_calibration<IsBeammap>(todproc, rawobs, logger);
}

}  // namespace citlali::pipeline
