#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/array_properties_table.h>

namespace citlali::pipeline {

template <class Engine, class RawObs, class Logger>
void load_astrometry_config(Engine &engine, const RawObs &rawobs,
                            const Logger &logger) {
    logger->debug("getting astrometry config");
    engine.get_astrometry_config(rawobs.astrometry_calib_info().config());
}

template <class Engine, class RawObs>
void load_photometry_config(Engine &engine, const RawObs &rawobs) {
    engine.get_photometry_config(rawobs.photometry_calib_info().config());
}

template <class Engine>
bool should_make_apt_from_raw_files(const Engine &engine) {
    return engine.typed_config.mapmaking.grouping ==
               citlali::config::MapGrouping::detector ||
           engine.typed_config.mapmaking.grouping ==
               citlali::config::MapGrouping::automatic;
}

template <class TodProc, class RawObs, class Logger>
void make_apt_from_raw_files(TodProc &todproc, const RawObs &rawobs,
                             const Logger &logger) {
    logger->info("making apt file from raw nc files");
    todproc.get_apt_from_files(rawobs);
}

template <bool IsBeammap, class TodProc, class RawObs, class Logger>
void configure_observation_calibration(TodProc &todproc, const RawObs &rawobs,
                                       const Logger &logger) {
    auto &engine = todproc.engine();

    load_astrometry_config(engine, rawobs, logger);

    if constexpr (IsBeammap) {
        load_photometry_config(engine, rawobs);
        if (should_make_apt_from_raw_files(engine)) {
            make_apt_from_raw_files(todproc, rawobs, logger);
            return;
        }
    }

    load_array_properties_table(engine, rawobs, logger);
}

}  // namespace citlali::pipeline
