#pragma once

#include <citlali/core/pipeline/array_properties_table.h>

namespace citlali::pipeline {

template <bool IsBeammap, class TodProc, class RawObs, class Logger>
void configure_observation_calibration(TodProc &todproc, const RawObs &rawobs,
                                       const Logger &logger) {
    auto &engine = todproc.engine();

    logger->debug("getting astrometry config");
    engine.get_astrometry_config(rawobs.astrometry_calib_info().config());

    if constexpr (IsBeammap) {
        engine.get_photometry_config(rawobs.photometry_calib_info().config());
        if (engine.map_grouping == "detector" ||
            engine.map_grouping == "auto") {
            logger->info("making apt file from raw nc files");
            todproc.get_apt_from_files(rawobs);
            return;
        }
    }

    load_array_properties_table(engine, rawobs, logger);
}

}  // namespace citlali::pipeline
