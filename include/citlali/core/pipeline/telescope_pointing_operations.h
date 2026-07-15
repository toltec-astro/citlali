#pragma once

#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <cstddef>

namespace citlali::pipeline {

template <class Engine, class Logger>
void calculate_tangent_plane_pointing(Engine &engine, const Logger &logger) {
    logger->info("calculating tangent plane pointing");
    engine.telescope.calc_tan_pointing();
}

template <class TodProc, class Logger>
void interpolate_pointing_offsets(TodProc &todproc, const Logger &logger) {
    logger->info("calculating pointing offsets");
    todproc.interp_pointing();
    auto &engine = todproc.engine();
    if constexpr (has_astrometry_plan_v<decltype(engine)>) {
        record_astrometry_applied(
            astrometry_plan(engine),
            static_cast<std::size_t>(
                engine.telescope.tel_data["TelTime"].size()));
    }
}

template <class TodProc, class Logger>
void calculate_telescope_pointing(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    calculate_tangent_plane_pointing(engine, logger);
    interpolate_pointing_offsets(todproc, logger);
}

}  // namespace citlali::pipeline
