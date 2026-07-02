#pragma once

#include <citlali/core/pipeline/map_center_config.h>
#include <citlali/core/pipeline/map_center_header.h>

namespace citlali::pipeline {

template <class Engine, class Logger>
void overwrite_map_center_if_configured(Engine &engine, const Logger &logger) {
    if (has_map_center_override(engine)) {
        log_map_center_override(map_center_ra_degrees(engine),
                                map_center_dec_degrees(engine), logger);
        const double map_center_ra_rad =
            degrees_to_radians(map_center_ra_degrees(engine));
        const double map_center_dec_rad =
            degrees_to_radians(map_center_dec_degrees(engine));
        set_map_center_header(
            engine, map_center_ra_rad, map_center_dec_rad);
    }
}

}  // namespace citlali::pipeline
