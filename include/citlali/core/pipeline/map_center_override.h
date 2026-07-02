#pragma once

namespace citlali::pipeline {

inline double degrees_to_radians(double degrees) {
    constexpr double deg_to_rad = 0.017453292519943295769;
    return degrees * deg_to_rad;
}

template <class Engine>
bool has_map_center_override(const Engine &engine) {
    return engine.omb.crval_config[0] != 0 &&
           engine.omb.crval_config[1] != 0;
}

template <class Engine>
double map_center_ra_degrees(const Engine &engine) {
    return engine.omb.crval_config[0];
}

template <class Engine>
double map_center_dec_degrees(const Engine &engine) {
    return engine.omb.crval_config[1];
}

template <class Engine, class Logger>
void overwrite_map_center_if_configured(Engine &engine, const Logger &logger) {
    if (has_map_center_override(engine)) {
        logger->info("overwriting map center to ({}, {})",
                     engine.omb.crval_config[0], engine.omb.crval_config[1]);
        const double map_center_ra_rad =
            degrees_to_radians(engine.omb.crval_config[0]);
        const double map_center_dec_rad =
            degrees_to_radians(engine.omb.crval_config[1]);
        engine.telescope.tel_header["Header.Source.Ra"].setConstant(
            map_center_ra_rad);
        engine.telescope.tel_header["Header.Source.Dec"].setConstant(
            map_center_dec_rad);
    }
}

}  // namespace citlali::pipeline
