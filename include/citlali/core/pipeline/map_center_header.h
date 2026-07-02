#pragma once

namespace citlali::pipeline {

template <class Logger>
void log_map_center_override(double ra_degrees, double dec_degrees,
                             const Logger &logger) {
    logger->info("overwriting map center to ({}, {})",
                 ra_degrees, dec_degrees);
}

template <class Engine>
void set_map_center_header(Engine &engine, double ra_radians,
                           double dec_radians) {
    engine.telescope.tel_header["Header.Source.Ra"].setConstant(ra_radians);
    engine.telescope.tel_header["Header.Source.Dec"].setConstant(dec_radians);
}

}  // namespace citlali::pipeline
