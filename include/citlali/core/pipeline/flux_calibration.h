#pragma once

namespace citlali::pipeline {

template <class Engine, class Logger>
void calculate_flux_calibration(Engine &engine, const Logger &logger) {
    logger->info("calculating flux calibration");
    engine.calib.calc_flux_calibration(engine.omb.sig_unit,
                                       engine.omb.pixel_size_rad);
}

template <class Engine, class Logger>
void calculate_reduction_observation_flux_calibration(
    Engine &engine, const Logger &logger) {
    calculate_flux_calibration(engine, logger);
}

}  // namespace citlali::pipeline
