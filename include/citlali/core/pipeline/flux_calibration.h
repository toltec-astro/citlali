#pragma once

namespace citlali::pipeline {

template <class Engine>
decltype(auto) flux_calibration_signal_unit(const Engine &engine) {
    return engine.omb.sig_unit;
}

template <class Engine>
decltype(auto) flux_calibration_pixel_size_rad(const Engine &engine) {
    return engine.omb.pixel_size_rad;
}

template <class Engine, class Logger>
void calculate_flux_calibration(Engine &engine, const Logger &logger) {
    logger->info("calculating flux calibration");
    engine.calib.calc_flux_calibration(flux_calibration_signal_unit(engine),
                                       engine.omb.pixel_size_rad);
}

template <class Engine, class Logger>
void calculate_reduction_observation_flux_calibration(
    Engine &engine, const Logger &logger) {
    calculate_flux_calibration(engine, logger);
}

}  // namespace citlali::pipeline
