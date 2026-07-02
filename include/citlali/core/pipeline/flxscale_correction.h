#pragma once

#include <cmath>

namespace citlali::pipeline {

template <class RawObs>
auto flxscale_correction_metadata(const RawObs &rawobs) {
    return rawobs.flxscale_correction();
}

template <class FlxscaleCorrection>
bool has_flxscale_correction(const FlxscaleCorrection *flxscale_corr) {
    return flxscale_corr != nullptr;
}

template <class FlxscaleCorrection>
double flxscale_correction_factor(const FlxscaleCorrection &flxscale_corr) {
    return flxscale_corr.value();
}

inline bool is_valid_flxscale_correction_factor(double factor) {
    return std::isfinite(factor) && factor > 0.0;
}

template <class Engine>
bool has_apt_flxscale_column(const Engine &engine) {
    return engine.calib.apt.count("flxscale") != 0;
}

template <class RawObs, class Logger>
void log_invalid_flxscale_correction_factor(double factor,
                                            const RawObs &rawobs,
                                            const Logger &logger) {
    logger->error(
        "invalid flxscale_correction={} for observation {}; "
        "factor must be finite and > 0",
        factor, rawobs.name());
}

template <class Engine, class RawObs, class Logger>
bool apply_flxscale_correction(Engine &engine, const RawObs &rawobs,
                               const Logger &logger) {
    const auto *flxscale_corr = flxscale_correction_metadata(rawobs);
    if (!has_flxscale_correction(flxscale_corr)) {
        return true;
    }

    const double factor = flxscale_correction_factor(*flxscale_corr);
    if (!is_valid_flxscale_correction_factor(factor)) {
        logger->error(
            "invalid flxscale_correction={} for observation {}; "
            "factor must be finite and > 0",
            factor, rawobs.name());
        return false;
    }
    if (!has_apt_flxscale_column(engine)) {
        logger->error(
            "flxscale column missing from APT while applying "
            "flxscale_correction for observation {}",
            rawobs.name());
        return false;
    }

    engine.calib.apt["flxscale"].array() *= factor;
    logger->info("applied flxscale correction factor={} for observation {}",
                 factor, rawobs.name());
    return true;
}

}  // namespace citlali::pipeline
