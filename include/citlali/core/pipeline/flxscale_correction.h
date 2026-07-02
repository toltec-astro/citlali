#pragma once

#include <citlali/core/pipeline/flxscale_correction_logging.h>
#include <citlali/core/pipeline/flxscale_correction_metadata.h>

namespace citlali::pipeline {

template <class Engine>
bool has_apt_flxscale_column(const Engine &engine) {
    return engine.calib.apt.count("flxscale") != 0;
}

template <class Engine>
void multiply_apt_flxscale_column(Engine &engine, double factor) {
    engine.calib.apt["flxscale"].array() *= factor;
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
        log_invalid_flxscale_correction_factor(factor, rawobs, logger);
        return false;
    }
    if (!has_apt_flxscale_column(engine)) {
        log_missing_flxscale_column(rawobs, logger);
        return false;
    }

    multiply_apt_flxscale_column(engine, factor);
    log_applied_flxscale_correction(factor, rawobs, logger);
    return true;
}

}  // namespace citlali::pipeline
