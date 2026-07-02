#pragma once

namespace citlali::pipeline {

template <class RawObs, class Logger>
void log_invalid_flxscale_correction_factor(double factor,
                                            const RawObs &rawobs,
                                            const Logger &logger) {
    logger->error(
        "invalid flxscale_correction={} for observation {}; "
        "factor must be finite and > 0",
        factor, rawobs.name());
}

template <class RawObs, class Logger>
void log_missing_flxscale_column(const RawObs &rawobs,
                                 const Logger &logger) {
    logger->error(
        "flxscale column missing from APT while applying "
        "flxscale_correction for observation {}",
        rawobs.name());
}

template <class RawObs, class Logger>
void log_applied_flxscale_correction(double factor, const RawObs &rawobs,
                                     const Logger &logger) {
    logger->info("applied flxscale correction factor={} for observation {}",
                 factor, rawobs.name());
}

}  // namespace citlali::pipeline
