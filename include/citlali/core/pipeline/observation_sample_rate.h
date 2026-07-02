#pragma once

#include <citlali/core/pipeline/downsample_config.h>

namespace citlali::pipeline {

template <class Engine, class Logger>
bool configure_effective_sample_rate(Engine &engine, const Logger &logger) {
    if (should_run_downsample(engine)) {
        if (downsample_factor_requires_frequency(engine)) {
            if (!validate_requested_downsample_frequency(engine, logger)) {
                return false;
            }
            derive_downsample_factor_from_frequency(engine);
        }
        if (!validate_downsample_factor(engine, logger)) {
            return false;
        }

        const double downsample_nyquist_Hz = downsample_nyquist_hz(engine);
        if (!validate_downsample_antialias_filter(
                engine, downsample_nyquist_Hz, logger)) {
            return false;
        }
        apply_downsampled_sample_rate(engine);
    }
    else {
        apply_native_sample_rate(engine);
    }
    return true;
}

}  // namespace citlali::pipeline
