#pragma once

#include <cmath>

namespace citlali::pipeline {

template <class Engine>
bool downsample_factor_requires_frequency(const Engine &engine) {
    return engine.rtcproc.downsampler.factor <= 0;
}

template <class Engine, class Logger>
bool configure_effective_sample_rate(Engine &engine, const Logger &logger) {
    if (engine.rtcproc.run_downsample) {
        if (engine.rtcproc.downsampler.factor <= 0) {
            if (engine.rtcproc.downsampler.downsampled_freq_Hz <= 0) {
                logger->error(
                    "downsampled freq ({} Hz) must be > 0 when downsample "
                    "factor <= 0",
                    engine.rtcproc.downsampler.downsampled_freq_Hz);
                return false;
            }
            if (engine.rtcproc.downsampler.downsampled_freq_Hz >
                engine.telescope.fsmp) {
                logger->error(
                    "downsampled freq ({} Hz) must be less than sample rate "
                    "({} Hz)",
                    engine.rtcproc.downsampler.downsampled_freq_Hz,
                    engine.telescope.fsmp);
                return false;
            }
            engine.rtcproc.downsampler.factor = std::floor(
                engine.telescope.fsmp /
                engine.rtcproc.downsampler.downsampled_freq_Hz);
        }
        if (engine.rtcproc.downsampler.factor <= 0) {
            logger->error("downsample factor ({}) must be > 0",
                          engine.rtcproc.downsampler.factor);
            return false;
        }

        const double downsample_nyquist_Hz =
            engine.telescope.fsmp /
            (2.0 * engine.rtcproc.downsampler.factor);
        if (engine.rtcproc.filter.freq_high_Hz > downsample_nyquist_Hz) {
            logger->error(
                "invalid anti-alias setup: filter freq_high_Hz ({} Hz) "
                "exceeds downsample Nyquist ({} Hz)",
                engine.rtcproc.filter.freq_high_Hz,
                downsample_nyquist_Hz);
            return false;
        }
        engine.telescope.d_fsmp =
            engine.telescope.fsmp / engine.rtcproc.downsampler.factor;
    }
    else {
        engine.telescope.d_fsmp = engine.telescope.fsmp;
    }
    return true;
}

}  // namespace citlali::pipeline
