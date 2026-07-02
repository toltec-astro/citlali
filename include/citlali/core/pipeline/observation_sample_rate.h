#pragma once

#include <cmath>

namespace citlali::pipeline {

template <class Engine>
bool should_run_downsample(const Engine &engine) {
    return engine.rtcproc.run_downsample;
}

template <class Engine>
auto downsample_factor(const Engine &engine) {
    return engine.rtcproc.downsampler.factor;
}

template <class Engine>
auto requested_downsample_frequency_hz(const Engine &engine) {
    return engine.rtcproc.downsampler.downsampled_freq_Hz;
}

template <class Engine>
auto native_sample_rate_hz(const Engine &engine) {
    return engine.telescope.fsmp;
}

template <class Engine>
bool downsample_factor_requires_frequency(const Engine &engine) {
    return downsample_factor(engine) <= 0;
}

template <class Engine, class Logger>
bool validate_downsample_frequency_positive(const Engine &engine,
                                            const Logger &logger) {
    if (requested_downsample_frequency_hz(engine) <= 0) {
        logger->error(
            "downsampled freq ({} Hz) must be > 0 when downsample "
            "factor <= 0",
            requested_downsample_frequency_hz(engine));
        return false;
    }
    return true;
}

template <class Engine, class Logger>
bool validate_downsample_frequency_below_sample_rate(
    const Engine &engine, const Logger &logger) {
    if (requested_downsample_frequency_hz(engine) >
        native_sample_rate_hz(engine)) {
        logger->error(
            "downsampled freq ({} Hz) must be less than sample rate "
            "({} Hz)",
            requested_downsample_frequency_hz(engine),
            native_sample_rate_hz(engine));
        return false;
    }
    return true;
}

template <class Engine, class Logger>
bool validate_requested_downsample_frequency(const Engine &engine,
                                             const Logger &logger) {
    return validate_downsample_frequency_positive(engine, logger) &&
           validate_downsample_frequency_below_sample_rate(engine, logger);
}

template <class Engine>
void derive_downsample_factor_from_frequency(Engine &engine) {
    engine.rtcproc.downsampler.factor = std::floor(
        engine.telescope.fsmp /
        engine.rtcproc.downsampler.downsampled_freq_Hz);
}

template <class Engine, class Logger>
bool validate_downsample_factor(const Engine &engine, const Logger &logger) {
    if (downsample_factor(engine) <= 0) {
        logger->error("downsample factor ({}) must be > 0",
                      downsample_factor(engine));
        return false;
    }
    return true;
}

template <class Engine>
double downsample_nyquist_hz(const Engine &engine) {
    return engine.telescope.fsmp /
           (2.0 * downsample_factor(engine));
}

template <class Engine, class Logger>
bool validate_downsample_antialias_filter(const Engine &engine,
                                          double downsample_nyquist_Hz,
                                          const Logger &logger) {
    if (engine.rtcproc.filter.freq_high_Hz > downsample_nyquist_Hz) {
        logger->error(
            "invalid anti-alias setup: filter freq_high_Hz ({} Hz) "
            "exceeds downsample Nyquist ({} Hz)",
            engine.rtcproc.filter.freq_high_Hz,
            downsample_nyquist_Hz);
        return false;
    }
    return true;
}

template <class Engine>
void apply_downsampled_sample_rate(Engine &engine) {
    engine.telescope.d_fsmp =
        engine.telescope.fsmp / downsample_factor(engine);
}

template <class Engine>
void apply_native_sample_rate(Engine &engine) {
    engine.telescope.d_fsmp = engine.telescope.fsmp;
}

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
