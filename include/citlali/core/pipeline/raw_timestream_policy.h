#pragma once

#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::pipeline {

template <class Engine>
bool raw_kernel_enabled(const Engine &engine) {
    return raw_time_chunk_config(engine).kernel.enabled;
}

template <class Engine>
bool raw_flux_calibration_enabled(const Engine &engine) {
    return raw_time_chunk_config(engine).flux_calibration_enabled;
}

template <class Engine>
bool raw_extinction_correction_enabled(const Engine &engine) {
    return raw_time_chunk_config(engine).extinction_correction_enabled;
}

template <class Engine>
bool raw_fir_filter_enabled(const Engine &engine) {
    return raw_time_chunk_config(engine).filter.enabled;
}

template <class Engine>
bool raw_notch_filter_enabled(const Engine &engine) {
    const auto &filter = raw_time_chunk_config(engine).filter;
    return filter.enabled && filter.notch.enabled;
}

template <class Engine>
bool raw_iir_filter_enabled(const Engine &engine) {
    return raw_time_chunk_config(engine).iir_filter.enabled;
}

template <class Engine>
double raw_iir_filter_frequency_hz(const Engine &engine) {
    return raw_time_chunk_config(engine).iir_filter.freq_Hz;
}

template <class Engine>
bool raw_iir_filter_below_nyquist(const Engine &engine) {
    return raw_iir_filter_frequency_hz(engine) < engine.telescope.fsmp / 2.0;
}

}  // namespace citlali::pipeline
