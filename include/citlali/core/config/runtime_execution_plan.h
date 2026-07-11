#pragma once

#include <citlali/core/config/runtime_config.h>

namespace citlali::config {

struct RuntimeThreadPlan {
    int requested_threads = 1;
    int omp_threads = 1;
    int eigen_threads = 1;
    int fftw_plan_threads = 1;
    bool wiener_filter_omp = false;
};

struct EffectiveRuntimeConfig {
    RuntimeConfig values;
    RuntimeThreadPlan threads;
};

struct RealizedRuntimeConfig {
    int omp_threads = 0;
    int eigen_threads = 0;
    int fftw_plan_threads = 0;
    bool fftw_threads_initialized = false;
};

struct RuntimeConfigProvenance {
    RuntimeConfig requested;
    EffectiveRuntimeConfig effective;
    RealizedRuntimeConfig realized;
    bool initialized = false;
};

inline RuntimeThreadPlan make_runtime_thread_plan(
    int requested_threads, bool wiener_filter_omp) {
    return RuntimeThreadPlan{
        requested_threads,
        requested_threads,
        1,
        wiener_filter_omp ? 1 : requested_threads,
        wiener_filter_omp,
    };
}

inline EffectiveRuntimeConfig make_effective_runtime_config(
    const RuntimeConfig &requested, bool wiener_filter_omp) {
    return EffectiveRuntimeConfig{
        requested,
        make_runtime_thread_plan(requested.n_threads, wiener_filter_omp),
    };
}

inline RuntimeConfigProvenance make_runtime_config_provenance(
    const RuntimeConfig &requested, bool wiener_filter_omp) {
    return RuntimeConfigProvenance{
        requested,
        make_effective_runtime_config(requested, wiener_filter_omp),
        RealizedRuntimeConfig{},
        true,
    };
}

}  // namespace citlali::config
