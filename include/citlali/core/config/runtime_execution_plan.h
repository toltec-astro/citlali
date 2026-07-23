#pragma once

#include <citlali/core/config/runtime_config.h>

#include <algorithm>
#include <string>
#include <utility>

namespace citlali::config {

struct RuntimeResourceAvailability {
    int slurm_cpus_per_task = 0;
    int affinity_cpus = 0;
    int hardware_cpus = 0;
    int available_threads = 0;
    std::string source = "unavailable";
};

struct RuntimeThreadPlan {
    int requested_threads = 1;
    int effective_threads = 1;
    int omp_threads = 1;
    int eigen_threads = 1;
    int fftw_plan_threads = 1;
    bool wiener_filter_omp = false;
    RuntimeResourceAvailability availability;
    bool adjusted = false;
    std::string adjustment_reason;
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
    ParallelPolicy parallel_policy = ParallelPolicy::seq;
    ReductionType reduction_type = ReductionType::science;
};

struct RuntimeConfigProvenance {
    RuntimeConfig requested;
    EffectiveRuntimeConfig effective;
    RealizedRuntimeConfig realized;
    bool initialized = false;
};

inline RuntimeThreadPlan make_runtime_thread_plan(
    int requested_threads, bool wiener_filter_omp,
    RuntimeResourceAvailability availability = {}) {
    const int effective_threads =
        availability.available_threads > 0
            ? std::min(requested_threads, availability.available_threads)
            : requested_threads;
    const bool adjusted = effective_threads != requested_threads;
    return RuntimeThreadPlan{
        requested_threads,
        effective_threads,
        effective_threads,
        1,
        wiener_filter_omp ? 1 : effective_threads,
        wiener_filter_omp,
        std::move(availability),
        adjusted,
        adjusted ? "requested threads exceed available CPU resources" : "",
    };
}

inline EffectiveRuntimeConfig make_effective_runtime_config(
    const RuntimeConfig &requested, bool wiener_filter_omp,
    RuntimeResourceAvailability availability = {}) {
    auto effective = requested;
    auto threads = make_runtime_thread_plan(
        requested.n_threads, wiener_filter_omp, std::move(availability));
    effective.n_threads = threads.effective_threads;
    return EffectiveRuntimeConfig{effective, std::move(threads)};
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
