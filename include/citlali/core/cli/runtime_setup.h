#pragma once

#include <Eigen/Core>
#include <fftw3.h>
#include <omp.h>

#include <citlali/core/config/runtime_execution_plan.h>
#include <citlali/core/cli/runtime_resources.h>
#include <citlali/core/pipeline/runtime_policy.h>

namespace citlali::cli {

inline int fftw_threads_for_runtime(int requested_threads,
                                    bool use_wiener_filter_omp) {
    return citlali::config::make_runtime_thread_plan(
               requested_threads, use_wiener_filter_omp)
        .fftw_plan_threads;
}

template <class Logger, class SetOmpThreads, class SetEigenThreads,
          class InitFftwThreads, class PlanFftwThreads>
citlali::config::RealizedRuntimeConfig configure_runtime_threads(
    const citlali::config::RuntimeThreadPlan &plan, const Logger &logger,
    SetOmpThreads &&set_omp_threads, SetEigenThreads &&set_eigen_threads,
    InitFftwThreads &&init_fftw_threads,
    PlanFftwThreads &&plan_fftw_threads) {
    citlali::config::RealizedRuntimeConfig realized;
    set_omp_threads(plan.omp_threads);
    realized.omp_threads = plan.omp_threads;
    set_eigen_threads(plan.eigen_threads);
    realized.eigen_threads = plan.eigen_threads;

    const int fftw_init_ok = init_fftw_threads();
    if (!fftw_init_ok) {
        logger->warn(
            "unable to initialize FFTW threading; using default FFTW behavior");
        return realized;
    }

    realized.fftw_threads_initialized = true;
    plan_fftw_threads(plan.fftw_plan_threads);
    realized.fftw_plan_threads = plan.fftw_plan_threads;
    logger->info("configured FFTW plan threads={}", plan.fftw_plan_threads);
    return realized;
}

template <class Engine, class Logger>
void resolve_citlali_runtime_resources(
    Engine &engine, const Logger &logger,
    const citlali::config::RuntimeResourceAvailability &availability) {
    auto &provenance =
        citlali::pipeline::runtime_config_provenance(engine);
    provenance.effective = citlali::config::make_effective_runtime_config(
        provenance.requested,
        provenance.effective.threads.wiener_filter_omp,
        availability);
    const auto &effective = provenance.effective;
    if (effective.threads.adjusted) {
        logger->warn(
            "requested runtime threads={} exceed available CPU resources={} "
            "(source={}); using {} threads",
            effective.threads.requested_threads,
            effective.threads.availability.available_threads,
            effective.threads.availability.source,
            effective.threads.effective_threads);
    }
}

template <class Engine, class Logger>
void configure_citlali_runtime_threads(Engine &engine,
                                       const Logger &logger) {
    resolve_citlali_runtime_resources(
        engine, logger, discover_runtime_resource_availability());
    auto &provenance =
        citlali::pipeline::runtime_config_provenance(engine);
    const auto &effective = provenance.effective;
    auto realized = configure_runtime_threads(
            effective.threads, logger,
            [](int n_threads) { omp_set_num_threads(n_threads); },
            [](int n_threads) { Eigen::setNbThreads(n_threads); },
            []() { return fftw_init_threads(); },
            [](int n_threads) { fftw_plan_with_nthreads(n_threads); });
    realized.parallel_policy = effective.values.parallel_policy;
    realized.reduction_type = effective.values.reduction_type;
    provenance.realized = realized;
}

}  // namespace citlali::cli
