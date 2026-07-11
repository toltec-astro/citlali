#pragma once

#include <Eigen/Core>
#include <fftw3.h>
#include <omp.h>

#include <citlali/core/config/runtime_execution_plan.h>
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
void configure_citlali_runtime_threads(Engine &engine,
                                       const Logger &logger) {
    const auto &effective =
        citlali::pipeline::effective_runtime_config(engine);
    auto realized = configure_runtime_threads(
            effective.threads, logger,
            [](int n_threads) { omp_set_num_threads(n_threads); },
            [](int n_threads) { Eigen::setNbThreads(n_threads); },
            []() { return fftw_init_threads(); },
            [](int n_threads) { fftw_plan_with_nthreads(n_threads); });
    realized.parallel_policy = effective.values.parallel_policy;
    realized.reduction_type = effective.values.reduction_type;
    citlali::pipeline::runtime_config_provenance(engine).realized = realized;
}

}  // namespace citlali::cli
