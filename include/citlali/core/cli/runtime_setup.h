#pragma once

namespace citlali::cli {

inline int fftw_threads_for_runtime(int requested_threads,
                                    bool use_wiener_filter_omp) {
    return use_wiener_filter_omp ? 1 : requested_threads;
}

template <class Engine, class Logger, class SetOmpThreads,
          class SetEigenThreads, class InitFftwThreads,
          class PlanFftwThreads>
void configure_runtime_threads(const Engine &engine, const Logger &logger,
                               bool use_wiener_filter_omp,
                               SetOmpThreads &&set_omp_threads,
                               SetEigenThreads &&set_eigen_threads,
                               InitFftwThreads &&init_fftw_threads,
                               PlanFftwThreads &&plan_fftw_threads) {
    set_omp_threads(engine.n_threads);
    set_eigen_threads(1);

    const int fftw_init_ok = init_fftw_threads();
    if (!fftw_init_ok) {
        logger->warn(
            "unable to initialize FFTW threading; using default FFTW behavior");
        return;
    }

    const int fftw_n_threads =
        fftw_threads_for_runtime(engine.n_threads, use_wiener_filter_omp);
    plan_fftw_threads(fftw_n_threads);
    logger->info("configured FFTW plan threads={}", fftw_n_threads);
}

}  // namespace citlali::cli
