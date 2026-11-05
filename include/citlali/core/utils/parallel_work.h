#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace citlali::utils {

// Execution policy only: no scientific identity, threshold or process state.
// Workers own distinct indexed outputs; callbacks must not mutate shared data.
struct ParallelExecution { unsigned workers = 1; };
struct ParallelWorkReport {
    std::size_t jobs = 0;
    unsigned started_workers = 0;
    unsigned peak_active_jobs = 0;
};

template<class Work>
ParallelWorkReport parallel_for(std::size_t count, ParallelExecution execution, Work work) {
    if (!execution.workers) throw std::invalid_argument("parallel worker budget must be positive");
    if (!count) return {};
    const auto workers = static_cast<unsigned>(std::min<std::size_t>(count, execution.workers));
    std::atomic<std::size_t> next{0};
    std::atomic<unsigned> active{0}, peak{0};
    std::vector<std::exception_ptr> failures(count);
    auto run = [&] {
#ifdef _OPENMP
        // OpenMP control is per calling thread; do not inherit a nested pool.
        const auto prior_omp_threads = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        for (;;) {
            const auto i = next.fetch_add(1);
            if (i >= count) break;
            const auto n = active.fetch_add(1) + 1;
            auto prior = peak.load();
            while (prior < n && !peak.compare_exchange_weak(prior, n)) {}
            try { work(i); } catch (...) { failures[i] = std::current_exception(); }
            active.fetch_sub(1);
        }
#ifdef _OPENMP
        omp_set_num_threads(prior_omp_threads);
#endif
    };
    // The caller participates: at most workers computational threads, not N+1.
    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    for (unsigned i = 1; i < workers; ++i) threads.emplace_back(run);
    run();
    for (auto &thread : threads) thread.join();
    // Every task has finished. Deterministic first-index failure; never publish
    // a partially successful stage or let an exception escape a worker thread.
    for (const auto &failure : failures) if (failure) std::rethrow_exception(failure);
    return {count, workers, peak.load()};
}

// At most worker-budget results/scratch sets live before ordered consumption.
// The consumer runs after the whole batch joins, so it may own serial reduction
// and publication state without locks or scheduling-dependent summation.
template<class Produce, class Consume>
ParallelWorkReport ordered_parallel_batches(std::size_t count, ParallelExecution execution,
                                           Produce produce, Consume consume) {
    if (!execution.workers) throw std::invalid_argument("parallel worker budget must be positive");
    using Result = decltype(produce(std::size_t{}));
    ParallelWorkReport total{count, 0, 0};
    for (std::size_t first = 0; first < count;) {
        const auto size = std::min<std::size_t>(execution.workers, count - first);
        std::vector<std::optional<Result>> results(size);
        const auto report = parallel_for(size, execution, [&](std::size_t i) {
            results[i].emplace(produce(first + i));
        });
        total.started_workers = std::max(total.started_workers, report.started_workers);
        total.peak_active_jobs = std::max(total.peak_active_jobs, report.peak_active_jobs);
        for (std::size_t i = 0; i < size; ++i) consume(first + i, std::move(*results[i]));
        first += size;
    }
    return total;
}
} // namespace citlali::utils

