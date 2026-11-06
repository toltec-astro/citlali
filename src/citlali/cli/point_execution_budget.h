// Private application resource adapter; scientific owners receive a worker budget.
#pragma once
#include <citlali/core/cli/runtime_resources.h>
#include <citlali/core/utils/parallel_work.h>
#include <chrono>
#include <ctime>
#include <sys/resource.h>

namespace citlali::cli::detail {
struct PointExecutionBudget {
    unsigned requested;
    citlali::config::RuntimeResourceAvailability available;
    citlali::utils::ParallelExecution execution;
};
inline PointExecutionBudget point_execution_budget(unsigned requested,unsigned maximum=12) {
    if(requested<1 || requested>maximum || maximum>32)throw std::invalid_argument("configured worker budget outside profile");
    auto available=citlali::cli::discover_runtime_resource_availability();
    const auto plan=citlali::config::make_runtime_thread_plan(static_cast<int>(requested),false,available);
    return {requested,available,{static_cast<unsigned>(plan.effective_threads)}};
}
struct PointPhaseClock {
    std::chrono::steady_clock::time_point wall=std::chrono::steady_clock::now();
    std::clock_t cpu=std::clock();
    double wall_seconds() const {return std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count();}
    double cpu_seconds() const {return double(std::clock()-cpu)/CLOCKS_PER_SEC;}
};
inline long long point_process_peak_rss_bytes() {
    struct rusage usage{};
    if(getrusage(RUSAGE_SELF,&usage)!=0)return -1;
#ifdef __APPLE__
    return usage.ru_maxrss;
#else
    return static_cast<long long>(usage.ru_maxrss)*1024;
#endif
}
} // namespace citlali::cli::detail

