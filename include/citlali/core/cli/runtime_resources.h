#pragma once

#include <citlali/core/config/runtime_execution_plan.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace citlali::cli {

inline int parse_positive_cpu_count(const char *value) {
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0 ||
        parsed > INT_MAX) {
        return 0;
    }
    return static_cast<int>(parsed);
}

inline citlali::config::RuntimeResourceAvailability
select_runtime_resource_availability(int slurm_cpus, int affinity_cpus,
                                     int hardware_cpus) {
    citlali::config::RuntimeResourceAvailability result;
    result.slurm_cpus_per_task = slurm_cpus > 0 ? slurm_cpus : 0;
    result.affinity_cpus = affinity_cpus > 0 ? affinity_cpus : 0;
    result.hardware_cpus = hardware_cpus > 0 ? hardware_cpus : 0;

    if (result.slurm_cpus_per_task > 0 && result.affinity_cpus > 0) {
        result.available_threads =
            std::min(result.slurm_cpus_per_task, result.affinity_cpus);
        result.source = "slurm+affinity";
    } else if (result.affinity_cpus > 0) {
        result.available_threads = result.affinity_cpus;
        result.source = "affinity";
    } else if (result.slurm_cpus_per_task > 0) {
        result.available_threads = result.slurm_cpus_per_task;
        result.source = "slurm";
    } else if (result.hardware_cpus > 0) {
        result.available_threads = result.hardware_cpus;
        result.source = "hardware";
    }
    return result;
}

inline int process_affinity_cpu_count() {
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        return CPU_COUNT(&mask);
    }
#endif
    return 0;
}

inline citlali::config::RuntimeResourceAvailability
discover_runtime_resource_availability() {
    return select_runtime_resource_availability(
        parse_positive_cpu_count(std::getenv("SLURM_CPUS_PER_TASK")),
        process_affinity_cpu_count(),
        static_cast<int>(std::thread::hardware_concurrency()));
}

}  // namespace citlali::cli
