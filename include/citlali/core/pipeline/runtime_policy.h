#pragma once

#include <string>

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::pipeline {

template <class Engine>
bool verbose_runtime_enabled(const Engine &engine) {
    return effective_runtime_values(engine).verbose;
}

template <class Engine>
int runtime_thread_count(const Engine &engine) {
    return effective_runtime_config(engine).threads.omp_threads;
}

template <class Engine>
const std::string &runtime_output_dir(const Engine &engine) {
    return effective_runtime_values(engine).output_dir;
}

template <class Engine>
citlali::config::ReductionType runtime_reduction_type(const Engine &engine) {
    return effective_runtime_values(engine).reduction_type;
}

inline std::string runtime_parallel_policy_name(
    const citlali::config::RuntimeConfig &runtime_config) {
    return std::string(citlali::config::to_string(
        runtime_config.parallel_policy));
}

template <class Engine>
std::string runtime_parallel_policy_name(const Engine &engine) {
    return runtime_parallel_policy_name(effective_runtime_values(engine));
}

}  // namespace citlali::pipeline
