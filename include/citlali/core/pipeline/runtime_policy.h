#pragma once

#include <string>

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::pipeline {

template <class Engine>
bool verbose_runtime_enabled(const Engine &engine) {
    return runtime_config(engine).verbose;
}

template <class Engine>
int runtime_thread_count(const Engine &engine) {
    return runtime_config(engine).n_threads;
}

template <class Engine>
const std::string &runtime_output_dir(const Engine &engine) {
    return runtime_config(engine).output_dir;
}

inline std::string runtime_parallel_policy_name(
    const citlali::config::RuntimeConfig &runtime_config) {
    return std::string(citlali::config::to_string(
        runtime_config.parallel_policy));
}

template <class Engine>
std::string runtime_parallel_policy_name(const Engine &engine) {
    return runtime_parallel_policy_name(runtime_config(engine));
}

}  // namespace citlali::pipeline
