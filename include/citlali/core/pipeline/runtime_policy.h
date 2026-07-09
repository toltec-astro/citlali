#pragma once

#include <string>

#include <citlali/core/config/runtime_config.h>

namespace citlali::pipeline {

template <class Engine>
bool verbose_runtime_enabled(const Engine &engine) {
    return engine.typed_config.runtime.verbose;
}

template <class Engine>
int runtime_thread_count(const Engine &engine) {
    return engine.typed_config.runtime.n_threads;
}

inline std::string runtime_parallel_policy_name(
    const citlali::config::RuntimeConfig &runtime_config) {
    return std::string(citlali::config::to_string(
        runtime_config.parallel_policy));
}

template <class Engine>
std::string runtime_parallel_policy_name(const Engine &engine) {
    return runtime_parallel_policy_name(engine.typed_config.runtime);
}

}  // namespace citlali::pipeline
