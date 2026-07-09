#pragma once

namespace citlali::pipeline {

template <class Engine>
bool verbose_runtime_enabled(const Engine &engine) {
    return engine.typed_config.runtime.verbose;
}

template <class Engine>
int runtime_thread_count(const Engine &engine) {
    return engine.typed_config.runtime.n_threads;
}

}  // namespace citlali::pipeline
