#pragma once

namespace citlali::pipeline {

template <class Engine>
bool verbose_runtime_enabled(const Engine &engine) {
    return engine.typed_config.runtime.verbose;
}

}  // namespace citlali::pipeline
