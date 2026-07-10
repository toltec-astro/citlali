#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/runtime_config_read.h>

template<typename CT>
citlali::config::RuntimeConfig Engine::get_runtime_config(CT &config) {
    auto &diagnostics = citlali::pipeline::config_diagnostics(*this);
    return citlali::pipeline::read_runtime_config(config, diagnostics);
}
