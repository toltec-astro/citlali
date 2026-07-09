#pragma once

#include <citlali/core/config/reduction_config.h>
#include <citlali/core/pipeline/config_diagnostics_state.h>

namespace citlali::pipeline {

struct ReductionConfigState {
    ConfigDiagnosticsState config_diagnostics;
    citlali::config::ReductionConfig typed_config;
};

}  // namespace citlali::pipeline
