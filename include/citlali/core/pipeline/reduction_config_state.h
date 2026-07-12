#pragma once

#include <citlali/core/config/reduction_config.h>
#include <citlali/core/config/runtime_execution_plan.h>
#include <citlali/core/pipeline/config_diagnostics_state.h>
#include <citlali/core/pipeline/processed_timestream_execution_plan.h>

namespace citlali::pipeline {

struct ReductionConfigState {
    ConfigDiagnosticsState config_diagnostics;
    citlali::config::ReductionConfig typed_config;
    citlali::config::RuntimeConfigProvenance runtime_config_provenance;
    ProcessedTimestreamExecutionPlan processed_timestream_plan;
};

}  // namespace citlali::pipeline
