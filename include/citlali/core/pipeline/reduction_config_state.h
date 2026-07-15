#pragma once

#include <citlali/core/config/reduction_config.h>
#include <citlali/core/config/runtime_execution_plan.h>
#include <citlali/core/pipeline/beammap_execution_plan.h>
#include <citlali/core/pipeline/coadd_execution_plan.h>
#include <citlali/core/pipeline/config_diagnostics_state.h>
#include <citlali/core/pipeline/kids_external_config.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>
#include <citlali/core/pipeline/noise_execution_plan.h>
#include <citlali/core/pipeline/pointing_execution_plan.h>
#include <citlali/core/pipeline/post_processing_execution_plan.h>
#include <citlali/core/pipeline/processed_timestream_execution_plan.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>

namespace citlali::pipeline {

struct ReductionConfigState {
    ConfigDiagnosticsState config_diagnostics;
    citlali::config::ReductionConfig typed_config;
    citlali::config::RuntimeConfigProvenance runtime_config_provenance;
    KidsExternalConfigPlan kids_external_plan;
    RawTimestreamExecutionPlan raw_timestream_plan;
    ProcessedTimestreamExecutionPlan processed_timestream_plan;
    MapmakingExecutionPlan mapmaking_plan;
    CoaddExecutionPlan coadd_plan;
    NoiseExecutionPlan noise_plan;
    PointingExecutionPlan pointing_plan;
    PostProcessingExecutionPlan post_processing_plan;
    BeammapExecutionPlan beammap_plan;
};

}  // namespace citlali::pipeline
