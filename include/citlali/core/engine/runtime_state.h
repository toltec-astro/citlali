#pragma once

#include <citlali/core/engine/reduction_progress_state.h>
#include <citlali/core/pipeline/logging_state.h>
#include <citlali/core/pipeline/observation_runtime_state.h>
#include <citlali/core/pipeline/reduction_config_state.h>
#include <citlali/core/pipeline/reduction_output_state.h>

struct EngineRuntimeState : public citlali::pipeline::LoggingState,
                            public citlali::pipeline::ReductionConfigState,
                            public citlali::pipeline::ReductionOutputState,
                            public citlali::pipeline::ObservationRuntimeState,
                            public ReductionProgressState {};
