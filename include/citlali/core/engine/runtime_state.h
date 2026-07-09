#pragma once

#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/fruit_loop_iteration_state.h>
#include <citlali/core/pipeline/logging_state.h>
#include <citlali/core/pipeline/map_index_state.h>
#include <citlali/core/pipeline/observation_runtime_state.h>
#include <citlali/core/pipeline/reduction_config_state.h>
#include <citlali/core/pipeline/reduction_output_state.h>

struct EngineRuntimeState : public citlali::pipeline::LoggingState,
                            public citlali::pipeline::ReductionConfigState,
                            public citlali::pipeline::ReductionOutputState,
                            public citlali::pipeline::ObservationRuntimeState {
    // map count and per-map index translations
    citlali::pipeline::MapIndexState map_indices;

    // current fruit-loop iteration counter
    citlali::pipeline::FruitLoopRuntimeState iteration;

    // shared state learned across RTC, PTC, and mapmaking phases
    ReductionLearningState learning;
};
