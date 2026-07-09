#pragma once

#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/fruit_loop_iteration_state.h>
#include <citlali/core/pipeline/map_index_state.h>

struct ReductionProgressState {
    // map count and per-map index translations
    citlali::pipeline::MapIndexState map_indices;

    // current fruit-loop iteration counter
    citlali::pipeline::FruitLoopRuntimeState iteration;

    // shared state learned across RTC, PTC, and mapmaking phases
    ReductionLearningState learning;
};
