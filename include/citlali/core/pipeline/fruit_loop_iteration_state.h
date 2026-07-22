#pragma once

#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::pipeline {

struct FruitLoopRuntimeState {
    int fruit_iter = 0;
};

struct ReductionIterationState {
    bool fruit_loops_converged = false;
    bool restarted = false;
    int start_iteration = 0;
};

inline void reset_reduction_iteration_state(ReductionIterationState &state) {
    state.fruit_loops_converged = false;
    state.restarted = false;
    state.start_iteration = 0;
}

template <class Engine>
bool fruit_loop_iteration_pending(const Engine &engine,
                                  bool fruit_loops_converged) {
    return (engine.iteration.fruit_iter <
            fruit_loops_config(engine).max_iters) &&
           !fruit_loops_converged;
}

template <class Engine>
bool fruit_loop_iteration_pending(const Engine &engine,
                                  const ReductionIterationState &state) {
    return fruit_loop_iteration_pending(engine,
                                        state.fruit_loops_converged);
}

}  // namespace citlali::pipeline
