#pragma once

namespace citlali::pipeline {

struct FruitLoopRuntimeState {
    int fruit_iter = 0;
};

struct ReductionIterationState {
    bool fruit_loops_converged = false;
};

inline void reset_reduction_iteration_state(ReductionIterationState &state) {
    state.fruit_loops_converged = false;
}

template <class Engine>
bool fruit_loop_iteration_pending(const Engine &engine,
                                  bool fruit_loops_converged) {
    return (engine.iteration.fruit_iter < engine.ptcproc.fruit_loops_iters) &&
           !fruit_loops_converged;
}

template <class Engine>
bool fruit_loop_iteration_pending(const Engine &engine,
                                  const ReductionIterationState &state) {
    return fruit_loop_iteration_pending(engine,
                                        state.fruit_loops_converged);
}

}  // namespace citlali::pipeline
