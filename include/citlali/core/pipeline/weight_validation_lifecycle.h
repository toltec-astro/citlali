#pragma once

namespace citlali::pipeline {

template <class Engine>
void begin_iteration_weight_validation(Engine &engine) {
    engine.ptcproc.begin_weight_validation_iteration(engine.fruit_iter);
}

template <class Engine>
void finalize_iteration_weight_validation(Engine &engine) {
    engine.ptcproc.finalize_weight_validation_iteration(engine.fruit_iter);
}

}  // namespace citlali::pipeline
