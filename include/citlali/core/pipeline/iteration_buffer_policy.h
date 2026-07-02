#pragma once

namespace citlali::pipeline {

template <class Engine>
bool should_allocate_coadd_noise_buffer(const Engine &engine) {
    return engine.run_noise;
}

template <class Engine>
bool should_prepare_coadd_iteration_buffers(const Engine &engine) {
    return engine.run_coadd;
}

}  // namespace citlali::pipeline
