#pragma once

#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool should_allocate_coadd_noise_buffer(const Engine &engine) {
    return noise_maps_enabled(engine);
}

template <class Engine>
bool should_prepare_coadd_iteration_buffers(const Engine &engine) {
    return coadd_outputs_enabled(engine);
}

}  // namespace citlali::pipeline
