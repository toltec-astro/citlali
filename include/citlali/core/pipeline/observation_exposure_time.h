#pragma once

#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
auto observation_start_time(Engine &engine) {
    return engine.telescope.tel_data["TelTime"](0);
}

template <class Engine>
auto observation_stop_time(Engine &engine) {
    return engine.telescope.tel_data["TelTime"](
        engine.telescope.tel_data["TelTime"].size() - 1);
}

template <class Engine>
auto calculate_observation_exposure_time(Engine &engine) {
    return observation_stop_time(engine) - observation_start_time(engine);
}

template <class Engine>
void update_observation_exposure_time(Engine &engine) {
    engine.omb.exposure_time = calculate_observation_exposure_time(engine);
    if (coadd_outputs_enabled(engine) &&
        !science_map_v1_profile_available(engine)) {
        // Preserve the legacy direct/polarized coadd lifecycle. Supported v1
        // profiles commit exposure atomically with the observation bundle.
        engine.cmb.exposure_time += engine.omb.exposure_time;
    }
}

template <class Engine>
void update_reduction_observation_exposure_time(Engine &engine) {
    update_observation_exposure_time(engine);
}

}  // namespace citlali::pipeline
