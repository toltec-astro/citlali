#pragma once

namespace citlali::pipeline {

template <class Engine>
auto observation_start_time(const Engine &engine) {
    return engine.telescope.tel_data["TelTime"](0);
}

template <class Engine>
auto observation_stop_time(const Engine &engine) {
    return engine.telescope.tel_data["TelTime"](
        engine.telescope.tel_data["TelTime"].size() - 1);
}

template <class Engine>
auto calculate_observation_exposure_time(const Engine &engine) {
    return observation_stop_time(engine) - observation_start_time(engine);
}

template <class Engine>
bool should_accumulate_coadd_exposure_time(const Engine &engine) {
    return engine.run_coadd;
}

template <class Engine>
void accumulate_coadd_exposure_time(Engine &engine) {
    engine.cmb.exposure_time =
        engine.cmb.exposure_time + engine.omb.exposure_time;
}

template <class Engine>
void update_observation_exposure_time(Engine &engine) {
    engine.omb.exposure_time = calculate_observation_exposure_time(engine);
    if (should_accumulate_coadd_exposure_time(engine)) {
        accumulate_coadd_exposure_time(engine);
    }
}

template <class Engine>
void update_reduction_observation_exposure_time(Engine &engine) {
    update_observation_exposure_time(engine);
}

}  // namespace citlali::pipeline
