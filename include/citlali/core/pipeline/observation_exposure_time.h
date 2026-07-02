#pragma once

namespace citlali::pipeline {

template <class Engine>
void update_observation_exposure_time(Engine &engine) {
    auto t0 = engine.telescope.tel_data["TelTime"](0);
    auto tn = engine.telescope.tel_data["TelTime"](
        engine.telescope.tel_data["TelTime"].size() - 1);

    engine.omb.exposure_time = tn - t0;
    if (engine.run_coadd) {
        engine.cmb.exposure_time =
            engine.cmb.exposure_time + engine.omb.exposure_time;
    }
}

template <class Engine>
void update_reduction_observation_exposure_time(Engine &engine) {
    update_observation_exposure_time(engine);
}

}  // namespace citlali::pipeline
