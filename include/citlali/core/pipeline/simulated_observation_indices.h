#pragma once

namespace citlali::pipeline {

template <class Engine, class RawObs>
void reset_simulated_observation_indices(Engine &engine,
                                         const RawObs &rawobs) {
    engine.start_indices.clear();
    engine.end_indices.clear();

    for (const auto &data_item : rawobs.kidsdata()) {
        (void)data_item;
        engine.start_indices.push_back(0);
        engine.start_indices.push_back(0);
    }

    if (engine.calib.run_hwpr) {
        engine.hwpr_start_indices = 0;
        engine.hwpr_end_indices = 0;
    }
}

}  // namespace citlali::pipeline
