#pragma once

#include <citlali/core/pipeline/timestream_alignment_state.h>

namespace citlali::pipeline {

template <class Engine, class RawObs>
void reset_simulated_observation_indices(Engine &engine,
                                         const RawObs &rawobs) {
    clear_alignment_windows(engine.alignment);

    for (const auto &data_item : rawobs.kidsdata()) {
        (void)data_item;
        engine.alignment.start_indices.push_back(0);
        engine.alignment.start_indices.push_back(0);
    }

    if (engine.calib.run_hwpr) {
        engine.alignment.hwpr_start_index = 0;
        engine.alignment.hwpr_end_index = 0;
    }
}

}  // namespace citlali::pipeline
