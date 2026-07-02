#pragma once

namespace citlali::pipeline {

template <class Engine>
void reset_coadd_iteration_accumulators(Engine &engine) {
    engine.cmb.obsnums.clear();
    engine.cmb.exposure_time = 0;
}

template <class Engine>
void clear_iteration_observation_dates(Engine &engine) {
    engine.date_obs.clear();
}

}  // namespace citlali::pipeline
