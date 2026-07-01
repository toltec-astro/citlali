#pragma once

#include <citlali/core/pipeline/kids_metadata.h>
#include <citlali/core/pipeline/observation_calibration.h>
#include <citlali/core/pipeline/observation_input_checks.h>
#include <citlali/core/pipeline/observation_sample_rate.h>
#include <citlali/core/pipeline/observation_telescope.h>
#include <citlali/core/pipeline/observation_timing.h>
#include <citlali/core/pipeline/rawobs_data_items.h>
#include <citlali/core/pipeline/reduction_config.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class TodProc, class RawObs, class Logger>
void load_raw_detector_diagnostics(TodProc &todproc, const RawObs &rawobs,
                                   const Logger &logger) {
    logger->debug("getting tone frequencies");
    todproc.get_tone_freqs_from_files(rawobs);

    if (!todproc.engine().telescope.sim_obs) {
        logger->debug("getting adc snap data");
        todproc.get_adc_snap_from_files(rawobs);
    }
}

template <class Engine, class Logger>
void configure_fruit_loop_iteration_policy(Engine &engine,
                                           const Logger &logger) {
    if (engine.ptcproc.run_fruit_loops && !engine.run_noise) {
        logger->warn("noise maps are not enabled for fruit loops");
    }

    if (!engine.ptcproc.run_fruit_loops || engine.redu_type == "beammap") {
        engine.ptcproc.fruit_loops_iters = 1;
        engine.ptcproc.save_all_iters = true;
    }
}

}  // namespace citlali::pipeline
