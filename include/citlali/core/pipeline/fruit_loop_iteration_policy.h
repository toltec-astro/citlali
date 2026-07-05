#pragma once

#include <string>

namespace citlali::pipeline {

template <class PtcProc, class Logger>
void configure_fruit_loop_interpolation_mode(
    PtcProc &ptcproc, const std::string &map_method, const Logger &logger) {
    const std::string fruit_interp_default =
        (map_method == "jinc") ? "jinc" : "bilinear";
    ptcproc.fruit_loops_interp_mode = fruit_interp_default;
    if (ptcproc.run_fruit_loops &&
        ptcproc.fruit_loops_interp_mode_override != "auto") {
        ptcproc.fruit_loops_interp_mode =
            ptcproc.fruit_loops_interp_mode_override;
    }
    if (ptcproc.fruit_loops_interp_mode == "jinc" &&
        map_method != "jinc") {
        logger->warn(
            "fruit_loops.interp_mode_override='jinc' requires mapmaking.method='jinc'; using bilinear");
        ptcproc.fruit_loops_interp_mode = "bilinear";
    }
    logger->info(
        "fruit loops interpolation mode: {} (default from mapmaking.method='{}' is {})",
        ptcproc.fruit_loops_interp_mode, map_method, fruit_interp_default);
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
