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

template <class PtcProc, class Logger>
void log_fruit_loop_runtime_policy(const PtcProc &ptcproc,
                                   const Logger &logger) {
    logger->info("fruit loops center convention: {}",
                 ptcproc.fruit_loops_legacy_center
                     ? "legacy n/2"
                     : "current (n-1)/2");
    logger->info("fruit loops post-addback weight mode: {}",
                 ptcproc.fruit_loops_recompute_weights_after_addback
                     ? "recompute from add-back TOD"
                     : "keep source-subtracted");
    logger->info(
        "fruit loops weight feedback: enabled={} reference={} relative=[{}, {}]",
        ptcproc.fruit_loops_weight_feedback_enabled,
        ptcproc.fruit_loops_weight_feedback_reference,
        ptcproc.fruit_loops_weight_feedback_low_relative_weight,
        ptcproc.fruit_loops_weight_feedback_high_relative_weight);
}

template <class PtcProc>
void reset_fruit_loop_jinc_kernel_config(PtcProc &ptcproc) {
    ptcproc.fruit_loops_jinc_r_max = 0.0;
    ptcproc.fruit_loops_jinc_subpixel_n = 1;
    ptcproc.fruit_loops_jinc_shape_params.clear();
}

template <class JincMapmaker, class PtcProc>
void mirror_jinc_mapmaker_config_to_fruit_loops(const JincMapmaker &jinc_mm,
                                                PtcProc &ptcproc) {
    ptcproc.fruit_loops_jinc_r_max = jinc_mm.r_max;
    ptcproc.fruit_loops_jinc_subpixel_n = jinc_mm.subpixel_n;
    ptcproc.fruit_loops_jinc_shape_params = jinc_mm.shape_params;
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
