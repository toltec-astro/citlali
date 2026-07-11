#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <string>

namespace citlali::pipeline {

template <class PtcProc, class Logger>
void configure_fruit_loop_interpolation_mode(
    PtcProc &ptcproc, citlali::config::MapMethod map_method,
    const Logger &logger) {
    const std::string map_method_name{citlali::config::to_string(map_method)};
    const std::string fruit_interp_default{
        citlali::config::to_string(
            citlali::config::is_jinc_map_method(map_method)
                ? citlali::config::FruitLoopsInterpModeOverride::jinc
                : citlali::config::FruitLoopsInterpModeOverride::bilinear)};
    ptcproc.fruit_loops_interp_mode = fruit_interp_default;
    if (ptcproc.run_fruit_loops &&
        !citlali::config::is_fruit_loops_auto_interp_mode(
            ptcproc.fruit_loops_interp_mode_override)) {
        ptcproc.fruit_loops_interp_mode =
            ptcproc.fruit_loops_interp_mode_override;
    }
    if (citlali::config::is_fruit_loops_jinc_interp_mode(
            ptcproc.fruit_loops_interp_mode) &&
        !citlali::config::is_jinc_map_method(map_method)) {
        logger->warn(
            "fruit_loops.interp_mode_override='jinc' requires mapmaking.method='jinc'; using bilinear");
        ptcproc.fruit_loops_interp_mode = std::string{
            citlali::config::to_string(
                citlali::config::FruitLoopsInterpModeOverride::bilinear)};
    }
    logger->info(
        "fruit loops interpolation mode: {} (default from mapmaking.method='{}' is {})",
        ptcproc.fruit_loops_interp_mode, map_method_name,
        fruit_interp_default);
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
    auto &config = fruit_loops_config(engine);
    if (config.enabled && !noise_maps_enabled(engine)) {
        logger->warn("noise maps are not enabled for fruit loops");
    }

    if (!config.enabled ||
        runtime_reduction_type(engine) ==
            citlali::config::ReductionType::beammap) {
        config.max_iters = 1;
        config.save_all_iters = true;
    }
    engine.ptcproc.fruit_loops_iters = config.max_iters;
    engine.ptcproc.save_all_iters = config.save_all_iters;
}

}  // namespace citlali::pipeline
