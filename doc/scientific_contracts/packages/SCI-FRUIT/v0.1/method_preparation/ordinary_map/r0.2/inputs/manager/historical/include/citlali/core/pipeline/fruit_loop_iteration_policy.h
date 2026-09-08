#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <string>

namespace citlali::pipeline {

struct FruitLoopIterationResolution {
    int effective_max_iters = 1;
    bool effective_save_all_iters = false;
    bool forced_single_iteration_while_disabled = false;
    bool forced_single_iteration_for_beammap = false;
};

struct FruitLoopInterpolationResolution {
    citlali::config::FruitLoopsInterpModeOverride requested =
        citlali::config::FruitLoopsInterpModeOverride::automatic;
    citlali::config::FruitLoopsInterpModeOverride mapmaking_default =
        citlali::config::FruitLoopsInterpModeOverride::bilinear;
    citlali::config::FruitLoopsInterpModeOverride effective =
        citlali::config::FruitLoopsInterpModeOverride::bilinear;
    bool override_applied = false;
    bool jinc_fell_back_to_bilinear = false;
};

inline FruitLoopInterpolationResolution resolve_fruit_loop_interpolation(
    const citlali::config::TimestreamFruitLoopsConfig &requested,
    citlali::config::MapMethod map_method) {
    const auto mapmaking_default = citlali::config::is_jinc_map_method(
                                       map_method)
        ? citlali::config::FruitLoopsInterpModeOverride::jinc
        : citlali::config::FruitLoopsInterpModeOverride::bilinear;
    FruitLoopInterpolationResolution resolution{
        requested.interp_mode_override,
        mapmaking_default,
        mapmaking_default,
    };
    if (requested.enabled &&
        !citlali::config::is_fruit_loops_auto_interp_mode(
            requested.interp_mode_override)) {
        resolution.effective = requested.interp_mode_override;
        resolution.override_applied = true;
    }
    if (citlali::config::is_fruit_loops_jinc_interp_mode(
            resolution.effective) &&
        !citlali::config::is_jinc_map_method(map_method)) {
        resolution.effective =
            citlali::config::FruitLoopsInterpModeOverride::bilinear;
        resolution.jinc_fell_back_to_bilinear = true;
    }
    return resolution;
}

inline FruitLoopIterationResolution resolve_fruit_loop_iteration_policy(
    const citlali::config::TimestreamFruitLoopsConfig &requested,
    citlali::config::ReductionType reduction_type) {
    FruitLoopIterationResolution resolution{
        requested.max_iters,
        requested.save_all_iters,
        !requested.enabled,
        citlali::config::is_beammap_reduction_type(reduction_type),
    };
    if (resolution.forced_single_iteration_while_disabled ||
        resolution.forced_single_iteration_for_beammap) {
        resolution.effective_max_iters = 1;
        resolution.effective_save_all_iters = true;
    }
    return resolution;
}

template <class Engine, class Logger>
void configure_fruit_loop_interpolation_mode(
    Engine &engine, citlali::config::MapMethod map_method,
    const Logger &logger) {
    const auto &config = fruit_loops_config(engine);
    auto &ptcproc = engine.ptcproc;
    const std::string map_method_name{citlali::config::to_string(map_method)};
    const auto resolution = resolve_fruit_loop_interpolation(
        config, map_method);
    auto &plan = processed_timestream_plan(engine);
    if (plan.initialized) {
        plan.effective_resolutions.fruit_loop_interpolation = resolution;
    }
    const std::string fruit_interp_default{
        citlali::config::to_string(resolution.mapmaking_default)};
    ptcproc.fruit_loops_interp_mode =
        std::string{citlali::config::to_string(resolution.effective)};
    if (resolution.jinc_fell_back_to_bilinear) {
        logger->warn(
            "fruit_loops.interp_mode_override='jinc' requires mapmaking.method='jinc'; using bilinear");
    }
    logger->info(
        "fruit loops interpolation mode: {} (default from mapmaking.method='{}' is {})",
        ptcproc.fruit_loops_interp_mode, map_method_name,
        fruit_interp_default);
}

template <class Engine, class Logger>
void log_fruit_loop_runtime_policy(const Engine &engine,
                                   const Logger &logger) {
    const auto &config = fruit_loops_config(engine);
    logger->info("fruit loops center convention: {}",
                 config.legacy_center
                     ? "legacy n/2"
                     : "current (n-1)/2");
    logger->info("fruit loops post-addback weight mode: {}",
                 config.recompute_weights_after_addback
                     ? "recompute from add-back TOD"
                     : "keep source-subtracted");
    logger->info(
        "fruit loops weight feedback: enabled={} reference={} relative=[{}, {}]",
        config.weight_feedback.enabled,
        citlali::config::to_string(config.weight_feedback.reference),
        config.weight_feedback.low_relative_weight,
        config.weight_feedback.high_relative_weight);
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
    (void) logger;
    auto &config = fruit_loops_config(engine);

    const auto resolution = resolve_fruit_loop_iteration_policy(
        config, runtime_reduction_type(engine));
    auto &plan = processed_timestream_plan(engine);
    if (plan.initialized) {
        plan.effective_resolutions.fruit_loop_iterations = resolution;
    }
    config.max_iters = resolution.effective_max_iters;
    config.save_all_iters = resolution.effective_save_all_iters;
    engine.ptcproc.fruit_loops_iters = config.max_iters;
    engine.ptcproc.save_all_iters = config.save_all_iters;
}

}  // namespace citlali::pipeline
