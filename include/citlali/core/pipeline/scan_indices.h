#pragma once

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_scan_context.h>

#include <type_traits>
#include <utility>

namespace citlali::pipeline {

template <class Engine, class = void>
struct has_sci_align_gap_plan_state : std::false_type {};

template <class Engine>
struct has_sci_align_gap_plan_state<
    Engine,
    std::void_t<
        decltype(std::declval<Engine &>().alignment),
        decltype(std::declval<Engine &>().telescope.scan_plan),
        decltype(std::declval<Engine &>()
                     .rtcproc.filter_edge_guard.context_samples)>>
    : std::true_type {};

template <class Logger>
void log_scan_index_calculation(const Logger &logger) {
    logger->info("calculating scan indices");
}

template <class Engine>
void calculate_telescope_scan_indices(Engine &engine) {
    if constexpr (has_sci_align_gap_plan_state<Engine>::value) {
        const sci_align::HalfOpenInterval governing_support{
            governing_consumer_local_start(engine.alignment),
            governing_consumer_local_stop(engine.alignment)};
        engine.telescope.calc_scan_indices(
            timestream_config(engine).chunking, governing_support);
        finalize_alignment_gap_processing_plan(
            engine.alignment, engine.telescope.scan_plan,
            engine.rtcproc.filter_edge_guard.context_samples,
            timestream_config(engine).type);
    }
    else {
        engine.telescope.calc_scan_indices(
            timestream_config(engine).chunking);
    }
}

template <class Engine, class Logger>
void calculate_scan_indices(Engine &engine, const Logger &logger) {
    log_scan_index_calculation(logger);
    calculate_telescope_scan_indices(engine);
}

template <class Engine, class Logger>
void calculate_scan_indices_if_needed(Engine &engine, bool should_calculate,
                                      const Logger &logger) {
    if (!should_calculate) {
        return;
    }

    calculate_scan_indices(engine, logger);
}

template <class Engine, class Logger>
void calculate_reduction_observation_scan_indices_if_needed(
    Engine &engine, bool should_calculate, const Logger &logger) {
    calculate_scan_indices_if_needed(engine, should_calculate, logger);
}

}  // namespace citlali::pipeline
