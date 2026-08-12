#pragma once

#include <citlali/core/pipeline/raw_map_outputs.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace citlali::pipeline {

template <class Engine, class = void>
struct has_tod_only_finalized_header_publication : std::false_type {};

template <class Engine>
struct has_tod_only_finalized_header_publication<
    Engine, std::void_t<decltype(
        std::declval<Engine &>().add_tod_header(
            std::declval<decltype(
                std::addressof(std::declval<Engine &>().omb)) &>()))>>
    : std::true_type {};

template <class Engine>
bool should_output_raw_observation_maps(const Engine &engine) {
    return mapmaking_outputs_enabled(engine);
}

template <class Engine, class Logger>
void calculate_raw_observation_noise_products_if_needed(
    Engine &engine, StageProfileCollector &stage_profile,
    const Logger &logger) {
    calculate_unfiltered_map_noise_products_if_needed(
        engine, engine.omb, stage_profile, logger, true,
        "calculating raw obs empirical noise products");
}

template <auto RawObsMap, class Engine, class Logger>
void output_raw_observation_maps_if_needed(Engine &engine,
                                           StageProfileCollector &stage_profile,
                                           const Logger &logger) {
    const auto should_output = should_output_raw_observation_maps(engine);
    if (should_output) {
        engine.create_obs_map_files();
    }
    output_unfiltered_maps_if_needed<RawObsMap>(
        engine, stage_profile, logger, should_output,
        "outputting raw obs files",
        "mapmaking disabled; skipping raw obs map output");
}

template <auto RawObsMap, class TodProc, class Logger>
void write_raw_observation_outputs(TodProc &todproc,
                                   StageProfileCollector &stage_profile,
                                   const Logger &logger) {
    auto &engine = todproc.engine();
    (void)stage_profile;
    const auto profile_scope =
        profile_stage(stage_profile, "raw_observation.outputs", logger);

    calculate_raw_observation_noise_products_if_needed(
        engine, stage_profile, logger);
    output_raw_observation_maps_if_needed<RawObsMap>(
        engine, stage_profile, logger);
    using EngineType = std::remove_reference_t<decltype(engine)>;
    if constexpr (
        has_tod_only_finalized_header_publication<EngineType>::value) {
        if (!should_output_raw_observation_maps(engine) &&
            tod_output_files_available(engine)) {
            // Map writers attach the finalized link when mapmaking is active.
            // TOD-only operation must publish the same finalized metadata.
            auto *observation_map = std::addressof(engine.omb);
            engine.add_tod_header(observation_map);
        }
    }
}

}  // namespace citlali::pipeline
