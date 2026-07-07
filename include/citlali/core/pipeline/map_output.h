#pragma once

#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <auto MapType, class Engine, class Logger>
void output_map_with_log(Engine &engine, const Logger &logger,
                         const char *log_message) {
    logger->info("{}", log_message);
    const auto profile_scope =
        profile_stage("map.output", logger, log_message);
    engine.template output<MapType>();
}

template <auto MapType, class Engine, class Logger>
void output_map_if_needed(Engine &engine, const Logger &logger,
                          bool should_output,
                          const char *output_log_message,
                          const char *skip_log_message) {
    if (should_output) {
        output_map_with_log<MapType>(
            engine, logger, output_log_message);
    }
    else {
        logger->info("{}", skip_log_message);
    }
}

}  // namespace citlali::pipeline
