#pragma once

namespace citlali::pipeline {

template <auto MapType, class Engine, class Logger>
void output_map_with_log(Engine &engine, const Logger &logger,
                         const char *log_message) {
    logger->info("{}", log_message);
    engine.template output<MapType>();
}

}  // namespace citlali::pipeline
