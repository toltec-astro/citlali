#pragma once

namespace citlali::pipeline {

template <auto MapType, class Engine, class MapBuffer, class Logger>
void find_map_sources_with_log(Engine &engine, MapBuffer &map_buffer,
                               const Logger &logger,
                               const char *log_message) {
    logger->info("{}", log_message);
    engine.template find_sources<MapType>(map_buffer);
}

}  // namespace citlali::pipeline
