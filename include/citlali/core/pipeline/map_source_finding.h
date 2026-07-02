#pragma once

namespace citlali::pipeline {

template <auto MapType, class Engine, class MapBuffer, class Logger>
void find_map_sources_with_log(Engine &engine, MapBuffer &map_buffer,
                               const Logger &logger,
                               const char *log_message) {
    logger->info("{}", log_message);
    engine.template find_sources<MapType>(map_buffer);
}

template <auto MapType, class Engine, class MapBuffer, class Logger>
void find_map_sources_if_needed(Engine &engine, MapBuffer &map_buffer,
                                const Logger &logger, bool should_find,
                                const char *log_message) {
    if (should_find) {
        find_map_sources_with_log<MapType>(
            engine, map_buffer, logger, log_message);
    }
}

}  // namespace citlali::pipeline
