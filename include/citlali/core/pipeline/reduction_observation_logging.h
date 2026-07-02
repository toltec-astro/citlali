#pragma once

#include <cstddef>

namespace citlali::pipeline {

template <class Logger>
void log_reduction_observation_start(std::size_t observation_index,
                                     std::size_t observation_count,
                                     const Logger &logger) {
    logger->info("starting reduction of observation {}/{}",
                 observation_index + 1, observation_count);
}

}  // namespace citlali::pipeline
