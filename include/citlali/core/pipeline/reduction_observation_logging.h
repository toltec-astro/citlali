#pragma once

#include <citlali/core/utils/process_resource_snapshot.h>

#include <cstddef>
#include <string>

namespace citlali::pipeline {

template <class Logger>
void log_reduction_observation_start(std::size_t observation_index,
                                     std::size_t observation_count,
                                     const Logger &logger) {
    logger->info("starting reduction of observation {}/{}",
                 observation_index + 1, observation_count);
    citlali::utils::log_process_resource_snapshot(
        logger, "observation " + std::to_string(observation_index + 1) +
                    " start");
}

}  // namespace citlali::pipeline
