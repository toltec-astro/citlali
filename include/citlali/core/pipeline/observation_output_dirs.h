#pragma once

#include <filesystem>

namespace citlali::pipeline {

template <class Engine>
bool should_create_filtered_observation_output_dir(const Engine &engine) {
    return !engine.run_coadd && engine.run_map_filter;
}

template <class Engine, class Logger>
void create_observation_output_dirs(const Engine &engine,
                                    const Logger &logger) {
    namespace fs = std::filesystem;

    logger->debug("creating obsnum directory");
    fs::create_directories(engine.obsnum_dir_name);

    logger->debug("creating obsnum raw directory");
    fs::create_directories(engine.obsnum_dir_name + "raw/");

    if (!engine.run_coadd) {
        if (engine.run_map_filter) {
            logger->debug("creating obsnum filtered directory");
            fs::create_directories(engine.obsnum_dir_name + "filtered/");
        }
    }
    if (engine.verbose_mode) {
        logger->debug("creating obsnum logs directory");
        fs::create_directories(engine.obsnum_dir_name + "logs/");
    }
}

}  // namespace citlali::pipeline
