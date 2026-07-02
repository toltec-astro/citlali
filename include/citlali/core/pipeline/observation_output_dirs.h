#pragma once

#include <filesystem>
#include <string>

namespace citlali::pipeline {

template <class Engine>
bool should_create_filtered_observation_output_dir(const Engine &engine) {
    return !engine.run_coadd && engine.run_map_filter;
}

template <class Engine>
bool should_create_observation_logs_dir(const Engine &engine) {
    return engine.verbose_mode;
}

inline void create_output_dir(const std::string &path) {
    namespace fs = std::filesystem;
    fs::create_directories(path);
}

template <class Engine, class Logger>
void create_observation_output_dirs(const Engine &engine,
                                    const Logger &logger) {
    namespace fs = std::filesystem;

    logger->debug("creating obsnum directory");
    create_output_dir(engine.obsnum_dir_name);

    logger->debug("creating obsnum raw directory");
    fs::create_directories(engine.obsnum_dir_name + "raw/");

    if (should_create_filtered_observation_output_dir(engine)) {
        logger->debug("creating obsnum filtered directory");
        fs::create_directories(engine.obsnum_dir_name + "filtered/");
    }
    if (should_create_observation_logs_dir(engine)) {
        logger->debug("creating obsnum logs directory");
        fs::create_directories(engine.obsnum_dir_name + "logs/");
    }
}

}  // namespace citlali::pipeline
