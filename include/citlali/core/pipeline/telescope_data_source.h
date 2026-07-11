#pragma once

#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <string>

namespace citlali::pipeline {

template <class RawObs>
std::string telescope_data_filepath(const RawObs &rawobs) {
    return rawobs.teldata().filepath();
}

template <class Engine, class Logger>
void load_telescope_data_file(Engine &engine, std::string filepath,
                              const Logger &logger) {
    logger->info("getting telescope file {}", filepath);
    engine.telescope.get_tel_data(
        filepath, timestream_config(engine).chunking);
}

}  // namespace citlali::pipeline
