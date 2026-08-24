#pragma once

#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <string>

namespace citlali::pipeline {

template <class Telescope>
void reset_telescope_observation_input_state(Telescope &telescope) {
    // Telescope is a reduction-lifetime compatibility object, while these
    // containers are observation-owned. In particular, calc_tan_pointing()
    // adds derived *_phys series that must not enter the next observation's
    // raw telescope trajectory, and an optional field absent from a later
    // file must not inherit the earlier observation's value.
    telescope.tel_data.clear();
    telescope.tel_header.clear();
}

template <class RawObs>
std::string telescope_data_filepath(const RawObs &rawobs) {
    return rawobs.teldata().filepath();
}

template <class Engine, class Logger>
void load_telescope_data_file(Engine &engine, std::string filepath,
                              const Logger &logger) {
    logger->info("getting telescope file {}", filepath);
    reset_telescope_observation_input_state(engine.telescope);
    engine.telescope.get_tel_data(
        filepath, timestream_config(engine).chunking);
}

}  // namespace citlali::pipeline
