#pragma once

#include <citlali/core/pipeline/obsnum_format.h>

namespace citlali::pipeline {

template <class Engine>
void set_observation_output_obsnum(Engine &engine, int obsnum) {
    engine.obsnum = format_obsnum(obsnum);
}

template <class Engine>
void set_observation_output_dir_name(Engine &engine) {
    engine.obsnum_dir_name = engine.redu_dir_name + "/" + engine.obsnum + "/";
}

template <class Engine>
void set_observation_map_output_obsnum(Engine &engine) {
    engine.omb.obsnums.clear();
    engine.omb.obsnums.push_back(engine.obsnum);
}

template <class Engine>
void configure_observation_output_layout(Engine &engine, int obsnum) {
    set_observation_output_obsnum(engine, obsnum);
    set_observation_output_dir_name(engine);

    set_observation_map_output_obsnum(engine);

    if (engine.run_coadd) {
        engine.cmb.obsnums.push_back(engine.obsnum);
    }
}

}  // namespace citlali::pipeline
