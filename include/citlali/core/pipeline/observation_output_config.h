#pragma once

#include <citlali/core/pipeline/obsnum_format.h>
#include <citlali/core/pipeline/output_policy.h>

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
bool should_record_coadd_output_obsnum(const Engine &engine) {
    return coadd_outputs_enabled(engine);
}

template <class Engine>
void record_coadd_output_obsnum(Engine &engine) {
    engine.cmb.obsnums.push_back(engine.obsnum);
}

template <class Engine>
void configure_observation_output_layout(Engine &engine, int obsnum) {
    set_observation_output_obsnum(engine, obsnum);
    set_observation_output_dir_name(engine);

    set_observation_map_output_obsnum(engine);

    if (should_record_coadd_output_obsnum(engine)) {
        record_coadd_output_obsnum(engine);
    }
}

}  // namespace citlali::pipeline
