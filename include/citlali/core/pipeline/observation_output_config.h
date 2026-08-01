#pragma once

#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/obsnum_format.h>
#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
void set_observation_output_obsnum(Engine &engine, int obsnum) {
    engine.observation_identity.obsnum = format_obsnum(obsnum);
}

template <class Engine>
void set_observation_output_dir_name(Engine &engine) {
    engine.output_paths.obsnum_dir_name = engine.output_paths.redu_dir_name +
                                          "/" +
                                          engine.observation_identity.obsnum +
                                          "/";
}

template <class Engine>
void set_observation_map_output_obsnum(Engine &engine) {
    engine.omb.obsnums.clear();
    engine.omb.obsnums.push_back(engine.observation_identity.obsnum);
}

template <class Engine>
void configure_observation_output_layout(Engine &engine, int obsnum) {
    set_observation_output_obsnum(engine, obsnum);
    set_observation_output_dir_name(engine);

    set_observation_map_output_obsnum(engine);

    if (coadd_outputs_enabled(engine) &&
        !science_map_v1_profile_available(engine)) {
        // Polarized and other non-v1 profiles retain their pre-repair
        // membership path; v1 membership is part of atomic F009 admission.
        engine.cmb.obsnums.push_back(engine.observation_identity.obsnum);
    }
}

}  // namespace citlali::pipeline
