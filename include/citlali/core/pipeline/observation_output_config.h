#pragma once

#include <citlali/core/pipeline/obsnum_format.h>

namespace citlali::pipeline {

template <class Engine>
void configure_observation_output_layout(Engine &engine, int obsnum) {
    engine.obsnum = format_obsnum(obsnum);
    engine.obsnum_dir_name = engine.redu_dir_name + "/" + engine.obsnum + "/";

    engine.omb.obsnums.clear();
    engine.omb.obsnums.push_back(engine.obsnum);

    if (engine.run_coadd) {
        engine.cmb.obsnums.push_back(engine.obsnum);
    }
}

}  // namespace citlali::pipeline
