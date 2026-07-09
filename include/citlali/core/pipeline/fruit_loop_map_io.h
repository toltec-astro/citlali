#pragma once

#include <string>

#include <citlali/core/pipeline/map_grouping_policy.h>

namespace citlali::pipeline {

template <class Engine>
void load_fruit_loop_maps(Engine &engine, const std::string &fruit_dir) {
    engine.ptcproc.tod_mb.cov_cut = engine.omb.cov_cut;
    engine.ptcproc.load_mb(fruit_dir, fruit_dir, engine.calib,
                           active_map_grouping_name(engine),
                           engine.telescope.pixel_axes,
                           engine.omb.pixel_size_rad);
}

}  // namespace citlali::pipeline
