#pragma once

namespace citlali::pipeline {

template <class Engine>
bool should_write_filtered_outputs(const Engine &engine) {
    return engine.run_map_filter;
}

template <class Engine>
bool should_write_iteration_coadd_outputs(const Engine &engine) {
    return engine.run_coadd;
}

}  // namespace citlali::pipeline
