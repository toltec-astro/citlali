#pragma once

#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <string>

#include <citlali/core/pipeline/map_grouping_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool unit_sum_convolved_fruit_loop_feedback_requested(
    const Engine &engine) {
    const auto &fruit_loops = fruit_loops_config(engine);
    const auto &post_processing = effective_post_processing_config(engine);
    return fruit_loops.enabled &&
           citlali::config::is_filtered_fruit_loops_type(
               fruit_loops.type) &&
           citlali::config::map_filtering_active(post_processing) &&
           citlali::config::map_filter_uses_unit_sum_convolution(
               post_processing.map_filtering);
}

template <class Engine>
void require_fruit_loop_feedback_product_contract(
    const Engine &engine) {
    if (unit_sum_convolved_fruit_loop_feedback_requested(engine)) {
        throw citlali::error::invalid_config(
            "unit-sum convolved maps are withheld from fruit-loop feedback "
            "until their support and response contract passes production "
            "validation; use obsnum/raw or coadd/raw feedback");
    }
}

template <class Engine>
void load_fruit_loop_maps(Engine &engine, const std::string &fruit_dir) {
    require_fruit_loop_feedback_product_contract(engine);
    const bool require_filtered_feedback_provenance =
        citlali::config::is_filtered_fruit_loops_type(
            fruit_loops_config(engine).type);
    engine.ptcproc.tod_mb.cov_cut = engine.omb.cov_cut;
    engine.ptcproc.load_mb(fruit_dir, fruit_dir, engine.calib,
                           active_map_grouping_name(engine),
                           engine.telescope.pixel_axes,
                           engine.omb.pixel_size_rad,
                           require_filtered_feedback_provenance);
}

}  // namespace citlali::pipeline
