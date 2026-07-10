#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/engine/detail/map_filter_config_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_map_filter_config(CT &config) {
    logger->info("getting map filtering config options");
    auto &post_processing_config =
        citlali::pipeline::post_processing_config(*this);
    const auto &reduction_config = citlali::pipeline::reduction_config(*this);
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    // get wiener filter config options
    citlali::engine_detail::read_processor_config(
        wiener_filter, config, config_diag);

    citlali::engine_detail::mirror_wiener_filter_config(
        wiener_filter, RAD_TO_ASEC, post_processing_config);
    citlali::engine_detail::apply_map_filter_runtime_policy(
        reduction_config, rtcproc, map_fitter,
        citlali::pipeline::runtime_parallel_policy_name(*this),
        wiener_filter, logger);
}
