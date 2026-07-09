#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/map_filter_config_policy.h>

template<typename CT>
void Engine::get_map_filter_config(CT &config) {
    logger->info("getting map filtering config options");
    auto &post_processing_config = typed_config.post_processing;
    // get wiener filter config options
    wiener_filter.get_config(config, missing_keys, invalid_keys);

    citlali::engine_detail::mirror_wiener_filter_config(
        wiener_filter, RAD_TO_ASEC, post_processing_config);
    citlali::engine_detail::apply_map_filter_runtime_policy(
        typed_config, rtcproc, map_fitter, parallel_policy, wiener_filter,
        logger);
}
