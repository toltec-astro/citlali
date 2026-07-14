#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/map_filter_config_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

inline void Engine::get_map_filter_config() {
    logger->info("getting map filtering config options");
    const auto &map_filter_config =
        citlali::pipeline::effective_post_processing_config(*this)
            .map_filtering;
    const auto &effective_noise_config =
        citlali::pipeline::noise_config(*this);

    citlali::pipeline::adapt_map_filter_config_one_way(
        map_filter_config, ASEC_TO_RAD, wiener_filter);
    citlali::pipeline::apply_map_filter_runtime_policy(
        effective_noise_config, map_filter_config,
        rtcproc, map_fitter,
        citlali::pipeline::runtime_parallel_policy_name(*this),
        wiener_filter, logger);
}
