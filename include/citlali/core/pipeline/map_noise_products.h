#pragma once

#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class MapBuffer, class Logger>
void calculate_map_noise_products_with_log(
    MapBuffer &map_buffer, bool apply_empirical_noise_weights,
    StageProfileCollector &stage_profile, const Logger &logger,
    const char *log_message) {
    (void)stage_profile;
    logger->info("{}", log_message);
    const auto profile_scope =
        profile_stage("map.noise_products", logger, log_message);
    map_buffer.calc_noise_products(apply_empirical_noise_weights);
}

template <class MapBuffer, class Logger>
void calculate_map_noise_products_if_needed(
    MapBuffer &map_buffer, bool should_calculate,
    bool apply_empirical_noise_weights, StageProfileCollector &stage_profile,
    const Logger &logger,
    const char *log_message) {
    if (should_calculate) {
        calculate_map_noise_products_with_log(
            map_buffer, apply_empirical_noise_weights, stage_profile, logger,
            log_message);
    }
}

}  // namespace citlali::pipeline
