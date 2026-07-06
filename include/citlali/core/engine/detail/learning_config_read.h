#pragma once

#include <citlali/core/engine/detail/config_parse_tracking.h>

#include <type_traits>
#include <vector>

namespace citlali::engine_detail {

template <class Config, class Param, class Target, class Key, class KeyVec>
void read_optional_learning_config(Config &config, const Key &key,
                                   Param &param, Target &target,
                                   KeyVec &missing_keys,
                                   KeyVec &invalid_keys,
                                   std::vector<std::decay_t<Param>> min_val = {},
                                   std::vector<std::decay_t<Param>> max_val = {}) {
    citlali::engine_detail::read_optional_mirrored_config_value(
        config, key, param, target, missing_keys, invalid_keys,
        {}, std::move(min_val), std::move(max_val));
}

template <class LearningOptions>
bool learning_map_contribution_diagnostics_enabled(
    const LearningOptions &options) {
    return options.enabled && options.diagnostics_enabled &&
           options.map_pixel_outlier_diagnostics_enabled &&
           options.map_pixel_outlier_contributor_diagnostics_enabled;
}

template <class OutputMapBlock, class CoaddMapBlock>
void set_learning_map_contribution_diagnostics(bool enabled,
                                               OutputMapBlock &omb,
                                               CoaddMapBlock &cmb) {
    omb.contribution_diag_enabled = enabled;
    cmb.contribution_diag_enabled = enabled;
}

}  // namespace citlali::engine_detail
