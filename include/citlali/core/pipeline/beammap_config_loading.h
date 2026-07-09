#pragma once

#include <citlali/core/config/config_value.h>
#include <citlali/core/config/beammap_config.h>

#include <cstddef>
#include <algorithm>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>


namespace citlali::pipeline {

template <class Config, class Value, class MissingKeys, class InvalidKeys,
          class Key>
void read_optional_beammap_config_value(
    Config &config, Value &value, MissingKeys &missing_keys,
    InvalidKeys &invalid_keys, const Key &key,
    std::vector<std::decay_t<Value>> accepted_values = {},
    std::vector<std::decay_t<Value>> min_values = {},
    std::vector<std::decay_t<Value>> max_values = {}) {
    using value_type = std::decay_t<Value>;
    if (!config.template has_typed<value_type>(key)) {
        return;
    }
    ::get_config_value(config, value, missing_keys, invalid_keys, key,
                       std::move(accepted_values), std::move(min_values),
                       std::move(max_values));
}

#include <citlali/core/pipeline/beammap_config_core_loading.h>
#include <citlali/core/pipeline/beammap_config_fitting_flagging.h>
#include <citlali/core/pipeline/beammap_config_split_outputs.h>
#include <citlali/core/pipeline/beammap_config_priors_loading.h>
#include <citlali/core/pipeline/beammap_config_tod_mirror.h>

}  // namespace citlali::pipeline
