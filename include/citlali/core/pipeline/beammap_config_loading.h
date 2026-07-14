#pragma once

#include <citlali/core/config/config_value.h>
#include <citlali/core/config/beammap_config.h>
#include <citlali/core/pipeline/beammap_execution_plan.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <cstddef>
#include <algorithm>
#include <map>
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

struct BeammapConfigReadResult {
    citlali::config::BeammapConfig request;
    BeammapRequestPresence presence;
};

template <class Config>
BeammapRequestPresence read_beammap_request_presence(Config &config) {
    BeammapRequestPresence presence;
    presence.max_d2_iter0 = config.template has_typed<double>(
        std::tuple{"beammap", "priors", "max_d2_iter0"});
    presence.max_d2_after_iter0 = config.template has_typed<double>(
        std::tuple{"beammap", "priors", "max_d2_after_iter0"});
    presence.score_lambda_iter0 = config.template has_typed<double>(
        std::tuple{"beammap", "priors", "score_lambda_iter0"});
    presence.score_lambda_after_iter0 = config.template has_typed<double>(
        std::tuple{"beammap", "priors", "score_lambda_after_iter0"});
    presence.split_flag_values =
        config.template has_typed<std::vector<int>>(
            std::tuple{"beammap", "split_fits_by_flag", "flag_values"});
    return presence;
}

template <class Config, class Diagnostics>
BeammapConfigReadResult read_beammap_request_config(
    Config &config, Diagnostics &diagnostics, std::size_t n_arrays) {
    BeammapConfigReadResult result;
    result.presence = read_beammap_request_presence(config);
    const auto core = read_beammap_core_config(config, diagnostics);
    const auto fitting = read_beammap_fitting_config(config, diagnostics);
    const auto scan_band =
        read_beammap_scan_band_mask_config(config, diagnostics);
    const auto split = read_beammap_split_fits_config(config, diagnostics);
    const auto priors = read_beammap_priors_config(config, diagnostics);
    const auto flagging =
        read_beammap_flagging_config(config, diagnostics, n_arrays);
    const auto sensitivity = read_beammap_sensitivity_config(
        config, diagnostics.invalid_key_paths());
    const auto detector_tod =
        read_beammap_detector_tod_output_config(config, diagnostics);
    apply_beammap_typed_config(
        result.request, core, fitting, scan_band, split, priors,
        detector_tod, flagging, sensitivity);
    return result;
}

}  // namespace citlali::pipeline
