#pragma once

#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <tuple>

namespace citlali::pipeline {

template <class Config, class KeyList, class PostProcessingConfig>
void read_post_processing_activation_config(
    Config &config, bool &run_map_filter, bool &run_source_finder,
    PostProcessingConfig &legacy_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    read_config_value_if_clean(
        config, std::tuple{"post_processing", "map_filtering", "enabled"},
        run_map_filter,
        [&legacy_post_processing_config](bool enabled) {
            citlali::config::set_map_filtering_enabled(
                legacy_post_processing_config, enabled);
        },
        missing_keys, invalid_keys);

    read_config_value_if_clean(
        config, std::tuple{"post_processing", "source_finding", "enabled"},
        run_source_finder,
        [&legacy_post_processing_config](bool enabled) {
            citlali::config::set_source_finding_enabled(
                legacy_post_processing_config, enabled);
        },
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class PostProcessingConfig>
void read_post_processing_activation_config(
    Config &config, bool &run_map_filter, bool &run_source_finder,
    PostProcessingConfig &legacy_post_processing_config,
    Diagnostics &diagnostics) {
    read_post_processing_activation_config(
        config, run_map_filter, run_source_finder,
        legacy_post_processing_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

}  // namespace citlali::pipeline
