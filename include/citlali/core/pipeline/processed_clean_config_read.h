#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <tuple>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_processed_clean_core_config(
    Config &config, citlali::config::ProcessedTimeChunkCleanConfig &typed,
    Diagnostics &diagnostics) {
    bool enabled = typed.enabled;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean", "enabled"},
        enabled, typed.enabled, diagnostics);
    if (!typed.enabled) {
        typed.active = citlali::config::ProcessedTimeChunkCleanerMode::none;
        return;
    }

    auto grouping = typed.grouping;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean", "grouping"},
        grouping, typed.grouping, diagnostics);

    double mask_radius_arcsec = typed.mask_radius_arcsec;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "mask_radius_arcsec"},
        mask_radius_arcsec, typed.mask_radius_arcsec, diagnostics);
    double tau = typed.tau;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "clean", "tau"},
        tau, typed.tau, diagnostics);

    const auto standard_key = std::tuple{
        "timestream", "processed_time_chunk", "clean", "standard_pca",
        "enabled"};
    const bool have_standard_block =
        config.template has_typed<bool>(standard_key);
    bool standard_enabled = typed.standard_pca.enabled;
    read_optional_mirrored_config_value(
        config, standard_key, standard_enabled, typed.standard_pca.enabled,
        diagnostics);

    auto read_mode_enabled = [&](const auto &key, bool &target) {
        bool value = target;
        read_optional_mirrored_config_value(
            config, key, value, target, diagnostics);
    };
    read_mode_enabled(
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "null_model", "enabled"},
        typed.null_model.enabled);
    read_mode_enabled(
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "marchenko_pastur", "enabled"},
        typed.marchenko_pastur.enabled);
    read_mode_enabled(
        std::tuple{"timestream", "processed_time_chunk", "clean",
                   "adaptive_selector", "enabled"},
        typed.adaptive_selector.enabled);
    if (!have_standard_block) {
        typed.standard_pca.enabled =
            !(typed.null_model.enabled || typed.marchenko_pastur.enabled ||
              typed.adaptive_selector.enabled);
    }

    typed.active = citlali::config::ProcessedTimeChunkCleanerMode::none;
    if (typed.standard_pca.enabled) {
        typed.active =
            citlali::config::ProcessedTimeChunkCleanerMode::standard_pca;
    } else if (typed.null_model.enabled) {
        typed.active =
            citlali::config::ProcessedTimeChunkCleanerMode::null_model;
    } else if (typed.marchenko_pastur.enabled) {
        typed.active =
            citlali::config::ProcessedTimeChunkCleanerMode::marchenko_pastur;
    } else if (typed.adaptive_selector.enabled) {
        typed.active =
            citlali::config::ProcessedTimeChunkCleanerMode::adaptive_selector;
    }
}

}  // namespace citlali::pipeline
