#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_fruit_loops_core_config(
    Config &config,
    citlali::config::TimestreamFruitLoopsConfig &typed_config,
    Diagnostics &diagnostics) {
    bool enabled = typed_config.enabled;
    read_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "enabled"},
        enabled, typed_config.enabled, diagnostics);
    if (!typed_config.enabled) {
        return;
    }

    bool diagnostics_enabled = typed_config.diagnostics_enabled;
    read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "diagnostics_enabled"},
        diagnostics_enabled, typed_config.diagnostics_enabled, diagnostics);

    bool save_all_iters = typed_config.save_all_iters;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "save_all_iters"},
        save_all_iters, typed_config.save_all_iters, diagnostics);

    std::string path = typed_config.path;
    read_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "path"},
        path, typed_config.path, diagnostics);

    std::string restart_path = typed_config.restart_path;
    read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "restart_path"},
        restart_path, typed_config.restart_path, diagnostics);

    std::string type = typed_config.type;
    read_config_value_if_clean(
        config, std::tuple{"timestream", "fruit_loops", "type"}, type,
        [&typed_config](const std::string &value) {
            typed_config.type = std::string{
                citlali::config::canonical_fruit_loops_type(value)};
        },
        diagnostics);

    std::string mode{citlali::config::to_string(typed_config.mode)};
    read_parsed_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "mode"}, mode,
        typed_config.mode, citlali::config::parse_fruit_loops_mode,
        diagnostics, {"upper", "lower", "both"});

    double sig2noise_limit = typed_config.sig2noise_limit;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "sig2noise_limit"},
        sig2noise_limit, typed_config.sig2noise_limit, diagnostics);

    auto array_flux_limit = typed_config.array_flux_limit;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "array_flux_limit"},
        array_flux_limit, typed_config.array_flux_limit, diagnostics);

    int max_iters = typed_config.max_iters;
    read_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "max_iters"},
        max_iters, typed_config.max_iters, diagnostics);

    auto read_optional_double = [&](const auto &key, double &target,
                                    double minimum = 0.0) {
        double value = target;
        read_optional_mirrored_config_value(
            config, key, value, target, diagnostics, {}, {minimum});
    };
    read_optional_double(
        std::tuple{"timestream", "fruit_loops", "peak_fraction_limit"},
        typed_config.peak_fraction_limit);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops", "local_snr_floor"},
        typed_config.local_snr_floor);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops",
                   "local_sigma_inner_radius_arcsec"},
        typed_config.local_sigma_inner_radius_arcsec);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops",
                   "local_sigma_outer_radius_arcsec"},
        typed_config.local_sigma_outer_radius_arcsec);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops", "local_sigma_inner_fwhm"},
        typed_config.local_sigma_inner_fwhm);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops", "local_sigma_outer_fwhm"},
        typed_config.local_sigma_outer_fwhm);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops",
                   "local_sigma_edge_guard_arcsec"},
        typed_config.local_sigma_edge_guard_arcsec);

    int local_sigma_min_pixels = typed_config.local_sigma_min_pixels;
    read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "local_sigma_min_pixels"},
        local_sigma_min_pixels, typed_config.local_sigma_min_pixels,
        diagnostics, {}, {1});

    read_optional_double(
        std::tuple{"timestream", "fruit_loops",
                   "adaptive_support_radius_arcsec"},
        typed_config.adaptive_support_radius_arcsec);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops",
                   "adaptive_support_radius_fwhm"},
        typed_config.adaptive_support_radius_fwhm);

    std::string source_center_mode{
        citlali::config::to_string(typed_config.source_center_mode)};
    read_optional_parsed_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "source_center_mode"},
        source_center_mode, typed_config.source_center_mode,
        citlali::config::parse_fruit_loops_source_center_mode,
        diagnostics, {"auto", "header", "peak", "map_center"});

    auto &feedback = typed_config.weight_feedback;
    bool feedback_enabled = feedback.enabled;
    read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "weight_feedback",
                   "enabled"},
        feedback_enabled, feedback.enabled, diagnostics);
    std::string feedback_reference{
        citlali::config::to_string(feedback.reference)};
    read_optional_parsed_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "weight_feedback",
                   "reference"},
        feedback_reference, feedback.reference,
        citlali::config::parse_fruit_loops_weight_feedback_reference,
        diagnostics);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops", "weight_feedback",
                   "low_relative_weight"},
        feedback.low_relative_weight);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops", "weight_feedback",
                   "high_relative_weight"},
        feedback.high_relative_weight);
    read_optional_double(
        std::tuple{"timestream", "fruit_loops", "center_keep_radius_arcsec"},
        typed_config.center_keep_radius_arcsec);

    std::string interp_mode{
        citlali::config::to_string(typed_config.interp_mode_override)};
    read_optional_parsed_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "interp_mode_override"},
        interp_mode, typed_config.interp_mode_override,
        citlali::config::parse_fruit_loops_interp_mode_override,
        diagnostics);

    bool legacy_center = typed_config.legacy_center;
    read_optional_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "legacy_center"},
        legacy_center, typed_config.legacy_center, diagnostics);
    bool recompute_weights = typed_config.recompute_weights_after_addback;
    read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops",
                   "recompute_weights_after_addback"},
        recompute_weights, typed_config.recompute_weights_after_addback,
        diagnostics);
}

}  // namespace citlali::pipeline
