#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <tuple>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_processed_weighting_core_config(
    Config &config,
    citlali::config::ProcessedTimeChunkWeightingConfig &weighting,
    citlali::config::ProcessedTimeChunkFlaggingConfig &flagging,
    Diagnostics &diagnostics) {
    std::string weighting_type{
        citlali::config::to_string(weighting.type)};
    read_parsed_mirrored_config_value(
        config,
        std::tuple{"timestream", "processed_time_chunk", "weighting",
                   "type"},
        weighting_type, weighting.type,
        citlali::config::parse_processed_weighting_type, diagnostics,
        {"full", "approximate", "hybrid", "validated", "const"});

    auto read_weighting_double = [&](const char *name, double &target) {
        double value = target;
        read_mirrored_config_value(
            config,
            std::tuple{"timestream", "processed_time_chunk", "weighting",
                       name},
            value, target, diagnostics);
    };
    read_weighting_double(
        "median_map_weight_factor", weighting.median_map_weight_factor);
    read_weighting_double(
        "lower_map_weight_factor", weighting.lower_map_weight_factor);
    read_weighting_double(
        "upper_map_weight_factor", weighting.upper_map_weight_factor);

    auto read_flagging_double = [&](const char *name, double &target) {
        double value = target;
        read_mirrored_config_value(
            config,
            std::tuple{"timestream", "processed_time_chunk", "flagging",
                       name},
            value, target, diagnostics);
    };
    read_flagging_double(
        "lower_tod_inv_var_factor", flagging.lower_tod_inv_var_factor);
    read_flagging_double(
        "upper_tod_inv_var_factor", flagging.upper_tod_inv_var_factor);

    auto read_optional_weighting_double = [&](const char *name,
                                               double &target) {
        double value = target;
        read_optional_mirrored_config_value(
            config,
            std::tuple{"timestream", "processed_time_chunk", "weighting",
                       name},
            value, target, diagnostics);
    };
    read_optional_weighting_double(
        "source_mask_radius_arcsec", weighting.source_mask_radius_arcsec);
    read_optional_weighting_double(
        "hybrid_correction_min_factor",
        weighting.hybrid_correction_min_factor);
    read_optional_weighting_double(
        "hybrid_correction_max_factor",
        weighting.hybrid_correction_max_factor);
}

template <class Config, class Diagnostics>
void read_processed_weight_validation_config(
    Config &config,
    citlali::config::ProcessedTimeChunkWeightValidationConfig &validation,
    Diagnostics &diagnostics) {
    const auto key = [](const char *name) {
        return std::tuple{"timestream", "processed_time_chunk", "weighting",
                          "validation", name};
    };
    auto read_bool = [&](const char *name, bool &target) {
        bool value = target;
        read_optional_mirrored_config_value(
            config, key(name), value, target, diagnostics);
    };
    auto read_int = [&](const char *name, int &target) {
        int value = target;
        read_optional_mirrored_config_value(
            config, key(name), value, target, diagnostics);
    };
    auto read_double = [&](const char *name, double &target) {
        double value = target;
        read_optional_mirrored_config_value(
            config, key(name), value, target, diagnostics);
    };

    read_bool("enabled", validation.enabled);
    if (!validation.enabled) {
        return;
    }
    read_int("accumulation_iters", validation.accumulation_iters);
    read_int("apply_start_iter", validation.apply_start_iter);
    read_int("min_valid_scans", validation.min_valid_scans);
    read_double("min_factor", validation.min_factor);
    read_double("unvalidated_factor", validation.unvalidated_factor);
    read_bool(
        "require_fruitloops_model", validation.require_fruitloops_model);
    read_bool("transient_ratio_enabled", validation.transient_ratio_enabled);
    read_double("ratio_power", validation.ratio_power);
    read_double("transient_ratio_power", validation.transient_ratio_power);
    read_bool("upward_enabled", validation.upward_enabled);
    read_double("upward_max_factor", validation.upward_max_factor);
    read_double("upward_power", validation.upward_power);
    read_double(
        "upward_min_base_factor", validation.upward_min_base_factor);
    read_bool(
        "upward_require_atmospheric",
        validation.upward_require_atmospheric);
    read_double(
        "upward_min_atmospheric_factor",
        validation.upward_min_atmospheric_factor);
    read_bool(
        "atmospheric_correlation_enabled",
        validation.atmospheric_correlation_enabled);
    std::string atmospheric_grouping{
        citlali::config::to_string(validation.atmospheric_grouping)};
    read_optional_parsed_mirrored_config_value(
        config, key("atmospheric_grouping"), atmospheric_grouping,
        validation.atmospheric_grouping,
        citlali::config::parse_processed_weight_grouping, diagnostics,
        {"array", "network"});
    read_int(
        "atmospheric_min_detectors", validation.atmospheric_min_detectors);
    read_double("atmospheric_ref", validation.atmospheric_ref);
    read_double("atmospheric_span", validation.atmospheric_span);
    read_double("atmospheric_power", validation.atmospheric_power);
    read_double("min_good_frac", validation.min_good_frac);
    read_int("min_overlap", validation.min_overlap);
    read_int("max_samples", validation.max_samples);
    read_bool(
        "high_weight_validation_enabled",
        validation.high_weight_validation_enabled);
    read_bool("high_weight_apply_caps", validation.high_weight_apply_caps);
    std::string high_weight_grouping{
        citlali::config::to_string(validation.high_weight_grouping)};
    read_optional_parsed_mirrored_config_value(
        config, key("high_weight_grouping"), high_weight_grouping,
        validation.high_weight_grouping,
        citlali::config::parse_processed_weight_grouping, diagnostics,
        {"array", "network"});
    read_int(
        "high_weight_min_group_detectors",
        validation.high_weight_min_group_detectors);
    read_double(
        "high_weight_log_robust_z", validation.high_weight_log_robust_z);
    read_double(
        "high_weight_max_median_factor",
        validation.high_weight_max_median_factor);
    read_double(
        "high_weight_cap_median_factor",
        validation.high_weight_cap_median_factor);
    read_double(
        "high_weight_min_validated_factor",
        validation.high_weight_min_validated_factor);
}

}  // namespace citlali::pipeline
