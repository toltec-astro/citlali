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

}  // namespace citlali::pipeline
