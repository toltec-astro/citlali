#pragma once

#include <citlali/core/config/timestream_config.h>

#include <optional>
#include <utility>

namespace citlali::pipeline {

struct ProcessedWeightingResolution {
    citlali::config::ProcessedTimeChunkWeightingConfig effective;
    bool validation_forced_by_weighting_type = false;
    bool busy_row_disabled_without_second_pass = false;
};

struct ProcessedWeightingSourceMaskResolution {
    std::optional<double> requested;
    double effective = 0.0;
    bool inherited_from_cleaning = false;
};

inline ProcessedWeightingSourceMaskResolution
resolve_processed_weighting_source_mask(
    std::optional<double> requested, double clean_mask_radius_arcsec) {
    return ProcessedWeightingSourceMaskResolution{
        requested,
        requested.value_or(clean_mask_radius_arcsec),
        !requested.has_value(),
    };
}

inline ProcessedWeightingResolution resolve_processed_weighting(
    const citlali::config::ProcessedTimeChunkWeightingConfig &requested,
    const citlali::config::ProcessedTimeChunkFlaggingConfig &flagging) {
    ProcessedWeightingResolution resolution{requested};
    auto &effective = resolution.effective;

    if (citlali::config::is_validated_processed_weighting_type(
            effective.type) &&
        !effective.validation.enabled) {
        effective.validation.enabled = true;
        resolution.validation_forced_by_weighting_type = true;
    }
    if (effective.busy_row_suppression.enabled &&
        !flagging.second_pass_local.enabled) {
        effective.busy_row_suppression.enabled = false;
        resolution.busy_row_disabled_without_second_pass = true;
    }
    return resolution;
}

template <class Logger>
void log_processed_weighting_resolution(
    const ProcessedWeightingResolution &resolution, const Logger &logger) {
    if (resolution.validation_forced_by_weighting_type) {
        logger->warn(
            "weighting.type='validated' forces weighting.validation.enabled=true");
    }
    if (resolution.busy_row_disabled_without_second_pass) {
        logger->warn(
            "weighting.busy_row_suppression requires flagging.second_pass_local.enabled; disabling busy-row suppression");
    }
}

template <class Logger>
void resolve_processed_weighting_dependencies(
    citlali::config::ProcessedTimeChunkWeightingConfig &weighting,
    const citlali::config::ProcessedTimeChunkFlaggingConfig &flagging,
    const Logger &logger) {
    auto resolution = resolve_processed_weighting(weighting, flagging);
    log_processed_weighting_resolution(resolution, logger);
    weighting = std::move(resolution.effective);
}

}  // namespace citlali::pipeline
