#pragma once

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>

#include <string>

namespace citlali::pipeline {

struct SourceProtectionActivationResolution {
    bool source_aware_reduction = false;
    bool raw_activation_requested = false;
    bool processed_activation_requested = false;
    bool raw_active = false;
    bool processed_active = false;
};

inline SourceProtectionActivationResolution resolve_source_protection(
    citlali::config::ReductionType reduction_type,
    const citlali::config::TimestreamConfig &config) {
    const auto &raw = config.raw_time_chunk.despike;
    const auto &processed =
        config.processed_time_chunk.flagging.second_pass_local;
    const bool source_aware_reduction =
        citlali::config::is_pointing_reduction_type(reduction_type);
    const bool raw_activation_requested =
        raw.enabled && raw.source_protection.enabled;
    const bool processed_activation_requested =
        processed.enabled && processed.source_protection.enabled;
    return SourceProtectionActivationResolution{
        source_aware_reduction,
        raw_activation_requested,
        processed_activation_requested,
        raw_activation_requested && source_aware_reduction,
        processed_activation_requested && source_aware_reduction,
    };
}

template <class RtcProc, class PtcProc, class TimestreamConfig, class Logger>
void apply_source_protection_activation(
    citlali::config::ReductionType reduction_type, RtcProc &rtcproc,
    PtcProc &ptcproc, TimestreamConfig &typed_timestream_config,
    const Logger &logger) {
    const auto resolution = resolve_source_protection(
        reduction_type, typed_timestream_config);
    auto &raw = typed_timestream_config.raw_time_chunk.despike;
    auto &processed = typed_timestream_config.processed_time_chunk.flagging
                          .second_pass_local;
    raw.source_protection.active = resolution.raw_active;
    processed.source_protection.active = resolution.processed_active;

    rtcproc.despiker.source_protection_enabled =
        raw.source_protection.active;
    rtcproc.despiker.source_protection_radius_arcsec =
        raw.source_protection.radius_arcsec;
    ptcproc.second_pass_local.source_protection_enabled =
        processed.source_protection.active;
    ptcproc.second_pass_local.source_protection_radius_arcsec =
        processed.source_protection.radius_arcsec;

    if (raw.enabled && raw.source_protection.enabled) {
        logger->info(
            "raw_time_chunk.despike source protection active={} reduction_type={} radius_arcsec={:.4g}",
            raw.source_protection.active,
            citlali::config::to_string(reduction_type),
            raw.source_protection.radius_arcsec);
    }
    if (processed.enabled && processed.source_protection.enabled) {
        logger->info(
            "processed_time_chunk.flagging.second_pass_local source protection active={} reduction_type={} radius_arcsec={:.4g}",
            processed.source_protection.active,
            citlali::config::to_string(reduction_type),
            processed.source_protection.radius_arcsec);
    }
}

}  // namespace citlali::pipeline
