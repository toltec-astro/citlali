#pragma once

#include <citlali/core/config/runtime_config.h>

#include <string>

namespace citlali::pipeline {

template <class RtcProc, class PtcProc, class TimestreamConfig, class Logger>
void apply_source_protection_activation(
    citlali::config::ReductionType reduction_type, RtcProc &rtcproc,
    PtcProc &ptcproc, TimestreamConfig &typed_timestream_config,
    const Logger &logger) {
    // The pointing pipeline also covers PSF-preserving focus and holography-style reductions.
    const bool source_aware_reduction =
        citlali::config::is_pointing_reduction_type(reduction_type);
    auto &raw = typed_timestream_config.raw_time_chunk.despike;
    auto &processed = typed_timestream_config.processed_time_chunk.flagging
                          .second_pass_local;
    raw.source_protection.active =
        raw.enabled && raw.source_protection.enabled && source_aware_reduction;
    processed.source_protection.active = processed.enabled &&
        processed.source_protection.enabled && source_aware_reduction;

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
