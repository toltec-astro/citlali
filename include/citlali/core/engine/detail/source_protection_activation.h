#pragma once

#include <citlali/core/config/runtime_config.h>

#include <string>

namespace citlali::engine_detail {

template <class RtcProc, class PtcProc, class TimestreamConfig, class Logger>
void apply_source_protection_activation(
    citlali::config::ReductionType reduction_type, RtcProc &rtcproc,
    PtcProc &ptcproc, TimestreamConfig &typed_timestream_config,
    const Logger &logger) {
    // The pointing pipeline also covers PSF-preserving focus and holography-style reductions.
    const bool source_aware_reduction =
        citlali::config::is_pointing_reduction_type(reduction_type);
    rtcproc.despiker.source_protection_enabled =
        rtcproc.run_despike &&
        rtcproc.despike_source_protection_config_enabled &&
        source_aware_reduction;
    ptcproc.second_pass_local.source_protection_enabled =
        ptcproc.second_pass_local.enabled &&
        ptcproc.second_pass_local.source_protection_config_enabled &&
        source_aware_reduction;
    typed_timestream_config.raw_time_chunk.despike.source_protection.active =
        rtcproc.despiker.source_protection_enabled;
    typed_timestream_config.processed_time_chunk.flagging.second_pass_local
        .source_protection.active =
        ptcproc.second_pass_local.source_protection_enabled;

    if (rtcproc.run_despike &&
        rtcproc.despike_source_protection_config_enabled) {
        logger->info(
            "raw_time_chunk.despike source protection active={} reduction_type={} radius_arcsec={:.4g}",
            rtcproc.despiker.source_protection_enabled,
            citlali::config::to_string(reduction_type),
            rtcproc.despiker.source_protection_radius_arcsec);
    }
    if (ptcproc.second_pass_local.enabled &&
        ptcproc.second_pass_local.source_protection_config_enabled) {
        logger->info(
            "processed_time_chunk.flagging.second_pass_local source protection active={} reduction_type={} radius_arcsec={:.4g}",
            ptcproc.second_pass_local.source_protection_enabled,
            citlali::config::to_string(reduction_type),
            ptcproc.second_pass_local.source_protection_radius_arcsec);
    }
}

}  // namespace citlali::engine_detail
