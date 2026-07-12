#pragma once

#include <citlali/core/pipeline/raw_timestream_execution_plan.h>
#include <citlali/core/pipeline/timestream_config_adapter_raw_filtering.h>
#include <citlali/core/pipeline/timestream_config_adapter_raw_flagging.h>
#include <citlali/core/pipeline/timestream_config_adapter_raw_line_audit.h>

namespace citlali::pipeline {

template <class RtcProc>
void adapt_raw_timestream_config_one_way(
    const citlali::config::RawTimeChunkConfig &raw, RtcProc &rtcproc,
    double arcsec_to_rad, double fwhm_to_std) {
    adapt_raw_filtering_config_one_way(
        raw, rtcproc, arcsec_to_rad, fwhm_to_std);
    adapt_raw_flagging_config_one_way(raw, rtcproc);
    adapt_raw_line_audit_config_one_way(raw.line_audit, rtcproc.line_audit);
}

template <class RtcProc>
void adapt_raw_timestream_observation_state_one_way(
    const RawTimestreamObservationState &observation, RtcProc &rtcproc) {
    if (observation.native_sample_rate_hz) {
        rtcproc.despiker.fsmp = *observation.native_sample_rate_hz;
    }
    if (observation.downsample_factor) {
        rtcproc.downsampler.factor = *observation.downsample_factor;
    }
    if (observation.filter_edge_guard_samples) {
        rtcproc.filter_edge_guard.guard_samples =
            *observation.filter_edge_guard_samples;
    }
    if (observation.filter_outer_context_samples) {
        rtcproc.filter_edge_guard.context_samples =
            *observation.filter_outer_context_samples;
    }
    if (observation.source_protection_active) {
        rtcproc.despiker.source_protection_enabled =
            *observation.source_protection_active;
    }
    if (observation.extinction_active) {
        rtcproc.run_extinction = *observation.extinction_active;
    }
    if (observation.extinction_model) {
        rtcproc.calibration.extinction_model =
            *observation.extinction_model;
    }
}

}  // namespace citlali::pipeline
