#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>
#include <citlali/core/pipeline/timestream_config_adapter_raw.h>

namespace citlali::pipeline {

template <class RtcProc>
void initialize_raw_timestream_authority(
    const citlali::config::RawTimeChunkConfig &request,
    const citlali::config::InterfaceSyncOffsetConfig
        &interface_sync_request,
    RawTimestreamExecutionPlan &plan,
    citlali::config::RawTimeChunkConfig &effective_config,
    RtcProc &rtcproc, double native_sample_rate_hz,
    double arcsec_to_rad, double fwhm_to_std) {
    rtcproc.reset_phase_independent_state();
    plan.reset_from_request(request, interface_sync_request);
    plan.requested_rtc_contract.assigned_grid_authority =
        std::string{RtcProc::assigned_grid_authority};
    plan.effective_rtc_contract = plan.requested_rtc_contract;
    effective_config = plan.effective;
    adapt_raw_timestream_config_one_way(
        effective_config, rtcproc, arcsec_to_rad, fwhm_to_std);
    rtcproc.configure_filter_edge_guard(native_sample_rate_hz);
}

}  // namespace citlali::pipeline
