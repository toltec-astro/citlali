#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_config_read.h>
#include <citlali/core/pipeline/timestream_config_mirror.h>

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    // get rtcproc config
    citlali::pipeline::read_processor_config(
        rtcproc, config, config_diag);
    auto &raw_config =
        citlali::pipeline::timestream_config(*this).raw_time_chunk;
    citlali::pipeline::mirror_raw_despike_config(
        raw_config.despike, rtcproc);

    citlali::pipeline::mirror_raw_flagging_config(
        raw_config.flagging, rtcproc);

    citlali::pipeline::mirror_raw_kernel_config(
        raw_config.kernel, rtcproc, RAD_TO_ASEC);

    citlali::pipeline::mirror_raw_altaz_destripe_config(
        raw_config.altaz_destripe, rtcproc);

    citlali::pipeline::mirror_raw_line_audit_config(
        raw_config.line_audit, rtcproc.line_audit);

    citlali::pipeline::mirror_raw_downsample_config(
        raw_config.downsample, rtcproc);

    auto &typed_filter = raw_config.filter;
    citlali::pipeline::mirror_raw_filter_config(typed_filter, rtcproc);

    citlali::pipeline::mirror_raw_iir_filter_config(
        raw_config.iir_filter, rtcproc);

    citlali::pipeline::mirror_raw_correction_flags(raw_config, rtcproc);

    rtcproc.configure_filter_edge_guard(telescope.fsmp);
    citlali::pipeline::mirror_raw_filter_edge_guard_config(
        typed_filter.edge_guard, rtcproc.filter_edge_guard);
    citlali::pipeline::configure_raw_tod_output_context(telescope, rtcproc);

    // ignore hwpr?
    auto &polarimetry_config =
        citlali::pipeline::polarimetry_config(*this);
    citlali::pipeline::read_polarimetry_hwpr_policy_config(
        config, calib.ignore_hwpr, polarimetry_config, config_diag);
    citlali::pipeline::mirror_polarimetry_config(
        polarimetry_config, rtcproc);
}
