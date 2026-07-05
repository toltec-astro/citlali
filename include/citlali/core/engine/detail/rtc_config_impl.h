#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/timestream_config_mirror.h>

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    // get rtcproc config
    rtcproc.get_config(config, missing_keys, invalid_keys);
    citlali::pipeline::mirror_raw_despike_config(
        typed_timestream_config.raw_time_chunk.despike, rtcproc);

    auto &typed_raw = typed_timestream_config.raw_time_chunk;
    citlali::pipeline::mirror_raw_flagging_config(
        typed_raw.flagging, rtcproc);

    citlali::pipeline::mirror_raw_kernel_config(
        typed_raw.kernel, rtcproc, RAD_TO_ASEC);

    citlali::pipeline::mirror_raw_altaz_destripe_config(
        typed_raw.altaz_destripe, rtcproc);

    citlali::pipeline::mirror_raw_line_audit_config(
        typed_raw.line_audit, rtcproc.line_audit);

    citlali::pipeline::mirror_raw_downsample_config(
        typed_raw.downsample, rtcproc);

    auto &typed_filter = typed_raw.filter;
    citlali::pipeline::mirror_raw_filter_config(typed_filter, rtcproc);

    citlali::pipeline::mirror_raw_iir_filter_config(
        typed_raw.iir_filter, rtcproc);

    citlali::pipeline::mirror_raw_correction_flags(typed_raw, rtcproc);

    rtcproc.configure_filter_edge_guard(telescope.fsmp);
    citlali::pipeline::mirror_raw_filter_edge_guard_config(
        typed_filter.edge_guard, rtcproc.filter_edge_guard);
    citlali::pipeline::configure_raw_tod_output_context(telescope, rtcproc);

    // ignore hwpr?
    get_config_value(config, calib.ignore_hwpr, missing_keys, invalid_keys,
                     std::tuple{"timestream","polarimetry", "ignore_hwpr"});
}
