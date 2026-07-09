#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/timestream_config_mirror.h>

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    logger->info("getting rtc config options");
    // get rtcproc config
    rtcproc.get_config(config, config_diagnostics.missing_keys, config_diagnostics.invalid_keys);
    auto &raw_config = typed_config.timestream.raw_time_chunk;
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
    get_config_value(config, calib.ignore_hwpr, config_diagnostics.missing_keys, config_diagnostics.invalid_keys,
                     std::tuple{"timestream","polarimetry", "ignore_hwpr"});
}
