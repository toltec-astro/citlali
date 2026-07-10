#pragma once

// Included by timestream_config_mirror.h inside namespace citlali::pipeline.

template <class PolarimetryConfig, class RtcProc>
void mirror_polarimetry_config(PolarimetryConfig &target,
                               const RtcProc &rtcproc,
                               std::string_view ignore_hwpr) {
    target.enabled = rtcproc.run_polarization;
    if (const auto grouping = citlali::config::parse_polarimetry_grouping(
            rtcproc.polarization.grouping)) {
        target.grouping = *grouping;
    }
    if (const auto hwpr_policy =
            citlali::config::parse_polarimetry_hwpr_policy(ignore_hwpr)) {
        target.hwpr_policy = *hwpr_policy;
    }
}
