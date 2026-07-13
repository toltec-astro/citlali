#pragma once

#include <citlali/core/config/timestream_config.h>

namespace citlali::pipeline {

template <class PolarimetryConfig, class RtcProc>
void mirror_polarimetry_config(PolarimetryConfig &target,
                               const RtcProc &rtcproc) {
    target.enabled = rtcproc.run_polarization;
    if (const auto grouping = citlali::config::parse_polarimetry_grouping(
            rtcproc.polarization.grouping)) {
        target.grouping = *grouping;
    }
}

}  // namespace citlali::pipeline
