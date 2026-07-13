#pragma once

#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>

namespace citlali::pipeline {

template <class Config, class RtcProc, class Diagnostics>
void read_legacy_polarimetry_runtime_config(
    Config &config, RtcProc &rtcproc, Diagnostics &diagnostics) {
    read_config_value(
        config, rtcproc.run_polarization, diagnostics,
        std::tuple{"timestream", "polarimetry", "enabled"});

    rtcproc.polarization.stokes_params.clear();
    if (rtcproc.run_polarization) {
        rtcproc.polarization.stokes_params = {
            {0, "I"}, {1, "Q"}, {2, "U"}};
        read_config_value(
            config, rtcproc.polarization.grouping, diagnostics,
            std::tuple{"timestream", "polarimetry", "grouping"});
    } else {
        rtcproc.polarization.stokes_params = {{0, "I"}};
    }
}

}  // namespace citlali::pipeline
