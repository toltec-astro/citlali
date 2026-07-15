#pragma once

#include <citlali/core/config/timestream_config.h>

#include <string>

namespace citlali::pipeline {

template <class RtcProc, class Calib>
void adapt_polarimetry_config(
    const citlali::config::TimestreamPolarimetryConfig &config,
    RtcProc &rtcproc, Calib &calib) {
    rtcproc.run_polarization = config.enabled;
    rtcproc.polarization.grouping =
        std::string{citlali::config::to_string(config.grouping)};
    rtcproc.polarization.stokes_params.clear();
    rtcproc.polarization.stokes_params = config.enabled
        ? decltype(rtcproc.polarization.stokes_params){
              {0, "I"}, {1, "Q"}, {2, "U"}}
        : decltype(rtcproc.polarization.stokes_params){{0, "I"}};
    calib.ignore_hwpr =
        std::string{citlali::config::to_string(config.hwpr_policy)};
}

}  // namespace citlali::pipeline
