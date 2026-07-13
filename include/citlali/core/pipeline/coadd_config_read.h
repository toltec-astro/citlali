#pragma once

#include <citlali/core/config/coadd_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <tuple>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_coadd_request_config(
    Config &config, citlali::config::CoaddConfig &request,
    Diagnostics &diagnostics) {
    read_config_value(
        config, request.enabled, diagnostics,
        std::tuple{"coadd", "enabled"});
}

}  // namespace citlali::pipeline
