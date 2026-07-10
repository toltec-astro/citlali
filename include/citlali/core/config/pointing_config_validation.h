#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/pointing_config.h>

namespace citlali::config {

inline void validate(const PointingConfig &config, ValidationReport &report) {
    check_minimum(config.header_max_radius_arcsec, 0.0,
                  {"pointing", "source_strategy", "header_max_radius_arcsec"},
                  report);
}

}  // namespace citlali::config
