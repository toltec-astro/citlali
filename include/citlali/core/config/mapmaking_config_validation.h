#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/mapmaking_config.h>

namespace citlali::config {

inline void validate(const MapmakingConfig &config, ValidationReport &report) {
    check_greater_than(config.pixel_size_arcsec, 0.0,
                       {"mapmaking", "pixel_size_arcsec"}, report);
    check_minimum(config.x_size_pix, 0, {"mapmaking", "x_size_pix"}, report);
    check_minimum(config.y_size_pix, 0, {"mapmaking", "y_size_pix"}, report);
}

}  // namespace citlali::config
