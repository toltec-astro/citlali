#pragma once

#include <citlali/core/config/config_error.h>

namespace citlali::config {

struct CoaddConfig {
    bool enabled = false;
};

inline void set_coadd_enabled(CoaddConfig &config, bool enabled) {
    config.enabled = enabled;
}

inline bool coadd_active(const CoaddConfig &config) {
    return config.enabled;
}

inline void validate(const CoaddConfig &, ValidationReport &) {}

}  // namespace citlali::config
