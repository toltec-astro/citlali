#pragma once

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

}  // namespace citlali::config
