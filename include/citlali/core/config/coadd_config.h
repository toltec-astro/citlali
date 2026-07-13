#pragma once

namespace citlali::config {

struct CoaddConfig {
    bool enabled = false;
};

inline bool coadd_active(const CoaddConfig &config) {
    return config.enabled;
}

}  // namespace citlali::config
