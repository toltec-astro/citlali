#pragma once

#include <citlali/core/config/config_error.h>

namespace citlali::config {

struct CoaddConfig {
    bool enabled = false;
};

inline void validate(const CoaddConfig &, ValidationReport &) {}

}  // namespace citlali::config
