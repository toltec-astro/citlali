#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/runtime_config.h>

namespace citlali::config {

struct ReductionConfig {
    RuntimeConfig runtime;
    MapmakingConfig mapmaking;
    NoiseConfig noise;
    ValidationReport validation;
};

inline ValidationReport validate(const ReductionConfig &config) {
    ValidationReport report;
    validate(config.runtime, report);
    validate(config.mapmaking, report);
    validate(config.noise, report);
    return report;
}

}  // namespace citlali::config
