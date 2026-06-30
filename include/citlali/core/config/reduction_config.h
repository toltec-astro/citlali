#pragma once

#include <citlali/core/config/coadd_config.h>
#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/pointing_config.h>
#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>

namespace citlali::config {

struct ReductionConfig {
    RuntimeConfig runtime;
    TimestreamConfig timestream;
    MapmakingConfig mapmaking;
    CoaddConfig coadd;
    NoiseConfig noise;
    PostProcessingConfig post_processing;
    PointingConfig pointing;
    ValidationReport validation;
};

inline ValidationReport validate(const ReductionConfig &config) {
    ValidationReport report;
    validate(config.runtime, report);
    validate(config.timestream, report);
    validate(config.mapmaking, report);
    validate(config.noise, report);
    validate(config.post_processing, report);
    validate(config.pointing, report);
    return report;
}

}  // namespace citlali::config
