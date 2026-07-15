#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/calibration_config.h>
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
    BeammapConfig beammap;
    BeammapPhotometryConfig beammap_photometry;
    AstrometryConfig astrometry;
};

}  // namespace citlali::config
