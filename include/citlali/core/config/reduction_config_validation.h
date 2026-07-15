#pragma once

#include <citlali/core/config/beammap_config_validation.h>
#include <citlali/core/config/calibration_config_validation.h>
#include <citlali/core/config/coadd_config_validation.h>
#include <citlali/core/config/interface_sync_config_validation.h>
#include <citlali/core/config/mapmaking_config_validation.h>
#include <citlali/core/config/noise_config_validation.h>
#include <citlali/core/config/pointing_config_validation.h>
#include <citlali/core/config/post_processing_config_validation.h>
#include <citlali/core/config/reduction_config.h>
#include <citlali/core/config/runtime_config_validation.h>
#include <citlali/core/config/timestream_config_validation.h>

namespace citlali::config {

inline ValidationReport validate(const ReductionConfig &config) {
    ValidationReport report;
    validate(config.runtime, report);
    validate(config.interface_sync, report);
    validate(config.timestream, report);
    validate(config.mapmaking, report);
    validate(config.coadd, report);
    validate(config.noise, report);
    validate(config.post_processing, report);
    validate(config.pointing, report);
    validate(config.beammap, report);
    validate(config.beammap_photometry, report);
    validate(config.astrometry, report);
    if (config.beammap.detector_tod_output.enabled &&
        config.mapmaking.grouping != MapGrouping::detector) {
        report.add_error(
            {"beammap", "detector_tod_output", "enabled"},
            "requires mapmaking.grouping=detector");
    }
    return report;
}

}  // namespace citlali::config
