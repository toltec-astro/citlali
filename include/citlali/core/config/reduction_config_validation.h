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
    if (config.timestream.fruit_loops.injected_source_test.enabled &&
        config.runtime.reduction_type != ReductionType::pointing) {
        report.add_error(
            {"timestream", "fruit_loops", "injected_source_test", "enabled"},
            "is diagnostic-only and supported only by pointing/OOF reductions");
    }
    if (config.beammap.detector_tod_output.enabled &&
        config.mapmaking.grouping != MapGrouping::detector) {
        report.add_error(
            {"beammap", "detector_tod_output", "enabled"},
            "requires mapmaking.grouping=detector");
    }
    if (config.runtime.reduction_type == ReductionType::beammap &&
        config.beammap.direction_mode == BeammapDirectionMode::all &&
        config.mapmaking.grouping != MapGrouping::detector) {
        report.add_error(
            {"beammap", "direction_mode"},
            "all requires mapmaking.grouping=detector because it emits standard, left, and right detector maps and APTs from one reduction");
    }
    return report;
}

}  // namespace citlali::config
