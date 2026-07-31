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
    const bool filtered_noise_products_requested =
        map_filtering_active(config.post_processing) &&
        config.post_processing.map_filtering.normalize_errors;
    const bool empirical_noise_products_requested =
        config.noise.enabled && config.noise.products_enabled;
    if (filtered_noise_products_requested && !config.noise.enabled) {
        report.add_error(
            {"post_processing", "map_filtering", "normalize_errors"},
            "requires noise_maps.enabled=true");
    }
    if (config.noise.enabled &&
        (empirical_noise_products_requested ||
         filtered_noise_products_requested) &&
        config.noise.n_noise_maps >= 0 &&
        config.noise.n_noise_maps < 2) {
        report.add_error(
            {"noise_maps", "n_noise_maps"},
            "must be at least 2 when empirical uncertainty products are requested");
    }
    if (config.timestream.fruit_loops.injected_source_test.enabled &&
        config.runtime.reduction_type != ReductionType::pointing) {
        report.add_error(
            {"timestream", "fruit_loops", "injected_source_test", "enabled"},
            "is diagnostic-only and supported only by pointing/OOF reductions");
    }
    if (config.timestream.fruit_loops.enabled &&
        is_filtered_fruit_loops_type(config.timestream.fruit_loops.type) &&
        map_filtering_active(config.post_processing) &&
        map_filter_uses_unit_sum_convolution(
            config.post_processing.map_filtering)) {
        report.add_error(
            {"timestream", "fruit_loops", "type"},
            "unit-sum convolved maps are withheld from fruit-loop feedback "
            "until their support and response contract passes production "
            "validation; use obsnum/raw or coadd/raw feedback");
    }
    if (config.beammap.detector_tod_output.enabled &&
        config.mapmaking.grouping != MapGrouping::detector) {
        report.add_error(
            {"beammap", "detector_tod_output", "enabled"},
            "requires mapmaking.grouping=detector");
    }
    return report;
}

}  // namespace citlali::config
