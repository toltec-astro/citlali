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
    const auto &response_arm = config.timestream.learning.fruit_response_arm;
    if (response_arm != "disabled" && response_arm != "H" && response_arm != "Half" && response_arm != "Hold")
        report.add_error({"timestream", "learning", "fruit_response_arm"}, "expected disabled, H, Half or Hold");
    if (response_arm != "disabled") {
        const auto &learning = config.timestream.learning;
        const auto &fruit = config.timestream.fruit_loops;
        const auto &outlier = learning.map_pixel_outlier;
        const ConfigPath path{"timestream", "learning", "fruit_response_arm"};
        if (config.runtime.reduction_type != ReductionType::pointing ||
            config.runtime.n_threads != 1 || config.runtime.parallel_policy != ParallelPolicy::seq ||
            !config.timestream.enabled || config.timestream.polarimetry.enabled ||
            !config.mapmaking.enabled || !is_jinc_map_method(config.mapmaking.method) ||
            config.coadd.enabled || config.post_processing.map_filtering.enabled ||
            config.mapmaking.jinc_accounting.enabled)
            report.add_error(path, "EL-F12 requires the registered serial raw-observation pointing JINC route");
        if (!learning.enabled || !learning.diagnostics_enabled || !outlier.diagnostics_enabled ||
            !outlier.targeted_contributor_diagnostics_enabled || !outlier.detector_exclusion_enabled ||
            outlier.detector_exclusion_feedback_bypass_enabled || outlier.detector_exclusion_min_pixels != 4 ||
            outlier.detector_exclusion_application != MapPixelOutlierDetectorExclusionApplication::pre_cleaning)
            report.add_error(path, "EL-F12 requires unchanged complete-map pre-cleaning exclusion evidence");
        if (!fruit.enabled || !fruit.diagnostics_enabled || !fruit.save_all_iters ||
            !is_obsnum_raw_fruit_loops_type(fruit.type) || fruit.relaxation_alpha != 1.0 || fruit.max_iters > 7)
            report.add_error(path, "EL-F12 requires the historical alpha-one recurrence and absolute horizon 6");
    }
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
    return report;
}

}  // namespace citlali::config
