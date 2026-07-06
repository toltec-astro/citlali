#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/calibration_config.h>
#include <citlali/core/config/coadd_config.h>
#include <citlali/core/config/config_error.h>
#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/pointing_config.h>
#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>

#include <string_view>

namespace citlali::engine_detail {

inline std::string_view config_diagnostic_severity_label(
    citlali::config::DiagnosticSeverity severity) {
    switch (severity) {
    case citlali::config::DiagnosticSeverity::warning:
        return "warning";
    case citlali::config::DiagnosticSeverity::error:
        return "error";
    }
    return "unknown";
}

template <typename Config>
void validate_typed_config_section(const Config &config,
                                   citlali::config::ValidationReport &report) {
    citlali::config::validate(config, report);
}

template <typename Logger>
void log_typed_config_validation_report(
    const citlali::config::ValidationReport &report, const Logger &logger) {
    if (report.diagnostics().empty()) {
        logger->debug("typed config mirror validation passed");
        return;
    }

    logger->warn(
        "typed config mirror validation reported {} error(s) and {} warning(s); "
        "legacy config parsing remains authoritative",
        report.error_count(), report.warning_count());
    for (const auto &diagnostic : report.diagnostics()) {
        logger->warn("typed config mirror {}: {}: {}",
                     config_diagnostic_severity_label(diagnostic.severity),
                     citlali::config::format_path(diagnostic.path),
                     diagnostic.message);
    }
}

template <typename Logger>
void validate_typed_config_mirrors(
    const citlali::config::RuntimeConfig &runtime_config,
    const citlali::config::TimestreamConfig &timestream_config,
    const citlali::config::MapmakingConfig &mapmaking_config,
    const citlali::config::CoaddConfig &coadd_config,
    const citlali::config::NoiseConfig &noise_config,
    const citlali::config::PostProcessingConfig &post_processing_config,
    const citlali::config::PointingConfig &pointing_config,
    const citlali::config::BeammapConfig &beammap_config,
    const citlali::config::AstrometryConfig &astrometry_config,
    std::string_view reduction_type, const Logger &logger) {
    citlali::config::ValidationReport report;
    validate_typed_config_section(runtime_config, report);
    validate_typed_config_section(timestream_config, report);
    validate_typed_config_section(mapmaking_config, report);
    validate_typed_config_section(coadd_config, report);
    validate_typed_config_section(noise_config, report);
    validate_typed_config_section(post_processing_config, report);
    validate_typed_config_section(astrometry_config, report);

    if (reduction_type == "pointing") {
        validate_typed_config_section(pointing_config, report);
    }
    if (reduction_type == "beammap") {
        validate_typed_config_section(beammap_config, report);
    }

    log_typed_config_validation_report(report, logger);
}

}  // namespace citlali::engine_detail
