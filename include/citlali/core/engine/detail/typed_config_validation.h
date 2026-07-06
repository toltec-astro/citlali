#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/reduction_config.h>

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

inline citlali::config::ReductionConfig make_typed_reduction_config_mirror(
    const citlali::config::RuntimeConfig &runtime_config,
    const citlali::config::TimestreamConfig &timestream_config,
    const citlali::config::MapmakingConfig &mapmaking_config,
    const citlali::config::CoaddConfig &coadd_config,
    const citlali::config::NoiseConfig &noise_config,
    const citlali::config::PostProcessingConfig &post_processing_config,
    const citlali::config::PointingConfig &pointing_config,
    const citlali::config::BeammapConfig &beammap_config,
    const citlali::config::AstrometryConfig &astrometry_config) {
    citlali::config::ReductionConfig config;
    config.runtime = runtime_config;
    config.timestream = timestream_config;
    config.mapmaking = mapmaking_config;
    config.coadd = coadd_config;
    config.noise = noise_config;
    config.post_processing = post_processing_config;
    config.pointing = pointing_config;
    config.beammap = beammap_config;
    config.astrometry = astrometry_config;
    return config;
}

template <typename Logger>
void validate_typed_config_mirrors(
    const citlali::config::ReductionConfig &config, const Logger &logger) {
    auto report = citlali::config::validate(config);

    log_typed_config_validation_report(report, logger);
}

}  // namespace citlali::engine_detail
