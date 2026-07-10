#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/reduction_config_validation.h>

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

template <typename Logger>
void validate_typed_config_mirrors(
    const citlali::config::ReductionConfig &config, const Logger &logger) {
    auto report = citlali::config::validate(config);

    log_typed_config_validation_report(report, logger);
}

}  // namespace citlali::engine_detail
