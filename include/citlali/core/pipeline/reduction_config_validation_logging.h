#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/reduction_config_validation.h>

#include <string_view>

namespace citlali::pipeline {

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
        logger->debug("typed config validation passed");
        return;
    }

    if (report.error_count() > 0) {
        logger->error(
            "typed config validation reported {} error(s) and {} warning(s)",
            report.error_count(), report.warning_count());
    } else {
        logger->warn(
            "typed config validation reported {} warning(s)",
            report.warning_count());
    }
    for (const auto &diagnostic : report.diagnostics()) {
        if (diagnostic.severity ==
            citlali::config::DiagnosticSeverity::error) {
            logger->error(
                "typed config {}: {}: {}",
                config_diagnostic_severity_label(diagnostic.severity),
                citlali::config::format_path(diagnostic.path),
                diagnostic.message);
        } else {
            logger->warn(
                "typed config {}: {}: {}",
                config_diagnostic_severity_label(diagnostic.severity),
                citlali::config::format_path(diagnostic.path),
                diagnostic.message);
        }
    }
}

template <typename Diagnostics, typename Logger>
void validate_typed_config(
    const citlali::config::ReductionConfig &config,
    Diagnostics &diagnostics, const Logger &logger) {
    auto report = citlali::config::validate(config);
    log_typed_config_validation_report(report, logger);
    for (const auto &error : report.errors()) {
        diagnostics.invalid_key_paths().push_back(error.path);
    }
}

}  // namespace citlali::pipeline
