#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace citlali::config {

using ConfigPath = std::vector<std::string>;

enum class DiagnosticSeverity {
    warning,
    error
};

struct Diagnostic {
    DiagnosticSeverity severity = DiagnosticSeverity::error;
    ConfigPath path;
    std::string message;
};

inline std::string format_path(const ConfigPath &path) {
    if (path.empty()) {
        return "<config>";
    }

    std::ostringstream stream;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i > 0) {
            stream << ".";
        }
        stream << path[i];
    }
    return stream.str();
}

inline ConfigPath append_config_path(ConfigPath path,
                                     std::initializer_list<std::string> suffix) {
    path.insert(path.end(), suffix.begin(), suffix.end());
    return path;
}

class ValidationReport {
public:
    void add(DiagnosticSeverity severity, ConfigPath path, std::string message) {
        diagnostics_.push_back(Diagnostic{severity, std::move(path), std::move(message)});
    }

    void add_error(ConfigPath path, std::string message) {
        add(DiagnosticSeverity::error, std::move(path), std::move(message));
    }

    void add_warning(ConfigPath path, std::string message) {
        add(DiagnosticSeverity::warning, std::move(path), std::move(message));
    }

    [[nodiscard]] bool ok() const {
        return error_count() == 0;
    }

    [[nodiscard]] std::size_t error_count() const {
        return static_cast<std::size_t>(
            std::count_if(diagnostics_.begin(), diagnostics_.end(), [](const auto &diagnostic) {
                return diagnostic.severity == DiagnosticSeverity::error;
            }));
    }

    [[nodiscard]] std::size_t warning_count() const {
        return static_cast<std::size_t>(
            std::count_if(diagnostics_.begin(), diagnostics_.end(), [](const auto &diagnostic) {
                return diagnostic.severity == DiagnosticSeverity::warning;
            }));
    }

    [[nodiscard]] const std::vector<Diagnostic> &diagnostics() const {
        return diagnostics_;
    }

    [[nodiscard]] std::vector<Diagnostic> errors() const {
        std::vector<Diagnostic> result;
        std::copy_if(diagnostics_.begin(), diagnostics_.end(), std::back_inserter(result),
                     [](const auto &diagnostic) {
                         return diagnostic.severity == DiagnosticSeverity::error;
                     });
        return result;
    }

    [[nodiscard]] std::string format_for_cli() const {
        std::ostringstream stream;
        for (const auto &diagnostic : diagnostics_) {
            stream << (diagnostic.severity == DiagnosticSeverity::error ? "error" : "warning")
                   << ": " << format_path(diagnostic.path) << ": " << diagnostic.message << "\n";
        }
        return stream.str();
    }

private:
    std::vector<Diagnostic> diagnostics_;
};

template <typename T>
void check_minimum(const T &value, const T &minimum, const ConfigPath &path,
                   ValidationReport &report) {
    if (value < minimum) {
        report.add_error(path, "must be greater than or equal to " + std::to_string(minimum));
    }
}

template <typename T>
void check_greater_than(const T &value, const T &minimum, const ConfigPath &path,
                        ValidationReport &report) {
    if (value <= minimum) {
        report.add_error(path, "must be greater than " + std::to_string(minimum));
    }
}

template <typename T>
void check_maximum(const T &value, const T &maximum, const ConfigPath &path,
                   ValidationReport &report) {
    if (value > maximum) {
        report.add_error(path, "must be less than or equal to " + std::to_string(maximum));
    }
}

inline void check_optional_minimum(const double value, const double minimum,
                                   const ConfigPath &path,
                                   ValidationReport &report) {
    if (std::isfinite(value)) {
        check_minimum(value, minimum, path, report);
    }
}

}  // namespace citlali::config
