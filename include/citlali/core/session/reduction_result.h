#pragma once

#include <citlali/core/error/error.h>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace citlali::session {

enum class ReductionStatus {
    succeeded,
    invalid_request,
    processor_selection_failed,
    io_failed,
    execution_failed,
    output_failed,
    unhandled_exception,
    invalid_session_state
};

struct ReductionDiagnostic {
    std::string code;
    std::string message;
    std::vector<std::string> path;
};

struct ReductionResult {
    ReductionStatus status = ReductionStatus::execution_failed;
    std::vector<ReductionDiagnostic> diagnostics;
    std::vector<std::filesystem::path> product_roots;
    std::vector<std::filesystem::path> provenance_artifacts;

    bool succeeded() const noexcept {
        return status == ReductionStatus::succeeded;
    }

    explicit operator bool() const noexcept {
        return succeeded();
    }

    void add_diagnostic(std::string code, std::string message,
                        std::vector<std::string> path = {}) {
        diagnostics.push_back(
            {std::move(code), std::move(message), std::move(path)});
    }
};

inline ReductionResult successful_reduction_result() {
    ReductionResult result;
    result.status = ReductionStatus::succeeded;
    return result;
}

inline ReductionResult failed_reduction_result(
    ReductionStatus status, std::string code, std::string message,
    std::vector<std::string> path = {}) {
    ReductionResult result;
    result.status = status;
    result.add_diagnostic(
        std::move(code), std::move(message), std::move(path));
    return result;
}

inline ReductionResult failed_reduction_result(
    const citlali::error::Error &error) {
    switch (error.code()) {
    case citlali::error::Code::invalid_config:
        return failed_reduction_result(
            ReductionStatus::invalid_request, "config.invalid", error.what());
    case citlali::error::Code::io:
        return failed_reduction_result(
            ReductionStatus::io_failed, "io.failed", error.what());
    case citlali::error::Code::output:
        return failed_reduction_result(
            ReductionStatus::output_failed, "output.failed", error.what());
    case citlali::error::Code::runtime:
        return failed_reduction_result(
            ReductionStatus::execution_failed, "runtime.failed", error.what());
    case citlali::error::Code::internal:
        return failed_reduction_result(
            ReductionStatus::execution_failed, "internal.failed", error.what());
    }
    return failed_reduction_result(
        ReductionStatus::execution_failed, "internal.failed", error.what());
}

}  // namespace citlali::session
