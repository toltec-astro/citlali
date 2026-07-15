#pragma once

#include <citlali/core/session/reduction_result.h>

#include <cstddef>
#include <cstdlib>
#include <ostream>

namespace citlali::cli {

inline int reduction_result_exit_code(
    const citlali::session::ReductionResult &result) noexcept {
    return result.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}

inline void report_reduction_result_diagnostics(
    const citlali::session::ReductionResult &result, std::ostream &os) {
    for (const auto &diagnostic : result.diagnostics) {
        os << diagnostic.code;
        if (!diagnostic.path.empty()) {
            os << " [";
            for (std::size_t index = 0; index < diagnostic.path.size();
                 ++index) {
                if (index != 0) {
                    os << '.';
                }
                os << diagnostic.path[index];
            }
            os << ']';
        }
        if (!diagnostic.message.empty()) {
            os << ": " << diagnostic.message;
        }
        os << '\n';
    }
}

}  // namespace citlali::cli
