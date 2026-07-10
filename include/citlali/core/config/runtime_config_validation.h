#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/runtime_config.h>

namespace citlali::config {

inline void validate(const RuntimeConfig &config, ValidationReport &report) {
    check_minimum(config.n_threads, 1, {"runtime", "n_threads"}, report);
    if (!config.interp_over_gaps) {
        report.add_error({"runtime", "interp_over_gaps"},
                         "false is not supported by the current pipeline");
    }
    if (config.output_dir.empty()) {
        report.add_error({"runtime", "output_dir"}, "must not be empty");
    }
}

}  // namespace citlali::config
