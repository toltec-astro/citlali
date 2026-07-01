#pragma once

#include <citlali/core/cli/argument_errors.h>
#include <citlali/core/cli/config_loading.h>
#include <tula/logging.h>

#include <cstdlib>
#include <ostream>

namespace citlali::cli {

template <class RuntimeConfig, class RunReduction>
int run_configured_process(const RuntimeConfig &runtime_config,
                           RunReduction &&run_reduction,
                           std::ostream &invalid_argument_stream) {
    if (has_config_files(runtime_config)) {
        tula::logging::scoped_timeit TULA_X{"Citlali Process"};
        return run_reduction(runtime_config);
    }

    report_missing_config_file_argument(invalid_argument_stream);
    return EXIT_FAILURE;
}

}  // namespace citlali::cli
