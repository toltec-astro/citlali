#include <citlali_config/config.h>
#include <citlali_config/gitversion.h>
#include <citlali_config/default_config.h>
#include <tula/config/core.h>
#include <tula/config/yamlconfig.h>
#include <tula/formatter/container.h>
#include <tula/logging.h>

#include <cstdlib>
#include <iostream>
#include <cmath>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/cli/argument_errors.h>
#include <citlali/core/cli/argument_parsing.h>
#include <citlali/core/cli/config_loading.h>
#include <citlali/core/cli/default_config_dump.h>
#include <citlali/core/cli/exception_reporting.h>
#include <citlali/core/cli/io_coordinator.h>
#include <citlali/core/cli/process_control.h>
#include <citlali/core/cli/run_environment.h>
#include <citlali/core/cli/standard_reduction_execution.h>

using rc_t = citlali::cli::RuntimeConfig;

// @brief Run citlali reduction.
/// @param rc The runtime config.
int run(const rc_t &rc) {
    auto run_environment = citlali::cli::configure_citlali_cli_run_environment();
    auto logger = run_environment.logger;
    // set pattern for logger
    //spdlog::set_pattern("[%H:%M:%S %z] [%s] %v");

    // start the main process
    auto exitcode = citlali::cli::load_and_run_default_citlali_reduction(
        rc, logger, std::cerr);

    // re-enable default logger
    citlali::cli::restore_citlali_cli_run_environment(run_environment);

    return exitcode;
}

int main(int argc, char *argv[]) {
     // to do the dump_config, we need to make sure the output is
    // not contaminated with any logging message. Therefore this has
    // to go first
    if (citlali::cli::dump_default_config_if_requested(
            argc, argv, CITLALI_GIT_VERSION, CITLALI_BUILD_TIMESTAMP,
            citlali::citlali_default_config_content)) {
        return EXIT_SUCCESS;
    }
    // now with normal CLI interface
    return citlali::cli::run_with_exception_reporting([&]() {
        tula::logging::init();
        auto rc = citlali::cli::parse_args(argc, argv);
        SPDLOG_INFO("rc {}", rc.pformat());
        return citlali::cli::run_configured_process(
            rc, [](const auto &runtime_config) { return run(runtime_config); },
            std::cout);
    });
}
