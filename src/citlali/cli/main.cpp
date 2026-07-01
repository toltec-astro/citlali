#include <citlali_config/config.h>
#include <citlali_config/gitversion.h>
#include <citlali_config/default_config.h>
#include <kids/core/kidsdata.h>
#include <kids/sweep/fitter.h>
#include <kids/timestream/solver.h>
#include <kids/toltec/toltec.h>
#include <tula/config/core.h>
#include <tula/config/yamlconfig.h>
#include <tula/formatter/container.h>
#include <tula/logging.h>

#include <cstdlib>
#include <cmath>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/cli/abort_backtrace.h>
#include <citlali/core/cli/argument_errors.h>
#include <citlali/core/cli/argument_parsing.h>
#include <citlali/core/cli/config_loading.h>
#include <citlali/core/cli/default_config_dump.h>
#include <citlali/core/cli/exception_reporting.h>
#include <citlali/core/cli/hdf5_diagnostics.h>
#include <citlali/core/cli/io_coordinator.h>
#include <citlali/core/cli/kids_data_spec.h>
#include <citlali/core/cli/process_control.h>
#include <citlali/core/cli/reduction_execution.h>
#include <citlali/core/cli/run_logging.h>
#include <citlali/core/cli/standard_reduction_selection.h>
#include <citlali/core/cli/standard_reduction_types.h>
#include <citlali/core/cli/tod_processor_selection.h>
#include <citlali/core/engine/lali.h>
#include <citlali/core/engine/pointing.h>
#include <citlali/core/engine/beammap.h>

using rc_t = citlali::cli::RuntimeConfig;

// @brief Run citlali reduction.
/// @param rc The runtime config.
int run(const rc_t &rc) {
    citlali::cli::suppress_optional_hdf5_diagnostics();

    // get current level
    auto log_level = spdlog::get_level();

    auto run_loggers = citlali::cli::configure_run_loggers(log_level);
    auto logger = run_loggers.logger;
    // set pattern for logger
    //spdlog::set_pattern("[%H:%M:%S %z] [%s] %v");

    citlali::cli::install_abort_backtrace_handler();

    citlali::cli::log_kids_data_spec(logger);

    auto loaded_config = citlali::cli::load_merged_yaml_config_files(
        rc, logger);
    auto &citlali_config = loaded_config.config;

    // set up the IO coorindator
    auto co =
        citlali::cli::make_io_coordinator_from_config<SeqIOCoordinator>(
            citlali_config);

    // set up KIDs data proc
    //auto kidsproc =
    //    KidsDataProc::from_config(citlali_config.get_config("kids"));

    // set up todproc
    using todproc_var_t = citlali::cli::StandardTodProcessorVariant;

    // declare todproc variable
    todproc_var_t todproc;

    // set todproc to variant depending on the config file reduction type
    if (!citlali::cli::select_standard_citlali_tod_processor_or_report_failure(
            todproc, citlali_config, logger, std::cerr)) {
        return EXIT_FAILURE;
    }

    // start the main process
    auto exitcode = citlali::cli::run_standard_cli_reduction_variant<
        citlali::cli::StandardBeammapTodProcessor,
        citlali::cli::StandardPointingTodProcessor, KidsDataProc>(
        todproc, co, citlali_config, loaded_config.filepaths, logger,
        std::cerr);

    // re-enable default logger
    citlali::cli::restore_default_sink_level(run_loggers, log_level);

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
