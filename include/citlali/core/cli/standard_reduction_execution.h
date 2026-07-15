#pragma once

#include <citlali/core/cli/reduction_execution.h>
#include <citlali/core/cli/reduction_result_reporting.h>
#include <citlali/core/cli/standard_reduction_inputs.h>
#include <citlali/core/cli/standard_reduction_selection.h>
#include <citlali/core/cli/standard_reduction_types.h>
#include <citlali/core/session/reduction_session.h>

#include <cstdlib>
#include <ostream>

namespace citlali::cli {

inline citlali::session::ReductionResult processor_selection_failure_result(
    TodProcessorSelectionStatus status) {
    const auto path = reduction_type_config_key_path();
    if (status == TodProcessorSelectionStatus::missing_reduction_type) {
        return citlali::session::failed_reduction_result(
            citlali::session::ReductionStatus::processor_selection_failed,
            "processor.missing_reduction_type",
            "runtime reduction type is required", path);
    }
    return citlali::session::failed_reduction_result(
        citlali::session::ReductionStatus::processor_selection_failed,
        "processor.invalid_reduction_type",
        "runtime reduction type is invalid", path);
}

template <class KidsDataProc, class IOCoordinator, class Config,
          class ConfigFilepaths, class Logger>
citlali::session::ReductionResult run_standard_citlali_reduction_variant(
    StandardTodProcessorVariant &todproc, const IOCoordinator &co,
    Config &config, const ConfigFilepaths &config_filepaths,
    const Logger &logger) {
    return run_standard_reduction_variant_session<
        StandardBeammapTodProcessor, StandardPointingTodProcessor,
        KidsDataProc>(todproc, co, config, config_filepaths, logger);
}

template <class KidsDataProc, class IOCoordinator, class Config,
          class ConfigFilepaths, class Logger>
citlali::session::ReductionResult select_and_run_standard_citlali_reduction(
    const IOCoordinator &co, Config &config,
    const ConfigFilepaths &config_filepaths, const Logger &logger) {
    StandardTodProcessorVariant todproc;
    const auto selection_status = select_standard_citlali_tod_processor(
        todproc, config, logger);
    if (selection_status != TodProcessorSelectionStatus::ok) {
        return processor_selection_failure_result(selection_status);
    }

    return run_standard_citlali_reduction_variant<KidsDataProc>(
        todproc, co, config, config_filepaths, logger);
}

template <class KidsDataProc, class Config, class IOCoordinator,
          class Logger>
citlali::session::ReductionResult run_standard_citlali_reduction_inputs(
    StandardReductionInputs<Config, IOCoordinator> &inputs,
    const Logger &logger) {
    return select_and_run_standard_citlali_reduction<KidsDataProc>(
        inputs.coordinator, inputs.loaded_config.config,
        inputs.loaded_config.filepaths, logger);
}

template <class KidsDataProc, class IOCoordinator, class RuntimeConfig,
          class Logger>
citlali::session::ReductionResult load_and_run_standard_citlali_reduction(
    citlali::session::ReductionSession &session,
    const RuntimeConfig &runtime_config, const Logger &logger) {
    return session.run([&] {
        auto inputs = load_standard_reduction_inputs<IOCoordinator>(
            runtime_config, logger);
        return run_standard_citlali_reduction_inputs<KidsDataProc>(
            inputs, logger);
    });
}

template <class RuntimeConfig, class Logger>
citlali::session::ReductionResult load_and_run_default_citlali_session(
    const RuntimeConfig &runtime_config, const Logger &logger) {
    citlali::session::ReductionSession session;
    return load_and_run_standard_citlali_reduction<
        StandardKidsDataProcessor, StandardIOCoordinator>(
        session, runtime_config, logger);
}

template <class RuntimeConfig, class Logger>
int load_and_run_default_citlali_reduction(
    const RuntimeConfig &runtime_config, const Logger &logger,
    std::ostream &os) {
    const auto result = load_and_run_default_citlali_session(
        runtime_config, logger);
    report_reduction_result_diagnostics(result, os);
    return reduction_result_exit_code(result);
}

}  // namespace citlali::cli
