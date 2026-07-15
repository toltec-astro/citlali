#pragma once

#include <citlali/core/cli/standard_reduction_types.h>

#include <ostream>

namespace citlali::cli {

template <class Config, class Logger>
TodProcessorSelectionStatus select_standard_citlali_tod_processor(
    StandardTodProcessorVariant &todproc, Config &config,
    const Logger &logger) {
    return select_tod_processor_from_config<
        StandardTodProcessorVariant, StandardScienceTodProcessor,
        StandardPointingTodProcessor, StandardBeammapTodProcessor>(
        todproc, config, logger);
}

template <class Config, class Logger>
bool select_standard_citlali_tod_processor_or_report_failure(
    StandardTodProcessorVariant &todproc, Config &config,
    const Logger &logger, std::ostream &os) {
    return !report_tod_processor_selection_failure(
        select_standard_citlali_tod_processor(
            todproc, config, logger),
        os);
}

}  // namespace citlali::cli
