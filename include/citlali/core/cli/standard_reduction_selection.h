#pragma once

#include <citlali/core/cli/standard_reduction_types.h>

#include <ostream>

namespace citlali::cli {

template <class Config, class Logger>
bool select_standard_citlali_tod_processor_or_report_failure(
    StandardTodProcessorVariant &todproc, Config &config,
    const Logger &logger, std::ostream &os) {
    return select_tod_processor_variant_or_report_failure<
        StandardScienceTodProcessor, StandardPointingTodProcessor,
        StandardBeammapTodProcessor>(todproc, config, logger, os);
}

}  // namespace citlali::cli
