#pragma once

#include <string>

namespace citlali::cli {

template <class TodProcVariant, class ScienceTodProc, class PointingTodProc,
          class BeammapTodProc, class Config, class Logger>
bool emplace_tod_processor_for_reduction_type(
    TodProcVariant &todproc, const std::string &reduction_type,
    Config &config, const Logger &logger) {
    if (reduction_type == "science") {
        logger->info("reducing in science mode");
        todproc.template emplace<ScienceTodProc>(
            ScienceTodProc::from_config(config));
        return true;
    }

    if (reduction_type == "pointing") {
        logger->info("reducing in pointing mode");
        todproc.template emplace<PointingTodProc>(
            PointingTodProc::from_config(config));
        return true;
    }

    if (reduction_type == "beammap") {
        logger->info("reducing in beammap mode");
        todproc.template emplace<BeammapTodProc>(
            BeammapTodProc::from_config(config));
        return true;
    }

    return false;
}

}  // namespace citlali::cli
