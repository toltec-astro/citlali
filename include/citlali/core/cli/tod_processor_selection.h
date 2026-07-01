#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace citlali::cli {

inline auto reduction_type_config_key() {
    return std::tuple{"runtime", "reduction_type"};
}

inline std::vector<std::string> reduction_type_config_key_path() {
    return {"runtime", "reduction_type"};
}

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
