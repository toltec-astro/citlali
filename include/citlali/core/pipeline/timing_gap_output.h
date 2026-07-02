#pragma once

#include <fstream>
#include <string>

namespace citlali::pipeline {

inline std::string gaps_log_filepath(const std::string &obsnum_dir_name) {
    return obsnum_dir_name + "/logs/gaps.log";
}

template <class Engine>
bool has_timing_gaps(const Engine &engine) {
    return engine.gaps.size() > 0;
}

template <class Engine, class Logger>
void record_timing_gaps_if_needed(const Engine &engine, const Logger &logger) {
    if (engine.gaps.size() > 0) {
        logger->warn("gaps found in obnsum {} data file timing!",
                     engine.obsnum);
        if (engine.verbose_mode) {
            logger->debug("writing gaps.log file");
            std::ofstream f;
            f.open(gaps_log_filepath(engine.obsnum_dir_name));
            f << "Summary of timing gaps\n";
            for (auto const &[key, val] : engine.gaps) {
                logger->debug("{} gaps: {}", key, val);
                f << "-" + key + " gaps: " << val << "\n";
            }
            f.close();
        }
    }
}

template <class Engine, class Logger>
void record_reduction_observation_timing_gaps_if_needed(
    const Engine &engine, const Logger &logger) {
    record_timing_gaps_if_needed(engine, logger);
}

}  // namespace citlali::pipeline
