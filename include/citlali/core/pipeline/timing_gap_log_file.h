#pragma once

#include <fstream>
#include <string>

namespace citlali::pipeline {

inline std::string gaps_log_filepath(const std::string &obsnum_dir_name) {
    return obsnum_dir_name + "/logs/gaps.log";
}

inline void write_timing_gaps_log_header(std::ofstream &stream) {
    stream << "Summary of timing gaps\n";
}

template <class Key, class Value, class Logger>
void log_timing_gap_entry(const Key &key, const Value &value,
                          const Logger &logger) {
    logger->debug("{} gaps: {}", key, value);
}

template <class Key, class Value>
void write_timing_gap_entry(std::ofstream &stream, const Key &key,
                            const Value &value) {
    stream << "-" + key + " gaps: " << value << "\n";
}

template <class Engine, class Logger>
void write_timing_gaps_log_file(const Engine &engine, const Logger &logger) {
    logger->debug("writing gaps.log file");
    std::ofstream stream;
    stream.open(gaps_log_filepath(engine.obsnum_dir_name));
    write_timing_gaps_log_header(stream);
    for (auto const &[key, value] : engine.gaps) {
        log_timing_gap_entry(key, value, logger);
        write_timing_gap_entry(stream, key, value);
    }
    stream.close();
}

}  // namespace citlali::pipeline
