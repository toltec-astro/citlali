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

}  // namespace citlali::pipeline
