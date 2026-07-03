#pragma once

#include <ostream>
#include <string>

namespace citlali::pipeline {

inline void write_pipeline_version_summary(std::ostream &stream,
                                           const std::string &citlali_version,
                                           const std::string &kids_version) {
    stream << "-Citlali version: " << citlali_version << "\n";
    stream << "-Kidscpp version: " << kids_version << "\n";
}

inline void write_chunk_time_summary(std::ostream &stream,
                                     const std::string &creation_time,
                                     const std::string &write_time) {
    stream << "-Time of time chunk creation: " << creation_time << "\n";
    stream << "-Time of file writing: " << write_time << "\n";
}

inline void write_file_time_summary(std::ostream &stream,
                                    const std::string &write_time) {
    stream << "-Time of file writing: " << write_time << "\n";
}

}  // namespace citlali::pipeline
