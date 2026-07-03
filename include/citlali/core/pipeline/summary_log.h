#pragma once

#include <ostream>
#include <string>
#include <string_view>

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

inline void write_chunk_identity_summary(std::ostream &stream,
                                         std::string_view reduction_type,
                                         std::string_view tod_type,
                                         std::string_view tod_unit,
                                         std::string_view chunk_type) {
    stream << "-Reduction type: " << reduction_type << "\n";
    stream << "-TOD type: " << tod_type << "\n";
    stream << "-TOD unit: " << tod_unit << "\n";
    stream << "-TOD chunk type: " << chunk_type << "\n";
}

inline void write_map_identity_summary(std::ostream &stream,
                                       const std::string &reduction_type,
                                       const std::string &map_type,
                                       const std::string &map_grouping,
                                       long long n_rows, long long n_cols,
                                       long long n_maps,
                                       const std::string &signal_unit) {
    stream << "-Reduction type: " << reduction_type << "\n";
    stream << "-Map type: " << map_type << "\n";
    stream << "-Map grouping: " << map_grouping << "\n";
    stream << "-Rows: " << n_rows << "\n";
    stream << "-Cols: " << n_cols << "\n";
    stream << "-Number of maps: " << n_maps << "\n";
    stream << "-Signal map unit: " << signal_unit << "\n";
    stream << "-Weight map unit: "
           << "1/(" + signal_unit + ")^2" << "\n";
}

template <class Status>
void write_chunk_processing_status_summary(std::ostream &stream,
                                           const Status &status) {
    stream << "-Calibrated: " << status.calibrated << "\n";
    stream << "-Extinction Corrected: " << status.extinction_corrected << "\n";
    stream << "-Demodulated: " << status.demodulated << "\n";
    stream << "-Kernel Generated: " << status.kernel_generated << "\n";
    stream << "-Despiked: " << status.despiked << "\n";
    stream << "-TOD filtered: " << status.tod_filtered << "\n";
    stream << "-Downsampled: " << status.downsampled << "\n";
    stream << "-Cleaned: " << status.cleaned << "\n";
}

}  // namespace citlali::pipeline
