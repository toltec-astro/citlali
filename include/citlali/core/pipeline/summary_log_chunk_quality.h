#pragma once

// Included by summary_log.h inside namespace citlali::pipeline.

inline void write_chunk_scan_shape_summary(std::ostream &stream,
                                           long long n_samples,
                                           long long n_detectors) {
    stream << "-Scan length: " << n_samples << "\n";
    stream << "-Number of detectors: " << n_detectors << "\n";
}

inline long long chunk_flagged_detector_count(long long n_low_weight,
                                              long long n_high_weight,
                                              long long n_apt_flagged) {
    return n_low_weight + n_high_weight + n_apt_flagged;
}

inline float chunk_flagged_detector_percent(long long n_flagged,
                                            long long n_detectors) {
    return 100 * static_cast<float>(n_flagged) /
           static_cast<float>(n_detectors);
}

inline void write_chunk_detector_flag_summary(std::ostream &stream,
                                              long long n_apt_flagged,
                                              long long n_low_weight,
                                              long long n_high_weight,
                                              long long n_detectors) {
    stream << "-Number of detectors flagged in APT table: "
           << n_apt_flagged << "\n";
    stream << "-Number of detectors flagged below weight limit: "
           << n_low_weight << "\n";
    stream << "-Number of detectors flagged above weight limit: "
           << n_high_weight << "\n";
    const auto n_flagged = chunk_flagged_detector_count(
        n_low_weight, n_high_weight, n_apt_flagged);
    stream << "-Number of detectors flagged: " << n_flagged << " ("
           << chunk_flagged_detector_percent(n_flagged, n_detectors)
           << "%)\n";
}

template <class Matrix>
void write_chunk_nonfinite_summary(std::ostream &stream,
                                   const Matrix &data) {
    stream << "-NaNs found: " << data.array().isNaN().count() << "\n";
    stream << "-Infs found: " << data.array().isInf().count() << "\n";
}

inline void write_chunk_data_stat_summary(
    std::ostream &stream, double min_value, double max_value,
    double mean_value, double median_value, double stddev_value,
    std::string_view unit) {
    stream << "-Data min: " << min_value << " " << unit << "\n";
    stream << "-Data max: " << max_value << " " << unit << "\n";
    stream << "-Data mean: " << mean_value << " " << unit << "\n";
    stream << "-Data median: " << median_value << " " << unit << "\n";
    stream << "-Data stddev: " << stddev_value << " " << unit << "\n";
}

template <class Kernel>
void write_chunk_kernel_summary_if_generated(std::ostream &stream,
                                             bool kernel_generated,
                                             const Kernel &kernel,
                                             std::string_view unit) {
    if (kernel_generated) {
        stream << "-Kernel max: " << kernel.data.maxCoeff() << " "
               << unit << "\n";
    }
}

