#pragma once

#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

constexpr int missing_source_location() {
    return -99;
}

template <class SourceCounts>
Eigen::Index count_map_sources(const SourceCounts &source_counts) {
    Eigen::Index n_sources = 0;
    for (const auto &sources : source_counts) {
        n_sources += sources;
    }
    return n_sources;
}

inline double source_fit_initial_fwhm_pixels(
    double array_fwhm_arcsec, double arcsec_to_rad, double pixel_size_rad) {
    return array_fwhm_arcsec * arcsec_to_rad / pixel_size_rad;
}

inline std::vector<std::string> source_table_header() {
    return {
        "array",
        "amp",
        "amp_err",
        "x_t",
        "x_t_err",
        "y_t",
        "y_t_err",
        "a_fwhm",
        "a_fwhm_err",
        "b_fwhm",
        "b_fwhm_err",
        "angle",
        "angle_err",
        "sig2noise"};
}

inline std::string source_position_units(const std::string &pixel_axes) {
    return pixel_axes == "radec" ? "deg" : "arcsec";
}

inline std::map<std::string, std::string> source_table_units(
    const std::string &signal_unit, const std::string &position_units) {
    return {
        {"array", "N/A"},
        {"amp", signal_unit},
        {"amp_err", signal_unit},
        {"x_t", position_units},
        {"x_t_err", position_units},
        {"y_t", position_units},
        {"y_t_err", position_units},
        {"a_fwhm", "arcsec"},
        {"a_fwhm_err", "arcsec"},
        {"b_fwhm", "arcsec"},
        {"b_fwhm_err", "arcsec"},
        {"angle", "rad"},
        {"angle_err", "rad"},
        {"sig2noise", "N/A"}};
}

inline Eigen::Index source_table_column_count(Eigen::Index n_params) {
    return 2 * n_params + 2;
}

inline Eigen::Index source_table_sig2noise_column(Eigen::Index n_params) {
    return 2 * n_params + 1;
}

template <class SourceCount>
std::vector<int> source_index_vector(SourceCount n_sources) {
    std::vector<int> source_indices(static_cast<std::size_t>(n_sources));
    std::iota(source_indices.begin(), source_indices.end(), 0);
    return source_indices;
}

template <auto MapType, class Engine, class MapBuffer, class Logger>
void find_map_sources_with_log(Engine &engine, MapBuffer &map_buffer,
                               const Logger &logger,
                               const char *log_message) {
    logger->info("{}", log_message);
    engine.template find_sources<MapType>(map_buffer);
}

template <auto MapType, class Engine, class MapBuffer, class Logger>
void find_map_sources_if_needed(Engine &engine, MapBuffer &map_buffer,
                                const Logger &logger, bool should_find,
                                const char *log_message) {
    if (should_find) {
        find_map_sources_with_log<MapType>(
            engine, map_buffer, logger, log_message);
    }
}

}  // namespace citlali::pipeline
