#pragma once

#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

constexpr int missing_source_location() {
    return -99;
}

template <class SourceCounts, class SourceLocations>
void append_missing_source_location(SourceCounts &source_counts,
                                    SourceLocations &row_source_locations,
                                    SourceLocations &col_source_locations) {
    source_counts.push_back(0);
    row_source_locations.push_back(Eigen::VectorXi::Ones(1));
    col_source_locations.push_back(Eigen::VectorXi::Ones(1));

    row_source_locations.back() *= missing_source_location();
    col_source_locations.back() *= missing_source_location();
}

inline bool has_sources(Eigen::Index n_sources) {
    return n_sources > 0;
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

inline double source_fit_pixel_to_arcsec(double rad_to_arcsec,
                                         double pixel_size_rad) {
    return rad_to_arcsec * pixel_size_rad;
}

inline double source_fit_fwhm_to_arcsec(double rad_to_arcsec,
                                        double std_to_fwhm,
                                        double pixel_size_rad) {
    return rad_to_arcsec * std_to_fwhm * pixel_size_rad;
}

template <class Params, class PErrors>
void rescale_source_fit_pixel_units(Params &params, PErrors &perrors,
                                    Eigen::Index n_rows, Eigen::Index n_cols,
                                    double pixel_to_arcsec,
                                    double source_fwhm_to_arcsec) {
    params(1) = pixel_to_arcsec * (params(1) - (n_cols - 1) / 2.0);
    params(2) = pixel_to_arcsec * (params(2) - (n_rows - 1) / 2.0);
    params(3) = source_fwhm_to_arcsec * params(3);
    params(4) = source_fwhm_to_arcsec * params(4);

    perrors(1) = pixel_to_arcsec * perrors(1);
    perrors(2) = pixel_to_arcsec * perrors(2);
    perrors(3) = source_fwhm_to_arcsec * perrors(3);
    perrors(4) = source_fwhm_to_arcsec * perrors(4);
}

inline bool source_fit_uses_radec_projection(const std::string &pixel_axes) {
    return pixel_axes == "radec";
}

template <class Params, class PErrors>
void rescale_source_fit_radec_errors(Params &params, PErrors &perrors,
                                     double ra_deg, double dec_deg,
                                     double arcsec_to_deg) {
    params(1) = ra_deg;
    params(2) = dec_deg;

    perrors(1) = perrors(1) * arcsec_to_deg;
    perrors(2) = perrors(2) * arcsec_to_deg;
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

inline std::string source_obsnum_meta_key(Eigen::Index obsnum_index) {
    return "obsnum" + std::to_string(obsnum_index);
}

inline std::string source_units_meta_entry(const std::string &unit) {
    return "units: " + unit;
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

inline Eigen::Index source_table_param_column(Eigen::Index source_param_index) {
    return 1 + 2 * source_param_index;
}

inline Eigen::Index source_table_error_column(Eigen::Index source_param_index) {
    return source_table_param_column(source_param_index) + 1;
}

inline float source_signal_to_noise(double source_amplitude,
                                    double map_std_dev) {
    return static_cast<float>(source_amplitude / map_std_dev);
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
