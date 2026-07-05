#pragma once

#include <string>

namespace citlali::pipeline {

inline bool map_filter_template_uses_fwhm(
    const std::string &template_type) {
    return template_type == "gaussian" || template_type == "airy";
}

inline double map_filter_initial_fwhm_pixels(
    double array_fwhm_arcsec, double arcsec_to_rad, double pixel_size_rad) {
    return array_fwhm_arcsec * arcsec_to_rad / pixel_size_rad;
}

template <class TemplateFwhmMap>
bool has_map_filter_template_fwhm(
    const TemplateFwhmMap &template_fwhm_rad,
    const std::string &array_name) {
    return template_fwhm_rad.find(array_name) != template_fwhm_rad.end();
}

template <class TemplateFwhmMap>
double map_filter_template_fwhm_or(
    const TemplateFwhmMap &template_fwhm_rad,
    const std::string &array_name, double fallback_value) {
    const auto it = template_fwhm_rad.find(array_name);
    return it == template_fwhm_rad.end() ? fallback_value : it->second;
}

inline bool should_calculate_map_filter_noise_products(
    bool write_filtered_maps_partial, bool run_noise_products,
    bool normalize_filtered_error) {
    return write_filtered_maps_partial &&
           (run_noise_products || normalize_filtered_error);
}

inline bool should_apply_map_filter_noise_scale(
    bool apply_empirical_noise_weights, bool normalize_filtered_error) {
    return apply_empirical_noise_weights || normalize_filtered_error;
}

template <class MapIndex, class SummarySize>
bool has_map_filter_noise_weight_summary(MapIndex map_index,
                                         SummarySize n_summary_values) {
    return map_index < n_summary_values;
}

inline bool should_destroy_filtered_fits_handle(
    bool next_map_opens_new_file, bool should_close_filtered_fits) {
    return next_map_opens_new_file && should_close_filtered_fits;
}

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void run_wiener_filter_with_log(Engine &engine, MapBuffer &map_buffer,
                                const Logger &logger,
                                const char *log_message) {
    logger->info("{}", log_message);
    engine.template run_wiener_filter<FilteredMap>(map_buffer);
}

}  // namespace citlali::pipeline
