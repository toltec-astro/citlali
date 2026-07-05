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

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void run_wiener_filter_with_log(Engine &engine, MapBuffer &map_buffer,
                                const Logger &logger,
                                const char *log_message) {
    logger->info("{}", log_message);
    engine.template run_wiener_filter<FilteredMap>(map_buffer);
}

}  // namespace citlali::pipeline
