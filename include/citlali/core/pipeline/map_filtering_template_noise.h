#pragma once

// Included by map_filtering.h inside namespace citlali::pipeline.

template <class NoiseCount>
auto map_filter_progress_stride(NoiseCount n_noise) {
    return n_noise / 100;
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

template <class TemplateFwhmMap, class Logger>
double map_filter_template_fwhm_or_exit(
    const std::string &template_type,
    const TemplateFwhmMap &template_fwhm_rad,
    const std::string &array_name, const Logger &logger) {
    double template_fwhm_rad_value = 0.0;
    const bool template_uses_fwhm =
        map_filter_template_uses_fwhm(template_type);
    if (!template_uses_fwhm) {
        return template_fwhm_rad_value;
    }

    const bool has_template_fwhm =
        has_map_filter_template_fwhm(template_fwhm_rad, array_name);
    if (!has_template_fwhm) {
        logger->error("missing Wiener template_fwhm_rad for array {}",
                      array_name);
        std::exit(EXIT_FAILURE);
    }

    return map_filter_template_fwhm_or(
        template_fwhm_rad, array_name, template_fwhm_rad_value);
}

template <class WienerFilter, class MapBuffer, class Apt, class MapIndex,
          class MapNumber, class MapCount, class Logger>
void build_map_filter_template(WienerFilter &wiener_filter,
                               MapBuffer &map_buffer, const Apt &apt,
                               MapIndex map_index, MapNumber map_number,
                               MapCount n_maps,
                               const std::string &array_name,
                               const char *map_label,
                               const Logger &logger) {
    logger->info(
        "building Wiener template for {} map {}/{} (array={})",
        map_label, map_number, n_maps, array_name);
    const double template_fwhm_rad =
        map_filter_template_fwhm_or_exit(
            wiener_filter.template_type,
            wiener_filter.template_fwhm_rad, array_name, logger);
    wiener_filter.make_template(
        map_buffer, apt, template_fwhm_rad, map_index);
    logger->info(
        "Wiener template ready for {} map {}/{} (array={})",
        map_label, map_number, n_maps, array_name);
}

template <class WienerFilter, class MapBuffer, class MapIndex,
          class MapNumber, class MapCount, class Logger>
void filter_map_filter_signal_map(WienerFilter &wiener_filter,
                                  MapBuffer &map_buffer,
                                  MapIndex map_index,
                                  MapNumber map_number,
                                  MapCount n_maps,
                                  const std::string &array_name,
                                  const char *map_label,
                                  const Logger &logger) {
    logger->info(
        "running Wiener filter core for {} map {}/{} (array={})",
        map_label, map_number, n_maps, array_name);
    wiener_filter.filter_maps(map_buffer, map_index);
    logger->info("map filtering complete for {} map {}/{}",
                 map_label, map_number, n_maps);
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

template <class MapBuffer, class MapIndex, class Logger>
void log_map_filter_noise_weight_summary_if_present(
    const MapBuffer &map_buffer, MapIndex map_index,
    const Logger &logger) {
    const bool has_noise_weight_summary =
        has_map_filter_noise_weight_summary(
            map_index, map_buffer.noise_weight_median_ratio.size());
    if (!has_noise_weight_summary) {
        return;
    }

    logger->info(
        "noise products: median(w_formal*var)={:.4g} "
        "scale={:.4g} noise_s2n_sigma={:.4g}",
        map_buffer.noise_weight_median_ratio(map_index),
        map_buffer.noise_weight_scale(map_index),
        map_buffer.noise_s2n_sigma(map_index));
}

template <class MapBuffer, class MapIndex, class MapNumber,
          class MapCount, class Logger>
void calculate_map_filter_noise_products_if_needed(
    MapBuffer &map_buffer, MapIndex map_index, MapNumber map_number,
    MapCount n_maps, bool write_filtered_maps_partial,
    bool run_noise_products, bool normalize_filtered_error,
    bool apply_empirical_noise_weights, const char *map_label,
    const Logger &logger) {
    const bool should_calculate_noise_products =
        should_calculate_map_filter_noise_products(
            write_filtered_maps_partial, run_noise_products,
            normalize_filtered_error);
    if (!should_calculate_noise_products) {
        return;
    }

    const bool apply_empirical_noise_scale =
        should_apply_map_filter_noise_scale(
            apply_empirical_noise_weights, normalize_filtered_error);
    logger->info(
        "calculating empirical noise products for {} map {}/{}",
        map_label, map_number, n_maps);
    map_buffer.calc_noise_products(map_index, apply_empirical_noise_scale);
    log_map_filter_noise_weight_summary_if_present(
        map_buffer, map_index, logger);
    map_buffer.calc_median_err();
    map_buffer.calc_median_rms();
}

