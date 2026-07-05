#pragma once

#include <cstdlib>
#include <string>

#include <Eigen/Core>
#include <tula/logging.h>

#include <citlali/core/mapmaking/map.h>

namespace citlali::pipeline {

template <class FitsVector>
struct MapFilterOutputTargets {
    FitsVector *filtered_fits_io;
    FitsVector *filtered_noise_fits_io;
    const char *map_label;
};

template <mapmaking::MapType map_t, class FitsVector>
MapFilterOutputTargets<FitsVector> map_filter_output_targets(
    FitsVector &filtered_fits_io_vec,
    FitsVector &filtered_noise_fits_io_vec,
    FitsVector &filtered_coadd_fits_io_vec,
    FitsVector &filtered_coadd_noise_fits_io_vec) {
    if constexpr (map_t == mapmaking::FilteredObs) {
        return {
            &filtered_fits_io_vec,
            &filtered_noise_fits_io_vec,
            "filtered obs maps"};
    }
    else {
        return {
            &filtered_coadd_fits_io_vec,
            &filtered_coadd_noise_fits_io_vec,
            "filtered coadded maps"};
    }
}

inline bool map_filter_template_uses_fwhm(
    const std::string &template_type) {
    return template_type == "gaussian" || template_type == "airy";
}

inline double map_filter_initial_fwhm_pixels(
    double array_fwhm_arcsec, double arcsec_to_rad, double pixel_size_rad) {
    return array_fwhm_arcsec * arcsec_to_rad / pixel_size_rad;
}

template <class MapIndex>
auto map_filter_display_number(MapIndex map_index) {
    return map_index + 1;
}

template <class NoiseContainer, class FitsContainer>
bool has_map_filter_noise_fits(const NoiseContainer &noise,
                               const FitsContainer &noise_fits) {
    return !noise.empty() && !noise_fits.empty();
}

template <class FitsVector, class MapBufferPtr, class Logger, class AddPhdu>
void prepare_map_filter_fits_headers(
    FitsVector *filtered_fits_io, FitsVector *filtered_noise_fits_io,
    MapBufferPtr map_buffer_ptr, const char *map_label,
    const Logger &logger, const AddPhdu &add_phdu) {
    const auto n_filtered_fits =
        static_cast<Eigen::Index>(filtered_fits_io->size());
    logger->info("preparing {} FITS headers ({} files)", map_label,
                 n_filtered_fits);
    const bool has_filtered_noise_fits =
        has_map_filter_noise_fits(
            map_buffer_ptr->noise, *filtered_noise_fits_io);
    for (Eigen::Index i = 0; i < n_filtered_fits; ++i) {
        add_phdu(filtered_fits_io, map_buffer_ptr, i);
        if (has_filtered_noise_fits) {
            add_phdu(filtered_noise_fits_io, map_buffer_ptr, i);
        }
    }
}

template <class FitsVector, class Logger>
void finalize_map_filter_fits_outputs(
    FitsVector *filtered_fits_io, FitsVector *filtered_noise_fits_io,
    const char *map_label, const Logger &logger) {
    logger->info("finalizing {} FITS handles", map_label);
    filtered_fits_io->clear();
    filtered_noise_fits_io->clear();
    logger->info("finished finalizing {} FITS handles", map_label);
}

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

inline bool should_destroy_filtered_fits_handle(
    bool next_map_opens_new_file, bool should_close_filtered_fits) {
    return next_map_opens_new_file && should_close_filtered_fits;
}

inline bool is_final_map_filter_polarization_stokes(
    const std::string &stokes_param) {
    return stokes_param == "U";
}

template <class MapIndex, class MapCount>
bool has_next_map_filter_output(MapIndex map_index, MapCount n_maps) {
    return map_index < n_maps - 1;
}

template <class MapIndex>
bool next_map_filter_output_opens_new_file(MapIndex current_map_index,
                                           MapIndex next_map_index) {
    return next_map_index > current_map_index;
}

template <class WienerFilter, class MapBuffer, class MapIndex,
          class MapNumber, class NoiseCount, class MapCount, class Logger>
void filter_map_filter_noise_maps(
    WienerFilter &wiener_filter, MapBuffer &map_buffer, MapIndex map_index,
    MapNumber map_number, NoiseCount n_wiener_noise_maps,
    const char *map_label, MapCount n_maps, const Logger &logger) {
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
    logger->info("filtering noise for {} map {}/{} (n_noise={})",
                 map_label, map_number, n_maps, n_wiener_noise_maps);
    #pragma omp parallel for schedule(dynamic)
    for (Eigen::Index j = 0; j < n_wiener_noise_maps; ++j) {
        wiener_filter.filter_noise_threadsafe(map_buffer, map_index, j);
    }
    logger->info("noise filtering complete for {} map {}/{}",
                 map_label, map_number, n_maps);
#else
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100,
        "filtering noise");
    const auto noise_progress_stride =
        map_filter_progress_stride(n_wiener_noise_maps);

    for (Eigen::Index j = 0; j < n_wiener_noise_maps; ++j) {
        wiener_filter.filter_noise(map_buffer, map_index, j);
        pb.count(n_wiener_noise_maps, noise_progress_stride);
    }
    logger->info("noise filtering complete for {} map {}/{}",
                 map_label, map_number, n_maps);
#endif
}

template <class Polarization, class MapsToStokes, class MapIndex>
bool should_close_current_map_filter_fits(
    bool run_polarization, Polarization &polarization,
    const MapsToStokes &maps_to_stokes, MapIndex map_index) {
    if (!run_polarization) {
        return true;
    }

    const auto &current_stokes_param =
        polarization.stokes_params[maps_to_stokes(map_index)];
    return is_final_map_filter_polarization_stokes(current_stokes_param);
}

template <class FitsVector, class MapIndex, class MapCount,
          class ArraysToMaps, class Logger>
void destroy_map_filter_fits_if_ready(
    FitsVector *filtered_fits_io, MapIndex map_i, MapIndex map_index,
    MapCount n_maps, const std::string &filtered_map_path,
    bool should_close_filtered_fits, const ArraysToMaps &arrays_to_maps,
    const Logger &logger) {
    const bool has_next_map =
        has_next_map_filter_output(map_i, n_maps);
    if (!has_next_map) {
        return;
    }

    const auto next_map_index = arrays_to_maps(map_i + 1);
    const bool next_map_opens_new_file =
        next_map_filter_output_opens_new_file(
            map_index, next_map_index);
    const bool should_destroy_filtered_fits =
        should_destroy_filtered_fits_handle(
            next_map_opens_new_file, should_close_filtered_fits);
    if (!should_destroy_filtered_fits) {
        return;
    }

    logger->info("closing FITS handle for {}", filtered_map_path);
    filtered_fits_io->at(map_index).pfits->destroy();
    logger->info("closed FITS handle for {}", filtered_map_path);
}

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void run_wiener_filter_with_log(Engine &engine, MapBuffer &map_buffer,
                                const Logger &logger,
                                const char *log_message) {
    logger->info("{}", log_message);
    engine.template run_wiener_filter<FilteredMap>(map_buffer);
}

}  // namespace citlali::pipeline
