#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>

#include <Eigen/Core>
#include <tula/logging.h>

#include <citlali/core/mapmaking/edge_guard_state.h>
#include <citlali/core/mapmaking/map.h>

namespace citlali::pipeline {

template <class FitsVector>
struct MapFilterOutputTargets {
    FitsVector *filtered_fits_io;
    FitsVector *filtered_noise_fits_io;
    const char *map_label;
};

template <class MapsToArrays, class MapsToStokes, class ArraysToMaps,
          class WriteMaps>
struct MapFilterCallbacks {
    MapsToArrays maps_to_arrays;
    MapsToStokes maps_to_stokes;
    ArraysToMaps arrays_to_maps;
    WriteMaps write_maps;
};

template <class MapsToArrays, class MapsToStokes, class ArraysToMaps,
          class WriteMaps>
MapFilterCallbacks<MapsToArrays, MapsToStokes, ArraysToMaps, WriteMaps>
make_map_filter_callbacks(const MapsToArrays &maps_to_arrays,
                          const MapsToStokes &maps_to_stokes,
                          const ArraysToMaps &arrays_to_maps,
                          const WriteMaps &write_maps) {
    return {maps_to_arrays, maps_to_stokes, arrays_to_maps, write_maps};
}

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

template <class WienerFilter, class ArrayFwhm, class ArrayIndex,
          class PixelSize>
void initialize_map_filter_fwhm(WienerFilter &wiener_filter,
                                ArrayFwhm &array_fwhm_arcsec,
                                ArrayIndex array_index,
                                double arcsec_to_rad,
                                PixelSize pixel_size_rad) {
    wiener_filter.init_fwhm =
        map_filter_initial_fwhm_pixels(
            array_fwhm_arcsec[array_index], arcsec_to_rad,
            pixel_size_rad);
}

template <class MapIndex>
auto map_filter_display_number(MapIndex map_index) {
    return map_index + 1;
}

template <class ArrayNames, class ArrayIndex>
decltype(auto) map_filter_array_name(ArrayNames &array_names,
                                     ArrayIndex array_index) {
    return array_names[array_index];
}

template <class MapNumber, class MapCount, class Logger>
void log_map_filter_map_start(const char *map_label,
                              MapNumber map_number, MapCount n_maps,
                              const std::string &array_name,
                              const Logger &logger) {
    logger->info("starting {} map {}/{} (array={})",
                 map_label, map_number, n_maps, array_name);
}

template <class MapNumber, class MapCount, class Logger>
void log_map_filter_map_completed(const char *map_label,
                                  MapNumber map_number, MapCount n_maps,
                                  const Logger &logger) {
    logger->info("completed {} map {}/{}", map_label, map_number, n_maps);
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

template <class MapBuffer>
void reset_map_filter_edge_guard_storage(MapBuffer &map_buffer) {
    const auto n_maps =
        static_cast<std::size_t>(map_buffer.signal.size());
    mapmaking::reset_edge_guard_storage(map_buffer, n_maps);
}

template <mapmaking::MapType map_t, class FitsVector,
          class MapBufferPtr, class Logger, class AddPhdu>
MapFilterOutputTargets<FitsVector> prepare_map_filter_outputs(
    FitsVector &filtered_fits_io_vec,
    FitsVector &filtered_noise_fits_io_vec,
    FitsVector &filtered_coadd_fits_io_vec,
    FitsVector &filtered_coadd_noise_fits_io_vec,
    MapBufferPtr map_buffer_ptr, const Logger &logger,
    const AddPhdu &add_phdu) {
    auto filter_outputs =
        map_filter_output_targets<map_t>(
            filtered_fits_io_vec, filtered_noise_fits_io_vec,
            filtered_coadd_fits_io_vec, filtered_coadd_noise_fits_io_vec);

    prepare_map_filter_fits_headers(
        filter_outputs.filtered_fits_io,
        filter_outputs.filtered_noise_fits_io,
        map_buffer_ptr, filter_outputs.map_label, logger, add_phdu);

    return filter_outputs;
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

template <class FitsVector, class Logger>
void finalize_map_filter_fits_outputs_if_needed(
    bool write_filtered_maps_partial,
    FitsVector *filtered_fits_io, FitsVector *filtered_noise_fits_io,
    const char *map_label, const Logger &logger) {
    if (write_filtered_maps_partial) {
        finalize_map_filter_fits_outputs(
            filtered_fits_io, filtered_noise_fits_io, map_label, logger);
    }
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

template <class FitsVector, class MapBufferPtr, class MapIndex,
          class MapNumber, class MapCount, class Polarization,
          class MapsToStokes, class ArraysToMaps, class WriteMaps,
          class Logger>
void write_map_filter_output(
    FitsVector *filtered_fits_io, FitsVector *filtered_noise_fits_io,
    MapBufferPtr map_buffer_ptr, MapIndex map_i, MapNumber map_number,
    MapIndex map_index, MapCount n_maps, const char *map_label,
    bool run_polarization, Polarization &polarization,
    const MapsToStokes &maps_to_stokes, const ArraysToMaps &arrays_to_maps,
    const WriteMaps &write_maps, const Logger &logger) {
    logger->info("writing {} map {}/{} to disk",
                 map_label, map_number, n_maps);
    write_maps(filtered_fits_io, filtered_noise_fits_io, map_buffer_ptr,
               map_i);

    const auto &filtered_map_path =
        filtered_fits_io->at(map_index).filepath;
    logger->info("file has been written to:");
    logger->info("{}.fits", filtered_map_path);

    const bool should_close_filtered_fits =
        should_close_current_map_filter_fits(
            run_polarization, polarization, maps_to_stokes, map_i);
    destroy_map_filter_fits_if_ready(
        filtered_fits_io, map_i, map_index, n_maps, filtered_map_path,
        should_close_filtered_fits, arrays_to_maps, logger);
}

template <class WienerFilter, class MapBuffer, class MapCount,
          class ArrayNames, class ArrayFwhm, class Apt,
          class FitsVector, class MapBufferPtr, class Polarization,
          class Callbacks, class Logger>
void run_map_filter_loop(
    WienerFilter &wiener_filter, MapBuffer &map_buffer, MapCount n_maps,
    const char *map_label, ArrayNames &array_names, ArrayFwhm &array_fwhm_arcsec,
    double arcsec_to_rad, const Apt &apt, bool run_noise,
    bool write_filtered_maps_partial, bool run_noise_products,
    bool apply_empirical_noise_weights,
    FitsVector *filtered_fits_io, FitsVector *filtered_noise_fits_io,
    MapBufferPtr map_buffer_ptr, bool run_polarization,
    Polarization &polarization, const Callbacks &callbacks,
    const Logger &logger) {
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const auto map_number = map_filter_display_number(i);
        const auto array = callbacks.maps_to_arrays(i);
        const auto &array_name =
            map_filter_array_name(array_names, array);
        const auto map_index = callbacks.arrays_to_maps(i);

        log_map_filter_map_start(
            map_label, map_number, n_maps, array_name, logger);
        initialize_map_filter_fwhm(
            wiener_filter, array_fwhm_arcsec, array, arcsec_to_rad,
            map_buffer.pixel_size_rad);
        build_map_filter_template(
            wiener_filter, map_buffer, apt, i, map_number, n_maps,
            array_name, map_label, logger);
        filter_map_filter_signal_map(
            wiener_filter, map_buffer, i, map_number, n_maps, array_name,
            map_label, logger);

        const auto n_wiener_noise_maps = map_buffer.n_noise;
        if (run_noise) {
            filter_map_filter_noise_maps(
                wiener_filter, map_buffer, i, map_number,
                n_wiener_noise_maps, map_label, n_maps, logger);
            calculate_map_filter_noise_products_if_needed(
                map_buffer, i, map_number, n_maps,
                write_filtered_maps_partial, run_noise_products,
                wiener_filter.normalize_error, apply_empirical_noise_weights,
                map_label, logger);
        }

        if (write_filtered_maps_partial) {
            write_map_filter_output(
                filtered_fits_io, filtered_noise_fits_io, map_buffer_ptr,
                i, map_number, map_index, n_maps, map_label,
                run_polarization, polarization, callbacks.maps_to_stokes,
                callbacks.arrays_to_maps, callbacks.write_maps, logger);
        }

        log_map_filter_map_completed(map_label, map_number, n_maps, logger);
    }
}

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void run_wiener_filter_with_log(Engine &engine, MapBuffer &map_buffer,
                                const Logger &logger,
                                const char *log_message) {
    logger->info("{}", log_message);
    engine.template run_wiener_filter<FilteredMap>(map_buffer);
}

}  // namespace citlali::pipeline
