#pragma once

// Included by map_filtering.h inside namespace citlali::pipeline.

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
          class FilterOutputs,
          class ArrayNames, class ArrayFwhm, class Apt,
          class MapBufferPtr, class Polarization,
          class Callbacks, class Logger>
void run_map_filter_loop(
    WienerFilter &wiener_filter, MapBuffer &map_buffer, MapCount n_maps,
    const FilterOutputs &filter_outputs, ArrayNames &array_names,
    ArrayFwhm &array_fwhm_arcsec, double arcsec_to_rad, const Apt &apt,
    const MapFilterRunOptions &options, MapBufferPtr map_buffer_ptr,
    bool run_polarization,
    Polarization &polarization, const Callbacks &callbacks,
    const Logger &logger) {
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const auto map_number = map_filter_display_number(i);
        const auto array = callbacks.maps_to_arrays(i);
        const auto &array_name =
            map_filter_array_name(array_names, array);
        const auto map_index = callbacks.arrays_to_maps(i);

        log_map_filter_map_start(
            filter_outputs.map_label, map_number, n_maps, array_name, logger);
        initialize_map_filter_fwhm(
            wiener_filter, array_fwhm_arcsec, array, arcsec_to_rad,
            map_buffer.pixel_size_rad);
        build_map_filter_template(
            wiener_filter, map_buffer, apt, i, map_number, n_maps,
            array_name, filter_outputs.map_label, logger);
        filter_map_filter_signal_map(
            wiener_filter, map_buffer, i, map_number, n_maps, array_name,
            filter_outputs.map_label, logger);

        const auto n_wiener_noise_maps = map_buffer.n_noise;
        if (options.run_noise) {
            filter_map_filter_noise_maps(
                wiener_filter, map_buffer, i, map_number,
                n_wiener_noise_maps, filter_outputs.map_label, n_maps,
                logger);
            calculate_map_filter_noise_products_if_needed(
                map_buffer, i, map_number, n_maps,
                options.write_filtered_maps_partial,
                options.run_noise_products, wiener_filter.normalize_error,
                options.apply_empirical_noise_weights,
                filter_outputs.map_label, logger);
        }

        if (options.write_filtered_maps_partial) {
            write_map_filter_output(
                filter_outputs.filtered_fits_io,
                filter_outputs.filtered_noise_fits_io, map_buffer_ptr, i,
                map_number, map_index, n_maps, filter_outputs.map_label,
                run_polarization, polarization, callbacks.maps_to_stokes,
                callbacks.arrays_to_maps, callbacks.write_maps, logger);
        }

        log_map_filter_map_completed(
            filter_outputs.map_label, map_number, n_maps, logger);
    }
}

template <auto FilteredMap, class Engine, class MapBuffer, class Logger>
void run_wiener_filter_with_log(Engine &engine, MapBuffer &map_buffer,
                                const Logger &logger,
                                const char *log_message) {
    logger->info("{}", log_message);
    engine.template run_wiener_filter<FilteredMap>(map_buffer);
}

