#pragma once

// Included by map_filtering.h inside namespace citlali::pipeline.

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

