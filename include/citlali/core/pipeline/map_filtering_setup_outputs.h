#pragma once

// Included by map_filtering.h inside namespace citlali::pipeline.

inline bool map_filter_template_uses_fwhm(
    const std::string &template_type) {
    return citlali::config::map_filter_template_uses_fwhm(template_type);
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
