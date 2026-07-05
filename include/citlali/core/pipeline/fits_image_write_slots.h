#pragma once

// Included by fits_image_metadata.h inside namespace citlali::pipeline.

struct MapWriteIndices {
    Eigen::Index map_index;
    Eigen::Index stokes_index;
    Eigen::Index array_index;
};

template <class ArrayToMap, class MapToStokes, class MapToArray>
MapWriteIndices map_write_indices(Eigen::Index i,
                                  const ArrayToMap &arrays_to_maps,
                                  const MapToStokes &maps_to_stokes,
                                  const MapToArray &maps_to_arrays) {
    return {
        arrays_to_maps(i),
        maps_to_stokes(i),
        maps_to_arrays(i),
    };
}

template <class FitsEntry, class Wcs, class Data>
void add_map_hdu_with_wcs(FitsEntry &fits_entry, const std::string &hdu_name,
                          Data &data, const Wcs &wcs,
                          double source_epoch) {
    fits_entry.add_hdu(hdu_name, data);
    fits_entry.add_wcs(fits_entry.hdus.back(), wcs, source_epoch);
}

inline bool has_map_data_slots(Eigen::Index i, Eigen::Index signal_size,
                               Eigen::Index weight_size) {
    return i >= 0 && i < signal_size && i < weight_size;
}

inline bool has_output_file_slot(Eigen::Index map_index,
                                 Eigen::Index n_files) {
    return map_index >= 0 && map_index < n_files;
}

inline bool has_stokes_slot(Eigen::Index stokes_index,
                            Eigen::Index n_stokes) {
    return stokes_index >= 0 && stokes_index < n_stokes;
}

inline bool has_array_slot(Eigen::Index array_index,
                           Eigen::Index n_arrays) {
    return array_index >= 0 && array_index < n_arrays;
}

template <class Logger>
void require_map_data_slots(Eigen::Index i, Eigen::Index signal_size,
                            Eigen::Index weight_size,
                            const Logger &logger) {
    if (!has_map_data_slots(i, signal_size, weight_size)) {
        logger->error(
            "write_maps map index out of range: i={} signal_size={} weight_size={}",
            static_cast<long long>(i), static_cast<long long>(signal_size),
            static_cast<long long>(weight_size));
        std::exit(EXIT_FAILURE);
    }
}

template <class Logger>
void require_map_write_index_slots(
    Eigen::Index i, Eigen::Index map_index, Eigen::Index n_files,
    Eigen::Index stokes_index, Eigen::Index n_stokes,
    Eigen::Index array_index, Eigen::Index n_arrays, const Logger &logger) {
    if (!has_output_file_slot(map_index, n_files)) {
        logger->error(
            "write_maps file index out of range: map_index={} fits_io_size={} map_i={}",
            static_cast<long long>(map_index), static_cast<long long>(n_files),
            static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }
    if (!has_stokes_slot(stokes_index, n_stokes)) {
        logger->error(
            "write_maps stokes index out of range: stokes_index={} stokes_size={} map_i={}",
            static_cast<long long>(stokes_index),
            static_cast<long long>(n_stokes), static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }
    if (!has_array_slot(array_index, n_arrays)) {
        logger->error(
            "write_maps maps_to_arrays index out of range: maps_to_arrays(i)={} calib.arrays.size={} map_i={}",
            static_cast<long long>(array_index),
            static_cast<long long>(n_arrays), static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }
}

