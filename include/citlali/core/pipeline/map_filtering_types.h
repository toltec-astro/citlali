#pragma once

// Included by map_filtering.h inside namespace citlali::pipeline.

template <class FitsVector>
struct MapFilterOutputTargets {
    FitsVector *filtered_fits_io;
    FitsVector *filtered_noise_fits_io;
    const char *map_label;
};

struct MapFilterRunOptions {
    bool run_noise;
    bool write_filtered_maps_partial;
    bool run_noise_products;
    bool apply_empirical_noise_weights;
};

inline MapFilterRunOptions map_filter_run_options(
    bool run_noise, bool write_filtered_maps_partial,
    bool run_noise_products, bool apply_empirical_noise_weights) {
    return {
        run_noise,
        write_filtered_maps_partial,
        run_noise_products,
        apply_empirical_noise_weights};
}

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

