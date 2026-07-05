#pragma once

// Included by mapdiag_observation_weight.h inside namespace citlali::pipeline.

template <class DoubleValues, class IntValues>
void assign_mapdiag_single_obs_entry(
    std::size_t flat, double map_weight_sum, double map_core_weight_sum,
    int map_valid_pixels, int map_core_pixels, DoubleValues &obs_weight_sum,
    DoubleValues &obs_core_weight_sum, IntValues &obs_valid_pixels,
    IntValues &obs_core_pixels) {
    assign_mapdiag_obs_entry(
        flat, map_weight_sum, map_core_weight_sum, map_valid_pixels,
        map_core_pixels, obs_weight_sum, obs_core_weight_sum,
        obs_valid_pixels, obs_core_pixels);
}

inline void assign_mapdiag_single_obs_entry(
    std::size_t flat, double map_weight_sum, double map_core_weight_sum,
    int map_valid_pixels, int map_core_pixels, MapdiagObsTableRefs tables) {
    assign_mapdiag_obs_entry(
        flat, map_weight_sum, map_core_weight_sum, map_valid_pixels,
        map_core_pixels, tables);
}

template <class Context>
void assign_mapdiag_single_obs_entry(
    const Context &context, std::size_t map_index, double map_weight_sum,
    double map_core_weight_sum, int map_valid_pixels, int map_core_pixels,
    MapdiagObsTableRefs tables) {
    assign_mapdiag_single_obs_entry(
        map_index * context.n_obsnums, map_weight_sum, map_core_weight_sum,
        map_valid_pixels, map_core_pixels, tables);
}

