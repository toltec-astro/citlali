#pragma once

// Included by mapdiag_observation_weight.h inside namespace citlali::pipeline.

template <class DoubleValues, class IntValues>
void assign_mapdiag_obs_entry(
    std::size_t flat, double weight_sum, double core_weight_sum,
    int valid_pixels, int core_pixels, DoubleValues &obs_weight_sum,
    DoubleValues &obs_core_weight_sum, IntValues &obs_valid_pixels,
    IntValues &obs_core_pixels) {
    obs_weight_sum[flat] = weight_sum;
    obs_core_weight_sum[flat] = core_weight_sum;
    obs_valid_pixels[flat] = valid_pixels;
    obs_core_pixels[flat] = core_pixels;
}

inline void assign_mapdiag_obs_entry(
    std::size_t flat, double weight_sum, double core_weight_sum,
    int valid_pixels, int core_pixels, MapdiagObsTableRefs tables) {
    assign_mapdiag_obs_entry(
        flat, weight_sum, core_weight_sum, valid_pixels, core_pixels,
        tables.weight_sum, tables.core_weight_sum, tables.valid_pixels,
        tables.core_pixels);
}

template <class DoubleValues, class IntValues>
void zero_mapdiag_obs_entry(
    std::size_t flat, DoubleValues &obs_weight_sum,
    DoubleValues &obs_core_weight_sum, IntValues &obs_valid_pixels,
    IntValues &obs_core_pixels) {
    assign_mapdiag_obs_entry(
        flat, 0.0, 0.0, 0, 0, obs_weight_sum, obs_core_weight_sum,
        obs_valid_pixels, obs_core_pixels);
}

inline void zero_mapdiag_obs_entry(std::size_t flat,
                                   MapdiagObsTableRefs tables) {
    assign_mapdiag_obs_entry(flat, 0.0, 0.0, 0, 0, tables);
}

template <class Context>
void zero_mapdiag_obs_entry(const Context &context, std::size_t map_index,
                            std::size_t obs_index,
                            MapdiagObsTableRefs tables) {
    zero_mapdiag_obs_entry(
        map_index * context.n_obsnums + obs_index, tables);
}

