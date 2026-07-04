#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

struct MapdiagObsWeightTotals {
    double weight;
    double core_weight;
};

struct MapdiagObsTableRefs {
    std::vector<double> &weight_sum;
    std::vector<double> &core_weight_sum;
    std::vector<int> &valid_pixels;
    std::vector<int> &core_pixels;
};

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

template <class CoreMask, class ObsWeight, class DoubleValues, class IntValues>
void accumulate_mapdiag_obs_weight(
    Eigen::Index map_i, std::size_t n_obsnums, Eigen::Index map_n_rows,
    Eigen::Index map_n_cols, const CoreMask &core_mask,
    const ObsWeight &obs_weight, std::size_t obs_index,
    DoubleValues &obs_weight_sum, DoubleValues &obs_core_weight_sum,
    IntValues &obs_valid_pixels, IntValues &obs_core_pixels) {
    const Eigen::Index block_row = (map_n_rows - obs_weight.rows()) / 2;
    const Eigen::Index block_col = (map_n_cols - obs_weight.cols()) / 2;
    Eigen::Index row0 = std::max<Eigen::Index>(0, block_row);
    Eigen::Index col0 = std::max<Eigen::Index>(0, block_col);
    Eigen::Index src_row0 = std::max<Eigen::Index>(0, -block_row);
    Eigen::Index src_col0 = std::max<Eigen::Index>(0, -block_col);
    Eigen::Index rows = std::min<Eigen::Index>(
        map_n_rows - row0, obs_weight.rows() - src_row0);
    Eigen::Index cols = std::min<Eigen::Index>(
        map_n_cols - col0, obs_weight.cols() - src_col0);
    const std::size_t flat =
        static_cast<std::size_t>(map_i) * n_obsnums + obs_index;
    if (rows <= 0 || cols <= 0) {
        assign_mapdiag_obs_entry(
            flat, 0.0, 0.0, 0, 0, obs_weight_sum, obs_core_weight_sum,
            obs_valid_pixels, obs_core_pixels);
        return;
    }

    const auto block = obs_weight.block(src_row0, src_col0, rows, cols);
    const auto valid = (block.array() > 0.0).template cast<double>();
    const auto core_block = core_mask.block(row0, col0, rows, cols);
    obs_weight_sum[flat] = (block.array() * valid).sum();
    obs_core_weight_sum[flat] =
        (block.array() * valid * core_block).sum();
    obs_valid_pixels[flat] = static_cast<int>(valid.sum());
    obs_core_pixels[flat] = static_cast<int>((valid * core_block).sum());
}

template <class CoreMask, class ObsWeight>
void accumulate_mapdiag_obs_weight(
    Eigen::Index map_i, std::size_t n_obsnums, Eigen::Index map_n_rows,
    Eigen::Index map_n_cols, const CoreMask &core_mask,
    const ObsWeight &obs_weight, std::size_t obs_index,
    MapdiagObsTableRefs tables) {
    accumulate_mapdiag_obs_weight(
        map_i, n_obsnums, map_n_rows, map_n_cols, core_mask, obs_weight,
        obs_index, tables.weight_sum, tables.core_weight_sum,
        tables.valid_pixels, tables.core_pixels);
}

inline double mapdiag_fraction_or_fill(double value, double total,
                                       double fill_value) {
    return (total > 0.0) ? value / total : fill_value;
}

template <class SourceValues, class DestValues>
void assign_mapdiag_obs_fraction_entry(
    std::size_t flat, const SourceValues &source_values, double total,
    double fill_value, DestValues &fraction_values) {
    fraction_values[flat] =
        mapdiag_fraction_or_fill(source_values[flat], total, fill_value);
}

template <class SourceValues, class DestValues>
void assign_mapdiag_obs_fraction_series(
    const SourceValues &source_values, double total, double fill_value,
    std::size_t n_obsnums, std::size_t map_index,
    DestValues &fraction_values) {
    for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
        const std::size_t flat = map_index * n_obsnums + obs_idx;
        assign_mapdiag_obs_fraction_entry(
            flat, source_values, total, fill_value, fraction_values);
    }
}

template <class SourceValues, class DestValues>
void assign_mapdiag_obs_fraction_pair(
    const SourceValues &weight_sum, double total_weight,
    const SourceValues &core_weight_sum, double total_core_weight,
    double fill_value, std::size_t n_obsnums, std::size_t map_index,
    DestValues &weight_frac, DestValues &core_weight_frac) {
    assign_mapdiag_obs_fraction_series(
        weight_sum, total_weight, fill_value, n_obsnums, map_index,
        weight_frac);
    assign_mapdiag_obs_fraction_series(
        core_weight_sum, total_core_weight, fill_value, n_obsnums,
        map_index, core_weight_frac);
}

template <class Values>
double sum_mapdiag_obs_values(const Values &values, std::size_t n_obsnums,
                              std::size_t map_index) {
    double total = 0.0;
    for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
        const std::size_t flat = map_index * n_obsnums + obs_idx;
        total += values[flat];
    }
    return total;
}

template <class Values>
MapdiagObsWeightTotals sum_mapdiag_obs_weight_totals(
    const Values &weight_sum, const Values &core_weight_sum,
    std::size_t n_obsnums, std::size_t map_index) {
    return {sum_mapdiag_obs_values(weight_sum, n_obsnums, map_index),
            sum_mapdiag_obs_values(core_weight_sum, n_obsnums, map_index)};
}

template <class Values, class Context>
MapdiagObsWeightTotals sum_mapdiag_obs_weight_totals(
    const Values &weight_sum, const Values &core_weight_sum,
    const Context &context, std::size_t map_index) {
    return sum_mapdiag_obs_weight_totals(
        weight_sum, core_weight_sum, context.n_obsnums, map_index);
}

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

}  // namespace citlali::pipeline
