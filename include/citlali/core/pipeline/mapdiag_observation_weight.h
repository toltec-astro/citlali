#pragma once

#include <algorithm>
#include <cstddef>

#include <Eigen/Core>

namespace citlali::pipeline {

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

template <class DoubleValues, class IntValues>
void zero_mapdiag_obs_entry(
    std::size_t flat, DoubleValues &obs_weight_sum,
    DoubleValues &obs_core_weight_sum, IntValues &obs_valid_pixels,
    IntValues &obs_core_pixels) {
    assign_mapdiag_obs_entry(
        flat, 0.0, 0.0, 0, 0, obs_weight_sum, obs_core_weight_sum,
        obs_valid_pixels, obs_core_pixels);
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

inline double mapdiag_fraction_or_fill(double value, double total,
                                       double fill_value) {
    return (total > 0.0) ? value / total : fill_value;
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

}  // namespace citlali::pipeline
