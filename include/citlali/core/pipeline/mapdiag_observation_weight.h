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
        obs_weight_sum[flat] = 0.0;
        obs_core_weight_sum[flat] = 0.0;
        obs_valid_pixels[flat] = 0;
        obs_core_pixels[flat] = 0;
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

}  // namespace citlali::pipeline
