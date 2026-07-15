#pragma once

#include <citlali/core/error/error.h>

#include <Eigen/Core>
#include <fmt/format.h>

namespace citlali::pipeline {

inline void require_beammap_fit_map_geometry(
    Eigen::Index map_index, Eigen::Index signal_rows,
    Eigen::Index signal_cols, Eigen::Index weight_rows,
    Eigen::Index weight_cols, Eigen::Index expected_rows,
    Eigen::Index expected_cols) {
    if (signal_rows != expected_rows || signal_cols != expected_cols ||
        weight_rows != expected_rows || weight_cols != expected_cols) {
        throw citlali::error::internal(fmt::format(
            "beammap fit map {} geometry mismatch: signal={}x{} weight={}x{} expected={}x{}",
            map_index, signal_rows, signal_cols, weight_rows, weight_cols,
            expected_rows, expected_cols));
    }
}

}  // namespace citlali::pipeline
