#pragma once

#include <citlali/core/error/error.h>

#include <fmt/core.h>

#include <cmath>
#include <cstddef>

namespace citlali::pipeline {

inline void require_wiener_template_geometry(
    std::ptrdiff_t rows, std::ptrdiff_t cols,
    std::size_t row_axis_size, std::size_t col_axis_size) {
    if (rows < 2 || cols < 2 || row_axis_size < 2 || col_axis_size < 2) {
        throw citlali::error::runtime(fmt::format(
            "invalid Wiener template geometry: rows={} cols={} "
            "row_axis_size={} col_axis_size={}",
            rows, cols, row_axis_size, col_axis_size));
    }
}

inline void require_wiener_pixel_spacing(
    double row_spacing, double col_spacing) {
    if (!std::isfinite(row_spacing) || !std::isfinite(col_spacing) ||
        row_spacing <= 0.0 || col_spacing <= 0.0) {
        throw citlali::error::runtime(fmt::format(
            "invalid Wiener tangent-plane spacing: rows={} cols={}",
            row_spacing, col_spacing));
    }
}

inline void require_wiener_kernel_index(
    std::ptrdiff_t map_index, std::size_t kernel_count) {
    if (map_index < 0 ||
        static_cast<std::size_t>(map_index) >= kernel_count) {
        throw citlali::error::runtime(fmt::format(
            "Wiener kernel map index {} is outside [0, {})",
            map_index, kernel_count));
    }
}

inline void require_wiener_kernel_weight_index(
    std::ptrdiff_t map_index,
    std::size_t kernel_count,
    std::size_t weight_count) {
    if (map_index < 0 ||
        static_cast<std::size_t>(map_index) >= kernel_count ||
        static_cast<std::size_t>(map_index) >= weight_count) {
        throw citlali::error::runtime(fmt::format(
            "Wiener map index {} is invalid for {} kernels and {} weights",
            map_index, kernel_count, weight_count));
    }
}

inline void require_wiener_kernel_geometry(
    std::ptrdiff_t map_index,
    std::ptrdiff_t kernel_rows,
    std::ptrdiff_t kernel_cols,
    std::ptrdiff_t weight_rows,
    std::ptrdiff_t weight_cols,
    std::ptrdiff_t expected_rows,
    std::ptrdiff_t expected_cols) {
    if (kernel_rows != expected_rows || kernel_cols != expected_cols ||
        weight_rows != expected_rows || weight_cols != expected_cols) {
        throw citlali::error::runtime(fmt::format(
            "Wiener kernel/weight geometry mismatch for map {}: "
            "kernel=({}, {}) weight=({}, {}) expected=({}, {})",
            map_index, kernel_rows, kernel_cols, weight_rows, weight_cols,
            expected_rows, expected_cols));
    }
}

inline void require_finite_wiener_kernel_peak(
    double peak, std::ptrdiff_t map_index) {
    if (!std::isfinite(peak)) {
        throw citlali::error::runtime(fmt::format(
            "Wiener kernel peak is non-finite for map {}", map_index));
    }
}

inline void require_wiener_fftw_context(
    bool resources_ready, int rows, int cols) {
    if (!resources_ready) {
        throw citlali::error::runtime(fmt::format(
            "failed to allocate Wiener FFTW context for rows={} cols={}",
            rows, cols));
    }
}

}  // namespace citlali::pipeline
