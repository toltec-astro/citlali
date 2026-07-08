#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace citlali::pipeline {

struct MapCoordinateLimits {
    double min_row = std::numeric_limits<double>::max();
    double max_row = std::numeric_limits<double>::lowest();
    double min_col = std::numeric_limits<double>::max();
    double max_col = std::numeric_limits<double>::lowest();
};

inline int odd_dimension_from_config(int configured_dimension) {
    return (configured_dimension % 2 == 0) ? configured_dimension + 1
                                           : configured_dimension;
}

inline int symmetric_odd_pixel_count(double min_dim, double max_dim,
                                     double pixel_size_rad) {
    const int min_pix =
        static_cast<int>(std::ceil(std::abs(min_dim / pixel_size_rad)));
    const int max_pix =
        static_cast<int>(std::ceil(std::abs(max_dim / pixel_size_rad)));
    return 2 * std::max(min_pix, max_pix) + 1;
}

inline Eigen::VectorXd tangent_coordinate_vector(int n_dim,
                                                 double pixel_size_rad) {
    const double dim_center = (n_dim - 1) / 2.0;
    return Eigen::VectorXd::LinSpaced(n_dim, 0, n_dim - 1).array() *
               pixel_size_rad -
           dim_center * pixel_size_rad;
}

inline std::tuple<int, Eigen::VectorXd> dimension_and_tangent_coordinates(
    double min_dim, double max_dim, double pixel_size_rad) {
    const int n_dim =
        symmetric_odd_pixel_count(min_dim, max_dim, pixel_size_rad);
    return {n_dim, tangent_coordinate_vector(n_dim, pixel_size_rad)};
}

template <class MapCoords>
MapCoordinateLimits coordinate_limits(const MapCoords &map_coords) {
    MapCoordinateLimits limits;
    for (const auto &coord : map_coords) {
        limits.min_row = std::min(limits.min_row, coord.front().minCoeff());
        limits.max_row = std::max(limits.max_row, coord.front().maxCoeff());
        limits.min_col = std::min(limits.min_col, coord.back().minCoeff());
        limits.max_col = std::max(limits.max_col, coord.back().maxCoeff());
    }
    return limits;
}

}  // namespace citlali::pipeline
