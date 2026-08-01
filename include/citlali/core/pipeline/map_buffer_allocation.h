#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <Eigen/Core>

#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline bool science_map_v1_profile_available(
    citlali::config::MapMethod method,
    citlali::config::MapGrouping grouping, bool polarization_enabled) {
    return method == citlali::config::MapMethod::naive &&
           grouping == citlali::config::MapGrouping::array &&
           !polarization_enabled;
}

template <class Engine>
bool science_map_v1_profile_available(const Engine &engine) {
    const auto &config = mapmaking_config(engine);
    return science_map_v1_profile_available(
        config.method, config.grouping, engine.rtcproc.run_polarization);
}

inline std::string science_map_v1_profile_absence_reason(
    citlali::config::MapMethod method,
    citlali::config::MapGrouping grouping, bool polarization_enabled) {
    if (science_map_v1_profile_available(method, grouping,
                                         polarization_enabled)) {
        return {};
    }
    if (polarization_enabled) {
        return "polarization science-map product profile is unavailable";
    }
    if (method != citlali::config::MapMethod::naive) {
        return "method-specific contribution predicate unavailable";
    }
    if (grouping == citlali::config::MapGrouping::detector) {
        return "detector-grouping science-map product profile is unavailable";
    }
    return "non-array map-grouping science-map product profile is unavailable";
}

template <class MapBuffer>
void clear_map_matrix_products(MapBuffer &buffer) {
    std::vector<Eigen::MatrixXd>().swap(buffer.signal);
    std::vector<Eigen::MatrixXd>().swap(buffer.weight);
    std::vector<Eigen::MatrixXd>().swap(buffer.kernel);
    std::vector<Eigen::MatrixXd>().swap(buffer.coverage);
    std::vector<Eigen::MatrixXd>().swap(buffer.grid_weight);
    std::vector<Eigen::MatrixXd>().swap(buffer.pointing);
    buffer.science_products.clear();
    buffer.raw_science_parent.reset();
    buffer.clear_contribution_diag();
}

template <class MapBuffer, class MapExtent, class MapCoord>
void apply_observation_map_geometry(MapBuffer &buffer,
                                    const MapExtent &map_extent,
                                    const MapCoord &map_coord) {
    buffer.n_rows = map_extent[0];
    buffer.n_cols = map_extent[1];
    buffer.wcs.naxis[0] = buffer.n_cols;
    buffer.wcs.naxis[1] = buffer.n_rows;
    buffer.wcs.crpix[0] = (buffer.n_cols - 1) / 2.0;
    buffer.wcs.crpix[1] = (buffer.n_rows - 1) / 2.0;
    buffer.rows_tan_vec = map_coord[0];
    buffer.cols_tan_vec = map_coord[1];
}

template <class MapBuffer>
void allocate_map_matrices(MapBuffer &buffer, Eigen::Index n_maps,
                           bool allocate_grid_weight, bool allocate_kernel,
                           bool allocate_coverage,
                           bool allocate_science_products = true,
                           std::string science_product_absence_reason = {}) {
    const Eigen::MatrixXd zero_matrix =
        Eigen::MatrixXd::Zero(buffer.n_rows, buffer.n_cols);

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        buffer.signal.push_back(zero_matrix);
        buffer.weight.push_back(zero_matrix);

        if (allocate_grid_weight) {
            buffer.grid_weight.push_back(zero_matrix);
        }
        if (allocate_kernel) {
            buffer.kernel.push_back(zero_matrix);
        }
        if (allocate_coverage || allocate_science_products) {
            buffer.coverage.push_back(zero_matrix);
        }
    }

    buffer.science_products.allocate(
        n_maps, buffer.n_rows, buffer.n_cols, buffer.name == "cmb",
        allocate_science_products && !allocate_grid_weight,
        allocate_science_products,
        std::move(science_product_absence_reason));
}

template <class MapBuffer>
void allocate_polarization_pointing_matrices(MapBuffer &buffer,
                                             Eigen::Index n_maps,
                                             Eigen::Index n_stokes,
                                             bool run_polarization) {
    if (!run_polarization) {
        return;
    }

    for (Eigen::Index i = 0; i < n_maps / n_stokes; ++i) {
        buffer.pointing.emplace_back(buffer.n_rows * buffer.n_cols, 9);
        buffer.pointing.back().setZero();
    }
}

}  // namespace citlali::pipeline
