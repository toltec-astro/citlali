#pragma once

#include <Eigen/Core>

#include <vector>

namespace citlali::pipeline {

template <class MapBuffer>
void clear_map_matrix_products(MapBuffer &buffer) {
    std::vector<Eigen::MatrixXd>().swap(buffer.signal);
    std::vector<Eigen::MatrixXd>().swap(buffer.weight);
    std::vector<Eigen::MatrixXd>().swap(buffer.kernel);
    std::vector<Eigen::MatrixXd>().swap(buffer.coverage);
    std::vector<Eigen::MatrixXd>().swap(buffer.grid_weight);
    std::vector<Eigen::MatrixXd>().swap(buffer.pointing);
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
                           bool allocate_coverage) {
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
        if (allocate_coverage) {
            buffer.coverage.push_back(zero_matrix);
        }
    }
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
