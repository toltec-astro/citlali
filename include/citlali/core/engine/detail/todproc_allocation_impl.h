#pragma once

// Implementation detail included by todproc.h.

template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_omb(map_extent_t &map_extent, map_coord_t &map_coord) {
    auto& omb = engine().omb;

    std::vector<Eigen::MatrixXd>().swap(omb.signal);
    std::vector<Eigen::MatrixXd>().swap(omb.weight);
    std::vector<Eigen::MatrixXd>().swap(omb.kernel);
    std::vector<Eigen::MatrixXd>().swap(omb.coverage);
    std::vector<Eigen::MatrixXd>().swap(omb.grid_weight);
    std::vector<Eigen::MatrixXd>().swap(omb.pointing);
    omb.clear_contribution_diag();

    // set omb dimensions and wcs parameters
    omb.n_rows = map_extent[0];
    omb.n_cols = map_extent[1];
    omb.wcs.naxis[0] = omb.n_cols;
    omb.wcs.naxis[1] = omb.n_rows;
    omb.wcs.crpix[0] = (omb.n_cols - 1) / 2.0;
    omb.wcs.crpix[1] = (omb.n_rows - 1) / 2.0;
    // set tangent plane coordinate vectors
    omb.rows_tan_vec = map_coord[0];
    omb.cols_tan_vec = map_coord[1];

    Eigen::MatrixXd zero_matrix = Eigen::MatrixXd::Zero(omb.n_rows, omb.n_cols);

    for (Eigen::Index i=0; i<engine().n_maps; ++i) {
        omb.signal.push_back(zero_matrix);
        omb.weight.push_back(zero_matrix);

        if (engine().map_method == "jinc") {
            omb.grid_weight.push_back(zero_matrix);
        }

        if (engine().rtcproc.run_kernel) {
            omb.kernel.push_back(zero_matrix);
        }

        if (engine().map_grouping != "detector") {
            omb.coverage.push_back(zero_matrix);
        }
    }

    if (engine().rtcproc.run_polarization) {
        // allocate pointing matrix
        for (Eigen::Index i=0; i<engine().n_maps/engine().rtcproc.polarization.stokes_params.size(); ++i) {
            omb.pointing.emplace_back(omb.n_rows*omb.n_cols, 9);
            engine().omb.pointing.back().setZero();
        }
    }
}

// allocate the coadded map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_cmb() {
    auto& cmb = engine().cmb;

    // clear map vectors
    std::vector<Eigen::MatrixXd>().swap(cmb.signal);
    std::vector<Eigen::MatrixXd>().swap(cmb.weight);
    std::vector<Eigen::MatrixXd>().swap(cmb.kernel);
    std::vector<Eigen::MatrixXd>().swap(cmb.coverage);
    std::vector<Eigen::MatrixXd>().swap(cmb.grid_weight);
    std::vector<Eigen::MatrixXd>().swap(cmb.pointing);
    cmb.clear_contribution_diag();

    Eigen::MatrixXd zero_matrix = Eigen::MatrixXd::Zero(cmb.n_rows, cmb.n_cols);

    // loop through maps and allocate space
    for (Eigen::Index i=0; i<engine().n_maps; ++i) {
        cmb.signal.push_back(zero_matrix);
        cmb.weight.push_back(zero_matrix);

        if (engine().rtcproc.run_kernel) {
            // allocate kernel
            cmb.kernel.push_back(zero_matrix);
        }
        if (engine().map_grouping!="detector") {
            // allocate coverage
            cmb.coverage.push_back(zero_matrix);
        }
    }

    if (engine().rtcproc.run_polarization) {// && engine().run_noise) {
        // allocate pointing matrix
        for (Eigen::Index i=0; i<engine().n_maps/engine().rtcproc.polarization.stokes_params.size(); ++i) {
            cmb.pointing.emplace_back(cmb.n_rows*cmb.n_cols, 9);
            cmb.pointing.back().setZero();
        }
    }
}

template <class EngineType>
template <class map_buffer_t>
void TimeOrderedDataProc<EngineType>::allocate_nmb(map_buffer_t &nmb) {
    // clear noise map buffer
    std::vector<Eigen::Tensor<double,3>>().swap(nmb.noise);

    const double nmb_size_gb =
        8.0 * static_cast<double>(nmb.n_rows) * static_cast<double>(nmb.n_cols) *
        static_cast<double>(engine().n_maps) * static_cast<double>(nmb.n_noise) / 1e9;
    engine().logger->info("allocating {} noise realization cube: rows={} cols={} maps={} n_noise={} estimated_size={:.2f} GB",
                          nmb.name, static_cast<long long>(nmb.n_rows),
                          static_cast<long long>(nmb.n_cols),
                          static_cast<long long>(engine().n_maps),
                          static_cast<long long>(nmb.n_noise),
                          nmb_size_gb);

    // resize noise maps (n_maps, [n_rows, n_cols, n_noise])
    for (Eigen::Index i=0; i<engine().n_maps; ++i) {
        nmb.noise.emplace_back(nmb.n_rows, nmb.n_cols, nmb.n_noise);
        nmb.noise.at(i).setZero();
    }
}

// coadd maps

