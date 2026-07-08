#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/map_buffer_allocation.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_omb(map_extent_t &map_extent, map_coord_t &map_coord) {
    auto& omb = engine().omb;

    citlali::pipeline::clear_map_matrix_products(omb);
    citlali::pipeline::apply_observation_map_geometry(
        omb, map_extent, map_coord);
    citlali::pipeline::allocate_map_matrices(
        omb, engine().n_maps,
        engine().typed_config.mapmaking.method ==
            citlali::config::MapMethod::jinc,
        engine().rtcproc.run_kernel,
        engine().typed_config.mapmaking.grouping !=
            citlali::config::MapGrouping::detector);
    citlali::pipeline::allocate_polarization_pointing_matrices(
        omb, engine().n_maps,
        static_cast<Eigen::Index>(
            engine().rtcproc.polarization.stokes_params.size()),
        engine().rtcproc.run_polarization);
}

// allocate the coadded map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_cmb() {
    auto& cmb = engine().cmb;

    citlali::pipeline::clear_map_matrix_products(cmb);
    citlali::pipeline::allocate_map_matrices(
        cmb, engine().n_maps, false, engine().rtcproc.run_kernel,
        engine().typed_config.mapmaking.grouping !=
            citlali::config::MapGrouping::detector);
    citlali::pipeline::allocate_polarization_pointing_matrices(
        cmb, engine().n_maps,
        static_cast<Eigen::Index>(
            engine().rtcproc.polarization.stokes_params.size()),
        engine().rtcproc.run_polarization);
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
