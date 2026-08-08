#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <algorithm>
#include <stdexcept>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_omb(map_extent_t &map_extent, map_coord_t &map_coord) {
    auto& omb = engine().omb;
    const auto &mapmaking_settings =
        citlali::pipeline::mapmaking_config(engine());
    const bool allocate_science_products =
        citlali::pipeline::science_map_v1_profile_available(
            mapmaking_settings.method, mapmaking_settings.grouping,
            engine().rtcproc.run_polarization);
    const std::string science_product_absence_reason =
        citlali::pipeline::science_map_v1_profile_absence_reason(
            mapmaking_settings.method, mapmaking_settings.grouping,
            engine().rtcproc.run_polarization);

    citlali::pipeline::clear_map_matrix_products(omb);
    citlali::pipeline::apply_observation_map_geometry(
        omb, map_extent, map_coord);
    citlali::pipeline::allocate_map_matrices(
        omb, engine().map_indices.n_maps,
        mapmaking_settings.method == citlali::config::MapMethod::jinc,
        citlali::pipeline::raw_kernel_enabled(engine()),
        mapmaking_settings.method == citlali::config::MapMethod::jinc ||
            mapmaking_settings.grouping != citlali::config::MapGrouping::detector,
        allocate_science_products, science_product_absence_reason,
        mapmaking_settings.method == citlali::config::MapMethod::jinc);
    if (mapmaking_settings.method == citlali::config::MapMethod::jinc) {
        auto &provenance = omb.jinc_products.provenance;
        const auto &plan = citlali::pipeline::mapmaking_plan(engine());
        if (!plan.initialized ||
            plan.effective.method != citlali::config::MapMethod::jinc ||
            engine().jinc_mm.resolved_arrays.empty() ||
            engine().map_indices.maps_to_arrays.size() !=
                engine().map_indices.n_maps) {
            throw std::logic_error(
                "JINC observation allocation requires a resolved typed plan and map identity");
        }
        provenance.requested_digest =
            mapmaking::jinc_filter_config_digest(
                plan.requested.jinc_filter);
        provenance.effective_digest =
            mapmaking::jinc_filter_config_digest(
                plan.effective.jinc_filter);
        provenance.requested_r_max = plan.requested.jinc_filter.r_max;
        provenance.effective_r_max = plan.effective.jinc_filter.r_max;
        provenance.requested_subpixel_n =
            plan.requested.jinc_filter.subpixel_n;
        provenance.effective_subpixel_n =
            plan.effective.jinc_filter.subpixel_n;
        provenance.requested_shape_params =
            plan.requested.jinc_filter.shape_params;
        provenance.effective_shape_params =
            plan.effective.jinc_filter.shape_params;
        for (const auto &resolved : engine().jinc_mm.resolved_arrays) {
            bool selected = false;
            for (Eigen::Index slot = 0;
                 slot < engine().map_indices.maps_to_arrays.size(); ++slot) {
                if (engine().map_indices.maps_to_arrays(slot) ==
                    resolved.array_id) {
                    selected = true;
                    break;
                }
            }
            if (selected) {
                mapmaking::validate_jinc_resolved_array(resolved);
                provenance.resolved_arrays.push_back(resolved);
            }
        }
        for (Eigen::Index slot = 0;
             slot < engine().map_indices.maps_to_arrays.size(); ++slot) {
            const auto array_id =
                engine().map_indices.maps_to_arrays(slot);
            const bool resolved = std::any_of(
                provenance.resolved_arrays.begin(),
                provenance.resolved_arrays.end(), [&](const auto &record) {
                    return record.array_id == array_id;
                });
            if (!resolved) {
                throw std::logic_error(
                    "JINC selected map array lacks a resolved stable identity");
            }
        }
        provenance.kernel_template_identity =
            citlali::pipeline::raw_kernel_enabled(engine())
                ? mapmaking::jinc_kernel_identity
                : "disabled";
        provenance.processing_realization_identity =
            "requested-effective-jinc-digests-and-runtime-kernel-v1";
    }
    citlali::pipeline::allocate_polarization_pointing_matrices(
        omb, engine().map_indices.n_maps,
        static_cast<Eigen::Index>(
            engine().rtcproc.polarization.stokes_params.size()),
        engine().rtcproc.run_polarization);
}

// allocate the coadded map buffer
template <class EngineType>
void TimeOrderedDataProc<EngineType>::allocate_cmb() {
    auto& cmb = engine().cmb;
    const auto &mapmaking_settings =
        citlali::pipeline::mapmaking_config(engine());
    const bool allocate_science_products =
        citlali::pipeline::science_map_v1_profile_available(
            mapmaking_settings.method, mapmaking_settings.grouping,
            engine().rtcproc.run_polarization);
    const std::string science_product_absence_reason =
        citlali::pipeline::science_map_v1_profile_absence_reason(
            mapmaking_settings.method, mapmaking_settings.grouping,
            engine().rtcproc.run_polarization);

    citlali::pipeline::clear_map_matrix_products(cmb);
    citlali::pipeline::allocate_map_matrices(
        cmb, engine().map_indices.n_maps, false,
        citlali::pipeline::raw_kernel_enabled(engine()),
        mapmaking_settings.grouping != citlali::config::MapGrouping::detector,
        allocate_science_products, science_product_absence_reason, false);
    citlali::pipeline::allocate_polarization_pointing_matrices(
        cmb, engine().map_indices.n_maps,
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
        static_cast<double>(engine().map_indices.n_maps) * static_cast<double>(nmb.n_noise) / 1e9;
    engine().logger->info("allocating {} noise realization cube: rows={} cols={} maps={} n_noise={} estimated_size={:.2f} GB",
                          nmb.name, static_cast<long long>(nmb.n_rows),
                          static_cast<long long>(nmb.n_cols),
                          static_cast<long long>(engine().map_indices.n_maps),
                          static_cast<long long>(nmb.n_noise),
                          nmb_size_gb);

    // resize noise maps (n_maps, [n_rows, n_cols, n_noise])
    for (Eigen::Index i=0; i<engine().map_indices.n_maps; ++i) {
        nmb.noise.emplace_back(nmb.n_rows, nmb.n_cols, nmb.n_noise);
        nmb.noise.at(i).setZero();
    }
}

// coadd maps
