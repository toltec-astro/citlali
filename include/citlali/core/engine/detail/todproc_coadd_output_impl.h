#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/observation_coadd_accumulation.h>
#include <citlali/core/pipeline/coadd_execution_plan.h>
#include <citlali/core/pipeline/observation_map_files.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/product_index_file.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::coadd() {
    auto &current_engine = engine();
    const auto n_maps = current_engine.map_indices.n_maps;
    const bool run_kernel =
        citlali::pipeline::raw_kernel_enabled(current_engine);
    const bool science_profile =
        citlali::pipeline::science_map_v1_coadd_profile_enabled(
            current_engine.cmb, current_engine.omb, n_maps);
    if (!science_profile) {
        citlali::pipeline::accumulate_legacy_observation_into_coadd(
            current_engine.cmb, current_engine.omb, n_maps, run_kernel);
        return;
    }
    auto admission = citlali::pipeline::preflight_observation_for_coadd(
        current_engine.cmb, current_engine.omb, n_maps, run_kernel,
        current_engine.observation_identity.obsnum,
        current_engine.omb.exposure_time);

    // Provenance is staged and fully validated before the numerical buffer is
    // touched. The live plan moves only after the already-admitted arithmetic
    // commit, so an identity or membership failure changes neither authority.
    if constexpr (citlali::pipeline::has_coadd_plan_v<EngineType>) {
        auto staged_plan = citlali::pipeline::coadd_plan(current_engine);
        const auto common_identity =
            citlali::pipeline::coadd_bundle_identity_for_embedding(
                *current_engine.omb.science_products.bundle_identity,
                current_engine.cmb.n_rows, current_engine.cmb.n_cols,
                admission.delta_row, admission.delta_col);
        staged_plan.resolve_common_identity(common_identity);
        staged_plan.record_admission(admission);
        citlali::pipeline::commit_observation_to_coadd(
            current_engine.cmb, current_engine.omb, n_maps, run_kernel,
            std::move(admission));
        citlali::pipeline::coadd_plan(current_engine) =
            std::move(staged_plan);
    }
    else {
        citlali::pipeline::commit_observation_to_coadd(
            current_engine.cmb, current_engine.omb, n_maps, run_kernel,
            std::move(admission));
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::create_coadded_map_files() {
    // clear fits_io vectors
    citlali::pipeline::reset_coadd_map_fits_files(
        engine().map_fits_outputs.coadd, engine().map_fits_outputs.coadd_noise,
        engine().map_fits_outputs.filtered_coadd,
        engine().map_fits_outputs.filtered_coadd_noise);

    const bool write_noise_maps =
        citlali::pipeline::noise_maps_enabled(engine()) &&
        citlali::pipeline::noise_realization_outputs_enabled(engine());
    const std::string raw_dir =
        citlali::pipeline::raw_coadd_map_directory(engine().output_paths.coadd_dir_name);
    const std::string filtered_dir =
        citlali::pipeline::filtered_coadd_map_directory(
            engine().output_paths.coadd_dir_name);

    citlali::pipeline::append_coadd_array_products<
        engine_utils::toltecIO::raw>(
        engine().map_fits_outputs.coadd, engine().map_fits_outputs.coadd_noise,
        engine().toltec_io, raw_dir, engine().calib.arrays,
        engine().calib.n_arrays, engine().toltec_io.array_name_map,
        engine().telescope.sim_obs, write_noise_maps);

    // if map filtering are requested
    if (citlali::pipeline::map_filter_outputs_enabled(engine())) {
        citlali::pipeline::append_coadd_array_products<
            engine_utils::toltecIO::filtered>(
            engine().map_fits_outputs.filtered_coadd,
            engine().map_fits_outputs.filtered_coadd_noise, engine().toltec_io,
            filtered_dir, engine().calib.arrays, engine().calib.n_arrays,
            engine().toltec_io.array_name_map, engine().telescope.sim_obs,
            write_noise_maps);
    }
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::make_index_file(std::string filepath) {
    citlali::pipeline::write_product_index_file(filepath);
}
