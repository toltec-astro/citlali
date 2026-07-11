#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/observation_coadd_accumulation.h>
#include <citlali/core/pipeline/observation_map_files.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/product_index_file.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::coadd() {
    citlali::pipeline::accumulate_observation_into_coadd(
        engine().cmb, engine().omb, engine().map_indices.n_maps,
        citlali::pipeline::raw_kernel_enabled(engine()));
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
