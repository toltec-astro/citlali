#pragma once

// Engine output implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>

void Engine::create_obs_map_files() {
    // clear fits vectors for each observation
    citlali::pipeline::clear_observation_map_fits_files(
        map_fits_outputs.obs, map_fits_outputs.obs_noise, map_fits_outputs.filtered_obs,
        map_fits_outputs.filtered_obs_noise);
    const std::string raw_dir =
        citlali::pipeline::raw_observation_map_directory(output_paths.obsnum_dir_name);
    const std::string filtered_dir =
        citlali::pipeline::filtered_observation_map_directory(
            output_paths.obsnum_dir_name);
    auto make_fits_io = [](const std::string &filename) {
        return fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>(filename);
    };
    auto append_fits_file = [&](auto &fits_files,
                                const std::string &filename) {
        citlali::pipeline::append_observation_map_fits_file(
            fits_files, filename, make_fits_io);
    };
    const bool create_per_obs_outputs =
        citlali::pipeline::should_create_observation_per_obs_outputs(
            *this);
    const bool create_noise_maps =
        citlali::pipeline::should_create_observation_noise_maps(
            *this);
    const bool create_filtered_maps =
        citlali::pipeline::should_create_observation_filtered_maps(
            *this);
    const bool create_filtered_noise_maps =
        citlali::pipeline::should_create_observation_filtered_noise_maps(
            *this);

    // loop through arrays
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        // array index
        auto array = calib.arrays[i];
        // array name
        std::string array_name = toltec_io.array_name_map[array];
        // map filename
        const auto filename =
            citlali::pipeline::observation_output_filename<
                engine_utils::toltecIO::toltec, engine_utils::toltecIO::map,
                engine_utils::toltecIO::raw>(
                toltec_io, raw_dir, citlali::pipeline::runtime_config(*this).reduction_type,
                array_name, observation_identity.obsnum, telescope.sim_obs);
        append_fits_file(map_fits_outputs.obs, filename);

        // if noise maps are requested but coadding is not, populate noise fits vector
        if (create_per_obs_outputs) {
            if (create_noise_maps) {
                // noise map filename
                const auto noise_filename =
                    citlali::pipeline::observation_output_filename<
                        engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::noise,
                        engine_utils::toltecIO::raw>(
                        toltec_io, raw_dir,
                        citlali::pipeline::runtime_config(*this).reduction_type, array_name,
                        observation_identity.obsnum, telescope.sim_obs);
                append_fits_file(map_fits_outputs.obs_noise, noise_filename);
            }

            // map filtering
            if (create_filtered_maps) {
                // filtered map filename
                const auto filtered_filename =
                    citlali::pipeline::observation_output_filename<
                        engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::map,
                        engine_utils::toltecIO::filtered>(
                        toltec_io, filtered_dir,
                        citlali::pipeline::runtime_config(*this).reduction_type, array_name,
                        observation_identity.obsnum, telescope.sim_obs);
                append_fits_file(map_fits_outputs.filtered_obs, filtered_filename);

                // filtered noise maps
                if (create_filtered_noise_maps) {
                    // filtered noise map filename
                    const auto filtered_noise_filename =
                        citlali::pipeline::observation_output_filename<
                            engine_utils::toltecIO::toltec,
                            engine_utils::toltecIO::noise,
                            engine_utils::toltecIO::filtered>(
                            toltec_io, filtered_dir,
                            citlali::pipeline::runtime_config(*this).reduction_type, array_name,
                            observation_identity.obsnum, telescope.sim_obs);
                    append_fits_file(map_fits_outputs.filtered_obs_noise,
                                     filtered_noise_filename);
                }
            }
        }
    }
}
