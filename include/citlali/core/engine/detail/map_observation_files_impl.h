#pragma once

// Engine output implementation detail.
// Include this only after Engine has been declared.

void Engine::create_obs_map_files() {
    // clear fits vectors for each observation
    citlali::pipeline::clear_observation_map_fits_files(
        fits_io_vec, noise_fits_io_vec, filtered_fits_io_vec,
        filtered_noise_fits_io_vec);
    const std::string raw_dir =
        citlali::pipeline::raw_observation_map_directory(obsnum_dir_name);
    const std::string filtered_dir =
        citlali::pipeline::filtered_observation_map_directory(
            obsnum_dir_name);
    auto make_fits_io = [](const std::string &filename) {
        return fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>(filename);
    };
    auto append_fits_file = [&](auto &fits_files,
                                const std::string &filename) {
        citlali::pipeline::append_observation_map_fits_file(
            fits_files, filename, make_fits_io);
    };
    const std::string reduction_type_name{
        citlali::config::to_string(typed_config.runtime.reduction_type)};
    const bool create_per_obs_outputs =
        citlali::pipeline::should_create_observation_per_obs_outputs(
            run_coadd);
    const bool create_noise_maps =
        citlali::pipeline::should_create_observation_noise_maps(
            run_noise, write_noise_realizations);
    const bool create_filtered_maps =
        citlali::pipeline::should_create_observation_filtered_maps(
            run_map_filter);
    const bool create_filtered_noise_maps =
        citlali::pipeline::should_create_observation_filtered_noise_maps(
            run_noise, write_noise_realizations);

    // loop through arrays
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        // array index
        auto array = calib.arrays[i];
        // array name
        std::string array_name = toltec_io.array_name_map[array];
        // map filename
        const auto filename =
            toltec_io
                .create_filename<engine_utils::toltecIO::toltec,
                                 engine_utils::toltecIO::map,
                                 engine_utils::toltecIO::raw>(
                    raw_dir, reduction_type_name, array_name, obsnum,
                    telescope.sim_obs);
        append_fits_file(fits_io_vec, filename);

        // if noise maps are requested but coadding is not, populate noise fits vector
        if (create_per_obs_outputs) {
            if (create_noise_maps) {
                // noise map filename
                const auto noise_filename =
                    toltec_io
                        .create_filename<engine_utils::toltecIO::toltec,
                                         engine_utils::toltecIO::noise,
                                         engine_utils::toltecIO::raw>(
                            raw_dir, reduction_type_name, array_name, obsnum,
                            telescope.sim_obs);
                append_fits_file(noise_fits_io_vec, noise_filename);
            }

            // map filtering
            if (create_filtered_maps) {
                // filtered map filename
                const auto filtered_filename =
                    toltec_io
                        .create_filename<engine_utils::toltecIO::toltec,
                                         engine_utils::toltecIO::map,
                                         engine_utils::toltecIO::filtered>(
                            filtered_dir, reduction_type_name, array_name,
                            obsnum,
                            telescope.sim_obs);
                append_fits_file(filtered_fits_io_vec, filtered_filename);

                // filtered noise maps
                if (create_filtered_noise_maps) {
                    // filtered noise map filename
                    const auto filtered_noise_filename =
                        toltec_io
                            .create_filename<engine_utils::toltecIO::toltec,
                                             engine_utils::toltecIO::noise,
                                             engine_utils::toltecIO::filtered>(
                                filtered_dir, reduction_type_name, array_name,
                                obsnum,
                                telescope.sim_obs);
                    append_fits_file(filtered_noise_fits_io_vec,
                                     filtered_noise_filename);
                }
            }
        }
    }
}
