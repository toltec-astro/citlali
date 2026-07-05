#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::run_wiener_filter(map_buffer_t &mb) {
    citlali::pipeline::reset_map_filter_edge_guard_storage(mb);

    using FitsVector =
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>>;
    auto add_phdu_for_filter = [&](auto *fits, auto *buffer,
                                   Eigen::Index fits_index) {
        add_phdu(fits, buffer, fits_index);
    };
    auto filter_outputs =
        citlali::pipeline::prepare_map_filter_outputs<map_t, FitsVector>(
            filtered_fits_io_vec, filtered_noise_fits_io_vec,
            filtered_coadd_fits_io_vec, filtered_coadd_noise_fits_io_vec,
            &mb, logger, add_phdu_for_filter);
    FitsVector *filtered_fits_io = filter_outputs.filtered_fits_io;
    FitsVector *filtered_noise_fits_io =
        filter_outputs.filtered_noise_fits_io;
    const char *map_label = filter_outputs.map_label;

    const auto write_filter_maps =
        [&](auto *fits, auto *noise_fits, auto *buffer,
            Eigen::Index map_index) {
            write_maps(fits, noise_fits, buffer, map_index);
        };
    const auto map_to_stokes_index = [&](Eigen::Index map_index) {
        return maps_to_stokes(map_index);
    };
    const auto map_to_array_index = [&](Eigen::Index map_index) {
        return maps_to_arrays(map_index);
    };
    const auto array_to_map_index = [&](Eigen::Index array_index) {
        return arrays_to_maps(array_index);
    };
    const auto filter_callbacks =
        citlali::pipeline::make_map_filter_callbacks(
            map_to_array_index, map_to_stokes_index, array_to_map_index,
            write_filter_maps);
    const auto filter_options =
        citlali::pipeline::map_filter_run_options(
            run_noise, write_filtered_maps_partial, run_noise_products,
            apply_empirical_noise_weights);

    citlali::pipeline::run_map_filter_loop(
        wiener_filter, mb, n_maps, filter_outputs,
        toltec_io.array_name_map, toltec_io.array_fwhm_arcsec,
        ASEC_TO_RAD, calib.apt, filter_options, &mb,
        rtcproc.run_polarization, rtcproc.polarization,
        filter_callbacks, logger);

    citlali::pipeline::finalize_map_filter_fits_outputs_if_needed(
        write_filtered_maps_partial, filtered_fits_io,
        filtered_noise_fits_io, map_label, logger);
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::find_sources(map_buffer_t &mb) {
    citlali::pipeline::detect_map_sources(mb, n_maps, logger);

    citlali::pipeline::initialize_source_fit_tables(
        mb, map_fitter.n_params);

    const auto source_fit_constants =
        citlali::pipeline::source_fit_unit_constants(
            RAD_TO_ASEC, STD_TO_FWHM, ASEC_TO_RAD, RAD_TO_DEG,
            DEG_TO_RAD, ASEC_TO_DEG);
    const auto map_to_array_index = [&](Eigen::Index map_index) {
        return maps_to_arrays(map_index);
    };
    const auto init_fwhm_for_array = [&](Eigen::Index array) {
        return citlali::pipeline::source_fit_initial_fwhm_for_array(
            toltec_io.array_fwhm_arcsec, array,
            source_fit_constants.arcsec_to_rad,
            mb.pixel_size_rad);
    };
    const auto fit_map_sources =
        [&](Eigen::Index map_index, Eigen::Index n_map_sources,
            double init_fwhm, Eigen::Index source_row_start) {
            citlali::pipeline::fit_source_candidates(
                parallel_policy, n_map_sources, [&](auto j) {
                const auto init_position =
                    citlali::pipeline::source_initial_position(
                        mb, map_index, j);

                auto [params, perrors, good_fit] =
                    map_fitter.fit_to_gaussian<
                        engine_utils::mapFitter::pointing>(
                            mb.signal[map_index], mb.weight[map_index],
                            init_fwhm, init_position.row,
                            init_position.col);
                if (good_fit) {
                    const auto tangent_to_abs = [](auto &lat, auto &lon,
                                                   double crval_lat,
                                                   double crval_lon) {
                        return engine_utils::tangent_to_abs(
                            lat, lon, crval_lat, crval_lon);
                    };
                    citlali::pipeline::normalize_and_store_source_fit_result(
                        mb, source_row_start, j, params, perrors,
                        telescope.pixel_axes, source_fit_constants,
                        tangent_to_abs);
                }
            });
        };
    const auto source_fit_callbacks =
        citlali::pipeline::make_source_fit_callbacks(
            map_to_array_index, init_fwhm_for_array, fit_map_sources);
    citlali::pipeline::fit_detected_map_sources(
        mb, n_maps, source_fit_callbacks);
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_sources(map_buffer_t &mb, std::string dir_name) {
    // get filename for source table
    const std::string source_filename =
        setup_filenames<map_t, engine_utils::toltecIO::source,
                        engine_utils::toltecIO::map>(dir_name);

    const auto map_to_array_index = [&](Eigen::Index map_index) {
        return maps_to_arrays(map_index);
    };
    const auto calc_map_std_dev = [](auto &signal) {
        return engine_utils::calc_std_dev(signal);
    };
    const auto write_source_table =
        [&](const std::string &filename, auto &source_table,
            auto source_header, auto source_meta) {
            to_ecsv_from_matrix(
                filename, source_table, source_header, source_meta);
        };
    const auto source_table_callbacks =
        citlali::pipeline::make_source_table_callbacks(
            map_to_array_index, calc_map_std_dev, write_source_table);
    citlali::pipeline::write_source_table_output(
        source_filename, *mb, map_fitter.n_params, telescope.pixel_axes,
        telescope.source_name, engine_utils::current_date_time(),
        date_obs.back(), calib.apt_header_description,
        source_table_callbacks);
}
