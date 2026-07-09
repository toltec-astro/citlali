#pragma once

// Engine post-processing implementation detail.
// Include this only after Engine has been declared.

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::find_sources(map_buffer_t &mb) {
    citlali::pipeline::detect_map_sources(mb, map_indices.n_maps, logger);

    citlali::pipeline::initialize_source_fit_tables(
        mb, map_fitter.n_params);

    const auto source_fit_constants =
        citlali::pipeline::source_fit_unit_constants(
            RAD_TO_ASEC, STD_TO_FWHM, ASEC_TO_RAD, RAD_TO_DEG,
            DEG_TO_RAD, ASEC_TO_DEG);
    const auto map_to_array_index = [&](Eigen::Index map_index) {
        return map_indices.maps_to_arrays(map_index);
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
                citlali::pipeline::runtime_parallel_policy_name(*this),
                n_map_sources, [&](auto j) {
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
                        typed_config.mapmaking.pixel_axes_frame,
                        source_fit_constants,
                        tangent_to_abs);
                }
            });
        };
    const auto source_fit_callbacks =
        citlali::pipeline::make_source_fit_callbacks(
            map_to_array_index, init_fwhm_for_array, fit_map_sources);
    citlali::pipeline::fit_detected_map_sources(
        mb, map_indices.n_maps, source_fit_callbacks);
}
