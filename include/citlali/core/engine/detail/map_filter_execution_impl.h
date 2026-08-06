#pragma once

// Engine post-processing implementation detail.
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
            map_fits_outputs.filtered_obs, map_fits_outputs.filtered_obs_noise,
            map_fits_outputs.filtered_coadd, map_fits_outputs.filtered_coadd_noise,
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
        return map_indices.maps_to_stokes(map_index);
    };
    const auto map_to_array_index = [&](Eigen::Index map_index) {
        return map_indices.maps_to_arrays(map_index);
    };
    const auto array_to_map_index = [&](Eigen::Index array_index) {
        return map_indices.arrays_to_maps(array_index);
    };
    const auto filter_callbacks =
        citlali::pipeline::make_map_filter_callbacks(
            map_to_array_index, map_to_stokes_index, array_to_map_index,
            write_filter_maps);
    const auto filter_options =
        citlali::pipeline::map_filter_run_options(*this);

    citlali::pipeline::run_map_filter_loop(
        wiener_filter, mb, map_indices.n_maps, filter_outputs,
        toltec_io.array_name_map, toltec_io.array_fwhm_arcsec,
        ASEC_TO_RAD, calib.apt, filter_options, &mb,
        rtcproc.run_polarization, rtcproc.polarization,
        filter_callbacks, logger);

    const auto published_data_paths =
        citlali::pipeline::noise_fits_output_paths(*filtered_fits_io);
    const auto published_noise_paths =
        citlali::pipeline::noise_fits_output_paths(*filtered_noise_fits_io);

    citlali::pipeline::finalize_map_filter_fits_outputs_if_needed(
        filter_options.write_filtered_maps_partial, filtered_fits_io,
        filtered_noise_fits_io, map_label, logger);

    if (filter_options.write_filtered_maps_partial) {
        constexpr bool is_coadd = map_t == mapmaking::RawCoadd ||
            map_t == mapmaking::FilteredCoadd;
        constexpr bool is_filtered = map_t == mapmaking::FilteredObs ||
            map_t == mapmaking::FilteredCoadd;
        citlali::pipeline::record_noise_map_output_publication(
            citlali::pipeline::noise_plan(*this), is_coadd, is_filtered,
            mb, published_data_paths, published_noise_paths);
    }
}
