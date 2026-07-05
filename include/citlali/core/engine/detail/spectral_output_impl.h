#pragma once

// Engine output implementation detail.
// Include this only after Engine has been declared.

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_psd(map_buffer_t &mb, std::string dir_name) {
    // get filename
    const std::string filename =
        setup_filenames<map_t, engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::psd>(dir_name);

    write_netcdf_atomic(filename + ".nc", [&](netCDF::NcFile &fo) {

    auto map_name_for_index = [&](Eigen::Index i) {
        return get_map_name(i);
    };
    citlali::pipeline::add_spectral_psd_products_for_maps(
        fo, mb, toltec_io.array_name_map, calib.arrays,
        rtcproc.polarization.stokes_params, map_name_for_index,
        arrays_to_maps, maps_to_stokes);
    });
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_hist(map_buffer_t &mb, std::string dir_name) {
    const std::string filename =
        setup_filenames<map_t, engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::hist>(dir_name);

    write_netcdf_atomic(filename + ".nc", [&](netCDF::NcFile &fo) {
    netCDF::NcDim hist_bins_dim =
        citlali::pipeline::add_spectral_histogram_bins_dim(
            fo, mb->hist_n_bins);

    auto map_name_for_index = [&](Eigen::Index i) {
        return get_map_name(i);
    };
    citlali::pipeline::add_spectral_histogram_products_for_maps(
        fo, mb, hist_bins_dim, toltec_io.array_name_map, calib.arrays,
        rtcproc.polarization.stokes_params, map_name_for_index,
        arrays_to_maps, maps_to_stokes);
    });
}

