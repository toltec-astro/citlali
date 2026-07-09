#pragma once

// Engine diagnostic output implementation detail.
// Include this only after Engine has been declared.

void Engine::write_stats() {
    std::string stats_dir =
        citlali::pipeline::stats_raw_directory(obsnum_dir_name);
    const auto &tod_output_subdir_name =
        typed_config.timestream.output.subdir_name;
    // if using tod subdir, put stats file in it
    const bool has_tod_output_subdir =
        citlali::pipeline::stats_has_tod_output_subdir(
            tod_output_subdir_name);
    if (has_tod_output_subdir) {
        const auto stats_subdir_path =
            citlali::pipeline::stats_tod_output_subdir_path(
                stats_dir, tod_output_subdir_name);
        if (!fs::exists(fs::status(stats_subdir_path))) {
            fs::create_directories(stats_subdir_path);
            stats_dir =
                citlali::pipeline::stats_directory_from_subdir(
                    stats_subdir_path);
        }
    }
    const auto stats_netcdf_filename =
        citlali::pipeline::stats_output_netcdf_filename<
            engine_utils::toltecIO::toltec,
            engine_utils::toltecIO::stats,
            engine_utils::toltecIO::raw>(
            toltec_io, stats_dir, typed_config.runtime.reduction_type, obsnum,
            telescope.sim_obs);
    write_netcdf_atomic(stats_netcdf_filename, [&](netCDF::NcFile &fo) {

    citlali::pipeline::add_stats_file_outputs(
        fo, std::stoi(obsnum), calib, diagnostics, ptcproc.cleaner, logger,
        omb.sig_unit, telescope.scan_indices.cols(),
        citlali::pipeline::ptcdiag_fill_double());
    });
}
