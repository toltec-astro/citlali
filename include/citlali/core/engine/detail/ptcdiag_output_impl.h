#pragma once

// Engine diagnostic output implementation detail.
// Include this only after Engine has been declared.

void Engine::create_ptcdiag_file() {
    ptcdiag_filename =
        citlali::pipeline::diagnostic_output_netcdf_filename<
            engine_utils::toltecIO::toltec,
            engine_utils::toltecIO::ptcdiag,
            engine_utils::toltecIO::raw>(
            toltec_io, obsnum_dir_name,
            typed_config.timestream.output.subdir_name,
            typed_config.runtime.reduction_type, obsnum, telescope.sim_obs);

    write_netcdf_atomic(ptcdiag_filename, [&](netCDF::NcFile &fo) {
    const int fill_int = citlali::pipeline::ptcdiag_fill_int();
    const double fill_double = citlali::pipeline::ptcdiag_fill_double();
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    const auto ptcdiag_dims =
        citlali::pipeline::add_ptcdiag_dims(fo, n_scans, calib.n_dets);

    citlali::pipeline::add_diagnostic_file_identity_vars(
        fo, "ptcdiag", std::stoi(obsnum),
        telescope.tel_header["Header.Source.Ra"](0),
        telescope.tel_header["Header.Source.Dec"](0));

    citlali::pipeline::add_diagnostic_output_scan_index(
        fo, ptcdiag_dims.n_scans, n_scans, fill_int);

    citlali::pipeline::add_ptcdiag_detector_metadata_vars(
        fo, calib, ptcdiag_dims.n_dets, fill_int);

    citlali::pipeline::add_pipeline_identity_vars(
        fo, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
        telescope.project_id, typed_config.runtime.reduction_type,
        telescope.obs_goal, typed_config.timestream.type);
    add_netcdf_var(fo, "SAMPRATE", telescope.fsmp);

    citlali::pipeline::add_ptcdiag_file_config_vars(
        fo, ptcproc, reduction_learning);

    citlali::pipeline::add_ptcdiag_standard_detector_diag(
        fo, ptcdiag_dims.det, ptcdiag_dims.det_chunks,
        ptcdiag_dims.n_det_values, fill_int, fill_double);

    citlali::pipeline::add_ptcdiag_standard_network_blocks(
        fo, calib, ptcdiag_dims.n_scans, n_scans, fill_int, fill_double);
    });
}
