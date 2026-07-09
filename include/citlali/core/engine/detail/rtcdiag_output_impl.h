#pragma once

// Engine diagnostic output implementation detail.
// Include this only after Engine has been declared.

void Engine::create_rtcdiag_file() {
    rtcdiag_filename =
        citlali::pipeline::diagnostic_output_netcdf_filename<
            engine_utils::toltecIO::toltec,
            engine_utils::toltecIO::rtcdiag,
            engine_utils::toltecIO::raw>(
            toltec_io, obsnum_dir_name,
            typed_config.timestream.output.subdir_name,
            typed_config.runtime.reduction_type, obsnum, telescope.sim_obs);

    write_netcdf_atomic(rtcdiag_filename, [&](netCDF::NcFile &fo) {

    const int fill_int = citlali::pipeline::rtcdiag_fill_int();
    const double fill_double = citlali::pipeline::rtcdiag_fill_double();
    const Eigen::Index n_scans = telescope.scan_indices.cols();
    const double rtc_fsmp =
        citlali::pipeline::rtc_tod_stream_sample_rate(
            rtcproc, telescope.fsmp, telescope.d_fsmp);

    citlali::pipeline::add_diagnostic_file_identity_vars(
        fo, "rtcdiag", std::stoi(obsnum),
        telescope.tel_header["Header.Source.Ra"](0),
        telescope.tel_header["Header.Source.Dec"](0));

    const auto rtcdiag_dims =
        citlali::pipeline::add_rtcdiag_dims(
            fo, n_scans, calib.n_dets, calib.n_arrays, calib.n_nws);

    citlali::pipeline::add_diagnostic_output_scan_index(
        fo, rtcdiag_dims.n_scans, n_scans, fill_int);

    citlali::pipeline::add_rtcdiag_array_ids(
        fo, calib, rtcdiag_dims.n_arrays, fill_int);

    const auto scan_summary =
        citlali::pipeline::calculate_rtcdiag_scan_summary(
            telescope, n_scans, rtcdiag_dims.n_scan_values, RAD_TO_ASEC,
            fill_double, logger);
    citlali::pipeline::add_rtcdiag_scan_summary_outputs(
        fo, rtcdiag_dims.n_scans, rtcdiag_dims.scan_chunks, scan_summary);

    const auto scan_array_summary =
        citlali::pipeline::calculate_rtcdiag_scan_array_summary(
            calib, rtcproc, scan_summary.scan_speed_p995_arcsec_s,
            n_scans, rtcdiag_dims.n_array_values,
            rtcdiag_dims.n_scan_array_values, pi, FWHM_TO_STD,
            fill_double);
    citlali::pipeline::add_rtcdiag_scan_array_summary_outputs(
        fo, rtcdiag_dims.scan_array, rtcdiag_dims.scan_array_chunks,
        scan_array_summary);

    citlali::pipeline::add_rtcdiag_network_ids(
        fo, calib, rtcdiag_dims.n_nws, fill_int);

    citlali::pipeline::add_pipeline_identity_vars(
        fo, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
        telescope.project_id, typed_config.runtime.reduction_type,
        telescope.obs_goal, typed_config.timestream.type);
    add_netcdf_var(fo, "SAMPRATE", telescope.fsmp);
    citlali::pipeline::add_rtcdiag_file_config_vars(
        fo, rtcproc, reduction_learning, verbose_mode,
        telescope.outer_scans_chunk, rtc_fsmp);

    citlali::pipeline::add_rtcdiag_apt_double_vars(
        fo, calib, rtcdiag_dims.n_dets);

    citlali::pipeline::add_rtcdiag_standard_detector_outputs(
        fo, rtcdiag_dims.det, rtcdiag_dims.det_chunks,
        rtcdiag_dims.n_det_values, fill_int, fill_double);

    citlali::pipeline::add_rtcdiag_standard_network_outputs(
        fo, rtcdiag_dims.nw, rtcdiag_dims.nw_chunks,
        rtcdiag_dims.n_nw_values, fill_int, fill_double);

    citlali::pipeline::add_rtcdiag_impulsive_capture_file_outputs_if_needed(
        fo, rtcproc.impulsive_capture, rtcdiag_dims.n_scans,
        rtcdiag_dims.n_nws, n_scans, calib.n_nws, rtc_fsmp, fill_int,
        fill_double);

    });
}
