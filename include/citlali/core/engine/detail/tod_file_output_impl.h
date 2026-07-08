#pragma once

// Engine output implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/output_policy.h>

template <class map_buffer_t>
void Engine::add_tod_header(map_buffer_t &mb) {
    const std::string reduction_type_name{
        citlali::config::to_string(typed_config.runtime.reduction_type)};
    const std::string tod_type_name{
        citlali::config::to_string(typed_config.timestream.type)};
    const std::string map_grouping_name{
        citlali::config::to_string(typed_config.mapmaking.grouping)};
    const std::string map_method_name{
        citlali::config::to_string(typed_config.mapmaking.method)};
    const auto &beammap_iteration_config = typed_config.beammap.iteration;
    const auto &beammap_phase_config = typed_config.beammap.phase_strategy;
    const auto &beammap_reference_config = typed_config.beammap.reference;

    // loop through viles
    for (const auto & [fkey, fval]: tod_filename) {
        netCDF::NcFile fo(fval, netCDF::NcFile::write);

        // add unit conversions
        if (rtcproc.run_calibrate) {
            citlali::pipeline::add_tod_unit_conversion_vars(
                fo, calib, toltec_io, omb.sig_unit, omb.pixel_size_rad,
                MJY_SR_TO_mJY_ASEC, FWHM_TO_STD, ASEC_TO_RAD, pi);
        }

        citlali::pipeline::add_observation_date_source_vars(
            fo, date_obs.back(), telescope.source_name);

        // add source flux for beammaps
        if (typed_config.runtime.reduction_type ==
            citlali::config::ReductionType::beammap) {
            citlali::pipeline::add_beammap_tod_header_vars(
                fo, calib, toltec_io.array_name_map,
                beammap_fluxes_mJy_beam, beammap_fluxes_MJy_Sr,
                beammap_iteration_config.tolerance,
                beammap_iteration_config.convergence_radius_arcsec,
                beammap_iteration_config.max_iterations,
                beammap_phase_config.enabled,
                beammap_phase_config.locator_iter,
                beammap_phase_config.measurement_start_iter,
                beammap_reference_config.derotate,
                beammap_reference_config.subtract_reference_detector,
                beammap_reference_config.reference_detector);
        }

        citlali::pipeline::add_tod_identity_geometry_vars(
            fo, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
            telescope.project_id, reduction_type_name, telescope.obs_goal,
            tod_type_name, calib.run_hwpr, map_grouping_name,
            map_method_name, omb.exposure_time, telescope.pixel_axes,
            telescope.tel_header["Header.Source.Ra"][0],
            telescope.tel_header["Header.Source.Dec"][0],
            RAD_TO_DEG * telescope.tel_data["TelElAct"].mean(),
            RAD_TO_DEG * telescope.tel_data["TelAzAct"].mean(),
            RAD_TO_DEG * telescope.tel_data["ActParAng"].mean(),
            calib.arrays, calib.array_fwhms, calib.array_pas,
            toltec_io.array_name_map, RAD_TO_DEG, pi / 2, omb.sig_unit);

        citlali::pipeline::add_jinc_shape_config_vars_if_needed(
            fo, typed_config.mapmaking.method, calib.arrays, jinc_mm.shape_params,
            toltec_io.array_name_map, jinc_mm.r_max);

        citlali::pipeline::add_tod_mean_tau_vars(
            fo, rtcproc, telescope.tel_data, telescope.tau_225_GHz,
            calib, toltec_io.array_name_map);

        citlali::pipeline::add_tod_auxiliary_metadata_vars(
            fo, telescope.fsmp,
            citlali::pipeline::apt_table_header_name(
                calib.apt_filepath, logger),
            fruit_iter);

        // add control/runtime parameters
        citlali::pipeline::add_tod_initial_runtime_config_vars(
            fo, verbose_mode, rtcproc.run_polarization, rtcproc.run_despike);
        const bool run_any_tod_filter = rtcproc.run_tod_filter || rtcproc.run_tod_iir_highpass;
        citlali::pipeline::add_rtc_local_despike_config_vars(
            fo, rtcproc.despiker.local_residual);
        citlali::pipeline::add_tod_filter_runtime_config_vars(
            fo, rtcproc, run_any_tod_filter);
        citlali::pipeline::add_tod_filter_edge_guard_config_vars(
            fo, rtcproc.filter_edge_guard, telescope.outer_scans_chunk,
            rtcproc.tod_output_outer_context_samples);
        citlali::pipeline::add_tod_processing_config_vars(fo, rtcproc);
        citlali::pipeline::add_weight_selection_config_vars(fo, ptcproc);
        citlali::pipeline::add_reduction_learning_config_vars(
            fo, reduction_learning);
        add_netcdf_var(fo, "CONFIG.INV_VAR.RTC.WTLOW", rtcproc.lower_inv_var_factor);
        add_netcdf_var(fo, "CONFIG.INV_VAR.RTC.WTHIGH", rtcproc.upper_inv_var_factor);
        citlali::pipeline::add_rtc_event_mask_config_vars(fo, rtcproc);
        citlali::pipeline::add_rtc_line_audit_config_vars_if_absent(
            fo, rtcproc.line_audit);
        citlali::pipeline::add_ptc_cleaning_header_config_vars(
            fo, ptcproc, calib, toltec_io.array_name_map);

        citlali::pipeline::add_oof_header_vars_if_observed(
            fo, telescope.sim_obs, telescope.tel_header, mb,
            typed_config.runtime.reduction_type,
            citlali::pipeline::mapmaking_enabled(*this), calib,
            toltec_io.array_name_map, toltec_io.array_wavelength_map);

        citlali::pipeline::add_fruit_loop_header_config_vars(
            fo, ptcproc, calib, toltec_io.array_name_map);

        fo.close();
    }
}

template <engine_utils::toltecIO::ProdType prod_t>
void Engine::create_tod_files() {
    const std::string reduction_type_name{
        citlali::config::to_string(typed_config.runtime.reduction_type)};
    // name for std map
    const std::string dir_name = citlali::pipeline::tod_output_directory(
        obsnum_dir_name, tod_output_subdir_name);
    constexpr bool is_rtc_stream =
        prod_t == engine_utils::toltecIO::rtc_timestream;

    const std::string name =
        citlali::pipeline::register_tod_stream_output_file<
            engine_utils::toltecIO::toltec, prod_t,
            engine_utils::toltecIO::raw>(
            toltec_io, tod_filename, dir_name, reduction_type_name, obsnum,
            telescope.sim_obs, is_rtc_stream);

    write_netcdf_atomic(tod_filename[name], [&](netCDF::NcFile &fo) {

    citlali::pipeline::add_tod_stream_output_type_label(fo, is_rtc_stream);
    if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        citlali::pipeline::add_ptc_eigenvalue_dim(fo, ptcproc.cleaner.n_calc);
    }

    citlali::pipeline::add_observation_identity_vars(
        fo, std::stoi(obsnum), telescope.tel_header["Header.Source.Ra"](0),
        telescope.tel_header["Header.Source.Dec"](0));

    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        // Keep the RTC line-audit tuning alongside the RTC TOD so offline audits
        // can recover the exact per-run thresholds without the sidecar YAML.
        citlali::pipeline::add_rtc_line_audit_config_vars(
            fo, rtcproc.line_audit);
    }

    const auto tod_layout = citlali::pipeline::prepare_tod_file_layout(
        fo, is_rtc_stream, n_tod_output_scans_rtc,
        n_tod_output_scans_ptc, rtcproc, ptcproc, telescope.scan_indices,
        calib.n_dets);
    const auto &tod_dims = tod_layout.dims;
    const auto &chunking = tod_layout.chunking;
    const auto chunkMode = chunking.mode;
    const auto &chunkSizes = chunking.sizes;

    citlali::pipeline::add_tod_core_data_vars(
        fo, tod_dims.signal, tod_layout.stream.mini_output, omb.sig_unit,
        rtcproc.run_kernel, telescope.pixel_axes, chunkMode, chunkSizes);

    citlali::pipeline::add_tod_static_metadata_vars(
        fo, calib.apt, calib.apt_header_units, telescope.tel_data,
        pointing_offsets_arcsec, logger, tod_dims.n_dets, tod_dims.n_pts,
        chunkMode, chunkSizes);

    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        citlali::pipeline::add_rtc_tod_stream_diagnostic_outputs(
            fo, calib, rtcproc, tod_layout, telescope.fsmp,
            telescope.d_fsmp);
    }

    // add weights
    if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        citlali::pipeline::add_ptc_tod_stream_weight_and_diagnostic_outputs(
            fo, calib, ptcproc, tod_layout, omb.sig_unit);
    }

    citlali::pipeline::add_tod_hwpr_var_if_requested(
        fo, rtcproc.run_polarization, calib.run_hwpr, tod_dims.n_pts);

    // add tel header
    citlali::pipeline::add_telescope_header_vars(fo, telescope.tel_header);

    });
}

//template <TCDataKind tc_t>
