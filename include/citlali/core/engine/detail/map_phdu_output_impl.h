#pragma once

// Engine FITS map output implementation detail.
// Include this only after Engine has been declared.

template <typename fits_io_type, class map_buffer_t>
void Engine::add_phdu(fits_io_type &fits_io, map_buffer_t &mb, Eigen::Index i) {
    citlali::pipeline::require_phdu_output_slots(
        i, static_cast<Eigen::Index>(fits_io->size()),
        static_cast<Eigen::Index>(calib.arrays.size()), logger);

    const auto array_id = citlali::pipeline::phdu_array_id(calib.arrays, i);

    // array name
    std::string name = citlali::pipeline::phdu_array_name(
        toltec_io.array_name_map, array_id);
    auto &fits_entry = fits_io->at(i);

    try {
    logger->debug("adding unit conversions");

    // conversion to Rayleigh-Jeans uK brightness temperature
    const auto unit_conversion =
        citlali::pipeline::phdu_unit_conversion_factors(
            calib.array_fwhms[array_id], mb->pixel_size_rad, FWHM_TO_STD,
            ASEC_TO_RAD, pi);
    auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(
        1, toltec_io.array_freq_map[array_id],
        unit_conversion.mean_fwhm_arcsec);

    auto get_tel_header_scalar = [&](const std::string &key, double fallback) {
        return citlali::pipeline::telescope_header_scalar(
            telescope.tel_header, key, fallback, logger);
    };

    auto get_tel_data_mean = [&](const std::string &key, double fallback) {
        return citlali::pipeline::telescope_data_mean(
            telescope.tel_data, key, fallback, logger);
    };

    auto add_double_key = [&](const std::string &key, double value, const std::string &comment,
                              double fallback = 0.0) {
        citlali::pipeline::add_phdu_double_key(
            fits_entry, name, logger, key, value, comment, fallback);
    };

    // add unit conversions
    citlali::pipeline::add_phdu_unit_conversion_config(
        fits_entry, name, logger, rtcproc.run_calibrate, mb->sig_unit,
        calib.array_beam_areas[array_id]*MJY_SR_TO_mJY_ASEC,
        mJy_beam_to_uK, unit_conversion.mjy_beam_to_jy_pixel);

    // add source flux and tuning for beammaps
    citlali::pipeline::add_phdu_beammap_keys_if_needed(
        fits_entry, name, logger, redu_type, beammap_fluxes_mJy_beam,
        beammap_fluxes_MJy_Sr, beammap_iter_tolerance,
        beammap_convergence_radius_arcsec, beammap_iter_max,
        beammap_phase_split_enabled, beammap_locator_iter,
        beammap_measurement_start_iter, beammap_derotate,
        beammap_subtract_reference, calib, beammap_reference_det);

    logger->debug("adding obsnums");

    // add obsnums
    citlali::pipeline::add_phdu_obsnum_keys(fits_entry, mb->obsnums);

    // add date and time of obs
    citlali::pipeline::add_phdu_date_obs_keys(
        fits_entry, mb->obsnums, date_obs);

    logger->debug("adding obs info");

    citlali::pipeline::add_phdu_pipeline_identity_keys(
        fits_entry, telescope.source_name, calib.run_hwpr, name,
        CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
        telescope.project_id, redu_type, telescope.obs_goal, tod_type,
        map_grouping, map_method);
    const double source_ra = get_tel_header_scalar("Header.Source.Ra", 0.0);
    const double source_dec = get_tel_header_scalar("Header.Source.Dec", 0.0);
    citlali::pipeline::add_phdu_map_geometry_keys(
        fits_entry, name, logger, mb->exposure_time, telescope.pixel_axes,
        source_ra, source_dec, RAD_TO_DEG*get_tel_data_mean("TelElAct", 0.0),
        RAD_TO_DEG*get_tel_data_mean("TelAzAct", 0.0),
        RAD_TO_DEG*get_tel_data_mean("ActParAng", 0.0));

    logger->debug("adding beamsizes");

    // add beamsizes
    citlali::pipeline::add_phdu_beam_geometry_keys(
        fits_entry, name, logger, calib.array_fwhms[array_id],
        calib.array_pas[array_id], RAD_TO_DEG, pi/2);

    citlali::pipeline::add_phdu_auxiliary_scalar_keys(
        fits_entry, mb->sig_unit, telescope.fsmp, fruit_iter);

    // add jinc shape params
    if (map_method=="jinc") {
        logger->debug("adding jinc params");

        citlali::pipeline::add_phdu_jinc_shape_keys(
            fits_entry, name, logger, jinc_mm.r_max,
            jinc_mm.shape_params[array_id]);
    }

    // add mean tau
    logger->debug("adding extinction");
    const double mean_tau = citlali::pipeline::phdu_mean_tau(
        rtcproc, telescope, calib, i, logger);
    add_double_key("MEAN_TAU", mean_tau, "mean tau (" + name + ")");

    citlali::pipeline::add_phdu_apt_key_if_single_observation(
        fits_entry, mb->obsnums, calib.apt_filepath, logger);

    const double rms = citlali::pipeline::phdu_oof_rms(
        mb, i, redu_type, name, fits_io->at(i).filepath, logger);

    // out-of-focus holography parameters
    if (! telescope.sim_obs) {
        logger->debug("adding oof params");
        citlali::pipeline::add_phdu_oof_keys(
            fits_entry, name, logger, rms, mb->sig_unit,
            toltec_io.array_wavelength_map[array_id]/1000.,
            static_cast<int>(toltec_io.array_wavelength_map[array_id]*1000),
            get_tel_header_scalar("Header.M2.XReq", 0.0)/1000.*1e6,
            get_tel_header_scalar("Header.M2.YReq", 0.0)/1000.*1e6,
            get_tel_header_scalar("Header.M2.ZReq", 0.0)/1000.*1e6);
    }
    // add control/runtime parameters
    logger->debug("adding config params");
    const bool run_any_tod_filter =
        citlali::pipeline::phdu_any_tod_filter_enabled(rtcproc);
    citlali::pipeline::add_phdu_initial_runtime_config(
        fits_entry, verbose_mode, rtcproc.run_polarization,
        rtcproc.run_despike);
    citlali::pipeline::add_phdu_rtc_local_despike_config(
        fits_entry, name, logger, rtcproc.despiker.local_residual);
    citlali::pipeline::add_phdu_tod_filter_runtime_config(
        fits_entry, name, logger, rtcproc, run_any_tod_filter);
    citlali::pipeline::add_phdu_tod_edge_guard_config(
        fits_entry, rtcproc.filter_edge_guard, telescope.outer_scans_chunk);
    citlali::pipeline::add_phdu_tod_processing_config(fits_entry, rtcproc);
    citlali::pipeline::add_phdu_weight_selection_config(
        fits_entry, name, logger, ptcproc, rtcproc);
    citlali::pipeline::add_phdu_rtc_event_mask_config(
        fits_entry, name, logger, rtcproc);
    citlali::pipeline::add_phdu_reduction_learning_config(
        fits_entry, name, logger, reduction_learning);
    citlali::pipeline::add_phdu_weight_corr_penalty_config(
        fits_entry, name, logger, ptcproc.weight_corr_penalty);
    citlali::pipeline::add_phdu_busy_row_suppression_config(
        fits_entry, name, logger, ptcproc.busy_row_suppression);
    const auto n_eig_removed =
        ptcproc.run_clean ? ptcproc.cleaner.n_eig_to_cut[array_id].sum()
                          : 0;
    citlali::pipeline::add_phdu_cleaner_config(
        fits_entry, name, logger, ptcproc, n_eig_removed);

    const double fruit_loops_flux_limit =
        citlali::pipeline::phdu_fruit_loop_flux_limit(
            ptcproc, calib.arrays, i, array_id);
    citlali::pipeline::add_phdu_fruit_loops_config(
        fits_entry, name, logger, ptcproc, fruit_loops_flux_limit,
        mb->sig_unit);

    if (redu_type == "pointing") {
        citlali::pipeline::add_phdu_pointing_config(
            fits_entry, name, logger, pointing_source_strategy,
            pointing_fit_gaussian_enabled, pointing_fruitloops_center_mode,
            pointing_header_center_max_radius_arcsec,
            pointing_header_center_require_coverage);
    }

    citlali::pipeline::add_phdu_telescope_header_keys_if_single_observation(
        fits_entry, mb->obsnums, name, logger, telescope.tel_header);
    } catch (const CCfits::FitsError &e) {
        throw std::runtime_error(
            citlali::pipeline::phdu_write_error_message(
                name, fits_io->at(i).filepath, e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            citlali::pipeline::phdu_write_error_message(
                name, fits_io->at(i).filepath, e.what()));
    }
}
