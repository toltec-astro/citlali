#pragma once

// Engine output implementation detail.
// Include this only after Engine has been declared.

template <typename fits_io_type, class map_buffer_t>
void Engine::add_phdu(fits_io_type &fits_io, map_buffer_t &mb, Eigen::Index i) {
    if (i < 0 || i >= static_cast<Eigen::Index>(fits_io->size())) {
        logger->error("add_phdu index out of range: i={} fits_io_size={}",
                      static_cast<long long>(i), static_cast<long long>(fits_io->size()));
        std::exit(EXIT_FAILURE);
    }
    if (i >= calib.arrays.size()) {
        logger->error("add_phdu array index out of range: i={} calib.arrays.size={}",
                      static_cast<long long>(i), static_cast<long long>(calib.arrays.size()));
        std::exit(EXIT_FAILURE);
    }

    const auto array_id = citlali::pipeline::phdu_array_id(calib.arrays, i);

    // array name
    std::string name = citlali::pipeline::phdu_array_name(
        toltec_io.array_name_map, array_id);
    auto &fits_entry = fits_io->at(i);

    try {
    logger->debug("adding unit conversions");

    // conversion to Rayleigh-Jeans uK brightness temperature
    auto fwhm = citlali::pipeline::mean_beam_fwhm_arcsec(
        calib.array_fwhms[array_id]);
    auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(
        1, toltec_io.array_freq_map[array_id], fwhm);

    // beam area in steradians
    auto beam_area_rad = citlali::pipeline::gaussian_beam_area_sr(
        fwhm, FWHM_TO_STD, ASEC_TO_RAD, pi);
    // get Jy/pixel
    auto mJy_beam_to_Jy_px =
        citlali::pipeline::mjy_beam_to_jy_pixel_factor(
            beam_area_rad, mb->pixel_size_rad);

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
        mJy_beam_to_uK, mJy_beam_to_Jy_px);

    // add source flux for beammaps
    if (redu_type == "beammap") {
        citlali::pipeline::add_phdu_beammap_source_flux(
            fits_entry, name, logger, beammap_fluxes_mJy_beam[name],
            beammap_fluxes_MJy_Sr[name]);

        citlali::pipeline::add_phdu_beammap_tuning(
            fits_entry, name, logger, beammap_iter_tolerance,
            beammap_convergence_radius_arcsec, beammap_iter_max,
            beammap_phase_split_enabled, beammap_locator_iter,
            beammap_measurement_start_iter, beammap_derotate);
        // add reference detector information
        citlali::pipeline::BeammapReferenceHeaderValues reference_values;
        if (beammap_subtract_reference) {
            reference_values =
                citlali::pipeline::beammap_reference_header_values(
                    calib, beammap_reference_det);
        }
        citlali::pipeline::add_phdu_beammap_reference(
            fits_entry, name, logger, beammap_subtract_reference,
            reference_values);
    }

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
            fmt::format("failed to add PHDU/header for array '{}' (file={}): {}",
                        name, fits_io->at(i).filepath, e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("failed to add PHDU/header for array '{}' (file={}): {}",
                        name, fits_io->at(i).filepath, e.what()));
    }
}

template <typename fits_io_type, class map_buffer_t>
void Engine::write_maps(fits_io_type &fits_io, fits_io_type &noise_fits_io, map_buffer_t &mb, Eigen::Index i) {
    if (!citlali::pipeline::has_map_data_slots(
            i, static_cast<Eigen::Index>(mb->signal.size()),
            static_cast<Eigen::Index>(mb->weight.size()))) {
        logger->error("write_maps map index out of range: i={} signal_size={} weight_size={}",
                      static_cast<long long>(i),
                      static_cast<long long>(mb->signal.size()),
                      static_cast<long long>(mb->weight.size()));
        std::exit(EXIT_FAILURE);
    }

    // get name for extension layer
    std::string map_name = get_map_name(i);

    const auto write_indices =
        citlali::pipeline::map_write_indices(
            i, arrays_to_maps, maps_to_stokes, maps_to_arrays);
    const Eigen::Index map_index = write_indices.map_index;
    const Eigen::Index stokes_index = write_indices.stokes_index;
    const Eigen::Index array_index = write_indices.array_index;
    if (!citlali::pipeline::has_output_file_slot(
            map_index, static_cast<Eigen::Index>(fits_io->size()))) {
        logger->error("write_maps file index out of range: map_index={} fits_io_size={} map_i={}",
                      static_cast<long long>(map_index),
                      static_cast<long long>(fits_io->size()),
                      static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }
    if (!citlali::pipeline::has_stokes_slot(
            stokes_index,
            static_cast<Eigen::Index>(rtcproc.polarization.stokes_params.size()))) {
        logger->error("write_maps stokes index out of range: stokes_index={} stokes_size={} map_i={}",
                      static_cast<long long>(stokes_index),
                      static_cast<long long>(rtcproc.polarization.stokes_params.size()),
                      static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }
    if (!citlali::pipeline::has_array_slot(array_index, calib.arrays.size())) {
        logger->error("write_maps maps_to_arrays index out of range: maps_to_arrays(i)={} calib.arrays.size={} map_i={}",
                      static_cast<long long>(array_index),
                      static_cast<long long>(calib.arrays.size()),
                      static_cast<long long>(i));
        std::exit(EXIT_FAILURE);
    }

    const double source_epoch =
        citlali::pipeline::wcs_source_epoch_or_default(telescope.tel_header,
                                                       logger);

    // update wcs ctypes for frequency and stokes params
    citlali::pipeline::assign_map_wcs_spectral_axes(
        mb->wcs, toltec_io.array_freq_map, calib.arrays, array_index,
        stokes_index);
    const std::string &stokes_suffix = rtcproc.polarization.stokes_params[stokes_index];

    try {
        auto add_map_hdu_with_wcs = [&](const std::string &hdu_name, auto &data) {
            fits_io->at(map_index).add_hdu(hdu_name, data);
            fits_io->at(map_index).add_wcs(
                fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
        };

        // signal map
        add_map_hdu_with_wcs(
            citlali::pipeline::signal_map_hdu_name(map_name, stokes_suffix),
            mb->signal[i]);
        citlali::pipeline::add_signal_map_metadata(
            *fits_io->at(map_index).hdus.back(), mb->sig_unit);

        // weight map
        add_map_hdu_with_wcs(
            citlali::pipeline::weight_map_hdu_name(map_name, stokes_suffix),
            mb->weight[i]);
        const std::string weight_unit =
            citlali::pipeline::map_weight_unit(mb->sig_unit);
        const bool empirical_weight_calibration =
            citlali::pipeline::empirical_weight_calibration_enabled(
                run_noise_products, run_noise,
                apply_empirical_noise_weights);
        citlali::pipeline::add_weight_map_metadata(
            *fits_io->at(map_index).hdus.back(), weight_unit,
            empirical_weight_calibration);
        if (i < mb->noise_weight_scale.size()) {
            citlali::pipeline::add_empirical_weight_scale_key(
                *fits_io->at(map_index).hdus.back(), mb->noise_weight_scale(i));
        }
        if (i < mb->noise_weight_median_ratio.size()) {
            citlali::pipeline::add_weight_variance_median_key(
                *fits_io->at(map_index).hdus.back(),
                mb->noise_weight_median_ratio(i));
        }
        const bool is_beammap = redu_type == "beammap";
        const double median_err_value = mb->median_err(i);
        const double median_err =
            citlali::pipeline::map_median_error_or_zero(median_err_value,
                                                        is_beammap);
        if (citlali::pipeline::has_negative_map_median_error(
                median_err_value, is_beammap)) {
            logger->warn("negative median_err for map {} in {}; using 0", map_name,
                         fits_io->at(map_index).filepath);
        }
        citlali::pipeline::add_image_median_error_key(
            *fits_io->at(map_index).hdus.back(), median_err, mb->sig_unit);

        if (citlali::pipeline::has_map_image_slot(
                mb->weight_formal, i, mb->n_rows, mb->n_cols)) {
            add_map_hdu_with_wcs(
                citlali::pipeline::formal_weight_map_hdu_name(
                    map_name, stokes_suffix),
                mb->weight_formal[i]);
            citlali::pipeline::add_formal_weight_map_metadata(
                *fits_io->at(map_index).hdus.back(), weight_unit);
        }

        if (citlali::pipeline::has_map_image_slot(
                mb->noise_variance, i, mb->n_rows, mb->n_cols)) {
            add_map_hdu_with_wcs(
                citlali::pipeline::noise_variance_map_hdu_name(
                    map_name, stokes_suffix),
                mb->noise_variance[i]);
            const std::string variance_unit =
                citlali::pipeline::map_variance_unit(mb->sig_unit);
            citlali::pipeline::add_noise_variance_map_metadata(
                *fits_io->at(map_index).hdus.back(), variance_unit);
        }

        // kernel map
        if (rtcproc.run_kernel) {
            fits_io->at(map_index).add_hdu(
                citlali::pipeline::kernel_map_hdu_name(map_name, stokes_suffix),
                mb->kernel[i]);
            citlali::pipeline::add_image_type_key(
                *fits_io->at(map_index).hdus.back(), rtcproc.kernel.type,
                citlali::pipeline::kernel_type_comment());

            double fwhm = citlali::pipeline::kernel_fwhm_arcsec(
                rtcproc.kernel.type, rtcproc.kernel.fwhm_rad,
                calib.array_fwhms[calib.arrays(i)], RAD_TO_ASEC);
            if (citlali::pipeline::has_nonfinite_kernel_fwhm(fwhm)) {
                logger->warn("non-finite kernel FWHM for map {} in {}; using -99", map_name,
                             fits_io->at(map_index).filepath);
                fwhm = citlali::pipeline::invalid_kernel_fwhm_arcsec();
            }
            citlali::pipeline::add_kernel_fwhm_key(
                *fits_io->at(map_index).hdus.back(), fwhm);
            fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(), mb->wcs, source_epoch);
            citlali::pipeline::add_kernel_map_metadata(
                *fits_io->at(map_index).hdus.back(), mb->sig_unit);
        }

        // coverage map
        if (!mb->coverage.empty()) {
            add_map_hdu_with_wcs(
                citlali::pipeline::coverage_map_hdu_name(
                    map_name, stokes_suffix),
                mb->coverage[i]);
            citlali::pipeline::add_coverage_map_metadata(
                *fits_io->at(map_index).hdus.back());
        }

        /* coverage bool and signal-to-noise maps */
        if (!mb->coverage.empty()) {
            // get weight threshold for current map
            auto cov_region = mb->calc_cov_region(i);
            auto weight_threshold = std::get<0>(cov_region);
            if (citlali::pipeline::has_nonfinite_weight_threshold(
                    weight_threshold)) {
                logger->warn("non-finite weight threshold for map {} in {}; using 0", map_name,
                             fits_io->at(map_index).filepath);
            }
            weight_threshold =
                citlali::pipeline::weight_threshold_or_zero(weight_threshold);
            Eigen::MatrixXd coverage_bool =
                citlali::pipeline::coverage_mask_from_weight(
                    mb->weight[i], weight_threshold);

            // coverage bool map
            add_map_hdu_with_wcs(
                citlali::pipeline::coverage_mask_map_hdu_name(
                    map_name, stokes_suffix),
                coverage_bool);
            citlali::pipeline::add_coverage_mask_map_metadata(
                *fits_io->at(map_index).hdus.back());
            citlali::pipeline::add_image_weight_threshold_key(
                *fits_io->at(map_index).hdus.back(), weight_threshold);

            // legacy signal-to-noise map name retained for compatibility; this is pixel S/N.
            Eigen::MatrixXd sig2noise =
                citlali::pipeline::pixel_snr_image_or_fallback(
                    mb->sig2noise_pixel, i, mb->n_rows, mb->n_cols,
                    mb->signal[i], mb->weight[i]);
            add_map_hdu_with_wcs(
                citlali::pipeline::legacy_pixel_snr_map_hdu_name(
                    map_name, stokes_suffix),
                sig2noise);
            citlali::pipeline::add_legacy_pixel_snr_map_metadata(
                *fits_io->at(map_index).hdus.back());

            add_map_hdu_with_wcs(
                citlali::pipeline::pixel_snr_map_hdu_name(
                    map_name, stokes_suffix),
                sig2noise);
            citlali::pipeline::add_pixel_snr_map_metadata(
                *fits_io->at(map_index).hdus.back());

            const bool is_filtered_output =
                citlali::pipeline::is_filtered_map_output(
                    fits_io, filtered_fits_io_vec, filtered_coadd_fits_io_vec);
            if (is_filtered_output &&
                citlali::pipeline::has_map_image_slot(
                    mb->point_source_uncertainty, i, mb->n_rows,
                    mb->n_cols)) {
                add_map_hdu_with_wcs(
                    citlali::pipeline::point_source_flux_map_hdu_name(
                        map_name, stokes_suffix),
                    mb->signal[i]);
                citlali::pipeline::add_point_source_flux_map_metadata(
                    *fits_io->at(map_index).hdus.back(), mb->sig_unit);
                citlali::pipeline::add_point_source_response_norm_key(
                    *fits_io->at(map_index).hdus.back(), 1.0);

                add_map_hdu_with_wcs(
                    citlali::pipeline::point_source_uncertainty_map_hdu_name(
                        map_name, stokes_suffix),
                    mb->point_source_uncertainty[i]);
                citlali::pipeline::add_point_source_uncertainty_map_metadata(
                    *fits_io->at(map_index).hdus.back(), mb->sig_unit);

                add_map_hdu_with_wcs(
                    citlali::pipeline::point_source_snr_map_hdu_name(
                        map_name, stokes_suffix),
                    mb->sig2noise_point_source[i]);
                citlali::pipeline::add_point_source_snr_map_metadata(
                    *fits_io->at(map_index).hdus.back());
            }
        }

        // write noise maps
        if (citlali::pipeline::should_write_noise_maps(mb->noise,
                                                       noise_fits_io)) {
            if (!citlali::pipeline::has_noise_fits_slot(noise_fits_io,
                                                        map_index)) {
                logger->error("write_maps noise file index out of range: map_index={} noise_fits_io_size={} map_i={}",
                              static_cast<long long>(map_index),
                              static_cast<long long>(noise_fits_io->size()),
                              static_cast<long long>(i));
                std::exit(EXIT_FAILURE);
            }
            if (!citlali::pipeline::has_noise_map_slot(mb->noise, i)) {
                logger->error("write_maps noise map index out of range: i={} noise_size={}",
                              static_cast<long long>(i), static_cast<long long>(mb->noise.size()));
                std::exit(EXIT_FAILURE);
            }
            const double median_rms =
                citlali::pipeline::map_median_rms_or_zero(mb->median_rms, i);
            if (citlali::pipeline::has_nonfinite_map_median_rms(
                    mb->median_rms, i)) {
                logger->warn("non-finite median_rms for map {} in {}; using 0", map_name,
                             noise_fits_io->at(map_index).filepath);
            }
            auto add_noise_map_hdu_with_wcs = [&](const std::string &hdu_name, auto &data) {
                noise_fits_io->at(map_index).add_hdu(hdu_name, data);
                noise_fits_io->at(map_index).add_wcs(
                    noise_fits_io->at(map_index).hdus.back(), mb->wcs,
                    source_epoch);
            };
            for (Eigen::Index n=0; n<mb->n_noise; ++n) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(mb->noise[i].data() + n * mb->n_rows * mb->n_cols,
                                                                                               mb->n_rows, mb->n_cols);

                add_noise_map_hdu_with_wcs(
                    citlali::pipeline::noise_signal_map_hdu_name(
                        map_name, n, stokes_suffix),
                    noise_matrix);
                citlali::pipeline::add_noise_image_summary_keys(
                    *noise_fits_io->at(map_index).hdus.back(), mb->sig_unit,
                    median_rms);
            }
        }
    } catch (const CCfits::FitsError &e) {
        throw std::runtime_error(
            fmt::format("failed to write map '{}' (map_i={} file={} noise_file={}): {}",
                        map_name,
                        static_cast<long long>(i),
                        fits_io->at(map_index).filepath,
                        citlali::pipeline::noise_file_path_or_na(
                            mb->noise, noise_fits_io, map_index),
                        e.message()));
    } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("failed to write map '{}' (map_i={} file={} noise_file={}): {}",
                        map_name,
                        static_cast<long long>(i),
                        fits_io->at(map_index).filepath,
                        citlali::pipeline::noise_file_path_or_na(
                            mb->noise, noise_fits_io, map_index),
                        e.what()));
    }
}

