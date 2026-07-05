#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

void Engine::create_obs_map_files() {
    // clear fits vectors for each observation
    citlali::pipeline::clear_observation_map_fits_files(
        fits_io_vec, noise_fits_io_vec, filtered_fits_io_vec,
        filtered_noise_fits_io_vec);
    const std::string raw_dir =
        citlali::pipeline::raw_observation_map_directory(obsnum_dir_name);
    const std::string filtered_dir =
        citlali::pipeline::filtered_observation_map_directory(
            obsnum_dir_name);
    auto make_fits_io = [](const std::string &filename) {
        return fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>(filename);
    };
    auto append_fits_file = [&](auto &fits_files,
                                const std::string &filename) {
        citlali::pipeline::append_observation_map_fits_file(
            fits_files, filename, make_fits_io);
    };
    const bool create_per_obs_outputs =
        citlali::pipeline::should_create_observation_per_obs_outputs(
            run_coadd);
    const bool create_noise_maps =
        citlali::pipeline::should_create_observation_noise_maps(
            run_noise, write_noise_realizations);
    const bool create_filtered_maps =
        citlali::pipeline::should_create_observation_filtered_maps(
            run_map_filter);
    const bool create_filtered_noise_maps =
        citlali::pipeline::should_create_observation_filtered_noise_maps(
            run_noise, write_noise_realizations);

    // loop through arrays
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        // array index
        auto array = calib.arrays[i];
        // array name
        std::string array_name = toltec_io.array_name_map[array];
        // map filename
        const auto filename =
            toltec_io
                .create_filename<engine_utils::toltecIO::toltec,
                                 engine_utils::toltecIO::map,
                                 engine_utils::toltecIO::raw>(
                    raw_dir, redu_type, array_name, obsnum,
                    telescope.sim_obs);
        append_fits_file(fits_io_vec, filename);

        // if noise maps are requested but coadding is not, populate noise fits vector
        if (create_per_obs_outputs) {
            if (create_noise_maps) {
                // noise map filename
                const auto noise_filename =
                    toltec_io
                        .create_filename<engine_utils::toltecIO::toltec,
                                         engine_utils::toltecIO::noise,
                                         engine_utils::toltecIO::raw>(
                            raw_dir, redu_type, array_name, obsnum,
                            telescope.sim_obs);
                append_fits_file(noise_fits_io_vec, noise_filename);
            }

            // map filtering
            if (create_filtered_maps) {
                // filtered map filename
                const auto filtered_filename =
                    toltec_io
                        .create_filename<engine_utils::toltecIO::toltec,
                                         engine_utils::toltecIO::map,
                                         engine_utils::toltecIO::filtered>(
                            filtered_dir, redu_type, array_name, obsnum,
                            telescope.sim_obs);
                append_fits_file(filtered_fits_io_vec, filtered_filename);

                // filtered noise maps
                if (create_filtered_noise_maps) {
                    // filtered noise map filename
                    const auto filtered_noise_filename =
                        toltec_io
                            .create_filename<engine_utils::toltecIO::toltec,
                                             engine_utils::toltecIO::noise,
                                             engine_utils::toltecIO::filtered>(
                                filtered_dir, redu_type, array_name, obsnum,
                                telescope.sim_obs);
                    append_fits_file(filtered_noise_fits_io_vec,
                                     filtered_noise_filename);
                }
            }
        }
    }
}

template <class map_buffer_t>
void Engine::add_tod_header(map_buffer_t &mb) {
    // loop through viles
    for (const auto & [fkey, fval]: tod_filename) {
        netCDF::NcFile fo(fval, netCDF::NcFile::write);

        // add unit conversions
        if (rtcproc.run_calibrate) {
            citlali::pipeline::add_unit_conversion_basis_vars(fo);
            for (const auto &val: calib.arrays) {
                auto name = toltec_io.array_name_map[val];
                // conversion to Rayleigh-Jeans uK brightness temperature
                auto fwhm = (std::get<0>(calib.array_fwhms[val]) + std::get<1>(calib.array_fwhms[val]))/2;
                auto mJy_beam_to_uK = engine_utils::mJy_beam_to_uK(1, toltec_io.array_freq_map[val], fwhm);

                // beam area in steradians
                auto beam_area_rad = 2.*pi*pow(fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
                // get Jy/pixel
                auto mJy_beam_to_Jy_px = 1e-3/beam_area_rad*pow(omb.pixel_size_rad,2);

                citlali::pipeline::add_unit_conversion_array_vars(
                    fo, name, omb.sig_unit,
                    calib.array_beam_areas[val]*MJY_SR_TO_mJY_ASEC,
                    mJy_beam_to_uK, mJy_beam_to_Jy_px);
            }
        }

        citlali::pipeline::add_observation_date_source_vars(
            fo, date_obs.back(), telescope.source_name);

        // add source flux for beammaps
        if (redu_type == "beammap") {
            citlali::pipeline::add_beammap_tod_header_vars(
                fo, calib, toltec_io.array_name_map,
                beammap_fluxes_mJy_beam, beammap_fluxes_MJy_Sr,
                beammap_iter_tolerance, beammap_convergence_radius_arcsec,
                beammap_iter_max, beammap_phase_split_enabled,
                beammap_locator_iter, beammap_measurement_start_iter,
                beammap_derotate, beammap_subtract_reference,
                beammap_reference_det);
        }

        citlali::pipeline::add_tod_identity_geometry_vars(
            fo, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION, TULA_GIT_VERSION,
            telescope.project_id, redu_type, telescope.obs_goal, tod_type,
            calib.run_hwpr, map_grouping, map_method, omb.exposure_time,
            telescope.pixel_axes, telescope.tel_header["Header.Source.Ra"][0],
            telescope.tel_header["Header.Source.Dec"][0],
            RAD_TO_DEG * telescope.tel_data["TelElAct"].mean(),
            RAD_TO_DEG * telescope.tel_data["TelAzAct"].mean(),
            RAD_TO_DEG * telescope.tel_data["ActParAng"].mean(),
            calib.arrays, calib.array_fwhms, calib.array_pas,
            toltec_io.array_name_map, RAD_TO_DEG, pi / 2, omb.sig_unit);

        citlali::pipeline::add_jinc_shape_config_vars_if_needed(
            fo, map_method, calib.arrays, jinc_mm.shape_params,
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
            fo, telescope.sim_obs, telescope.tel_header, mb, redu_type,
            run_mapmaking, calib, toltec_io.array_name_map,
            toltec_io.array_wavelength_map);

        citlali::pipeline::add_fruit_loop_header_config_vars(
            fo, ptcproc, calib, toltec_io.array_name_map);

        fo.close();
    }
}

template <engine_utils::toltecIO::ProdType prod_t>
void Engine::create_tod_files() {
    // name for std map
    std::string name;
    const std::string dir_name = citlali::pipeline::tod_output_directory(
        obsnum_dir_name, tod_output_subdir_name);
    constexpr bool is_rtc_stream =
        prod_t == engine_utils::toltecIO::rtc_timestream;

    // rtc tod output filename setup
    if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::rtc_timestream,
                                                  engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                               obsnum, telescope.sim_obs);

        name = citlali::pipeline::register_tod_output_file(
            tod_filename,
            citlali::pipeline::tod_stream_output_key(is_rtc_stream),
            filename);
    }

    // ptc tod output filename setup
    else if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::ptc_timestream,
                                                  engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                               obsnum, telescope.sim_obs);

        name = citlali::pipeline::register_tod_output_file(
            tod_filename,
            citlali::pipeline::tod_stream_output_key(is_rtc_stream),
            filename);
    }

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
void Engine::cli_summary() {
    citlali::pipeline::log_reduction_map_summary(
        logger, obsnum, omb, rtcproc.run_polarization);
    const double mb_size_total =
        citlali::pipeline::log_map_memory_summary(
            logger, omb, cmb, run_coadd, run_noise);

    logger->info("estimated size of all maps {:.2f} GB", mb_size_total);
    logger->info("number of scans: {}",telescope.scan_indices.cols());
    if (run_tod_output) {
        citlali::pipeline::log_tod_output_selection_summary(
            logger, tod_output_type, n_tod_output_scans_rtc,
            rtcproc.tod_output_mini, rtcproc.tod_output_outer,
            n_tod_output_scans_ptc, ptcproc.tod_output_mini);
    }
    citlali::pipeline::log_diagnostics_sidecar_summary(logger);

    // test getting memory usage for fun
    /*struct sysinfo memInfo;
    long long totalPhysMem = memInfo.totalram;
    totalPhysMem *= memInfo.mem_unit;

    logger->info("total physical memory available {} GB", (totalPhysMem/1024)/1e7);*/
    auto phys_memory_kb = engine_utils::get_phys_memory();
    citlali::pipeline::log_physical_memory_summary(logger, phys_memory_kb);
}

template <TCDataKind tc_t>
void Engine::write_chunk_summary(TCData<tc_t, Eigen::MatrixXd> &in) {

    logger->debug("writing summary files for chunk {}",in.index.data);

    const auto filename =
        citlali::pipeline::chunk_summary_filename(in.index.data);

    // write summary log file
    std::ofstream f;
    f.open(citlali::pipeline::summary_log_path(obsnum_dir_name, filename));

    citlali::pipeline::write_chunk_summary_log(
        f, in, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION,
        engine_utils::current_date_time(), redu_type, tod_type,
        omb.sig_unit, rtcproc, telescope.outer_scans_chunk,
        (calib.apt["flag"].array()!=0).count(),
        tula::alg::median(in.scans.data),
        engine_utils::calc_std_dev(in.scans.data));

    f.close();
}

template <typename map_buffer_t>
void Engine::write_map_summary(map_buffer_t &mb) {

    logger->debug("writing map summary files");

    const auto filename = citlali::pipeline::map_summary_filename();
    std::ofstream f;
    f.open(citlali::pipeline::summary_log_path(obsnum_dir_name, filename));

    const auto nonfinite_counts =
        citlali::pipeline::count_map_summary_nonfinite(mb);
    citlali::pipeline::write_map_summary_log(
        f, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION,
        engine_utils::current_date_time(), redu_type, tod_type,
        map_grouping, n_maps, mb, nonfinite_counts);
}

template <mapmaking::MapType map_t, engine_utils::toltecIO::DataType data_t, engine_utils::toltecIO::ProdType prod_t>
auto Engine::setup_filenames(std::string dir_name) {
    return citlali::pipeline::map_output_filename<map_t, data_t, prod_t>(
        toltec_io, dir_name, redu_type, obsnum, telescope.sim_obs);
}

auto Engine::get_map_name(int i) {
    return citlali::pipeline::map_layer_name(i, map_grouping, calib);
}

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

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_mapdiag(map_buffer_t &mb, std::string dir_name) {
    const std::string filename =
        setup_filenames<map_t, engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::mapdiag>(dir_name);
    const auto mapdiag_context = citlali::pipeline::make_mapdiag_size_context(
        static_cast<std::size_t>(n_maps),
        std::max<std::size_t>(1, mb->obsnums.size()),
        map_t == mapmaking::RawCoadd || map_t == mapmaking::FilteredCoadd);
    const double fill_double = citlali::pipeline::mapdiag_fill_double();
    const int fill_int = citlali::pipeline::mapdiag_fill_int();
    const auto n_mapdiag_maps = mapdiag_context.n_maps;

    std::vector<std::string> array_names(n_mapdiag_maps);
    std::vector<std::string> stokes_names(n_mapdiag_maps);
    std::vector<std::string> map_names(n_mapdiag_maps);
    std::vector<double> median_err(n_mapdiag_maps, fill_double);
    std::vector<double> median_rms(n_mapdiag_maps, fill_double);
    std::vector<double> weight_thresholds(n_mapdiag_maps, fill_double);
    std::vector<double> weight_sum(n_mapdiag_maps, fill_double);
    std::vector<double> core_weight_sum(n_mapdiag_maps, fill_double);
    std::vector<double> coverage_sum(n_mapdiag_maps, fill_double);
    std::vector<double> coverage_max(n_mapdiag_maps, fill_double);
    std::vector<double> coverage_median_core(n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagCoverageRefs coverage_refs{
        coverage_sum,
        coverage_max,
        coverage_median_core};
    std::vector<double> empirical_to_formal_noise_ratio(
        n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagFormalNoiseRefs formal_noise_refs{
        median_err,
        median_rms,
        empirical_to_formal_noise_ratio};
    std::vector<double> noise_weight_median_ratio(n_mapdiag_maps, fill_double);
    std::vector<double> noise_weight_scale(n_mapdiag_maps, fill_double);
    std::vector<double> noise_products_s2n_sigma(n_mapdiag_maps, fill_double);
    std::vector<double> noise_products_valid_pixels(
        n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagNoiseProductRefs noise_product_refs{
        noise_weight_median_ratio,
        noise_weight_scale,
        noise_products_s2n_sigma,
        noise_products_valid_pixels};
    std::vector<double> peak_signal(n_mapdiag_maps, fill_double);
    std::vector<double> peak_abs_sig2noise(n_mapdiag_maps, fill_double);
    std::vector<double> core_peak_abs_sig2noise(n_mapdiag_maps, fill_double);
    std::vector<double> noise_rms_p16(n_mapdiag_maps, fill_double);
    std::vector<double> noise_rms_p84(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_frac_abs3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_frac_pos3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_frac_neg3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_excess_abs3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_excess_pos3(n_mapdiag_maps, fill_double);
    std::vector<double> core_tail_excess_neg3(n_mapdiag_maps, fill_double);
    std::vector<double> core_sig2noise_skew(n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagCoreTailRefs core_tail_refs{
        core_tail_frac_abs3,
        core_tail_frac_pos3,
        core_tail_frac_neg3,
        core_tail_excess_abs3,
        core_tail_excess_pos3,
        core_tail_excess_neg3,
        core_sig2noise_skew};
    std::vector<double> noise_tail_frac_abs3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_frac_pos3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_frac_neg3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_excess_abs3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_excess_pos3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_tail_excess_neg3(n_mapdiag_maps, fill_double);
    std::vector<double> noise_sig2noise_skew(n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagNoiseTailRefs noise_tail_refs{
        noise_rms_p16,
        noise_rms_p84,
        noise_tail_frac_abs3,
        noise_tail_frac_pos3,
        noise_tail_frac_neg3,
        noise_tail_excess_abs3,
        noise_tail_excess_pos3,
        noise_tail_excess_neg3,
        noise_sig2noise_skew};
    std::vector<double> edge_guard_weight_thresholds(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_hits_thresholds(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_background_levels(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_science_frac(n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_support_frac(n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_guardband_rms_pre(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_guardband_rms_post(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_exterior_rms_pre(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_exterior_rms_post(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_exterior_max_abs_pre(
        n_mapdiag_maps, fill_double);
    std::vector<double> edge_guard_exterior_max_abs_post(
        n_mapdiag_maps, fill_double);
    citlali::pipeline::MapdiagEdgeGuardDoubleRefs edge_guard_double_refs{
        edge_guard_weight_thresholds,
        edge_guard_hits_thresholds,
        edge_guard_background_levels,
        edge_guard_science_frac,
        edge_guard_support_frac,
        edge_guard_guardband_rms_pre,
        edge_guard_guardband_rms_post,
        edge_guard_exterior_rms_pre,
        edge_guard_exterior_rms_post,
        edge_guard_exterior_max_abs_pre,
        edge_guard_exterior_max_abs_post};
    std::vector<int> n_valid_pixels(n_mapdiag_maps, 0);
    std::vector<int> n_core_pixels(n_mapdiag_maps, 0);
    citlali::pipeline::MapdiagWeightRefs weight_refs{
        weight_sum,
        core_weight_sum,
        n_valid_pixels,
        n_core_pixels};
    std::vector<int> peak_row(n_mapdiag_maps, fill_int);
    std::vector<int> peak_col(n_mapdiag_maps, fill_int);
    citlali::pipeline::MapdiagPeakRefs peak_refs{
        peak_abs_sig2noise,
        core_peak_abs_sig2noise,
        peak_row,
        peak_col};
    std::vector<int> edge_guard_applied(n_mapdiag_maps, 0);
    std::vector<int> edge_guard_support_radius_pix(n_mapdiag_maps, 0);
    std::vector<int> edge_guard_science_npix(n_mapdiag_maps, 0);
    std::vector<int> edge_guard_support_npix(n_mapdiag_maps, 0);
    std::vector<int> edge_guard_guardband_npix(n_mapdiag_maps, 0);
    citlali::pipeline::MapdiagEdgeGuardIntRefs edge_guard_int_refs{
        edge_guard_applied,
        edge_guard_support_radius_pix,
        edge_guard_science_npix,
        edge_guard_support_npix,
        edge_guard_guardband_npix};
    citlali::pipeline::MapdiagMapIntValues map_int_values{
        n_valid_pixels,
        n_core_pixels,
        peak_row,
        peak_col,
        edge_guard_applied,
        edge_guard_support_radius_pix,
        edge_guard_science_npix,
        edge_guard_support_npix,
        edge_guard_guardband_npix};

    const std::size_t obs_table_size =
        citlali::pipeline::mapdiag_obs_table_size(mapdiag_context);
    std::vector<double> obs_weight_sum(obs_table_size, fill_double);
    std::vector<double> obs_weight_frac(obs_table_size, fill_double);
    std::vector<double> obs_core_weight_sum(obs_table_size, fill_double);
    std::vector<double> obs_core_weight_frac(obs_table_size, fill_double);
    std::vector<int> obs_valid_pixels(obs_table_size, fill_int);
    std::vector<int> obs_core_pixels(obs_table_size, fill_int);
    citlali::pipeline::MapdiagObsTableRefs obs_tables{
        obs_weight_sum,
        obs_core_weight_sum,
        obs_valid_pixels,
        obs_core_pixels};
    citlali::pipeline::MapdiagObservationDoubleValues obs_double_values{
        obs_weight_sum,
        obs_weight_frac,
        obs_core_weight_sum,
        obs_core_weight_frac};
    citlali::pipeline::MapdiagObservationIntValues obs_int_values{
        obs_valid_pixels,
        obs_core_pixels};
    citlali::pipeline::MapdiagMapDoubleValues map_double_values{
        median_err,
        median_rms,
        weight_thresholds,
        weight_sum,
        core_weight_sum,
        coverage_sum,
        coverage_max,
        coverage_median_core,
        empirical_to_formal_noise_ratio,
        noise_weight_median_ratio,
        noise_weight_scale,
        noise_products_s2n_sigma,
        noise_products_valid_pixels,
        peak_signal,
        peak_abs_sig2noise,
        core_peak_abs_sig2noise,
        noise_rms_p16,
        noise_rms_p84,
        core_tail_frac_abs3,
        core_tail_frac_pos3,
        core_tail_frac_neg3,
        core_tail_excess_abs3,
        core_tail_excess_pos3,
        core_tail_excess_neg3,
        core_sig2noise_skew,
        noise_tail_frac_abs3,
        noise_tail_frac_pos3,
        noise_tail_frac_neg3,
        noise_tail_excess_abs3,
        noise_tail_excess_pos3,
        noise_tail_excess_neg3,
        noise_sig2noise_skew,
        edge_guard_weight_thresholds,
        edge_guard_hits_thresholds,
        edge_guard_background_levels,
        edge_guard_science_frac,
        edge_guard_support_frac,
        edge_guard_guardband_rms_pre,
        edge_guard_guardband_rms_post,
        edge_guard_exterior_rms_pre,
        edge_guard_exterior_rms_post,
        edge_guard_exterior_max_abs_pre,
        edge_guard_exterior_max_abs_post};

    const std::string stage_name =
        citlali::pipeline::mapdiag_stage_name<map_t>();
    const auto mapdiag_metadata =
        citlali::pipeline::make_mapdiag_metadata_vars(
            stage_name, mb, map_regime, telescope.source_name,
            telescope.project_id, telescope.obs_goal, wiener_filter);
    const auto mapdiag_labels =
        citlali::pipeline::make_mapdiag_label_vars(
            array_names, stokes_names, map_names, mb->obsnums, obsnum,
            date_obs, mapdiag_context);
    const auto mapdiag_values =
        citlali::pipeline::make_mapdiag_value_vars(
            map_double_values, map_int_values, obs_double_values,
            obs_int_values);

    const citlali::pipeline::MapdiagStatsContext mapdiag_stats{fill_double};
    const std::string mapdiag_record_producer =
        citlali::pipeline::mapdiag_record_producer(stage_name);
    auto map_name_for_index = [&](Eigen::Index map_i) {
        return get_map_name(map_i);
    };

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const std::size_t idx = citlali::pipeline::mapdiag_size_index(i);
        const auto write_indices =
            citlali::pipeline::map_write_indices(
                i, arrays_to_maps, maps_to_stokes, maps_to_arrays);
        citlali::pipeline::assign_mapdiag_map_labels_from_indices(
            idx, i, write_indices, toltec_io.array_name_map, calib.arrays,
            rtcproc.polarization.stokes_params, map_name_for_index,
            {array_names, stokes_names, map_names});

        const auto cov_region = mb->calc_cov_region(i);
        auto weight_threshold = std::get<0>(cov_region);
        weight_threshold =
            citlali::pipeline::mapdiag_weight_threshold_or_zero(
                weight_threshold);
        weight_thresholds[idx] = weight_threshold;
        if (citlali::pipeline::mapdiag_has_edge_guard_entry(idx, *mb)) {
            citlali::pipeline::assign_mapdiag_edge_guard_int_entry(
                idx, *mb, edge_guard_int_refs);
            citlali::pipeline::assign_mapdiag_edge_guard_double_entry(
                idx, *mb, edge_guard_double_refs);
        }

        const auto weight_arr = mb->weight[i].array();
        const auto valid_mask =
            citlali::pipeline::mapdiag_valid_weight_mask(weight_arr);
        const auto core_mask =
            citlali::pipeline::mapdiag_core_weight_mask(
                weight_arr, weight_threshold);
        citlali::pipeline::assign_mapdiag_weight_stats(
            idx,
            citlali::pipeline::mapdiag_weight_stats(
                weight_arr, valid_mask, core_mask),
            weight_refs);

        citlali::pipeline::assign_mapdiag_formal_noise_stats(
            idx,
            citlali::pipeline::mapdiag_formal_noise_stats_or_fill(
                mb->median_err, mb->median_rms, i, fill_double),
            formal_noise_refs);
        const auto noise_product_stats =
            citlali::pipeline::mapdiag_noise_product_stats_or_fill(
                mb->noise_weight_median_ratio, mb->noise_weight_scale,
                mb->noise_s2n_sigma, mb->noise_valid_pixels, i,
                fill_double);
        citlali::pipeline::assign_mapdiag_noise_product_stats(
            idx, noise_product_stats, noise_product_refs);

        if (citlali::pipeline::mapdiag_has_coverage_map(
                mb->coverage, i)) {
            citlali::pipeline::assign_mapdiag_coverage_stats(
                idx, mb->coverage[i], core_mask, fill_double,
                coverage_refs);
        }
        peak_signal[idx] = citlali::pipeline::mapdiag_peak_signal_or_fill(
            mb->signal[i], fill_double);
        if (citlali::pipeline::mapdiag_has_signal_weight_samples(
                mb->signal[i], mb->weight[i])) {
            const Eigen::MatrixXd sig2noise =
                citlali::pipeline::mapdiag_sig2noise_image(
                    mb->signal[i], mb->weight[i]);
            citlali::pipeline::assign_mapdiag_peak_stats(
                idx,
                citlali::pipeline::mapdiag_peak_stats(
                    sig2noise, core_mask, n_core_pixels[idx], fill_double),
                peak_refs);
            const auto core_values =
                mapdiag_stats.collect_masked_values(sig2noise, core_mask);
            const auto signal_tail = mapdiag_stats.tail_stats(core_values);
            citlali::pipeline::assign_mapdiag_core_tail_stats(
                idx, signal_tail, core_tail_refs);

            if (citlali::pipeline::mapdiag_outlier_diagnostics_enabled(
                    reduction_learning)) {
                const auto source_distance_context =
                    citlali::pipeline::mapdiag_source_distance_context(
                        mb, RAD_TO_ASEC, fill_double);

                const double protect_radius =
                    citlali::pipeline::mapdiag_source_protect_radius_arcsec(
                        reduction_learning);
                const Eigen::ArrayXXd off_source_core_mask =
                    citlali::pipeline::mapdiag_off_source_core_mask(
                        core_mask, source_distance_context, protect_radius);

                const auto off_source_values =
                    mapdiag_stats.collect_masked_values(
                        sig2noise, off_source_core_mask);
                if (citlali::pipeline::mapdiag_has_enough_off_source_values(
                        off_source_values)) {
                    const auto robust_stats =
                        citlali::pipeline::mapdiag_robust_center_stats(
                            mapdiag_stats, off_source_values);
                    if (citlali::pipeline::
                            mapdiag_has_valid_robust_center_stats(
                                robust_stats)) {
                        auto candidates =
                            citlali::pipeline::make_mapdiag_pixel_candidates();
                        const bool has_contribution_products =
                            citlali::pipeline::
                                mapdiag_has_contribution_products(mb, i);
                        const double ptc_fs_hz = processed_time_chunk_fs_hz();
                        const Eigen::Index n_mapdiag_rows =
                            citlali::pipeline::mapdiag_n_rows(mb);
                        const Eigen::Index n_mapdiag_cols =
                            citlali::pipeline::mapdiag_n_cols(mb);
                        const double min_effective_samples =
                            citlali::pipeline::mapdiag_min_effective_samples(
                                reduction_learning);
                        const double min_abs_z =
                            citlali::pipeline::mapdiag_min_abs_z(
                                reduction_learning);

                        for (Eigen::Index r = 0; r < n_mapdiag_rows; ++r) {
                            for (Eigen::Index c = 0; c < n_mapdiag_cols; ++c) {
                                if (!citlali::pipeline::
                                        mapdiag_mask_pixel_is_selected(
                                            off_source_core_mask, r, c)) {
                                    continue;
                                }

                                const double value =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            mb->signal[i], r, c);
                                const double wt =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            mb->weight[i], r, c);
                                const double sn =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            sig2noise, r, c);
                                if (!citlali::pipeline::
                                        mapdiag_is_valid_outlier_pixel_value(
                                            value, wt, sn)) {
                                    continue;
                                }

                                const double n_eff =
                                    citlali::pipeline::
                                        mapdiag_effective_samples_or_fill(
                                            mb->coverage, i, r, c,
                                            mb->n_rows, mb->n_cols,
                                            ptc_fs_hz, fill_double);
                                if (!citlali::pipeline::
                                        mapdiag_passes_min_effective_samples(
                                            n_eff, min_effective_samples)) {
                                    continue;
                                }

                                const double z =
                                    citlali::pipeline::mapdiag_robust_z(
                                        sn, robust_stats);
                                if (!citlali::pipeline::
                                        mapdiag_passes_min_abs_z(z,
                                                                 min_abs_z)) {
                                    continue;
                                }

                                const double source_distance_arcsec =
                                    citlali::pipeline::
                                        mapdiag_source_distance_arcsec(
                                            r, c, source_distance_context);
                                auto candidate =
                                    citlali::pipeline::
                                        make_mapdiag_map_pixel_candidate(
                                            r, c, value, wt, n_eff, z,
                                            source_distance_arcsec,
                                            fill_int, fill_double);

                                if (has_contribution_products) {
                                    const auto contribution_map_index =
                                        citlali::pipeline::
                                            mapdiag_contribution_map_index(i);
                                    const int uid =
                                        citlali::pipeline::
                                            mapdiag_matrix_value(
                                                mb->contribution_uid[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_signal =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_signal[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_weight =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_weight[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_variance_weight =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_variance_weight[
                                                    contribution_map_index],
                                                r, c);
                                    if (citlali::pipeline::
                                            mapdiag_has_valid_contributor(
                                                uid, fill_int,
                                                contrib_signal)) {
                                        citlali::pipeline::
                                            assign_mapdiag_candidate_contributor_from_products(
                                                candidate, uid,
                                                mb->contribution_scan[
                                                    contribution_map_index],
                                                mb->contribution_sample[
                                                    contribution_map_index],
                                                r, c);
                                        const double total_signal =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_signal[
                                                        contribution_map_index],
                                                    r, c);
                                        const double total_weight =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_weight[
                                                        contribution_map_index],
                                                    r, c);
                                        const double total_variance_weight =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_variance_weight[
                                                        contribution_map_index],
                                                    r, c);
                                        const double remaining_weight =
                                            citlali::pipeline::
                                                mapdiag_remaining_contribution_weight(
                                                    total_weight,
                                                    contrib_weight);
                                        if (citlali::pipeline::
                                                mapdiag_has_full_leave_one_out_inputs(
                                                    total_signal,
                                                    total_weight,
                                                    contrib_weight,
                                                    contrib_variance_weight,
                                                    total_variance_weight,
                                                    remaining_weight)) {
                                            const double loo_value =
                                                citlali::pipeline::
                                                    mapdiag_full_leave_one_out_value(
                                                        total_signal,
                                                        contrib_signal,
                                                        remaining_weight);
                                            citlali::pipeline::
                                                mapdiag_assign_leave_one_out_z(
                                                    value, wt, loo_value,
                                                    candidate.leave_one_out_z);
                                        }
                                        else if (citlali::pipeline::
                                                     mapdiag_has_fallback_leave_one_out_inputs(
                                                         wt, contrib_weight)) {
                                            const double raw_sum =
                                                citlali::pipeline::
                                                    mapdiag_raw_weighted_signal(
                                                        value, wt);
                                            const double loo_value =
                                                citlali::pipeline::
                                                    mapdiag_fallback_leave_one_out_value(
                                                        raw_sum,
                                                        contrib_signal, wt,
                                                        contrib_weight);
                                            citlali::pipeline::
                                                mapdiag_assign_leave_one_out_z(
                                                    value, wt, loo_value,
                                                    candidate.leave_one_out_z);
                                        }
                                    }
                                }
                                citlali::pipeline::
                                    append_mapdiag_pixel_candidate(
                                        candidates, candidate);
                            }
                        }

                        citlali::pipeline::sort_mapdiag_pixel_candidates(
                            candidates);
                        const std::size_t candidate_top_n =
                            citlali::pipeline::mapdiag_candidate_top_n(
                                reduction_learning);
                        const std::size_t n_emitted_candidates =
                            citlali::pipeline::mapdiag_candidate_emit_count(
                                candidates.size(), candidate_top_n);
                        auto dominance =
                            citlali::pipeline::
                                make_mapdiag_detector_dominance_list();

                        for (std::size_t ci = 0; ci < n_emitted_candidates;
                             ++ci) {
                            const auto &candidate =
                                citlali::pipeline::mapdiag_emitted_candidate(
                                    candidates, ci);
                            const auto outlier_reason =
                                citlali::pipeline::
                                    mapdiag_map_pixel_outlier_reason(
                                        candidate, mb);
                            const auto record_map_index =
                                citlali::pipeline::mapdiag_record_map_index(i);
                            auto record =
                                citlali::pipeline::make_mapdiag_outlier_record<
                                    ReductionLearningState::MapPixelOutlier>(
                                    obsnum, mapdiag_record_producer,
                                    outlier_reason, fruit_iter,
                                    record_map_index, candidate);
                            reduction_learning.record_map_pixel_outlier(
                                std::move(record));
                            citlali::pipeline::
                                update_mapdiag_detector_dominance(
                                    dominance, candidate, fill_int);
                        }

                        const bool detector_exclusion_enabled =
                            citlali::pipeline::
                                mapdiag_detector_exclusion_enabled(
                                    reduction_learning);
                        if (detector_exclusion_enabled) {
                            const int detector_exclusion_min_pixels =
                                citlali::pipeline::
                                    mapdiag_detector_exclusion_min_pixels(
                                        reduction_learning);
                            const int array_id =
                                citlali::pipeline::mapdiag_array_id_or_default(
                                    write_indices.map_index, calib.arrays,
                                    -1);
                            for (const auto &entry : dominance) {
                                if (!citlali::pipeline::
                                        mapdiag_dominance_meets_min_pixels(
                                            entry,
                                            detector_exclusion_min_pixels)) {
                                    continue;
                                }
                                const auto penalty_reason =
                                    citlali::pipeline::
                                        mapdiag_detector_dominance_penalty_reason();
                                auto penalty =
                                    citlali::pipeline::
                                        make_mapdiag_detector_penalty<
                                            ReductionLearningState::
                                                DetectorPenalty>(
                                            obsnum, mapdiag_record_producer,
                                            penalty_reason,
                                            fruit_iter, entry, array_id);
                                reduction_learning.record_detector_penalty(
                                    std::move(penalty), true);
                                const auto display_scan_index =
                                    citlali::pipeline::
                                        mapdiag_display_scan_index(entry.scan);
                                logger->info(
                                    "mapdiag learned scan-local detector exclusion candidate stage={} iter={} map={} uid={} scan={} outlier_pixels={} max_abs_value={:.4g} max_abs_leave_one_out_z={:.4g}",
                                    stage_name, fruit_iter, i, entry.uid,
                                    display_scan_index,
                                    entry.count, entry.max_abs_value,
                                    entry.max_abs_leave_one_out_z);
                            }
                        }
                    }
                }
            }

            const bool has_noise_realizations =
                citlali::pipeline::mapdiag_has_noise_realizations(
                    mb->noise, i, mb->n_noise);
            if (has_noise_realizations) {
                auto noise_samples =
                    citlali::pipeline::make_mapdiag_noise_tail_samples(mb);

                const auto valid_core =
                    citlali::pipeline::mapdiag_valid_core_noise_mask(
                        core_mask);
                const double valid_core_count =
                    citlali::pipeline::mapdiag_valid_core_noise_count(
                        valid_core);
                const Eigen::Index n_noise_realizations =
                    citlali::pipeline::mapdiag_noise_realization_count(mb);
                for (Eigen::Index n = 0; n < n_noise_realizations; ++n) {
                    const auto noise_matrix =
                        citlali::pipeline::mapdiag_noise_matrix(mb, i, n);
                    citlali::pipeline::add_mapdiag_noise_realization_samples(
                        noise_samples, mapdiag_stats, noise_matrix,
                        valid_core, valid_core_count, core_mask);
                }
                citlali::pipeline::assign_mapdiag_noise_tail_samples(
                    idx, mapdiag_stats, noise_samples, noise_tail_refs);
            }
        }

        const bool is_single_observation_mapdiag = !mapdiag_context.is_coadd;
        if (is_single_observation_mapdiag) {
            citlali::pipeline::assign_mapdiag_single_obs_entry(
                mapdiag_context, idx, weight_sum[idx],
                core_weight_sum[idx], n_valid_pixels[idx],
                n_core_pixels[idx], obs_tables);
        }
        else {
            const auto n_obsnums = mb->obsnums.size();
            for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
                const auto &coadd_obsnum = mb->obsnums[obs_idx];
                const auto obs_dir =
                    citlali::pipeline::mapdiag_obs_raw_dir(
                        redu_dir_name, coadd_obsnum);
                const auto obs_weight_path =
                    toltec_io
                        .create_filename<engine_utils::toltecIO::toltec,
                                         engine_utils::toltecIO::map,
                                         engine_utils::toltecIO::raw>(
                        obs_dir, redu_type, array_names[idx], coadd_obsnum,
                        telescope.sim_obs) + ".fits";
                const auto weight_hdu_name =
                    citlali::pipeline::mapdiag_weight_hdu_name(
                        map_names[idx], stokes_names[idx]);
                try {
                    fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*>
                        obs_fits(obs_weight_path);
                    const auto obs_weight = obs_fits.get_hdu(weight_hdu_name);
                    citlali::pipeline::accumulate_mapdiag_obs_weight(
                        i, mapdiag_context.n_obsnums, mb->n_rows, mb->n_cols,
                        core_mask, obs_weight, obs_idx, obs_tables);
                } catch (const std::exception &e) {
                    logger->warn(
                        "failed to derive mapdiag contribution from {} [{}]: {}",
                        obs_weight_path, weight_hdu_name, e.what());
                    citlali::pipeline::zero_mapdiag_obs_entry(
                        mapdiag_context, idx, obs_idx, obs_tables);
                }
            }
        }
        const auto obs_totals =
            citlali::pipeline::sum_mapdiag_obs_weight_totals(
                obs_weight_sum, obs_core_weight_sum, mapdiag_context, idx);
        citlali::pipeline::assign_mapdiag_obs_fraction_pair(
            obs_weight_sum, obs_totals.weight, obs_core_weight_sum,
            obs_totals.core_weight, fill_double, mapdiag_context, idx,
            obs_weight_frac, obs_core_weight_frac);
    }

    write_netcdf_atomic(
        citlali::pipeline::mapdiag_netcdf_filename(filename),
        [&](netCDF::NcFile &fo) {
            const auto mapdiag_netcdf_vars =
                citlali::pipeline::make_mapdiag_netcdf_vars(
                    mapdiag_context, obsnum, mapdiag_metadata,
                    mapdiag_labels, mapdiag_values);
            citlali::pipeline::add_mapdiag_netcdf_vars(
                fo, mapdiag_netcdf_vars);
        });
}

void Engine::create_ptcdiag_file() {
    ptcdiag_filename =
        citlali::pipeline::diagnostic_output_netcdf_filename<
            engine_utils::toltecIO::toltec,
            engine_utils::toltecIO::ptcdiag,
            engine_utils::toltecIO::raw>(
            toltec_io, obsnum_dir_name, tod_output_subdir_name, redu_type,
            obsnum, telescope.sim_obs);

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
        telescope.project_id, redu_type, telescope.obs_goal, tod_type);
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

void Engine::create_rtcdiag_file() {
    rtcdiag_filename =
        citlali::pipeline::diagnostic_output_netcdf_filename<
            engine_utils::toltecIO::toltec,
            engine_utils::toltecIO::rtcdiag,
            engine_utils::toltecIO::raw>(
            toltec_io, obsnum_dir_name, tod_output_subdir_name, redu_type,
            obsnum, telescope.sim_obs);

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
        telescope.project_id, redu_type, telescope.obs_goal, tod_type);
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

void Engine::write_stats() {
    std::string stats_dir =
        citlali::pipeline::stats_raw_directory(obsnum_dir_name);
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
            toltec_io, stats_dir, redu_type, obsnum, telescope.sim_obs);
    write_netcdf_atomic(stats_netcdf_filename, [&](netCDF::NcFile &fo) {

    citlali::pipeline::add_stats_file_outputs(
        fo, std::stoi(obsnum), calib, diagnostics, ptcproc.cleaner, logger,
        omb.sig_unit, telescope.scan_indices.cols(),
        citlali::pipeline::ptcdiag_fill_double());
    });
}
