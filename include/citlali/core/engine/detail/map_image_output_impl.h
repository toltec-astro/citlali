#pragma once

// Engine FITS map output implementation detail.
// Include this only after Engine has been declared.

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

