#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

template <class KidsProc, class RawObs>
void Beammap::loop_pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // run iterative stage
    run_loop(kidsproc, rawobs);
    ptcproc.fruit_loops_kernel_feedback_enabled = true;

    // write map summary
    if (verbose_mode) {
        write_map_summary(omb);
    }

    // empty initial ptcdata vector to save memory
    ptcs0.clear();

    // set to input parallel policy
    parallel_policy = omb.parallel_policy;

    if (map_grouping=="detector") {
        logger->info("calculating sensitivity");
        // parallelize on detectors
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            Eigen::MatrixXd det_sens, noise_flux;
            // calc sensitivity within psd freq range
            calc_sensitivity(ptcs, det_sens, noise_flux, telescope.d_fsmp, i, {sens_psd_limits_Hz(0), sens_psd_limits_Hz(1)});
            // copy into apt table
            calib.apt["sens"](i) = tula::alg::median(det_sens);

            return 0;
        });
    }

    // apt and sensitivity only relevant if beammapping
    if (map_grouping=="detector") {
        // rescale fit params from pixel to on-sky units
        calib.apt["amp"] = params.col(0);
        calib.apt["x_t"] = RAD_TO_ASEC*omb.pixel_size_rad*(params.col(1).array() - (omb.n_cols - 1)/2.0);
        calib.apt["y_t"] = RAD_TO_ASEC*omb.pixel_size_rad*(params.col(2).array() - (omb.n_rows - 1)/2.0);
        calib.apt["a_fwhm"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params.col(3));
        calib.apt["b_fwhm"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params.col(4));
        calib.apt["angle"] = params.col(5);

        // rescale fit errors from pixel to on-sky units
        calib.apt["amp_err"] = perrors.col(0);
        calib.apt["x_t_err"] = RAD_TO_ASEC*omb.pixel_size_rad*(perrors.col(1));
        calib.apt["y_t_err"] = RAD_TO_ASEC*omb.pixel_size_rad*(perrors.col(2));
        calib.apt["a_fwhm_err"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(3));
        calib.apt["b_fwhm_err"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(4));
        calib.apt["angle_err"] = perrors.col(5);

        // add convergence iteration to apt table
        calib.apt["converge_iter"] = converge_iter.cast<double> ();

        if (rfi_mask_samples_flagged.size() == calib.n_dets) {
            calib.apt["rfi_masked_samples"] = rfi_mask_samples_flagged.cast<double>();
        }
        if (rfi_mask_scans_flagged.size() == calib.n_dets) {
            calib.apt["rfi_masked_scans"] = rfi_mask_scans_flagged.cast<double>();
        }
        if (scan_band_mask_samples_flagged.size() == calib.n_dets) {
            calib.apt["scan_band_masked_samples"] = scan_band_mask_samples_flagged.cast<double>();
        }
        if (scan_band_mask_rows_flagged.size() == calib.n_dets) {
            calib.apt["scan_band_masked_rows"] = scan_band_mask_rows_flagged.cast<double>();
        }
        if (scan_band_mask_edge_code.size() == calib.n_dets) {
            calib.apt["scan_band_masked_edge"] = scan_band_mask_edge_code.cast<double>();
        }
        if (scan_band_mask_rejected.size() == calib.n_dets) {
            calib.apt["scan_band_mask_rejected"] = scan_band_mask_rejected.cast<double>();
        }
        if (beammap_rfi_mask_enabled &&
            rfi_mask_samples_flagged.size() == calib.n_dets &&
            rfi_mask_scans_flagged.size() == calib.n_dets) {
            const Eigen::Index n_det_masked = (rfi_mask_scans_flagged.array() > 0).count();
            logger->info("beammap rfi mask summary: {} detectors affected, {} total samples masked",
                         n_det_masked, static_cast<long long>(rfi_mask_samples_flagged.cast<double>().sum()));
        }

        if (fit_diag_bound_nhit.size() == n_maps &&
            fit_diag_hit_lower.rows() == n_maps && fit_diag_hit_upper.rows() == n_maps &&
            fit_diag_hit_lower.cols() >= 6 && fit_diag_hit_upper.cols() >= 6) {
            const Eigen::Index n_bound_any = (fit_diag_bound_nhit.array() > 0).count();
            Eigen::VectorXi low_hits = fit_diag_hit_lower.colwise().sum().transpose();
            Eigen::VectorXi high_hits = fit_diag_hit_upper.colwise().sum().transpose();
            logger->info(
                "beammap final bound-hit summary: any_hit={}/{} amp(lo/hi)={}/{} x(lo/hi)={}/{} y(lo/hi)={}/{} a(lo/hi)={}/{} b(lo/hi)={}/{} angle(lo/hi)={}/{}",
                n_bound_any, n_maps,
                low_hits(0), high_hits(0),
                low_hits(1), high_hits(1),
                low_hits(2), high_hits(2),
                low_hits(3), high_hits(3),
                low_hits(4), high_hits(4),
                low_hits(5), high_hits(5));
        }

        // flag detectors in apt based on config limits
        set_apt_flags();

        // subtract reference detector position and derotate
        process_apt();
        apply_final_network_position_flags();
        update_final_prior_match_diagnostics();
        if (final_prior_slot_index_diag.size() == calib.n_dets) {
            calib.apt["final_prior_slot_index"] = final_prior_slot_index_diag.cast<double>();
        }
        if (final_prior_d2_diag.size() == calib.n_dets) {
            calib.apt["final_prior_d2"] = final_prior_d2_diag;
        }
        calib.setup();
        for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
            Eigen::Index array = calib.arrays(i);
            std::string array_name = toltec_io.array_name_map[array];
            beammap_fluxes_MJy_Sr[array_name] =
                mJY_ASEC_to_MJY_SR * (beammap_fluxes_mJy_beam[array_name]) / calib.array_beam_areas[array];
        }
        log_final_network_qc_summary();

        // add final apt table to timestream files
        if (run_tod_output && !tod_filename.empty()) {
            // vectors to hold tangent plane pointing for all ptcs (n_chunks x [n_pts x n_dets])
            std::vector<Eigen::MatrixXd> lat, lon;

            // recalculate tangent plane pointing for tod output
            for (Eigen::Index i=0; i<ptcs.size(); ++i) {
                // tangent plane pointing for each detector
                Eigen::MatrixXd ptc_lat(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
                Eigen::MatrixXd ptc_lon(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
                // loop through detectors
                grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto j) {
                    // det indices
                    auto det_index = j;
                    double az_off = calib.apt["x_t"](det_index);
                    double el_off = calib.apt["y_t"](det_index);

                    // get tangent pointing
                    auto [det_lat, det_lon] = engine_utils::calc_det_pointing(ptcs[i].tel_data.data, az_off,
                                                                              el_off, telescope.pixel_axes,
                                                                              ptcs[i].pointing_offsets_arcsec.data,
                                                                              map_grouping, true);
                    ptc_lat.col(j) = std::move(det_lat);
                    ptc_lon.col(j) = std::move(det_lon);

                    return 0;
                });
                lat.push_back(std::move(ptc_lat));
                lon.push_back(std::move(ptc_lon));
            }

            logger->info("adding final apt and detector pointing to tod files");
            // loop through tod files
            for (const auto & [key, val]: tod_filename) {
                netCDF::NcFile fo(val, netCDF::NcFile::write);
                // overwrite apt table
                for (auto const& x: calib.apt) {
                    if (x.first!="flag2") {
                        // start index for apt table
                        std::vector<std::size_t> start_index_apt = {0};
                        // size for apt
                        std::vector<std::size_t> size_apt = {1};
                        netCDF::NcVar apt_v = fo.getVar("apt_" + x.first);
                        if (!apt_v.isNull()) {
                            for (std::size_t i=0; i< TULA_SIZET(calib.n_dets); ++i) {
                                start_index_apt[0] = i;
                                apt_v.putVar(start_index_apt, size_apt, &calib.apt[x.first](i));
                            }
                        }
                    }
                }

                // detector tangent plane pointing
                netCDF::NcVar det_lat_v = fo.getVar("det_lat");
                netCDF::NcVar det_lon_v = fo.getVar("det_lon");

                // detector absolute pointing
                netCDF::NcVar det_ra_v = fo.getVar("det_ra");
                netCDF::NcVar det_dec_v = fo.getVar("det_dec");
                const bool write_tangent_pointing = !det_lat_v.isNull() && !det_lon_v.isNull();
                const bool write_abs_pointing = telescope.pixel_axes == "radec" &&
                                                !det_ra_v.isNull() && !det_dec_v.isNull();
                if (!write_tangent_pointing && !write_abs_pointing) {
                    logger->debug("tod file {} has no detector pointing variables; skipping final detector pointing update", val);
                    continue;
                }

                // start indices for data
                std::vector<std::size_t> start_index = {0, 0};
                // size for data
                std::vector<std::size_t> size = {1, TULA_SIZET(calib.n_dets)};
                std::size_t k = 0;
                // loop through ptcs
                for (Eigen::Index i=0; i<lat.size(); ++i) {
                    // loop through n_pts
                    for (std::size_t j=0; j < TULA_SIZET(lat[i].rows()); ++j) {
                        start_index[0] = k;
                        k++;
                        // append detector latitudes
                        Eigen::VectorXd lat_row = lat[i].row(j);

                        // append detector longitudes
                        Eigen::VectorXd lon_row = lon[i].row(j);
                        if (write_tangent_pointing) {
                            det_lat_v.putVar(start_index, size, lat_row.data());
                            det_lon_v.putVar(start_index, size, lon_row.data());
                        }

                        if (write_abs_pointing) {
                            // get absolute pointing
                            auto [dec, ra] = engine_utils::tangent_to_abs(lat_row, lon_row, telescope.tel_header["Header.Source.Ra"](0),
                                                                          telescope.tel_header["Header.Source.Dec"](0));
                            // append detector ra
                            det_ra_v.putVar(start_index, size, ra.data());

                            // append detector dec
                            det_dec_v.putVar(start_index, size, dec.data());
                        }
                    }
                }
            }

            // empty ptcdata vector to save memory
            ptcs.clear();
        }
    }

    else {
        // calculate map psds
        logger->info("calculating map psd");
        omb.calc_map_psd();
        // calculate map histograms
        logger->info("calculating map histogram");
        omb.calc_map_hist();
    }
}
