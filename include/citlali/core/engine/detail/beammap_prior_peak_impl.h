#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

bool Beammap::find_map_weighted_peak(Eigen::Index map_index, Eigen::Index &best_row,
                                     Eigen::Index &best_col, double &best_snr) const {
    best_row = -1;
    best_col = -1;
    best_snr = -std::numeric_limits<double>::infinity();

    if (map_index < 0 || map_index >= n_maps) {
        return false;
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 || wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        return false;
    }

    const double center_row = static_cast<double>(sig.rows() - 1) / 2.0;
    const double center_col = static_cast<double>(sig.cols() - 1) / 2.0;
    const double radius_pix = map_fitter.fitting_region_pix;
    const double radius2 = radius_pix * radius_pix;

    auto scan = [&](bool apply_radius) {
        bool found = false;
        for (Eigen::Index row = 0; row < sig.rows(); ++row) {
            for (Eigen::Index col = 0; col < sig.cols(); ++col) {
                const double s = sig(row, col);
                const double w = wt(row, col);
                if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                    continue;
                }
                if (apply_radius) {
                    const double dr = static_cast<double>(row) - center_row;
                    const double dc = static_cast<double>(col) - center_col;
                    if (dr * dr + dc * dc >= radius2) {
                        continue;
                    }
                }
                const double snr = s * std::sqrt(w);
                if (!std::isfinite(snr)) {
                    continue;
                }
                if (!found || snr > best_snr) {
                    best_row = row;
                    best_col = col;
                    best_snr = snr;
                    found = true;
                }
            }
        }
        return found;
    };

    if (radius_pix > 0.0 && scan(true)) {
        return true;
    }
    return scan(false);
}

void Beammap::configure_detector_source_centers_from_previous_fit() {
    if (map_grouping != "detector") {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        return;
    }

    if (!is_beammap_measurement_iter(current_iter)) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        logger->info(
            "beammap detector source centers unavailable on iter {} phase={}: locator pass has no previous fits "
            "(ptc_mask_radius={:.3f} arcsec)",
            current_iter, beammap_iter_phase_name(current_iter), ptcproc.mask_radius_arcsec);
        return;
    }

    if (p0.rows() != n_maps || p0.cols() < 3 || good_fits.size() != n_maps) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        logger->warn(
            "beammap detector source centers unavailable on iter {}: previous-fit state is incomplete "
            "(p0={}x{}, good_fits={})",
            current_iter, p0.rows(), p0.cols(), good_fits.size());
        return;
    }

    ptcproc.fruit_loops_source_lat = Eigen::VectorXd::Zero(n_maps);
    ptcproc.fruit_loops_source_lon = Eigen::VectorXd::Zero(n_maps);
    ptcproc.fruit_loops_source_valid = Eigen::VectorXi::Zero(n_maps);
    Eigen::VectorXd kernel_source_a_fwhm_rad = Eigen::VectorXd::Zero(n_maps);
    Eigen::VectorXd kernel_source_b_fwhm_rad = Eigen::VectorXd::Zero(n_maps);

    Eigen::Index n_valid = 0;
    Eigen::Index n_valid_fwhm = 0;
    std::vector<double> fwhm_arcsec_values;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        if (!good_fits(i) ||
            !std::isfinite(p0(i, 0)) || p0(i, 0) <= 0.0 ||
            !std::isfinite(p0(i, 1)) || !std::isfinite(p0(i, 2))) {
            continue;
        }
        ptcproc.fruit_loops_source_lat(i) =
            (p0(i, 2) - (omb.n_rows - 1) / 2.0) * omb.pixel_size_rad;
        ptcproc.fruit_loops_source_lon(i) =
            (p0(i, 1) - (omb.n_cols - 1) / 2.0) * omb.pixel_size_rad;
        ptcproc.fruit_loops_source_valid(i) = 1;
        n_valid++;

        if (p0.cols() > 4 &&
            std::isfinite(p0(i, 3)) && p0(i, 3) > 0.0 &&
            std::isfinite(p0(i, 4)) && p0(i, 4) > 0.0) {
            kernel_source_a_fwhm_rad(i) = STD_TO_FWHM * omb.pixel_size_rad * p0(i, 3);
            kernel_source_b_fwhm_rad(i) = STD_TO_FWHM * omb.pixel_size_rad * p0(i, 4);
            const double mean_fwhm_arcsec =
                RAD_TO_ASEC * (kernel_source_a_fwhm_rad(i) + kernel_source_b_fwhm_rad(i)) / 2.0;
            if (std::isfinite(mean_fwhm_arcsec) && mean_fwhm_arcsec > 0.0) {
                fwhm_arcsec_values.push_back(mean_fwhm_arcsec);
                n_valid_fwhm++;
            }
        }
    }

    logger->info(
        "beammap detector source centers using previous-fit centers for {}/{} detector maps "
        "on iter {} (ptc_mask_radius={:.3f} arcsec)",
        n_valid, n_maps, current_iter, ptcproc.mask_radius_arcsec);

    if (rtcproc.run_kernel) {
        double median_fwhm_arcsec = std::numeric_limits<double>::quiet_NaN();
        if (!fwhm_arcsec_values.empty()) {
            std::sort(fwhm_arcsec_values.begin(), fwhm_arcsec_values.end());
            median_fwhm_arcsec = fwhm_arcsec_values[fwhm_arcsec_values.size() / 2];
        }
        rtcproc.kernel.set_source_centers(ptcproc.fruit_loops_source_lat,
                                          ptcproc.fruit_loops_source_lon,
                                          ptcproc.fruit_loops_source_valid,
                                          kernel_source_a_fwhm_rad,
                                          kernel_source_b_fwhm_rad);
        logger->info(
            "beammap detector kernel placement using previous-fit centers for {}/{} detector maps on iter {}; fitted kernel FWHM available for {}/{} maps (median={:.3f} arcsec)",
            n_valid, n_maps, current_iter, n_valid_fwhm, n_maps, median_fwhm_arcsec);
    }
}

double Beammap::get_prior_derot_elev_rad() const {
    double derot_elev_rad = 0.0;
    auto tel_el_it = telescope.tel_data.find("TelElAct");
    if (tel_el_it != telescope.tel_data.end() && tel_el_it->second.size() > 0) {
        derot_elev_rad = tel_el_it->second.mean();
    }
    if (!std::isfinite(derot_elev_rad)) {
        derot_elev_rad = 0.0;
    }
    if (std::abs(derot_elev_rad) > pi) {
        derot_elev_rad *= DEG_TO_RAD;
    }
    return derot_elev_rad;
}

double Beammap::effective_prior_max_d2() const {
    return is_beammap_measurement_iter(current_iter)
               ? beammap_priors_max_d2_after_iter0
               : beammap_priors_max_d2_iter0;
}

double Beammap::effective_prior_score_lambda() const {
    return is_beammap_measurement_iter(current_iter)
               ? beammap_priors_score_lambda_after_iter0
               : beammap_priors_score_lambda_iter0;
}

bool Beammap::observed_to_prior_frame(int array, double x_raw_arcsec, double y_raw_arcsec,
                                      double derot_elev_rad, double &x_prior_arcsec,
                                      double &y_prior_arcsec, double *center_x_arcsec,
                                      double *center_y_arcsec,
                                      bool apply_empirical_alignment) const {
    if (!std::isfinite(x_raw_arcsec) || !std::isfinite(y_raw_arcsec)) {
        return false;
    }

    double x = x_raw_arcsec;
    double y = y_raw_arcsec;
    double center_x = std::numeric_limits<double>::quiet_NaN();
    double center_y = std::numeric_limits<double>::quiet_NaN();

    if (beammap_soft_priors_are_centered) {
        auto x_it = beammap_prior_array_center_x_arcsec.find(array);
        auto y_it = beammap_prior_array_center_y_arcsec.find(array);
        if (x_it == beammap_prior_array_center_x_arcsec.end() ||
            y_it == beammap_prior_array_center_y_arcsec.end() ||
            !std::isfinite(x_it->second) || !std::isfinite(y_it->second)) {
            return false;
        }
        center_x = x_it->second;
        center_y = y_it->second;
        x -= center_x;
        y -= center_y;
    }

    if (center_x_arcsec != nullptr) {
        *center_x_arcsec = center_x;
    }
    if (center_y_arcsec != nullptr) {
        *center_y_arcsec = center_y;
    }

    if (beammap_soft_priors_are_derotated && telescope.pixel_axes == "altaz") {
        if (!std::isfinite(derot_elev_rad)) {
            derot_elev_rad = 0.0;
        }
        if (std::abs(derot_elev_rad) > pi) {
            derot_elev_rad *= DEG_TO_RAD;
        }
        const double cos_rot = std::cos(-derot_elev_rad);
        const double sin_rot = std::sin(-derot_elev_rad);
        const double rot_az_off = cos_rot * x - sin_rot * y;
        const double rot_alt_off = sin_rot * x + cos_rot * y;
        x = -rot_az_off;
        y = -rot_alt_off;
    }

    if (apply_empirical_alignment) {
        auto align_it = beammap_prior_array_alignment.find(array);
        if (align_it != beammap_prior_array_alignment.end() && align_it->second.valid) {
            const auto &align = align_it->second;
            const double x_rot = align.cos_theta * x - align.sin_theta * y;
            const double y_rot = align.sin_theta * x + align.cos_theta * y;
            x = x_rot + align.dx_arcsec;
            y = y_rot + align.dy_arcsec;
        }
    }

    x_prior_arcsec = x;
    y_prior_arcsec = y;
    return std::isfinite(x_prior_arcsec) && std::isfinite(y_prior_arcsec);
}

bool Beammap::match_prior_slot(int array, int nw, double x_prior_arcsec, double y_prior_arcsec,
                               double &best_d2, int &best_slot, double *slot_x_arcsec,
                               double *slot_y_arcsec, double *slot_sx_arcsec,
                               double *slot_sy_arcsec) const {
    best_d2 = std::numeric_limits<double>::infinity();
    best_slot = -1;
    auto slots_it = beammap_soft_prior_slots.find({array, nw});
    if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty() ||
        !std::isfinite(x_prior_arcsec) || !std::isfinite(y_prior_arcsec)) {
        return false;
    }

    for (const auto &slot : slots_it->second) {
        if (!std::isfinite(slot.x_arcsec) || !std::isfinite(slot.y_arcsec) ||
            !std::isfinite(slot.sx_arcsec) || !std::isfinite(slot.sy_arcsec) ||
            slot.sx_arcsec <= 0.0 || slot.sy_arcsec <= 0.0) {
            continue;
        }
        const double dx = (x_prior_arcsec - slot.x_arcsec) / slot.sx_arcsec;
        const double dy = (y_prior_arcsec - slot.y_arcsec) / slot.sy_arcsec;
        const double d2 = dx * dx + dy * dy;
        if (std::isfinite(d2) && d2 < best_d2) {
            best_d2 = d2;
            best_slot = slot.slot_index;
            if (slot_x_arcsec != nullptr) {
                *slot_x_arcsec = slot.x_arcsec;
            }
            if (slot_y_arcsec != nullptr) {
                *slot_y_arcsec = slot.y_arcsec;
            }
            if (slot_sx_arcsec != nullptr) {
                *slot_sx_arcsec = slot.sx_arcsec;
            }
            if (slot_sy_arcsec != nullptr) {
                *slot_sy_arcsec = slot.sy_arcsec;
            }
        }
    }
    return std::isfinite(best_d2);
}
