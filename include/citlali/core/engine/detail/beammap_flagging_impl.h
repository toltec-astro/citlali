#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::set_apt_flags() {
    // setup bitwise flags
    flag2.resize(calib.n_dets);
    flag2.setConstant(AptFlags::Good);

    // track number of flagged detectors
    std::atomic<int> n_flagged_dets{0};

    logger->info("flagging detectors");
    // first flag based on fit values and signal-to-noise
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[array_index];

        // calculate map standard deviation
        double map_std_dev = calc_map_support_stddev(i, true);
        const bool valid_map_std = std::isfinite(map_std_dev) && map_std_dev > 0.0;

        // reject non-physical fit results before threshold checks
        const bool finite_params = params.row(i).array().isFinite().all();
        const bool finite_perrors = perrors.row(i).array().isFinite().all();
        const bool positive_amp = std::isfinite(params(i,0)) && params(i,0) > 0.0;
        const bool positive_fwhm =
            std::isfinite(calib.apt["a_fwhm"](i)) && std::isfinite(calib.apt["b_fwhm"](i)) &&
            calib.apt["a_fwhm"](i) > 0.0 && calib.apt["b_fwhm"](i) > 0.0;
        if (!(finite_params && finite_perrors && positive_amp && positive_fwhm && valid_map_std)) {
            good_fits(i) = false;
        }

        // set apt signal to noise
        if (std::isfinite(perrors(i,0)) && perrors(i,0) > 0) {
            calib.apt["sig2noise"](i) = params(i,0)/perrors(i,0);
        } else {
            calib.apt["sig2noise"](i) = 0;
        }

        // flag bad fits
        if (!good_fits(i)) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::BadFit;
        }
        // flag detectors with outlier a_fwhm values
        if (calib.apt["a_fwhm"](i) < lower_fwhm_arcsec[array_name] ||
            ((calib.apt["a_fwhm"](i) > upper_fwhm_arcsec[array_name]) && upper_fwhm_arcsec[array_name] > 0)) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::AzFWHM;
        }
        // flag detectors with outlier b_fwhm values
        if (calib.apt["b_fwhm"](i) < lower_fwhm_arcsec[array_name] ||
            ((calib.apt["b_fwhm"](i) > upper_fwhm_arcsec[array_name] && upper_fwhm_arcsec[array_name] > 0))) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::ElFWHM;
        }
        // flag detectors with outlier S/N values
        const double map_sig2noise = valid_map_std ? params(i,0)/map_std_dev : 0.0;
        if (!std::isfinite(map_sig2noise) ||
            (map_sig2noise < lower_sig2noise[array_name]) ||
            ((map_sig2noise > upper_sig2noise[array_name]) && (upper_sig2noise[array_name] > 0))) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::Sig2Noise;
        }
        return 0;
    });


    // median network sensitivity for flagging
    std::map<Eigen::Index, double> nw_median_sens;

    // calc median sens from unflagged detectors for each nw
    logger->debug("calculating mean sensitivities");
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        Eigen::Index nw = calib.nws(i);

        // nw sensitivity
        auto nw_sens = calib.apt["sens"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                    std::get<1>(calib.nw_limits[nw])-1));

        // number of good detectors
        Eigen::Index n_good_det = (calib.apt["flag"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                               std::get<1>(calib.nw_limits[nw])-1)).array()==0).count();

        if (n_good_det>0) {
            // to hold good detectors
            Eigen::VectorXd sens(n_good_det);

            // remove flagged dets
            Eigen::Index j = std::get<0>(calib.nw_limits[nw]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<nw_sens.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    sens(k) = nw_sens(m);
                    k++;
                }
                j++;
            }
            // calculate median sens
            nw_median_sens[nw] = tula::alg::median(sens);
        }
        else {
            nw_median_sens[nw] = tula::alg::median(nw_sens);
        }
    }


    // flag too low/high sensitivies based on the median unflagged sensitivity of each nw
    logger->debug("flagging sensitivities");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get nw of current detector
        auto nw_index = calib.apt["nw"](i);

        // flag outlier sensitivities
        if (calib.apt["sens"](i) < lower_sens_factor*nw_median_sens[nw_index] ||
            (calib.apt["sens"](i) > upper_sens_factor*nw_median_sens[nw_index] && upper_sens_factor > 0)) {
            if (calib.apt["flag"](i)==0) {
                calib.apt["flag"](i) = 1;
                n_flagged_dets++;
            }
            flag2(i) |= AptFlags::Sens;
        }

        return 0;
    });

    // std maps to hold median unflagged x and y positions
    std::map<std::string, double> array_median_x_t, array_median_y_t;

    // calc median x_t and y_t values from unflagged detectors for each arrays
    logger->debug("calculating array median positions");
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // x_t
        auto array_x_t = calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        // y_t
        auto array_y_t = calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        // number of good detectors
        Eigen::Index n_good_det = (calib.apt["flag"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                std::get<1>(calib.array_limits[array])-1)).array()==0).count();

        // to hold good detectors
        Eigen::VectorXd x_t, y_t;

        if (n_good_det>0) {
            x_t.resize(n_good_det);
            y_t.resize(n_good_det);

            // remove flagged dets
            Eigen::Index j = std::get<0>(calib.array_limits[array]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<array_x_t.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    x_t(k) = array_x_t(m);
                    y_t(k) = array_y_t(m);
                    k++;
                }
                j++;
            }
            // calculate medians
            array_median_x_t[array_name] = tula::alg::median(x_t);
            array_median_y_t[array_name] = tula::alg::median(y_t);
        }
        else {
            // if no good dets, use all dets to calculate median
            array_median_x_t[array_name] = tula::alg::median(array_x_t);
            array_median_y_t[array_name] = tula::alg::median(array_y_t);
        }
    }

    // remove detectors above distance limits
    logger->debug("flagging detector positions");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[array_index];

        // calculate distance of detector from mean position of all detectors
        double dist = sqrt(pow(calib.apt["x_t"](i) - array_median_x_t[array_name],2) +
                           pow(calib.apt["y_t"](i) - array_median_y_t[array_name],2));

        // flag detectors that are further than the mean value than the distance limit
        if (dist > max_dist_arcsec[array_name] && max_dist_arcsec[array_name] > 0) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::Position;
        }

        return 0;
    });

    const bool prior_dist_flag_enabled =
        beammap_flag_max_prior_d2 > 0.0 && beammap_soft_priors_loaded && !beammap_soft_prior_slots.empty();
    if (beammap_flag_max_prior_d2 > 0.0 && !prior_dist_flag_enabled) {
        logger->warn(
            "beammap.flagging.max_prior_d2={} requested but soft priors are unavailable; skipping prior-distance flagging",
            beammap_flag_max_prior_d2);
    }
    if (prior_dist_flag_enabled) {
        double prior_derot_elev_rad = telescope.tel_data["TelElAct"].mean();
        if (!std::isfinite(prior_derot_elev_rad)) {
            prior_derot_elev_rad = 0.0;
        }
        if (std::abs(prior_derot_elev_rad) > pi) {
            prior_derot_elev_rad *= DEG_TO_RAD;
        }
        const bool apply_derot = beammap_soft_priors_are_derotated && telescope.pixel_axes == "altaz";
        const double cos_rot = std::cos(-prior_derot_elev_rad);
        const double sin_rot = std::sin(-prior_derot_elev_rad);
        std::atomic<int> n_prior_dist_hits{0};

        logger->debug("flagging detector prior distances");
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            const int array_index = static_cast<int>(std::lround(calib.apt["array"](i)));
            const int nw_index = static_cast<int>(std::lround(calib.apt["nw"](i)));
            std::string array_name = toltec_io.array_name_map[array_index];

            auto slots_it = beammap_soft_prior_slots.find({array_index, nw_index});
            if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
                return 0;
            }

            double x_arcsec = calib.apt["x_t"](i);
            double y_arcsec = calib.apt["y_t"](i);
            if (!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) {
                return 0;
            }

            if (beammap_soft_priors_are_centered) {
                x_arcsec -= array_median_x_t[array_name];
                y_arcsec -= array_median_y_t[array_name];
            }

            if (apply_derot) {
                const double rot_az_off = cos_rot * x_arcsec - sin_rot * y_arcsec;
                const double rot_alt_off = sin_rot * x_arcsec + cos_rot * y_arcsec;
                x_arcsec = -rot_az_off;
                y_arcsec = -rot_alt_off;
            }

            double min_d2 = std::numeric_limits<double>::infinity();
            for (const auto &slot : slots_it->second) {
                if (!std::isfinite(slot.x_arcsec) || !std::isfinite(slot.y_arcsec) ||
                    !std::isfinite(slot.sx_arcsec) || !std::isfinite(slot.sy_arcsec) ||
                    slot.sx_arcsec <= 0.0 || slot.sy_arcsec <= 0.0) {
                    continue;
                }
                const double dx = (x_arcsec - slot.x_arcsec) / slot.sx_arcsec;
                const double dy = (y_arcsec - slot.y_arcsec) / slot.sy_arcsec;
                const double d2 = dx * dx + dy * dy;
                if (std::isfinite(d2) && d2 < min_d2) {
                    min_d2 = d2;
                }
            }
            if (!std::isfinite(min_d2) || min_d2 <= beammap_flag_max_prior_d2) {
                return 0;
            }

            n_prior_dist_hits++;
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::PriorDist;
            return 0;
        });

        logger->info("beammap prior-distance flagging: {} detectors exceeded max_prior_d2={}",
                     n_prior_dist_hits.load(), beammap_flag_max_prior_d2);
    }

    // print number of flagged detectors
    logger->info("{} detectors were flagged", n_flagged_dets.load());

    // Derive the calibration amplitude from an empirical array template where
    // possible.  The Gaussian fit amplitude remains in amp for morphology/QC.
    calc_empirical_template_calibration();

    // calculate fcf
    logger->debug("calculating flux conversion factors");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[array_index];

        const double template_cal_amp =
            (calib.apt.count("cal_amp") > 0 && calib.apt["cal_amp"].size() == calib.n_dets)
                ? calib.apt["cal_amp"](i)
                : std::numeric_limits<double>::quiet_NaN();
        const double amp =
            (std::isfinite(template_cal_amp) && template_cal_amp > 0.0)
                ? template_cal_amp
                : params(i,0);
        // calc flux scale (always in mJy/beam)
        if (calib.apt["flag"](i) == 0 && std::isfinite(amp) && amp > 0.0) {
            const double flxscale = beammap_fluxes_mJy_beam[array_name] / amp;
            if (std::isfinite(flxscale) && flxscale > 0.0) {
                calib.apt["flxscale"](i) = flxscale;
                calib.apt["sens"](i) = calib.apt["sens"](i) * flxscale;
            } else {
                calib.apt["flxscale"](i) = 0;
                calib.apt["sens"](i) = 0;
                calib.apt["flag"](i) = 1;
                flag2(i) |= AptFlags::Sens;
            }
        }
        // set fluxscale (fcf) to zero if flagged
        else {
            calib.apt["flxscale"](i) = 0;
            calib.apt["sens"](i) = 0;
        }
        return 0;
    });

    // re-run calib setup to get average fwhms and beam areas
    calib.setup();

    // calculate source flux in MJy/sr from average beamsizes
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // get source flux in MJy/Sr
        beammap_fluxes_MJy_Sr[array_name] = mJY_ASEC_to_MJY_SR*(beammap_fluxes_mJy_beam[array_name])/calib.array_beam_areas[array];
    }
}
