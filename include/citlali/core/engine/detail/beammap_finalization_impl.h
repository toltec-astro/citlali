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

void Beammap::process_apt() {
    // reference detector x and y
    double ref_det_x_t = 0;
    double ref_det_y_t = 0;

    // initial reference det
    beammap_reference_det_found = -99;

    // if particular reference detector is requested
    if (beammap_subtract_reference) {
        if (beammap_reference_det >= 0 && beammap_reference_det < calib.n_dets) {
            beammap_reference_det_found = beammap_reference_det;
            // set reference x_t and y_t
            ref_det_x_t = calib.apt["x_t"](beammap_reference_det_found);
            ref_det_y_t = calib.apt["y_t"](beammap_reference_det_found);
        }
        // else use detector closest to the median of selected networks
        else {
            if (beammap_reference_det >= 0) {
                logger->warn("configured beammap_reference_det={} is out of range [0, {}); using automatic reference selection",
                             beammap_reference_det, calib.n_dets);
            }
            logger->info("finding a reference detector");
            constexpr Eigen::Index min_reference_candidates = 25;
            auto nw_in_set = [](Eigen::Index nw, const std::vector<Eigen::Index> &set) {
                return std::find(set.begin(), set.end(), nw) != set.end();
            };

            using IndexVector = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
            auto gather_from_nws = [&](const std::vector<Eigen::Index> &ref_nws,
                                       Eigen::VectorXd &x_t, Eigen::VectorXd &y_t,
                                       IndexVector &det_indices) -> bool {
                Eigen::Index n_match = 0;
                for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                    if (calib.apt["flag"](i) == 0) {
                        auto nw = static_cast<Eigen::Index>(calib.apt["nw"](i));
                        const double x = calib.apt["x_t"](i);
                        const double y = calib.apt["y_t"](i);
                        if (nw_in_set(nw, ref_nws) && std::isfinite(x) && std::isfinite(y)) {
                            n_match++;
                        }
                    }
                }
                if (n_match < min_reference_candidates) {
                    return false;
                }

                x_t.resize(n_match);
                y_t.resize(n_match);
                det_indices.resize(n_match);
                Eigen::Index k = 0;
                for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                    if (calib.apt["flag"](i) == 0) {
                        auto nw = static_cast<Eigen::Index>(calib.apt["nw"](i));
                        const double x = calib.apt["x_t"](i);
                        const double y = calib.apt["y_t"](i);
                        if (nw_in_set(nw, ref_nws) && std::isfinite(x) && std::isfinite(y)) {
                            x_t(k) = x;
                            y_t(k) = y;
                            det_indices(k) = i;
                            k++;
                        }
                    }
                }
                return true;
            };

            Eigen::VectorXd x_t, y_t, dist;
            IndexVector det_indices;
            double med_x_t = 0.0;
            double med_y_t = 0.0;

            const std::vector<Eigen::Index> primary_nws = {3};
            const std::vector<Eigen::Index> fallback_nws = {2, 3, 4};

            bool have_ref = false;
            if (gather_from_nws(primary_nws, x_t, y_t, det_indices)) {
                logger->info("using median of nw=3 for reference");
                have_ref = true;
            }
            else if (gather_from_nws(fallback_nws, x_t, y_t, det_indices)) {
                logger->info("using median of nw=2,3,4 for reference");
                have_ref = true;
            }

            if (!have_ref) {
                logger->warn("no robust reference from nw=3 or nw=2,3,4; using all unflagged detectors");
                Eigen::Index n_unflagged = 0;
                for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                    if (calib.apt["flag"](i) == 0 &&
                        std::isfinite(calib.apt["x_t"](i)) &&
                        std::isfinite(calib.apt["y_t"](i))) {
                        n_unflagged++;
                    }
                }
                if (n_unflagged > 0) {
                    x_t.resize(n_unflagged);
                    y_t.resize(n_unflagged);
                    det_indices.resize(n_unflagged);
                    Eigen::Index k = 0;
                    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                        const double x = calib.apt["x_t"](i);
                        const double y = calib.apt["y_t"](i);
                        if (calib.apt["flag"](i) == 0 && std::isfinite(x) && std::isfinite(y)) {
                            x_t(k) = x;
                            y_t(k) = y;
                            det_indices(k) = i;
                            k++;
                        }
                    }
                    have_ref = true;
                }
            }

            if (!have_ref) {
                logger->warn("all detectors are flagged; disabling reference subtraction");
            } else {
                logger->info("beammap reference candidate count: {}", x_t.size());
                med_x_t = tula::alg::median(x_t);
                med_y_t = tula::alg::median(y_t);

                if (!std::isfinite(med_x_t) || !std::isfinite(med_y_t)) {
                    logger->warn("beammap reference median is non-finite ({},{}); disabling reference subtraction",
                                 med_x_t, med_y_t);
                    beammap_reference_det_found = -99;
                } else {
                    dist = (x_t.array() - med_x_t).square().matrix() +
                           (y_t.array() - med_y_t).square().matrix();
                    Eigen::Index nearest_candidate = -1;
                    dist.minCoeff(&nearest_candidate);
                    if (nearest_candidate >= 0 && nearest_candidate < det_indices.size()) {
                        beammap_reference_det_found = det_indices(nearest_candidate);

                        // set reference x_t and y_t to the median location
                        ref_det_x_t = med_x_t;
                        ref_det_y_t = med_y_t;
                    } else {
                        logger->warn("beammap reference nearest candidate index {} is invalid; disabling reference subtraction",
                                     nearest_candidate);
                        beammap_reference_det_found = -99;
                    }
                }
            }
        }
        if (beammap_reference_det_found >= 0 && beammap_reference_det_found < calib.n_dets) {
            double ref_det_actual_x_t = calib.apt["x_t"](beammap_reference_det_found);
            double ref_det_actual_y_t = calib.apt["y_t"](beammap_reference_det_found);
            logger->info("using reference median ({:.3f},{:.3f}) arcsec; nearest detector {} at ({:.3f},{:.3f}) arcsec",
                         ref_det_x_t, ref_det_y_t,
                         beammap_reference_det_found,
                         ref_det_actual_x_t, ref_det_actual_y_t);
            // record resolved reference detector for metadata; keep config value unchanged
            calib.apt_meta["reference_det"] = beammap_reference_det_found;
        } else {
            logger->warn("reference detector is invalid; leaving reference offsets at ({:.3f},{:.3f}) arcsec",
                         ref_det_x_t, ref_det_y_t);
        }
    }
    else {
        logger->info("no reference detector selected");
    }

    // add reference detector to APT meta data
    calib.apt_meta["reference_x_t"] = ref_det_x_t;
    calib.apt_meta["reference_y_t"] = ref_det_y_t;

    // raw (not derotated or reference detector subtracted) detector x and y values
    calib.apt["x_t_raw"] = calib.apt["x_t"];
    calib.apt["y_t_raw"] = calib.apt["y_t"];

    // per-detector derotation elevation for altaz beammaps
    calib.apt["derot_elev"].setConstant(telescope.tel_data["TelElAct"].mean());
    if (telescope.pixel_axes == "altaz" && map_grouping == "detector" && !ptcs.empty()) {
        Eigen::MatrixXd elev_best(omb.n_rows, omb.n_cols);
        Eigen::MatrixXd dist2_best(omb.n_rows, omb.n_cols);
        elev_best.setConstant(std::numeric_limits<double>::quiet_NaN());
        dist2_best.setConstant(std::numeric_limits<double>::infinity());

        for (const auto &ptc : ptcs) {
            const auto &alt = ptc.tel_data.data.at("alt_phys");
            const auto &az = ptc.tel_data.data.at("az_phys");
            const auto &el = ptc.tel_data.data.at("TelElAct");
            for (Eigen::Index k = 0; k < alt.size(); ++k) {
                double row = alt(k) / omb.pixel_size_rad + (omb.n_rows - 1) / 2.0;
                double col = az(k) / omb.pixel_size_rad + (omb.n_cols - 1) / 2.0;
                Eigen::Index ir = static_cast<Eigen::Index>(std::llround(row));
                Eigen::Index ic = static_cast<Eigen::Index>(std::llround(col));
                if ((ir >= 0) && (ir < omb.n_rows) && (ic >= 0) && (ic < omb.n_cols)) {
                    double lat_center = (static_cast<double>(ir) - (omb.n_rows - 1) / 2.0) * omb.pixel_size_rad;
                    double lon_center = (static_cast<double>(ic) - (omb.n_cols - 1) / 2.0) * omb.pixel_size_rad;
                    double dlat = alt(k) - lat_center;
                    double dlon = az(k) - lon_center;
                    double dist2 = dlat * dlat + dlon * dlon;
                    if (dist2 < dist2_best(ir, ic)) {
                        dist2_best(ir, ic) = dist2;
                        elev_best(ir, ic) = el(k);
                    }
                }
            }
        }

        for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
            double row = (calib.apt["y_t_raw"](i) * ASEC_TO_RAD) / omb.pixel_size_rad + (omb.n_rows - 1) / 2.0;
            double col = (calib.apt["x_t_raw"](i) * ASEC_TO_RAD) / omb.pixel_size_rad + (omb.n_cols - 1) / 2.0;
            Eigen::Index ir = static_cast<Eigen::Index>(std::llround(row));
            Eigen::Index ic = static_cast<Eigen::Index>(std::llround(col));
            if ((ir >= 0) && (ir < omb.n_rows) && (ic >= 0) && (ic < omb.n_cols)) {
                double elev = elev_best(ir, ic);
                if (std::isfinite(elev)) {
                    calib.apt["derot_elev"](i) = elev;
                }
            }
        }
    }

    // align to reference detector if specified and subtract its position from x and y
    calib.apt["x_t"] =  calib.apt["x_t"].array() - ref_det_x_t;
    calib.apt["y_t"] =  calib.apt["y_t"].array() - ref_det_y_t;

    // derotated detector x and y values
    calib.apt["x_t_derot"] = calib.apt["x_t"];
    calib.apt["y_t_derot"] = calib.apt["y_t"];

    // tolerate telescope streams that provide elevation in degrees.
    Eigen::VectorXd derot_elev_rad = calib.apt["derot_elev"];
    const double max_abs_elev = derot_elev_rad.array().abs().maxCoeff();
    if (std::isfinite(max_abs_elev) && max_abs_elev > 2.0 * pi + 0.1) {
        logger->warn("derot_elev appears to be in degrees (max |elev|={:.4g}); converting to radians", max_abs_elev);
        derot_elev_rad *= DEG_TO_RAD;
    }

    // calculate derotated positions
    Eigen::VectorXd rot_az_off = cos(-derot_elev_rad.array())*calib.apt["x_t_derot"].array() -
                                 sin(-derot_elev_rad.array())*calib.apt["y_t_derot"].array();
    Eigen::VectorXd rot_alt_off = sin(-derot_elev_rad.array())*calib.apt["x_t_derot"].array() +
                                  cos(-derot_elev_rad.array())*calib.apt["y_t_derot"].array();

    // overwrite x_t and y_t
    calib.apt["x_t_derot"] = -rot_az_off;
    calib.apt["y_t_derot"] = -rot_alt_off;

    if (beammap_derotate) {
        logger->info("derotating apt");
        // if derotation requested set default positions to derotated positions
        calib.apt["x_t"] = calib.apt["x_t_derot"];
        calib.apt["y_t"] = calib.apt["y_t_derot"];
    }
}

void Beammap::apply_final_network_position_flags() {
    if (map_grouping != "detector") {
        return;
    }

    bool enabled = false;
    for (const auto &[arr_index, arr_name] : toltec_io.array_name_map) {
        auto it = network_robust_z.find(arr_name);
        if (it != network_robust_z.end() && it->second > 0.0) {
            enabled = true;
            break;
        }
    }
    if (!enabled) {
        return;
    }

    struct NetworkStats {
        bool valid = false;
        double median_x = 0.0;
        double median_y = 0.0;
        double sigma_x = 0.0;
        double sigma_y = 0.0;
        double threshold = 0.0;
    };

    std::map<std::pair<int, int>, NetworkStats> stats_by_network;
    constexpr Eigen::Index min_network_samples = 16;

    logger->debug("flagging final detector network positions");
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];
        const double threshold = network_robust_z[array_name];
        if (!(threshold > 0.0)) {
            continue;
        }

        for (Eigen::Index j = 0; j < calib.n_nws; ++j) {
            Eigen::Index nw = calib.nws(j);
            if (std::get<0>(calib.nw_limits[nw]) < 0 ||
                std::get<1>(calib.nw_limits[nw]) <= std::get<0>(calib.nw_limits[nw])) {
                continue;
            }
            if (static_cast<Eigen::Index>(calib.apt["array"](std::get<0>(calib.nw_limits[nw]))) != array) {
                continue;
            }

            std::vector<double> x_vals;
            std::vector<double> y_vals;
            x_vals.reserve(static_cast<std::size_t>(std::get<1>(calib.nw_limits[nw]) - std::get<0>(calib.nw_limits[nw])));
            y_vals.reserve(x_vals.capacity());

            for (Eigen::Index k = std::get<0>(calib.nw_limits[nw]); k < std::get<1>(calib.nw_limits[nw]); ++k) {
                if (calib.apt["flag"](k) != 0) {
                    continue;
                }
                const double x = calib.apt["x_t"](k);
                const double y = calib.apt["y_t"](k);
                if (!std::isfinite(x) || !std::isfinite(y)) {
                    continue;
                }
                x_vals.push_back(x);
                y_vals.push_back(y);
            }
            if (static_cast<Eigen::Index>(x_vals.size()) < min_network_samples) {
                continue;
            }

            Eigen::Map<Eigen::VectorXd> x_vec(x_vals.data(), static_cast<Eigen::Index>(x_vals.size()));
            Eigen::Map<Eigen::VectorXd> y_vec(y_vals.data(), static_cast<Eigen::Index>(y_vals.size()));
            const double median_x = tula::alg::median(x_vec);
            const double median_y = tula::alg::median(y_vec);
            Eigen::VectorXd x_abs_dev = (x_vec.array() - median_x).abs().matrix();
            Eigen::VectorXd y_abs_dev = (y_vec.array() - median_y).abs().matrix();
            double sigma_x = 1.4826 * tula::alg::median(x_abs_dev);
            double sigma_y = 1.4826 * tula::alg::median(y_abs_dev);
            if (!std::isfinite(sigma_x) || sigma_x <= std::numeric_limits<double>::epsilon()) {
                sigma_x = engine_utils::calc_std_dev(x_vec);
            }
            if (!std::isfinite(sigma_y) || sigma_y <= std::numeric_limits<double>::epsilon()) {
                sigma_y = engine_utils::calc_std_dev(y_vec);
            }
            if (!std::isfinite(sigma_x) || !std::isfinite(sigma_y) ||
                sigma_x <= std::numeric_limits<double>::epsilon() ||
                sigma_y <= std::numeric_limits<double>::epsilon()) {
                continue;
            }

            stats_by_network[{static_cast<int>(array), static_cast<int>(nw)}] =
                {true, median_x, median_y, sigma_x, sigma_y, threshold};
        }
    }

    std::atomic<int> n_flagged{0};
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        if (calib.apt["flag"](i) != 0) {
            return 0;
        }

        const int array_index = static_cast<int>(std::lround(calib.apt["array"](i)));
        const int nw_index = static_cast<int>(std::lround(calib.apt["nw"](i)));
        auto it = stats_by_network.find({array_index, nw_index});
        if (it == stats_by_network.end() || !it->second.valid) {
            return 0;
        }

        const double x = calib.apt["x_t"](i);
        const double y = calib.apt["y_t"](i);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return 0;
        }

        const double zx = (x - it->second.median_x) / it->second.sigma_x;
        const double zy = (y - it->second.median_y) / it->second.sigma_y;
        const double z = std::sqrt(zx * zx + zy * zy);
        if (!std::isfinite(z) || z <= it->second.threshold) {
            return 0;
        }

        calib.apt["flag"](i) = 1;
        calib.apt["flxscale"](i) = 0.0;
        calib.apt["sens"](i) = 0.0;
        flag2(i) |= AptFlags::NetworkPos;
        n_flagged++;
        return 0;
    });

    if (n_flagged.load() > 0) {
        std::string by_array;
        for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
            Eigen::Index array = calib.arrays(i);
            std::string array_name = toltec_io.array_name_map[array];
            Eigen::Index n_array_flagged = 0;
            if (calib.array_limits.count(array) > 0) {
                for (Eigen::Index k = std::get<0>(calib.array_limits[array]);
                     k < std::get<1>(calib.array_limits[array]); ++k) {
                    if ((flag2(k) & AptFlags::NetworkPos) != 0) {
                        n_array_flagged++;
                    }
                }
            }
            if (!by_array.empty()) {
                by_array += ", ";
            }
            by_array += array_name + "=" + std::to_string(n_array_flagged);
        }
        logger->info(
            "beammap final network-position flagging: {} detectors exceeded per-array robust-z limits ({})",
            n_flagged.load(), by_array);
    }
}

void Beammap::update_final_prior_match_diagnostics() {
    final_prior_d2_diag = Eigen::VectorXd::Constant(
        calib.n_dets, std::numeric_limits<double>::quiet_NaN());
    final_prior_slot_index_diag = Eigen::VectorXi::Constant(calib.n_dets, -1);

    if (map_grouping != "detector" || !beammap_soft_priors_loaded || beammap_soft_prior_slots.empty()) {
        return;
    }

    struct ArrayCenter {
        bool valid = false;
        double x = 0.0;
        double y = 0.0;
    };

    std::map<int, ArrayCenter> centers;
    auto median_from = [](std::vector<double> &values, double &median) -> bool {
        if (values.empty()) {
            median = std::numeric_limits<double>::quiet_NaN();
            return false;
        }
        Eigen::Map<Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
        median = tula::alg::median(vec);
        return std::isfinite(median);
    };

    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        const Eigen::Index array = calib.arrays(i);
        std::vector<double> x_vals;
        std::vector<double> y_vals;

        auto gather = [&](bool unflagged_only) {
            x_vals.clear();
            y_vals.clear();
            for (Eigen::Index k = 0; k < calib.n_dets; ++k) {
                if (static_cast<Eigen::Index>(std::lround(calib.apt["array"](k))) != array) {
                    continue;
                }
                if (unflagged_only && calib.apt["flag"](k) != 0) {
                    continue;
                }
                const double x = calib.apt["x_t_raw"](k);
                const double y = calib.apt["y_t_raw"](k);
                if (!std::isfinite(x) || !std::isfinite(y)) {
                    continue;
                }
                x_vals.push_back(x);
                y_vals.push_back(y);
            }
        };

        gather(true);
        if (x_vals.size() < 8) {
            gather(false);
        }
        if (x_vals.empty()) {
            continue;
        }

        double median_x = std::numeric_limits<double>::quiet_NaN();
        double median_y = std::numeric_limits<double>::quiet_NaN();
        if (!median_from(x_vals, median_x) || !median_from(y_vals, median_y)) {
            continue;
        }
        centers[static_cast<int>(array)] = {true, median_x, median_y};
    }

    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        const int array = static_cast<int>(std::lround(calib.apt["array"](i)));
        const int nw = static_cast<int>(std::lround(calib.apt["nw"](i)));
        auto slots_it = beammap_soft_prior_slots.find({array, nw});
        if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
            continue;
        }

        double x_arcsec = calib.apt["x_t_raw"](i);
        double y_arcsec = calib.apt["y_t_raw"](i);
        if (!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) {
            continue;
        }

        if (beammap_soft_priors_are_centered) {
            auto center_it = centers.find(array);
            if (center_it != centers.end() && center_it->second.valid) {
                x_arcsec -= center_it->second.x;
                y_arcsec -= center_it->second.y;
            }
        }

        if (beammap_soft_priors_are_derotated && telescope.pixel_axes == "altaz") {
            double derot_elev_rad = calib.apt["derot_elev"](i);
            if (!std::isfinite(derot_elev_rad)) {
                derot_elev_rad = telescope.tel_data["TelElAct"].mean();
            }
            if (!std::isfinite(derot_elev_rad)) {
                derot_elev_rad = 0.0;
            }
            if (std::abs(derot_elev_rad) > pi) {
                derot_elev_rad *= DEG_TO_RAD;
            }
            const double rot_az_off = std::cos(-derot_elev_rad) * x_arcsec -
                                      std::sin(-derot_elev_rad) * y_arcsec;
            const double rot_alt_off = std::sin(-derot_elev_rad) * x_arcsec +
                                       std::cos(-derot_elev_rad) * y_arcsec;
            x_arcsec = -rot_az_off;
            y_arcsec = -rot_alt_off;
        }

        double best_d2 = std::numeric_limits<double>::infinity();
        int best_slot = -1;
        for (const auto &slot : slots_it->second) {
            const double sx = std::max(slot.sx_arcsec, std::numeric_limits<double>::epsilon());
            const double sy = std::max(slot.sy_arcsec, std::numeric_limits<double>::epsilon());
            const double dx = (x_arcsec - slot.x_arcsec) / sx;
            const double dy = (y_arcsec - slot.y_arcsec) / sy;
            const double d2 = dx * dx + dy * dy;
            if (std::isfinite(d2) && d2 < best_d2) {
                best_d2 = d2;
                best_slot = slot.slot_index;
            }
        }
        if (std::isfinite(best_d2)) {
            final_prior_d2_diag(i) = best_d2;
            final_prior_slot_index_diag(i) = best_slot;
        }
    }
}

void Beammap::log_final_network_qc_summary() {
    if (map_grouping != "detector") {
        return;
    }

    auto median_or_nan = [](std::vector<double> &values) {
        if (values.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        Eigen::Map<Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
        return tula::alg::median(vec);
    };

    logger->info("beammap final per-network qc summary follows");
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        const Eigen::Index array = calib.arrays(i);
        const std::string array_name = toltec_io.array_name_map[array];

        for (Eigen::Index j = 0; j < calib.n_nws; ++j) {
            const Eigen::Index nw = calib.nws(j);
            if (calib.nw_limits.count(nw) == 0) {
                continue;
            }
            const auto [k0, k1] = calib.nw_limits[nw];
            if (k0 < 0 || k1 <= k0) {
                continue;
            }
            if (static_cast<Eigen::Index>(std::lround(calib.apt["array"](k0))) != array) {
                continue;
            }

            std::vector<double> a_vals;
            std::vector<double> b_vals;
            std::vector<double> snr_vals;
            std::vector<double> prior_d2_vals;
            Eigen::Index n_total = 0;
            Eigen::Index n_good = 0;
            for (Eigen::Index k = k0; k < k1; ++k) {
                n_total++;
                if (calib.apt["flag"](k) != 0) {
                    continue;
                }
                n_good++;
                if (std::isfinite(calib.apt["a_fwhm"](k))) {
                    a_vals.push_back(calib.apt["a_fwhm"](k));
                }
                if (std::isfinite(calib.apt["b_fwhm"](k))) {
                    b_vals.push_back(calib.apt["b_fwhm"](k));
                }
                if (std::isfinite(calib.apt["sig2noise"](k))) {
                    snr_vals.push_back(calib.apt["sig2noise"](k));
                }
                if (final_prior_d2_diag.size() == calib.n_dets &&
                    std::isfinite(final_prior_d2_diag(k))) {
                    prior_d2_vals.push_back(final_prior_d2_diag(k));
                }
            }

            const double good_frac =
                static_cast<double>(n_good) / static_cast<double>(std::max<Eigen::Index>(1, n_total));
            logger->info(
                "beammap network qc: array={} nw={} good={}/{} ({:.3f}) med_a_fwhm={} med_b_fwhm={} med_sig2noise={} med_final_prior_d2={}",
                array_name,
                static_cast<int>(nw),
                n_good,
                n_total,
                good_frac,
                median_or_nan(a_vals),
                median_or_nan(b_vals),
                median_or_nan(snr_vals),
                median_or_nan(prior_d2_vals));
        }
    }
}
