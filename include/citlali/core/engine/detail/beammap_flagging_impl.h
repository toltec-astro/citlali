#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/runtime_policy.h>
#include <citlali/core/pipeline/beammap_config_fitting_flagging.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

void Beammap::mark_beammap_detector_flagged(
    Eigen::Index detector_index,
    AptFlags flag,
    std::atomic<int> &n_flagged_dets) {
    if (calib.apt["flag"](detector_index)==0) {
        n_flagged_dets++;
        calib.apt["flag"](detector_index) = 1;
    }
    flag2(detector_index) |= flag;
}

bool Beammap::beammap_fit_quality_values_valid(
    Eigen::Index detector_index,
    double map_std_dev) {
    const bool valid_map_std = std::isfinite(map_std_dev) && map_std_dev > 0.0;
    const bool finite_params = params.row(detector_index).array().isFinite().all();
    const bool finite_perrors = perrors.row(detector_index).array().isFinite().all();
    const bool positive_amp =
        std::isfinite(params(detector_index, 0)) && params(detector_index, 0) > 0.0;
    const bool positive_fwhm =
        std::isfinite(calib.apt["a_fwhm"](detector_index)) &&
        std::isfinite(calib.apt["b_fwhm"](detector_index)) &&
        calib.apt["a_fwhm"](detector_index) > 0.0 &&
        calib.apt["b_fwhm"](detector_index) > 0.0;
    return finite_params && finite_perrors && positive_amp && positive_fwhm && valid_map_std;
}

void Beammap::update_beammap_fit_sig2noise(Eigen::Index detector_index) {
    if (std::isfinite(perrors(detector_index, 0)) && perrors(detector_index, 0) > 0) {
        calib.apt["sig2noise"](detector_index) =
            params(detector_index, 0) / perrors(detector_index, 0);
    }
    else {
        calib.apt["sig2noise"](detector_index) = 0;
    }
}

bool Beammap::beammap_az_fwhm_outlier(
    Eigen::Index detector_index,
    const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
    const std::string &array_name) {
    return calib.apt["a_fwhm"](detector_index) < flag_limits.lower_fwhm_arcsec.at(array_name) ||
           (calib.apt["a_fwhm"](detector_index) > flag_limits.upper_fwhm_arcsec.at(array_name) &&
            flag_limits.upper_fwhm_arcsec.at(array_name) > 0);
}

bool Beammap::beammap_el_fwhm_outlier(
    Eigen::Index detector_index,
    const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
    const std::string &array_name) {
    return calib.apt["b_fwhm"](detector_index) < flag_limits.lower_fwhm_arcsec.at(array_name) ||
           (calib.apt["b_fwhm"](detector_index) > flag_limits.upper_fwhm_arcsec.at(array_name) &&
            flag_limits.upper_fwhm_arcsec.at(array_name) > 0);
}

bool Beammap::beammap_map_sig2noise_outlier(
    double map_sig2noise,
    const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
    const std::string &array_name) {
    return !std::isfinite(map_sig2noise) ||
           map_sig2noise < flag_limits.lower_sig2noise.at(array_name) ||
           (map_sig2noise > flag_limits.upper_sig2noise.at(array_name) &&
            flag_limits.upper_sig2noise.at(array_name) > 0);
}

void Beammap::flag_beammap_fit_quality_detector(
    Eigen::Index detector_index,
    const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
    std::atomic<int> &n_flagged_dets) {
    auto array_index = calib.apt["array"](detector_index);
    std::string array_name = toltec_io.array_name_map[array_index];

    const double map_std_dev = calc_map_support_stddev(detector_index, true);
    const bool valid_map_std = std::isfinite(map_std_dev) && map_std_dev > 0.0;
    if (!beammap_fit_quality_values_valid(detector_index, map_std_dev)) {
        good_fits(detector_index) = false;
    }

    update_beammap_fit_sig2noise(detector_index);

    if (!good_fits(detector_index)) {
        mark_beammap_detector_flagged(detector_index, AptFlags::BadFit, n_flagged_dets);
    }
    if (beammap_az_fwhm_outlier(detector_index, flag_limits, array_name)) {
        mark_beammap_detector_flagged(detector_index, AptFlags::AzFWHM, n_flagged_dets);
    }
    if (beammap_el_fwhm_outlier(detector_index, flag_limits, array_name)) {
        mark_beammap_detector_flagged(detector_index, AptFlags::ElFWHM, n_flagged_dets);
    }
    const double map_sig2noise =
        valid_map_std ? params(detector_index, 0) / map_std_dev : 0.0;
    if (beammap_map_sig2noise_outlier(map_sig2noise, flag_limits, array_name)) {
        mark_beammap_detector_flagged(detector_index, AptFlags::Sig2Noise, n_flagged_dets);
    }
}

void Beammap::flag_beammap_fit_quality_outliers(
    const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
    const std::string &runtime_parallel_policy,
    std::atomic<int> &n_flagged_dets) {
    logger->info("flagging detectors");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        flag_beammap_fit_quality_detector(i, flag_limits, n_flagged_dets);
        return 0;
    });
}

std::map<Eigen::Index, double> Beammap::beammap_network_median_sensitivities() {
    std::map<Eigen::Index, double> nw_median_sens;

    logger->debug("calculating mean sensitivities");
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        Eigen::Index nw = calib.nws(i);

        auto nw_sens = calib.apt["sens"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                    std::get<1>(calib.nw_limits[nw])-1));

        Eigen::Index n_good_det =
            (calib.apt["flag"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                          std::get<1>(calib.nw_limits[nw])-1)).array()==0).count();

        if (n_good_det>0) {
            Eigen::VectorXd sens(n_good_det);

            Eigen::Index j = std::get<0>(calib.nw_limits[nw]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<nw_sens.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    sens(k) = nw_sens(m);
                    k++;
                }
                j++;
            }
            nw_median_sens[nw] = tula::alg::median(sens);
        }
        else {
            nw_median_sens[nw] = tula::alg::median(nw_sens);
        }
    }

    return nw_median_sens;
}

void Beammap::flag_beammap_sensitivity_outliers(
    std::map<Eigen::Index, double> &nw_median_sens,
    double lower_sens_factor,
    double upper_sens_factor,
    const std::string &runtime_parallel_policy,
    std::atomic<int> &n_flagged_dets) {
    logger->debug("flagging sensitivities");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        auto nw_index = calib.apt["nw"](i);

        if (calib.apt["sens"](i) < lower_sens_factor*nw_median_sens[nw_index] ||
            (calib.apt["sens"](i) > upper_sens_factor*nw_median_sens[nw_index] && upper_sens_factor > 0)) {
            mark_beammap_detector_flagged(i, AptFlags::Sens, n_flagged_dets);
        }

        return 0;
    });
}

Beammap::BeammapArrayPositionMedians Beammap::beammap_array_position_medians() {
    BeammapArrayPositionMedians medians;

    logger->debug("calculating array median positions");
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        auto array_x_t = calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        auto array_y_t = calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        Eigen::Index n_good_det =
            (calib.apt["flag"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                          std::get<1>(calib.array_limits[array])-1)).array()==0).count();

        Eigen::VectorXd x_t, y_t;

        if (n_good_det>0) {
            x_t.resize(n_good_det);
            y_t.resize(n_good_det);

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
            medians.x_t[array_name] = tula::alg::median(x_t);
            medians.y_t[array_name] = tula::alg::median(y_t);
        }
        else {
            medians.x_t[array_name] = tula::alg::median(array_x_t);
            medians.y_t[array_name] = tula::alg::median(array_y_t);
        }
    }

    return medians;
}

void Beammap::flag_beammap_position_outliers(
    const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
    Beammap::BeammapArrayPositionMedians &array_position_medians,
    const std::string &runtime_parallel_policy,
    std::atomic<int> &n_flagged_dets) {
    logger->debug("flagging detector positions");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        auto array_index = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[array_index];

        double dist = sqrt(pow(calib.apt["x_t"](i) - array_position_medians.x_t[array_name],2) +
                           pow(calib.apt["y_t"](i) - array_position_medians.y_t[array_name],2));

        if (dist > flag_limits.max_dist_arcsec.at(array_name) &&
            flag_limits.max_dist_arcsec.at(array_name) > 0) {
            mark_beammap_detector_flagged(i, AptFlags::Position, n_flagged_dets);
        }

        return 0;
    });
}

void Beammap::flag_beammap_prior_distance_outliers(
    double max_prior_d2,
    const Beammap::BeammapArrayPositionMedians &array_position_medians,
    const std::string &runtime_parallel_policy,
    std::atomic<int> &n_flagged_dets) {
    const bool prior_dist_flag_enabled =
        max_prior_d2 > 0.0 && beammap_soft_priors_loaded &&
        !beammap_soft_prior_slots.empty();
    if (max_prior_d2 <= 0.0) {
        return;
    }
    if (!prior_dist_flag_enabled) {
        logger->warn(
            "beammap.flagging.max_prior_d2={} requested but soft priors are unavailable; skipping prior-distance flagging",
            max_prior_d2);
        return;
    }

    double prior_derot_elev_rad = telescope.tel_data["TelElAct"].mean();
    if (!std::isfinite(prior_derot_elev_rad)) {
        prior_derot_elev_rad = 0.0;
    }
    if (std::abs(prior_derot_elev_rad) > pi) {
        prior_derot_elev_rad *= DEG_TO_RAD;
    }
    const bool apply_derot =
        beammap_soft_priors_are_derotated &&
        citlali::config::is_altaz_map_pixel_axes(telescope.pixel_axes);
    const double cos_rot = std::cos(-prior_derot_elev_rad);
    const double sin_rot = std::sin(-prior_derot_elev_rad);
    std::atomic<int> n_prior_dist_hits{0};

    logger->debug("flagging detector prior distances");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
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
            auto x_it = array_position_medians.x_t.find(array_name);
            auto y_it = array_position_medians.y_t.find(array_name);
            const double median_x = (x_it != array_position_medians.x_t.end()) ? x_it->second : 0.0;
            const double median_y = (y_it != array_position_medians.y_t.end()) ? y_it->second : 0.0;
            x_arcsec -= median_x;
            y_arcsec -= median_y;
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
        if (!std::isfinite(min_d2) || min_d2 <= max_prior_d2) {
            return 0;
        }

        n_prior_dist_hits++;
        mark_beammap_detector_flagged(i, AptFlags::PriorDist, n_flagged_dets);
        return 0;
    });

    logger->info("beammap prior-distance flagging: {} detectors exceeded max_prior_d2={}",
                 n_prior_dist_hits.load(), max_prior_d2);
}

void Beammap::calculate_beammap_flux_conversion_factors(
    const std::string &runtime_parallel_policy) {
    logger->debug("calculating flux conversion factors");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
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
        if (calib.apt["flag"](i) == 0 && std::isfinite(amp) && amp > 0.0) {
            const double flxscale = source_flux_mJy_beam[array_name] / amp;
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
        else {
            calib.apt["flxscale"](i) = 0;
            calib.apt["sens"](i) = 0;
        }
        return 0;
    });

    calib.setup();

    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        source_flux_MJy_Sr[array_name] =
            mJY_ASEC_to_MJY_SR*(source_flux_mJy_beam[array_name])/calib.array_beam_areas[array];
    }
}

void Beammap::set_apt_flags() {
    // setup bitwise flags
    flag2.resize(calib.n_dets);
    flag2.setConstant(AptFlags::Good);

    // track number of flagged detectors
    std::atomic<int> n_flagged_dets{0};
    const auto &flagging_config =
        citlali::pipeline::beammap_config(*this).flagging;
    const auto flag_limits =
        citlali::pipeline::make_beammap_array_flagging_limits(
            toltec_io.array_name_map, flagging_config);
    const double lower_sens_factor = flagging_config.sens_factors[0];
    const double upper_sens_factor = flagging_config.sens_factors[1];
    const auto runtime_parallel_policy =
        citlali::pipeline::runtime_parallel_policy_name(*this);

    flag_beammap_fit_quality_outliers(
        flag_limits, runtime_parallel_policy, n_flagged_dets);

    auto nw_median_sens = beammap_network_median_sensitivities();
    flag_beammap_sensitivity_outliers(
        nw_median_sens, lower_sens_factor, upper_sens_factor,
        runtime_parallel_policy, n_flagged_dets);

    auto array_position_medians = beammap_array_position_medians();
    flag_beammap_position_outliers(
        flag_limits, array_position_medians,
        runtime_parallel_policy, n_flagged_dets);

    flag_beammap_prior_distance_outliers(
        flagging_config.max_prior_d2, array_position_medians,
        runtime_parallel_policy, n_flagged_dets);

    // print number of flagged detectors
    logger->info("{} detectors were flagged", n_flagged_dets.load());

    // Derive the calibration amplitude from an empirical array template where
    // possible.  The Gaussian fit amplitude remains in amp for morphology/QC.
    calc_empirical_template_calibration();

    calculate_beammap_flux_conversion_factors(runtime_parallel_policy);
}
