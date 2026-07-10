#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

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
