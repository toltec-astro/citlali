#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

double Beammap::beammap_detector_flux_calibration_amp(Eigen::Index detector_index) {
    const double template_cal_amp =
        (calib.apt.count("cal_amp") > 0 && calib.apt["cal_amp"].size() == calib.n_dets)
            ? calib.apt["cal_amp"](detector_index)
            : std::numeric_limits<double>::quiet_NaN();
    return (std::isfinite(template_cal_amp) && template_cal_amp > 0.0)
               ? template_cal_amp
               : params(detector_index, 0);
}

void Beammap::clear_beammap_detector_flux_conversion(Eigen::Index detector_index) {
    calib.apt["flxscale"](detector_index) = 0;
    calib.apt["sens"](detector_index) = 0;
}

void Beammap::reject_beammap_detector_flux_conversion(Eigen::Index detector_index) {
    clear_beammap_detector_flux_conversion(detector_index);
    calib.apt["flag"](detector_index) = 1;
    flag2(detector_index) |= AptFlags::Sens;
}

void Beammap::calculate_beammap_detector_flux_conversion(Eigen::Index detector_index) {
    auto array_index = calib.apt["array"](detector_index);
    std::string array_name = toltec_io.array_name_map[array_index];

    const double amp = beammap_detector_flux_calibration_amp(detector_index);
    if (calib.apt["flag"](detector_index) != 0 || !std::isfinite(amp) || amp <= 0.0) {
        clear_beammap_detector_flux_conversion(detector_index);
        return;
    }

    const double flxscale = source_flux_mJy_beam[array_name] / amp;
    if (std::isfinite(flxscale) && flxscale > 0.0) {
        calib.apt["flxscale"](detector_index) = flxscale;
        calib.apt["sens"](detector_index) = calib.apt["sens"](detector_index) * flxscale;
    }
    else {
        reject_beammap_detector_flux_conversion(detector_index);
    }
}

void Beammap::update_beammap_array_source_flux_density() {
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        source_flux_MJy_Sr[array_name] =
            mJY_ASEC_to_MJY_SR*(source_flux_mJy_beam[array_name])/calib.array_beam_areas[array];
    }
}

void Beammap::calculate_beammap_flux_conversion_factors(
    const std::string &runtime_parallel_policy) {
    logger->debug("calculating flux conversion factors");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        calculate_beammap_detector_flux_conversion(i);
        return 0;
    });

    calib.setup();
    update_beammap_array_source_flux_density();
}
