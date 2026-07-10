#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::populate_beammap_identity_metadata() {
    calib.apt_meta["obsnum"] = observation_identity.obsnum;
    calib.apt_meta["source"] = telescope.source_name;
    calib.apt_meta["project_id"] = telescope.project_id;
}

void Beammap::populate_beammap_phase_metadata() {
    const auto &beammap_config = citlali::pipeline::beammap_config(*this);
    const auto &beammap_phase_config = beammap_config.phase_strategy;
    calib.apt_meta["beammap_phase_split_enabled"] =
        beammap_phase_config.enabled;
    calib.apt_meta["beammap_locator_iter"] = beammap_phase_config.locator_iter;
    calib.apt_meta["beammap_measurement_start_iter"] =
        beammap_phase_config.measurement_start_iter;
}

void Beammap::populate_beammap_flux_metadata() {
    for (const auto &beammap_flux: source_flux_mJy_beam) {
        auto key = beammap_flux.first + "_flux";
        calib.apt_meta[key].push_back(beammap_flux.second);
        calib.apt_meta[key].push_back("units: mJy/beam");
        calib.apt_meta[key].push_back(beammap_flux.first + " flux density");
    }
}

void Beammap::populate_beammap_time_and_frame_metadata() {
    calib.apt_meta["creation_date"] = engine_utils::current_date_time();
    calib.apt_meta["date"] =
        citlali::pipeline::latest_observation_date(observation_dates);
    calib.apt_meta["mjd"] =
        engine_utils::unix_to_modified_julian_date(telescope.tel_data["TelTime"].mean());
    calib.apt_meta["Radesys"] = telescope.pixel_axes;
}

void Beammap::populate_beammap_tau_metadata() {
    if (rtcproc.run_extinction) {
        Eigen::VectorXd tau_el(1);
        tau_el << telescope.tel_data["TelElAct"].mean();
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);

        Eigen::Index i = 0;
        for (auto const& [key, val] : tau_freq) {
            calib.apt_meta[toltec_io.array_name_map[calib.arrays(i)]+"_tau"] = val[0];
            i++;
        }
    }
    else {
        for (Eigen::Index i=0; i<calib.arrays.size(); ++i) {
            calib.apt_meta[toltec_io.array_name_map[calib.arrays(i)]+"_tau"] = 0.;
        }
    }
}

void Beammap::populate_beammap_header_metadata() {
    for (const auto &[param,unit]: calib.apt_header_units) {
        calib.apt_meta[param].push_back("units: " + unit);
    }
    for (const auto &[param,description]: calib.apt_header_description) {
        calib.apt_meta[param].push_back(description);
    }

    calib.apt_meta["kids_tone"].push_back("units: N/A");
    calib.apt_meta["kids_tone"].push_back("index of tone in network");
}

void Beammap::populate_beammap_reference_metadata() {
    const auto &beammap_config = citlali::pipeline::beammap_config(*this);
    const auto &beammap_reference_config = beammap_config.reference;
    calib.apt_meta["is_derotated"] = beammap_reference_config.derotate;
    calib.apt_meta["reference_detector_subtracted"] =
        beammap_reference_config.subtract_reference_detector;
    calib.apt_meta["reference_det"] = beammap_reference_det_found;
}

void Beammap::populate_beammap_masking_metadata() {
    const auto &beammap_config = citlali::pipeline::beammap_config(*this);
    const auto &rfi_config = beammap_config.rfi_mask;
    calib.apt_meta["rfi_mask_enabled"] = rfi_config.enabled;
    calib.apt_meta["rfi_mask_block_size_samples"] =
        rfi_config.block_size_samples;
    calib.apt_meta["rfi_mask_min_good_samples"] =
        rfi_config.min_good_samples;
    calib.apt_meta["rfi_mask_dilate_blocks"] = rfi_config.dilate_blocks;
    calib.apt_meta["rfi_mask_sigma_threshold"] =
        rfi_config.sigma_threshold;
    calib.apt_meta["rfi_mask_sigma_floor"] = rfi_config.sigma_floor;
    calib.apt_meta["rfi_mask_max_flagged_fraction"] =
        rfi_config.max_flagged_fraction;
}

void Beammap::populate_beammap_weighting_and_fit_metadata() {
    const auto &beammap_config = citlali::pipeline::beammap_config(*this);
    calib.apt_meta["detector_weighting_mode"] =
        std::string(citlali::config::to_string(
            beammap_config.detector_weighting_mode));
    calib.apt_meta["beammap_fit_radius_fwhm"] =
        beammap_config.fitting.fit_radius_fwhm;
}

void Beammap::populate_beammap_setup_metadata() {
    calib.apt_meta.reset();
    populate_beammap_identity_metadata();
    populate_beammap_phase_metadata();
    populate_beammap_flux_metadata();
    populate_beammap_time_and_frame_metadata();
    populate_beammap_tau_metadata();
    populate_beammap_header_metadata();
    populate_beammap_reference_metadata();
    populate_beammap_masking_metadata();
    populate_beammap_weighting_and_fit_metadata();
}
