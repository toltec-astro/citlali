#pragma once

#include <citlali/core/timestream/calibration_product.h>

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/raw_timestream_policy.h>

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
    if (citlali::pipeline::raw_extinction_correction_enabled(*this)) {
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
    calib.apt_meta["calibration_operator_id"] =
        std::string{rtcproc.calibration.operator_id()};
    calib.apt_meta["calibration_operator_contract_sha256"] =
        std::string{rtcproc.calibration.operator_contract_sha256()};
    calib.apt_meta["calibration_node_table_sha256"] =
        std::string{rtcproc.calibration.operator_nodes_sha256()};
    calib.apt_meta["calibration_passband_set_id"] =
        std::string{rtcproc.calibration.passband_set_id()};
    calib.apt_meta["calibration_reference_profile_id"] =
        std::string{rtcproc.calibration.reference_profile_id()};
    calib.apt_meta["calibration_reference_spectral_index_alpha"] =
        rtcproc.calibration.effective_reference_spectral_index_alpha();
    calib.apt_meta["calibration_reference_spectral_index_default_applied"] =
        rtcproc.calibration.reference_spectral_index_default_applied();
    calib.apt_meta["calibration_quality_regime"] =
        rtcproc.calibration.calibration_quality_regime;
    calib.apt_meta["calibration_valid"] =
        rtcproc.calibration.calibration_valid;
    calib.apt_meta["calibration_validity_reason"] =
        rtcproc.calibration.calibration_validity_reason;
    const auto &product = rtcproc.calibration.product;
    calib.apt_meta["calibration_product_schema"] =
        std::string{product.schema_version};
    calib.apt_meta["calibration_validity_detail"] = product.validity_detail;
    calib.apt_meta["calibration_target_unit"] = product.target_unit;
    calib.apt_meta["calibration_photometry_policy"] =
        std::string{product.photometry_policy};
    calib.apt_meta["calibration_factor_composition"] =
        std::string{product.factor_composition};
    calib.apt_meta["calibration_factor_provenance"] =
        std::string{product.factor_provenance};
    calib.apt_meta["calibration_compatibility_fcf_semantics"] =
        std::string{product.compatibility_fcf_semantics};
    calib.apt_meta["calibration_weight_recipient_semantics"] =
        std::string{product.weight_recipient_semantics};
    calib.apt_meta["calibration_compact_covariance_state"] =
        std::string{product.compact_covariance_state};
    calib.apt_meta["calibration_apt_artifact_sha256"] =
        product.apt_artifact_sha256;
    calib.apt_meta["calibration_acquisition_binding_sha256"] =
        product.acquisition_binding_sha256;
    calib.apt_meta["calibration_raw_observation_identity"] =
        product.raw_observation_identity;
    calib.apt_meta["calibration_acquisition_binding_mode"] =
        product.acquisition_binding_mode;
    calib.apt_meta["calibration_acquisition_key_schema"] =
        product.acquisition_key_schema;
    calib.apt_meta["calibration_response_identity"] =
        product.response_identity;
    calib.apt_meta["calibration_conditional_variance_transfer"] =
        std::string{product.conditional_variance_transfer};
    calib.apt_meta["calibration_conditional_inverse_variance_transfer"] =
        std::string{product.conditional_inverse_variance_transfer};
    calib.apt_meta["calibration_precision_limitation"] =
        std::string{product.precision_limitation};
    calib.apt_meta["calibration_nuisance_states"] =
        timestream::calibration_nuisance_state_summary(product);
    const auto minimum_total_multiplier =
        timestream::minimum_total_signal_multiplier(product);
    const auto maximum_total_multiplier =
        timestream::maximum_total_signal_multiplier(product);
    const bool total_multiplier_extrema_available =
        std::isfinite(minimum_total_multiplier) &&
        std::isfinite(maximum_total_multiplier);
    calib.apt_meta["calibration_total_multiplier_extrema_available"] =
        total_multiplier_extrema_available;
    if (total_multiplier_extrema_available) {
        calib.apt_meta["calibration_minimum_total_multiplier"] =
            minimum_total_multiplier;
        calib.apt_meta["calibration_maximum_total_multiplier"] =
            maximum_total_multiplier;
    }
    calib.apt_meta["calibration_tau_frame"] =
        "line_of_sight_at_sample_elevation";
    calib.apt_meta["calibration_x_ref"] = 0.0;
    calib.apt_meta["flxscale_reference_plane"] = "top_of_atmosphere";
    calib.apt_meta["flxscale_extinction_application"] =
        "shared_tod_surface_once_before_fit_no_second_factor";
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
