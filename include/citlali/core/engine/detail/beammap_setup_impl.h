#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/map_grouping_policy.h>

void Beammap::setup() {
    // assign parallel policies
    map_parallel_policy = citlali::pipeline::runtime_parallel_policy_name(*this);

    // run obsnum setup
    obsnum_setup();

    // create kids tone apt row
    calib.apt["kids_tone"].resize(calib.n_dets);

    Eigen::Index j = 0;
    // set kids tone (det number on network)
    calib.apt["kids_tone"](0) = 0;
    for (Eigen::Index i=1; i<calib.n_dets; ++i) {
        if (calib.apt["nw"](i) > calib.apt["nw"](i-1)) {
            j = 0;
        }
        else {
            j++;
        }

        calib.apt["kids_tone"](i) = j;
    }

    // add kids tone to apt header
    calib.apt_header_keys.push_back("kids_tone");
    calib.apt_header_units["kids_tone"] = "N/A";

    // resize the PTCData vector to number of scans
    ptcs0.resize(telescope.scan_indices.cols());

    // resize the calib vector to number of scans
    calib_scans0.resize(telescope.scan_indices.cols());

    // resize the initial fit matrix
    p0.setZero(map_indices.n_maps, map_fitter.n_params);
    // resize the initial fit error matrix
    perror0.setZero(map_indices.n_maps, map_fitter.n_params);
    // resize the current fit matrix
    params.setZero(map_indices.n_maps, map_fitter.n_params);
    perrors.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_init_params.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_lower_limits.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_upper_limits.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_hit_lower.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_hit_upper.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_bound_code.setZero(map_indices.n_maps);
    fit_diag_bound_nhit.setZero(map_indices.n_maps);
    prior_diag_values.resize(map_indices.n_maps, n_prior_diag_cols);
    prior_diag_values.setConstant(std::numeric_limits<double>::quiet_NaN());

    // resize good fits
    good_fits.setZero(map_indices.n_maps);
    rfi_mask_samples_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    rfi_mask_scans_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_samples_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_rows_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_edge_code = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_rejected = Eigen::VectorXi::Zero(calib.n_dets);
    final_prior_d2_diag = Eigen::VectorXd::Constant(calib.n_dets, std::numeric_limits<double>::quiet_NaN());
    final_prior_slot_index_diag = Eigen::VectorXi::Constant(calib.n_dets, -1);

    // initially all detectors are unconverged
    converged.setZero(map_indices.n_maps);
    // convergence iteration
    converge_iter.resize(map_indices.n_maps);
    converge_iter.setConstant(1);
    // set the initial iteration
    current_iter = 0;

    /* update apt table meta data */
    calib.apt_meta.reset();

    // add obsnum to meta data
    calib.apt_meta["obsnum"] = obsnum;

    // add source name
    calib.apt_meta["source"] = telescope.source_name;

    // add project id to meta data
    calib.apt_meta["project_id"] = telescope.project_id;

    const auto &beammap_phase_config = typed_config.beammap.phase_strategy;
    calib.apt_meta["beammap_phase_split_enabled"] =
        beammap_phase_config.enabled;
    calib.apt_meta["beammap_locator_iter"] = beammap_phase_config.locator_iter;
    calib.apt_meta["beammap_measurement_start_iter"] =
        beammap_phase_config.measurement_start_iter;

    // add input source flux
    for (const auto &beammap_flux: beammap_fluxes_mJy_beam) {
        auto key = beammap_flux.first + "_flux";
        calib.apt_meta[key].push_back(beammap_flux.second);
        calib.apt_meta[key].push_back("units: mJy/beam");
        calib.apt_meta[key].push_back(beammap_flux.first + " flux density");
    }

    // add date of file creation
    calib.apt_meta["creation_date"] = engine_utils::current_date_time();

    // add observation date
    calib.apt_meta["date"] = date_obs.back();

    // mean Modified Julian Date
    calib.apt_meta["mjd"] = engine_utils::unix_to_modified_julian_date(telescope.tel_data["TelTime"].mean());

    // reference frame
    calib.apt_meta["Radesys"] = telescope.pixel_axes;

    // add mean tau to apt meta
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

    // add apt header keys
    for (const auto &[param,unit]: calib.apt_header_units) {
        calib.apt_meta[param].push_back("units: " + unit);
    }
    // add apt header descriptions
    for (const auto &[param,description]: calib.apt_header_description) {
        calib.apt_meta[param].push_back(description);
    }

    // kids tone
    calib.apt_meta["kids_tone"].push_back("units: N/A");
    calib.apt_meta["kids_tone"].push_back("index of tone in network");

    // diagnostics for robust sample masking of beammap RFI
    calib.apt["rfi_masked_samples"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["rfi_masked_samples"] = "samples";
    calib.apt_header_keys.push_back("rfi_masked_samples");
    calib.apt_meta["rfi_masked_samples"].push_back("units: samples");
    calib.apt_meta["rfi_masked_samples"].push_back("number of timestream samples masked by beammap rfi_mask");

    calib.apt["rfi_masked_scans"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["rfi_masked_scans"] = "scans";
    calib.apt_header_keys.push_back("rfi_masked_scans");
    calib.apt_meta["rfi_masked_scans"].push_back("units: scans");
    calib.apt_meta["rfi_masked_scans"].push_back("number of scans with at least one sample masked by beammap rfi_mask");

    calib.apt["scan_band_masked_samples"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["scan_band_masked_samples"] = "samples";
    calib.apt_header_keys.push_back("scan_band_masked_samples");
    calib.apt_meta["scan_band_masked_samples"].push_back("units: samples");
    calib.apt_meta["scan_band_masked_samples"].push_back(
        "number of timestream samples masked by beammap scan_band_mask");

    calib.apt["scan_band_masked_rows"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["scan_band_masked_rows"] = "rows";
    calib.apt_header_keys.push_back("scan_band_masked_rows");
    calib.apt_meta["scan_band_masked_rows"].push_back("units: rows");
    calib.apt_meta["scan_band_masked_rows"].push_back(
        "number of detector-map edge rows flagged by beammap scan_band_mask");

    calib.apt["scan_band_masked_edge"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["scan_band_masked_edge"] = "N/A";
    calib.apt_header_keys.push_back("scan_band_masked_edge");
    calib.apt_meta["scan_band_masked_edge"].push_back("units: N/A");
    calib.apt_meta["scan_band_masked_edge"].push_back(
        "scan-band edge code (0 none, 1 top, 2 bottom, 3 both)");
    calib.apt_meta["scan_band_masked_edge"].push_back("0=none");
    calib.apt_meta["scan_band_masked_edge"].push_back("1=top");
    calib.apt_meta["scan_band_masked_edge"].push_back("2=bottom");
    calib.apt_meta["scan_band_masked_edge"].push_back("3=both");

    calib.apt["scan_band_mask_rejected"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["scan_band_mask_rejected"] = "N/A";
    calib.apt_header_keys.push_back("scan_band_mask_rejected");
    calib.apt_meta["scan_band_mask_rejected"].push_back("units: N/A");
    calib.apt_meta["scan_band_mask_rejected"].push_back(
        "1 if scan_band_mask proposed a mask but rejected it due to max_flagged_fraction");

    calib.apt["final_prior_slot_index"] =
        Eigen::VectorXd::Constant(calib.n_dets, -1.0);
    calib.apt_header_units["final_prior_slot_index"] = "N/A";
    calib.apt_header_keys.push_back("final_prior_slot_index");
    calib.apt_meta["final_prior_slot_index"].push_back("units: N/A");
    calib.apt_meta["final_prior_slot_index"].push_back(
        "nearest prior slot index for final detector position in prior frame (-1 if unavailable)");

    calib.apt["final_prior_d2"] =
        Eigen::VectorXd::Constant(calib.n_dets, std::numeric_limits<double>::quiet_NaN());
    calib.apt_header_units["final_prior_d2"] = "N/A";
    calib.apt_header_keys.push_back("final_prior_d2");
    calib.apt_meta["final_prior_d2"].push_back("units: N/A");
    calib.apt_meta["final_prior_d2"].push_back(
        "nearest-slot Mahalanobis d^2 for final detector position in the soft-prior frame");

    init_empirical_template_calibration_columns();

    // bitwise flag
    calib.apt_meta["flag2"].push_back("units: N/A");
    calib.apt_meta["flag2"].push_back("bitwise flag");
    calib.apt_meta["flag2"].push_back("Good=0");
    calib.apt_meta["flag2"].push_back("BadFit=1");
    calib.apt_meta["flag2"].push_back("AzFWHM=2");
    calib.apt_meta["flag2"].push_back("ElFWHM=4");
    calib.apt_meta["flag2"].push_back("Sig2Noise=8");
    calib.apt_meta["flag2"].push_back("Sens=16");
    calib.apt_meta["flag2"].push_back("Position=32");
    calib.apt_meta["flag2"].push_back("PriorDist=64");
    calib.apt_meta["flag2"].push_back("NetworkPos=128");

    // add array mapping
    for (const auto &[arr_index,arr_name]: toltec_io.array_name_map) {
        calib.apt_meta["array_order"].push_back(std::to_string(arr_index) + ": " + arr_name);
    }

    calib.apt_header_units["flag2"] = "N/A";
    calib.apt_header_keys.push_back("flag2");

    const auto &beammap_reference_config = typed_config.beammap.reference;
    // is the detector rotated?
    calib.apt_meta["is_derotated"] = beammap_reference_config.derotate;
    // was a reference detector subtracted?
    calib.apt_meta["reference_detector_subtracted"] =
        beammap_reference_config.subtract_reference_detector;
    // reference detector
    calib.apt_meta["reference_det"] = beammap_reference_det_found;
    const auto &rfi_config = typed_config.beammap.rfi_mask;
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
    calib.apt_meta["detector_weighting_mode"] =
        std::string(citlali::config::to_string(
            typed_config.beammap.detector_weighting_mode));
    calib.apt_meta["beammap_fit_radius_fwhm"] =
        typed_config.beammap.fitting.fit_radius_fwhm;
    beammap_soft_prior_slots.clear();
    beammap_soft_priors_loaded = false;
    beammap_soft_priors_are_centered = false;
    beammap_soft_priors_are_derotated = false;
    beammap_prior_array_center_x_arcsec.clear();
    beammap_prior_array_center_y_arcsec.clear();
    beammap_prior_array_alignment.clear();
    auto &priors_config = typed_config.beammap.priors;
    if (priors_config.enabled) {
        if (typed_config.mapmaking.grouping !=
            citlali::config::MapGrouping::detector) {
            logger->warn("beammap priors requested but map_grouping={} (requires detector); disabling priors",
                         citlali::pipeline::active_map_grouping_name(*this));
            priors_config.enabled = false;
        }
        else if (!load_soft_priors()) {
            logger->warn("beammap priors failed to load; disabling prior-guided initialization");
            priors_config.enabled = false;
        }
    }
    calib.apt_meta["beammap_priors_enabled"] = priors_config.enabled;
    calib.apt_meta["beammap_priors_filepath"] = priors_config.filepath;
    calib.apt_meta["beammap_priors_candidate_top_n"] = priors_config.candidate_top_n;
    calib.apt_meta["beammap_priors_min_snr"] = priors_config.min_snr;
    calib.apt_meta["beammap_priors_max_d2"] = priors_config.max_d2;
    calib.apt_meta["beammap_priors_max_d2_iter0"] = priors_config.max_d2_iter0;
    calib.apt_meta["beammap_priors_max_d2_after_iter0"] = priors_config.max_d2_after_iter0;
    calib.apt_meta["beammap_priors_score_lambda"] = priors_config.score_lambda;
    calib.apt_meta["beammap_priors_score_lambda_iter0"] = priors_config.score_lambda_iter0;
    calib.apt_meta["beammap_priors_score_lambda_after_iter0"] = priors_config.score_lambda_after_iter0;
    calib.apt_meta["beammap_priors_fallback_blind"] = priors_config.fallback_blind;
    calib.apt_meta["beammap_priors_align_after_iter0"] = priors_config.align_after_iter0;
    calib.apt_meta["beammap_priors_alignment_scope"] =
        std::string(citlali::config::to_string(priors_config.alignment_scope));
    calib.apt_meta["beammap_priors_alignment_common_support"] =
        std::string(citlali::config::to_string(
            priors_config.alignment_common_support));
    calib.apt_meta["beammap_priors_alignment_common_support_quantile"] =
        priors_config.alignment_common_support_quantile;
    calib.apt_meta["beammap_priors_alignment_min_matches"] = priors_config.alignment_min_matches;
    calib.apt_meta["beammap_priors_alignment_max_d2"] = priors_config.alignment_max_d2;
    calib.apt_meta["beammap_priors_alignment_fit_rotation"] = priors_config.alignment_fit_rotation;
    calib.apt_meta["beammap_priors_alignment_max_rotation_deg"] = priors_config.alignment_max_rotation_deg;
}
