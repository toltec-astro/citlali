#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

struct reduControls {
    // interpolate over gaps in timestreams
    bool interp_over_gaps;
    // create reduction subdirectories
    bool use_subdir;

    // run or skip tod processing
    bool run_tod;

    // output timestreams
    bool run_tod_output;

    // controls for mapmaking
    bool run_mapmaking;
    bool run_coadd;
    bool run_noise;
    bool write_noise_realizations;
    bool run_noise_products;
    bool apply_empirical_noise_weights;
    bool run_map_filter;

    // run source finding
    bool run_source_finder;
};

struct beammapControls {
    // source name
    std::string beammap_source_name;

    // beammap source position
    double beammap_ra_rad, beammap_dec_rad;

    // fluxes and errs
    std::map<std::string, double> beammap_fluxes_mJy_beam, beammap_err_mJy_beam;
    std::map<std::string, double> beammap_fluxes_MJy_Sr, beammap_err_MJy_Sr;

    // maximum beammap iterations
    int beammap_iter_max;

    // beammap tolerance
    double beammap_iter_tolerance;

    // beammap convergence aperture radius
    double beammap_convergence_radius_arcsec = 10.0;

    // detector-beammap iteration phase controls
    bool beammap_phase_split_enabled = true;
    int beammap_locator_iter = 0;
    int beammap_measurement_start_iter = 1;

    // subtract reference detector
    bool beammap_subtract_reference;

    // beammap reference detector
    Eigen::Index beammap_reference_det;

    // derotate fitted detectors
    bool beammap_derotate;

    // optional robust sample-level RFI masking in detector-grouped beammaps
    bool beammap_rfi_mask_enabled = false;
    int beammap_rfi_mask_block_size_samples = 64;
    int beammap_rfi_mask_min_good_samples = 32;
    int beammap_rfi_mask_dilate_blocks = 1;
    double beammap_rfi_mask_sigma_threshold = 6.0;
    double beammap_rfi_mask_sigma_floor = 0.0;
    double beammap_rfi_mask_max_flagged_fraction = 0.35;

    // detector-map sample weighting policy
    std::string beammap_detector_weighting_mode = "const";

    // optional circular residual support for beammap Gaussian fits, in nominal FWHM units
    double beammap_fit_radius_fwhm = 0.0;

    // optional detector-map edge-band masking for coherent bad scan legs
    bool beammap_scan_band_mask_enabled = false;
    int beammap_scan_band_mask_edge_rows = 24;
    int beammap_scan_band_mask_min_row_pixels = 8;
    int beammap_scan_band_mask_min_contiguous_rows = 2;
    double beammap_scan_band_mask_row_median_sigma_threshold = 4.0;
    double beammap_scan_band_mask_row_sigma_ratio_threshold = 2.5;
    double beammap_scan_band_mask_max_flagged_fraction = 0.30;

    // optional beammap detector-map FITS splitting by detector quality flag
    bool beammap_split_fits_by_flag = false;
    std::vector<int> beammap_split_flag_values = {0, 1};

    // optional soft priors for beammap peak initialization
    bool beammap_priors_enabled = false;
    std::string beammap_priors_filepath = "null";
    int beammap_priors_candidate_top_n = 64;
    double beammap_priors_min_snr = 0.0;
    double beammap_priors_max_d2 = 25.0;
    double beammap_priors_max_d2_iter0 = 25.0;
    double beammap_priors_max_d2_after_iter0 = 25.0;
    double beammap_priors_score_lambda = 2.0;
    double beammap_priors_score_lambda_iter0 = 2.0;
    double beammap_priors_score_lambda_after_iter0 = 2.0;
    bool beammap_priors_fallback_blind = true;
    bool beammap_priors_align_after_iter0 = true;
    std::string beammap_priors_alignment_scope = "array";
    std::string beammap_priors_alignment_common_support = "all";
    double beammap_priors_alignment_common_support_quantile = 0.02;
    int beammap_priors_alignment_min_matches = 30;
    double beammap_priors_alignment_max_d2 = 25.0;
    bool beammap_priors_alignment_fit_rotation = true;
    double beammap_priors_alignment_max_rotation_deg = 8.0;

    // iteration to write out beammap PTC data; -1 means final attempted iteration
    int beammap_tod_output_iter = -1;

    // optional detector-specific PTC TOD diagnostic sidecar for beammaps
    bool beammap_detector_tod_output_enabled = false;
    std::string beammap_detector_tod_output_subdir_name = "source_crossing_tod";
    int beammap_detector_tod_output_n_uniform = 10;
    int beammap_detector_tod_output_n_source_dense = 10;

    // upper and lower limits of psd for sensitivity calc
    Eigen::VectorXd sens_psd_limits_Hz;

    // limits on fwhm, sig2noise, and distance from center for flagging
    std::map<std::string, double> lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise,
        upper_sig2noise, max_dist_arcsec, network_robust_z;
    double beammap_flag_max_prior_d2 = 0.0;

    // limits on sensitivity for flagging
    double lower_sens_factor, upper_sens_factor;
};

struct pointingControls {
    // source-aware pointing strategy.  Gaussian fits are optional diagnostics;
    // fruit loops remains empirical and uses previous maps.
    std::string pointing_source_strategy = "standard";
    bool pointing_fit_gaussian_enabled = true;
    std::string pointing_fruitloops_center_mode = "auto";
    double pointing_header_center_max_radius_arcsec = 0.0;
    bool pointing_header_center_require_coverage = true;
};
