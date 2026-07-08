#pragma once

#include <yaml-cpp/yaml.h>

#include <map>
#include <string>
#include <vector>

// Beammap detector fit-QC table schema helpers.

namespace beammap_fit_qc_schema {

inline std::vector<std::string> header_keys() {
    return {
        "uid",
        "array",
        "nw",
        "kids_tone",
        "good_fit",
        "converged",
        "converge_iter",
        "flag",
        "flag2",
        "amp",
        "amp_err",
        "cal_amp",
        "cal_amp_method",
        "template_amp",
        "template_offset",
        "template_resid_rms",
        "template_npix",
        "template_amp_over_fit_amp",
        "cal_amp_over_fit_amp",
        "map_peak_amp",
        "map_peak_amp_over_fit_amp",
        "fit_sig2noise",
        "map_rms",
        "map_sig2noise",
        "n_weight_pos",
        "fruitloops_source_x_t",
        "fruitloops_source_y_t",
        "fruitloops_local_sigma",
        "fruitloops_local_sigma_npix",
        "fruitloops_amp_ref",
        "fruitloops_peak_threshold",
        "fruitloops_snr_threshold",
        "fruitloops_adaptive_threshold",
        "fruitloops_support_radius_arcsec",
        "fruitloops_support_npix",
        "fruitloops_support_signal_sum",
        "fruitloops_support_x_span_arcsec",
        "fruitloops_support_y_span_arcsec",
        "rfi_masked_samples",
        "rfi_masked_scans",
        "scan_band_masked_samples",
        "scan_band_masked_rows",
        "scan_band_masked_edge",
        "scan_band_mask_rejected",
        "fit_bound_nhit",
        "fit_bound_code",
        "fit_bound_amp",
        "fit_bound_x",
        "fit_bound_y",
        "fit_bound_a",
        "fit_bound_b",
        "fit_bound_angle",
        "fit_init_amp",
        "fit_init_x_t",
        "fit_init_y_t",
        "fit_init_a_fwhm",
        "fit_init_b_fwhm",
        "fit_low_a_fwhm",
        "fit_high_a_fwhm",
        "fit_low_b_fwhm",
        "fit_high_b_fwhm",
        "prior_init_mode",
        "prior_used",
        "prior_fallback_blind",
        "prior_no_candidate_reason",
        "prior_slot_index",
        "prior_match_d2",
        "prior_match_score",
        "prior_candidate_snr",
        "prior_n_candidates",
        "prior_n_candidates_keep",
        "prior_n_candidates_gate",
        "prior_candidate_x_t_raw",
        "prior_candidate_y_t_raw",
        "prior_candidate_x_t_prior",
        "prior_candidate_y_t_prior",
        "prior_center_x_t",
        "prior_center_y_t",
        "prior_derot_elev",
        "prior_slot_x_t",
        "prior_slot_y_t",
        "prior_slot_sx",
        "prior_slot_sy",
        "final_prior_slot_index",
        "final_prior_d2",
        "x_t_raw",
        "y_t_raw",
        "x_t",
        "y_t",
        "x_t_derot",
        "y_t_derot",
        "a_fwhm",
        "a_fwhm_err",
        "b_fwhm",
        "b_fwhm_err",
        "angle",
        "angle_err",
        "flxscale",
        "sens"
    };
}

template <class TableAccess>
std::map<std::string, std::string> units(const TableAccess &table_access,
                                         const std::string &sig_unit) {
    return {
        {"uid", "N/A"},
        {"array", "N/A"},
        {"nw", "N/A"},
        {"kids_tone", "N/A"},
        {"good_fit", "N/A"},
        {"converged", "N/A"},
        {"converge_iter", "N/A"},
        {"flag", "N/A"},
        {"flag2", "N/A"},
        {"amp", table_access.unit("amp", sig_unit)},
        {"amp_err", table_access.unit("amp_err", sig_unit)},
        {"cal_amp", table_access.unit("cal_amp", sig_unit)},
        {"cal_amp_method", "N/A"},
        {"template_amp", table_access.unit("template_amp", sig_unit)},
        {"template_offset", table_access.unit("template_offset", sig_unit)},
        {"template_resid_rms", table_access.unit("template_resid_rms", sig_unit)},
        {"template_npix", "pix"},
        {"template_amp_over_fit_amp", "N/A"},
        {"cal_amp_over_fit_amp", "N/A"},
        {"map_peak_amp", table_access.unit("map_peak_amp", sig_unit)},
        {"map_peak_amp_over_fit_amp", "N/A"},
        {"fit_sig2noise", "N/A"},
        {"map_rms", sig_unit},
        {"map_sig2noise", "N/A"},
        {"n_weight_pos", "pix"},
        {"fruitloops_source_x_t", "arcsec"},
        {"fruitloops_source_y_t", "arcsec"},
        {"fruitloops_local_sigma", sig_unit},
        {"fruitloops_local_sigma_npix", "pix"},
        {"fruitloops_amp_ref", sig_unit},
        {"fruitloops_peak_threshold", sig_unit},
        {"fruitloops_snr_threshold", sig_unit},
        {"fruitloops_adaptive_threshold", sig_unit},
        {"fruitloops_support_radius_arcsec", "arcsec"},
        {"fruitloops_support_npix", "pix"},
        {"fruitloops_support_signal_sum", sig_unit},
        {"fruitloops_support_x_span_arcsec", "arcsec"},
        {"fruitloops_support_y_span_arcsec", "arcsec"},
        {"rfi_masked_samples", "samples"},
        {"rfi_masked_scans", "scans"},
        {"scan_band_masked_samples", "samples"},
        {"scan_band_masked_rows", "rows"},
        {"scan_band_masked_edge", "N/A"},
        {"scan_band_mask_rejected", "N/A"},
        {"fit_bound_nhit", "N/A"},
        {"fit_bound_code", "N/A"},
        {"fit_bound_amp", "N/A"},
        {"fit_bound_x", "N/A"},
        {"fit_bound_y", "N/A"},
        {"fit_bound_a", "N/A"},
        {"fit_bound_b", "N/A"},
        {"fit_bound_angle", "N/A"},
        {"fit_init_amp", table_access.unit("amp", sig_unit)},
        {"fit_init_x_t", "arcsec"},
        {"fit_init_y_t", "arcsec"},
        {"fit_init_a_fwhm", "arcsec"},
        {"fit_init_b_fwhm", "arcsec"},
        {"fit_low_a_fwhm", "arcsec"},
        {"fit_high_a_fwhm", "arcsec"},
        {"fit_low_b_fwhm", "arcsec"},
        {"fit_high_b_fwhm", "arcsec"},
        {"prior_init_mode", "N/A"},
        {"prior_used", "N/A"},
        {"prior_fallback_blind", "N/A"},
        {"prior_no_candidate_reason", "N/A"},
        {"prior_slot_index", "N/A"},
        {"prior_match_d2", "N/A"},
        {"prior_match_score", "N/A"},
        {"prior_candidate_snr", "N/A"},
        {"prior_n_candidates", "pix"},
        {"prior_n_candidates_keep", "pix"},
        {"prior_n_candidates_gate", "pix"},
        {"prior_candidate_x_t_raw", "arcsec"},
        {"prior_candidate_y_t_raw", "arcsec"},
        {"prior_candidate_x_t_prior", "arcsec"},
        {"prior_candidate_y_t_prior", "arcsec"},
        {"prior_center_x_t", "arcsec"},
        {"prior_center_y_t", "arcsec"},
        {"prior_derot_elev", "rad"},
        {"prior_slot_x_t", "arcsec"},
        {"prior_slot_y_t", "arcsec"},
        {"prior_slot_sx", "arcsec"},
        {"prior_slot_sy", "arcsec"},
        {"final_prior_slot_index", "N/A"},
        {"final_prior_d2", "N/A"},
        {"x_t_raw", table_access.unit("x_t", "arcsec")},
        {"y_t_raw", table_access.unit("y_t", "arcsec")},
        {"x_t", table_access.unit("x_t", "arcsec")},
        {"y_t", table_access.unit("y_t", "arcsec")},
        {"x_t_derot", table_access.unit("x_t", "arcsec")},
        {"y_t_derot", table_access.unit("y_t", "arcsec")},
        {"a_fwhm", table_access.unit("a_fwhm", "arcsec")},
        {"a_fwhm_err", table_access.unit("a_fwhm_err", "arcsec")},
        {"b_fwhm", table_access.unit("b_fwhm", "arcsec")},
        {"b_fwhm_err", table_access.unit("b_fwhm_err", "arcsec")},
        {"angle", table_access.unit("angle", "rad")},
        {"angle_err", table_access.unit("angle_err", "rad")},
        {"flxscale", table_access.unit("flxscale", "N/A")},
        {"sens", table_access.unit("sens", "N/A")}
    };
}

template <class TableAccess>
std::map<std::string, std::string> descriptions(const TableAccess &table_access) {
    return {
        {"uid", table_access.description("uid", "detector uid")},
        {"array", table_access.description("array", "array index")},
        {"nw", table_access.description("nw", "network index")},
        {"kids_tone", table_access.description("kids_tone", "index of tone in network")},
        {"good_fit", "fit returned a usable solution"},
        {"converged", "beammap iterative convergence flag"},
        {"converge_iter", table_access.description("converge_iter", "beammap convergence iteration")},
        {"flag", table_access.description("flag", "detector quality flag")},
        {"flag2", "bitwise detector quality flag"},
        {"amp", table_access.description("amp", "fitted beam amplitude")},
        {"amp_err", table_access.description("amp_err", "fitted beam amplitude uncertainty")},
        {"cal_amp", table_access.description("cal_amp", "amplitude used for beammap flux calibration")},
        {"cal_amp_method", table_access.description("cal_amp_method", "calibration amplitude method code")},
        {"template_amp", table_access.description("template_amp", "empirical-template matched amplitude")},
        {"template_offset", table_access.description("template_offset", "empirical-template fitted local offset")},
        {"template_resid_rms", table_access.description("template_resid_rms", "empirical-template residual RMS")},
        {"template_npix", table_access.description("template_npix", "number of pixels used by empirical-template amplitude fit")},
        {"template_amp_over_fit_amp", table_access.description("template_amp_over_fit_amp", "empirical-template amplitude divided by Gaussian fit amplitude")},
        {"cal_amp_over_fit_amp", table_access.description("cal_amp_over_fit_amp", "calibration amplitude divided by Gaussian fit amplitude")},
        {"map_peak_amp", table_access.description("map_peak_amp", "local map peak near Gaussian fit center")},
        {"map_peak_amp_over_fit_amp", table_access.description("map_peak_amp_over_fit_amp", "local map peak divided by Gaussian fit amplitude")},
        {"fit_sig2noise", "fitted amplitude divided by fitted amplitude uncertainty"},
        {"map_rms", "standard deviation of positive-weight detector map pixels, excluding fitted source core when possible"},
        {"map_sig2noise", "fitted amplitude divided by support-only detector map rms"},
        {"n_weight_pos", "number of detector-map pixels with positive weight"},
        {"fruitloops_source_x_t", "source-support x center used by adaptive fruit loops feedback"},
        {"fruitloops_source_y_t", "source-support y center used by adaptive fruit loops feedback"},
        {"fruitloops_local_sigma", "local annulus robust sigma used by adaptive fruit loops feedback"},
        {"fruitloops_local_sigma_npix", "number of local-annulus pixels used for fruit loops local sigma"},
        {"fruitloops_amp_ref", "amplitude reference used by adaptive fruit loops feedback"},
        {"fruitloops_peak_threshold", "peak-fraction threshold component for adaptive fruit loops feedback"},
        {"fruitloops_snr_threshold", "local-S/N threshold component for adaptive fruit loops feedback"},
        {"fruitloops_adaptive_threshold", "final adaptive threshold used for fruit loops map feedback"},
        {"fruitloops_support_radius_arcsec", "radial source-support limit used by adaptive fruit loops feedback"},
        {"fruitloops_support_npix", "number of final-map positive-weight pixels passing adaptive fruit loops threshold and support cuts"},
        {"fruitloops_support_signal_sum", "sum of final-map signal over pixels passing adaptive fruit loops threshold and support cuts"},
        {"fruitloops_support_x_span_arcsec", "x span of final-map pixels passing adaptive fruit loops threshold and support cuts"},
        {"fruitloops_support_y_span_arcsec", "y span of final-map pixels passing adaptive fruit loops threshold and support cuts"},
        {"rfi_masked_samples", "number of timestream samples masked by beammap rfi_mask"},
        {"rfi_masked_scans", "number of scans with at least one sample masked by beammap rfi_mask"},
        {"scan_band_masked_samples", "number of timestream samples masked by beammap scan_band_mask"},
        {"scan_band_masked_rows", "number of detector-map edge rows flagged by beammap scan_band_mask"},
        {"scan_band_masked_edge", "scan-band edge code (0 none, 1 top, 2 bottom, 3 both)"},
        {"scan_band_mask_rejected", "1 if scan_band_mask proposed a mask but rejected it due to max_flagged_fraction"},
        {"fit_bound_nhit", "number of fitted parameters at lower/upper bounds"},
        {"fit_bound_code", "bitmask of bound-hit parameters (see metadata legend)"},
        {"fit_bound_amp", "bound state for amplitude (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_x", "bound state for fitted x center (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_y", "bound state for fitted y center (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_a", "bound state for fitted a sigma/FWHM (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_b", "bound state for fitted b sigma/FWHM (-1 lower, 0 none, +1 upper)"},
        {"fit_bound_angle", "bound state for fitted angle (-1 lower, 0 none, +1 upper)"},
        {"fit_init_amp", "initial amplitude used by Gaussian fitter"},
        {"fit_init_x_t", "initial x position converted to arcsec offset"},
        {"fit_init_y_t", "initial y position converted to arcsec offset"},
        {"fit_init_a_fwhm", "initial a FWHM implied by fitter initialization"},
        {"fit_init_b_fwhm", "initial b FWHM implied by fitter initialization"},
        {"fit_low_a_fwhm", "active lower bound for a FWHM"},
        {"fit_high_a_fwhm", "active upper bound for a FWHM"},
        {"fit_low_b_fwhm", "active lower bound for b FWHM"},
        {"fit_high_b_fwhm", "active upper bound for b FWHM"},
        {"prior_init_mode", "prior-init mode code (0 blind, 1 previous, 2 prior, -1 skipped/not fit)"},
        {"prior_used", "1 if the final initialization seed came from priors, else 0"},
        {"prior_fallback_blind", "1 if priors were attempted but blind fallback was used, else 0"},
        {"prior_no_candidate_reason", "reason code when priors produced no accepted candidate (see metadata legend)"},
        {"prior_slot_index", "matched prior slot index for the chosen prior-guided seed"},
        {"prior_match_d2", "Mahalanobis d^2 of chosen prior-guided seed in prior frame"},
        {"prior_match_score", "prior ranking score of chosen prior-guided seed"},
        {"prior_candidate_snr", "S/N metric of chosen prior-guided seed candidate"},
        {"prior_n_candidates", "number of weighted pixels above prior min_snr before top-N truncation"},
        {"prior_n_candidates_keep", "number of top-ranked candidates retained for prior scoring"},
        {"prior_n_candidates_gate", "number of retained candidates that passed the prior d^2 gate"},
        {"prior_candidate_x_t_raw", "chosen prior-guided candidate x offset before prior-frame transforms"},
        {"prior_candidate_y_t_raw", "chosen prior-guided candidate y offset before prior-frame transforms"},
        {"prior_candidate_x_t_prior", "chosen prior-guided candidate x offset in the prior frame"},
        {"prior_candidate_y_t_prior", "chosen prior-guided candidate y offset in the prior frame"},
        {"prior_center_x_t", "array-center x offset subtracted before prior matching"},
        {"prior_center_y_t", "array-center y offset subtracted before prior matching"},
        {"prior_derot_elev", "derotation elevation used for prior-frame matching"},
        {"prior_slot_x_t", "matched prior slot x center in the prior frame"},
        {"prior_slot_y_t", "matched prior slot y center in the prior frame"},
        {"prior_slot_sx", "matched prior slot soft x sigma"},
        {"prior_slot_sy", "matched prior slot soft y sigma"},
        {"final_prior_slot_index", "nearest prior slot index for final detector position in the prior frame"},
        {"final_prior_d2", "nearest-slot Mahalanobis d^2 for final detector position in the soft-prior frame"},
        {"x_t_raw", "raw x position before reference subtraction/derotation"},
        {"y_t_raw", "raw y position before reference subtraction/derotation"},
        {"x_t", table_access.description("x_t", "detector x position")},
        {"y_t", table_access.description("y_t", "detector y position")},
        {"x_t_derot", "detector x position after derotation transform"},
        {"y_t_derot", "detector y position after derotation transform"},
        {"a_fwhm", table_access.description("a_fwhm", "fitted major-axis FWHM")},
        {"a_fwhm_err", table_access.description("a_fwhm_err", "fitted major-axis FWHM uncertainty")},
        {"b_fwhm", table_access.description("b_fwhm", "fitted minor-axis FWHM")},
        {"b_fwhm_err", table_access.description("b_fwhm_err", "fitted minor-axis FWHM uncertainty")},
        {"angle", table_access.description("angle", "fitted beam angle")},
        {"angle_err", table_access.description("angle_err", "fitted beam angle uncertainty")},
        {"flxscale", table_access.description("flxscale", "flux conversion factor")},
        {"sens", table_access.description("sens", "detector sensitivity")}
    };
}

inline void append_legends(YAML::Node &fit_qc_meta);

template <class BeammapState, class TableAccess>
YAML::Node make_metadata(const BeammapState &beammap,
                         const TableAccess &table_access,
                         const std::vector<std::string> &fit_qc_header) {
    YAML::Node fit_qc_meta;
    fit_qc_meta["obsnum"] = beammap.obsnum;
    fit_qc_meta["source"] = beammap.telescope.source_name;
    fit_qc_meta["creation_date"] = engine_utils::current_date_time();
    fit_qc_meta["date"] = beammap.date_obs.back();
    fit_qc_meta["map_grouping"] = beammap.map_grouping;
    const auto &iteration_config = beammap.typed_config.beammap.iteration;
    const auto &phase_config = beammap.typed_config.beammap.phase_strategy;
    fit_qc_meta["beammap_iter_max"] = iteration_config.max_iterations;
    fit_qc_meta["beammap_iter_tolerance"] = iteration_config.tolerance;
    fit_qc_meta["beammap_convergence_radius_arcsec"] =
        iteration_config.convergence_radius_arcsec;
    fit_qc_meta["beammap_phase_split_enabled"] =
        phase_config.enabled;
    fit_qc_meta["beammap_locator_iter"] = phase_config.locator_iter;
    fit_qc_meta["beammap_measurement_start_iter"] =
        phase_config.measurement_start_iter;
    const auto &reference_config = beammap.typed_config.beammap.reference;
    fit_qc_meta["reference_detector_subtracted"] =
        reference_config.subtract_reference_detector;
    fit_qc_meta["reference_det"] = beammap.beammap_reference_det_found;
    const auto &rfi_config = beammap.typed_config.beammap.rfi_mask;
    const auto &scan_band_config =
        beammap.typed_config.beammap.scan_band_mask;
    fit_qc_meta["rfi_mask_enabled"] = rfi_config.enabled;
    fit_qc_meta["rfi_mask_block_size_samples"] =
        rfi_config.block_size_samples;
    fit_qc_meta["rfi_mask_min_good_samples"] =
        rfi_config.min_good_samples;
    fit_qc_meta["rfi_mask_dilate_blocks"] =
        rfi_config.dilate_blocks;
    fit_qc_meta["rfi_mask_sigma_threshold"] =
        rfi_config.sigma_threshold;
    fit_qc_meta["rfi_mask_sigma_floor"] =
        rfi_config.sigma_floor;
    fit_qc_meta["rfi_mask_max_flagged_fraction"] =
        rfi_config.max_flagged_fraction;
    fit_qc_meta["detector_weighting_mode"] =
        std::string(citlali::config::to_string(
            beammap.typed_config.beammap.detector_weighting_mode));
    fit_qc_meta["beammap_fit_radius_fwhm"] =
        beammap.typed_config.beammap.fitting.fit_radius_fwhm;
    fit_qc_meta["rfi_mask_detectors_affected"] =
        static_cast<int>(
            (table_access.apt_or_zero("rfi_masked_scans").array() > 0.0)
                .count());
    fit_qc_meta["scan_band_mask_enabled"] =
        scan_band_config.enabled;
    fit_qc_meta["scan_band_mask_edge_rows"] =
        scan_band_config.edge_rows;
    fit_qc_meta["scan_band_mask_min_row_pixels"] =
        scan_band_config.min_row_pixels;
    fit_qc_meta["scan_band_mask_min_contiguous_rows"] =
        scan_band_config.min_contiguous_rows;
    fit_qc_meta["scan_band_mask_row_median_sigma_threshold"] =
        scan_band_config.row_median_sigma_threshold;
    fit_qc_meta["scan_band_mask_row_sigma_ratio_threshold"] =
        scan_band_config.row_sigma_ratio_threshold;
    fit_qc_meta["scan_band_mask_max_flagged_fraction"] =
        scan_band_config.max_flagged_fraction;
    fit_qc_meta["scan_band_mask_detectors_affected"] =
        static_cast<int>(
            (table_access.apt_or_zero("scan_band_masked_rows").array() > 0.0)
                .count());
    fit_qc_meta["scan_band_mask_detectors_rejected"] =
        static_cast<int>(
            (table_access.apt_or_zero("scan_band_mask_rejected").array() >
             0.0)
                .count());
    fit_qc_meta["fit_bound_any"] =
        static_cast<int>((beammap.fit_diag_bound_nhit.array() > 0).count());
    fit_qc_meta["beammap_priors_enabled"] =
        beammap.beammap_priors_enabled;
    fit_qc_meta["beammap_priors_filepath"] =
        beammap.beammap_priors_filepath;
    fit_qc_meta["beammap_priors_centered"] =
        beammap.beammap_soft_priors_are_centered;
    fit_qc_meta["beammap_priors_derotated"] =
        beammap.beammap_soft_priors_are_derotated;
    fit_qc_meta["beammap_priors_max_d2_iter0"] =
        beammap.beammap_priors_max_d2_iter0;
    fit_qc_meta["beammap_priors_max_d2_after_iter0"] =
        beammap.beammap_priors_max_d2_after_iter0;
    fit_qc_meta["beammap_priors_score_lambda_iter0"] =
        beammap.beammap_priors_score_lambda_iter0;
    fit_qc_meta["beammap_priors_score_lambda_after_iter0"] =
        beammap.beammap_priors_score_lambda_after_iter0;
    fit_qc_meta["beammap_priors_align_after_iter0"] =
        beammap.beammap_priors_align_after_iter0;
    fit_qc_meta["beammap_priors_alignment_scope"] =
        beammap.beammap_priors_alignment_scope;
    fit_qc_meta["beammap_priors_alignment_common_support"] =
        beammap.beammap_priors_alignment_common_support;
    fit_qc_meta["beammap_priors_alignment_common_support_quantile"] =
        beammap.beammap_priors_alignment_common_support_quantile;
    fit_qc_meta["beammap_priors_alignment_min_matches"] =
        beammap.beammap_priors_alignment_min_matches;
    fit_qc_meta["beammap_priors_alignment_max_d2"] =
        beammap.beammap_priors_alignment_max_d2;
    fit_qc_meta["beammap_priors_alignment_fit_rotation"] =
        beammap.beammap_priors_alignment_fit_rotation;
    fit_qc_meta["beammap_priors_alignment_max_rotation_deg"] =
        beammap.beammap_priors_alignment_max_rotation_deg;
    fit_qc_meta["beammap_priors_aligned_arrays"] =
        static_cast<int>(beammap.beammap_prior_array_alignment.size());

    auto fit_qc_units = units(table_access, beammap.omb.sig_unit);
    auto fit_qc_desc = descriptions(table_access);
    for (const auto &key: fit_qc_header) {
        fit_qc_meta[key].push_back("units: " + fit_qc_units[key]);
        fit_qc_meta[key].push_back(fit_qc_desc[key]);
    }
    append_legends(fit_qc_meta);
    return fit_qc_meta;
}

inline void append_legends(YAML::Node &fit_qc_meta) {
    fit_qc_meta["flag2"].push_back("Good=0");
    fit_qc_meta["flag2"].push_back("BadFit=1");
    fit_qc_meta["flag2"].push_back("AzFWHM=2");
    fit_qc_meta["flag2"].push_back("ElFWHM=4");
    fit_qc_meta["flag2"].push_back("Sig2Noise=8");
    fit_qc_meta["flag2"].push_back("Sens=16");
    fit_qc_meta["flag2"].push_back("Position=32");
    fit_qc_meta["flag2"].push_back("PriorDist=64");
    fit_qc_meta["flag2"].push_back("NetworkPos=128");
    fit_qc_meta["cal_amp_method"].push_back("0: Gaussian fit amplitude fallback");
    fit_qc_meta["cal_amp_method"].push_back("1: empirical array-template matched amplitude");
    fit_qc_meta["fit_bound_code"].push_back("bit 0: amp lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 1: amp upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 2: x lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 3: x upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 4: y lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 5: y upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 6: a lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 7: a upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 8: b lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 9: b upper");
    fit_qc_meta["fit_bound_code"].push_back("bit 10: angle lower");
    fit_qc_meta["fit_bound_code"].push_back("bit 11: angle upper");
    fit_qc_meta["prior_init_mode"].push_back("-1: skipped before fitting on last attempted iteration");
    fit_qc_meta["prior_init_mode"].push_back("0: blind seed");
    fit_qc_meta["prior_init_mode"].push_back("1: previous-iteration seed");
    fit_qc_meta["prior_init_mode"].push_back("2: prior-guided seed");
    fit_qc_meta["prior_no_candidate_reason"].push_back("0: none");
    fit_qc_meta["prior_no_candidate_reason"].push_back("1: no slot group for (array,nw)");
    fit_qc_meta["prior_no_candidate_reason"].push_back("2: no valid weighted pixels");
    fit_qc_meta["prior_no_candidate_reason"].push_back("3: invalid robust sigma estimate");
    fit_qc_meta["prior_no_candidate_reason"].push_back("4: no candidates above min_snr");
    fit_qc_meta["prior_no_candidate_reason"].push_back("5: all retained candidates failed max_d2 gate");
    fit_qc_meta["scan_band_masked_edge"].push_back("0: none");
    fit_qc_meta["scan_band_masked_edge"].push_back("1: top");
    fit_qc_meta["scan_band_masked_edge"].push_back("2: bottom");
    fit_qc_meta["scan_band_masked_edge"].push_back("3: both");
}

} // namespace beammap_fit_qc_schema
