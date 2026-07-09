#pragma once

// Beammap detector fit-QC metadata helpers.

#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

#include <citlali/core/engine/detail/beammap_fit_qc_units_descriptions.h>
#include <citlali/core/pipeline/map_grouping_policy.h>

namespace beammap_fit_qc_schema {

inline void append_legends(YAML::Node &fit_qc_meta);

template <class BeammapState, class TableAccess>
YAML::Node make_metadata(const BeammapState &beammap,
                         const TableAccess &table_access,
                         const std::vector<std::string> &fit_qc_header) {
    YAML::Node fit_qc_meta;
    fit_qc_meta["obsnum"] = beammap.observation_identity.obsnum;
    fit_qc_meta["source"] = beammap.telescope.source_name;
    fit_qc_meta["creation_date"] = engine_utils::current_date_time();
    fit_qc_meta["date"] =
        citlali::pipeline::latest_observation_date(beammap.observation_dates);
    fit_qc_meta["map_grouping"] =
        citlali::pipeline::active_map_grouping_name(beammap);
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
    const auto &priors_config = beammap.typed_config.beammap.priors;
    fit_qc_meta["beammap_priors_enabled"] =
        priors_config.enabled;
    fit_qc_meta["beammap_priors_filepath"] =
        priors_config.filepath;
    fit_qc_meta["beammap_priors_centered"] =
        beammap.beammap_soft_priors_are_centered;
    fit_qc_meta["beammap_priors_derotated"] =
        beammap.beammap_soft_priors_are_derotated;
    fit_qc_meta["beammap_priors_max_d2_iter0"] =
        priors_config.max_d2_iter0;
    fit_qc_meta["beammap_priors_max_d2_after_iter0"] =
        priors_config.max_d2_after_iter0;
    fit_qc_meta["beammap_priors_score_lambda_iter0"] =
        priors_config.score_lambda_iter0;
    fit_qc_meta["beammap_priors_score_lambda_after_iter0"] =
        priors_config.score_lambda_after_iter0;
    fit_qc_meta["beammap_priors_align_after_iter0"] =
        priors_config.align_after_iter0;
    fit_qc_meta["beammap_priors_alignment_scope"] =
        std::string(citlali::config::to_string(priors_config.alignment_scope));
    fit_qc_meta["beammap_priors_alignment_common_support"] =
        std::string(citlali::config::to_string(
            priors_config.alignment_common_support));
    fit_qc_meta["beammap_priors_alignment_common_support_quantile"] =
        priors_config.alignment_common_support_quantile;
    fit_qc_meta["beammap_priors_alignment_min_matches"] =
        priors_config.alignment_min_matches;
    fit_qc_meta["beammap_priors_alignment_max_d2"] =
        priors_config.alignment_max_d2;
    fit_qc_meta["beammap_priors_alignment_fit_rotation"] =
        priors_config.alignment_fit_rotation;
    fit_qc_meta["beammap_priors_alignment_max_rotation_deg"] =
        priors_config.alignment_max_rotation_deg;
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
