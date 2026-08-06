#pragma once

#include <citlali/core/config/beammap_config.h>

#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

namespace citlali::pipeline {

template <class Value>
YAML::Node beammap_sequence_node(const std::vector<Value> &values) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto &value : values) {
        node.push_back(value);
    }
    return node;
}

inline YAML::Node beammap_config_node(
    const citlali::config::BeammapConfig &config) {
    YAML::Node node;
    node["direction_mode"] = std::string{
        citlali::config::to_string(config.direction_mode)};
    node["iter_max"] = config.iteration.max_iterations;
    node["iter_tolerance"] = config.iteration.tolerance;
    node["convergence_radius_arcsec"] =
        config.iteration.convergence_radius_arcsec;
    node["phase_strategy"]["enabled"] = config.phase_strategy.enabled;
    node["phase_strategy"]["locator_iter"] =
        config.phase_strategy.locator_iter;
    node["phase_strategy"]["measurement_start_iter"] =
        config.phase_strategy.measurement_start_iter;
    node["reference_det"] = config.reference.reference_detector;
    node["subtract_reference_det"] =
        config.reference.subtract_reference_detector;
    node["derotate"] = config.reference.derotate;
    node["rfi_mask"]["enabled"] = config.rfi_mask.enabled;
    node["rfi_mask"]["block_size_samples"] =
        config.rfi_mask.block_size_samples;
    node["rfi_mask"]["min_good_samples"] =
        config.rfi_mask.min_good_samples;
    node["rfi_mask"]["dilate_blocks"] = config.rfi_mask.dilate_blocks;
    node["rfi_mask"]["sigma_threshold"] =
        config.rfi_mask.sigma_threshold;
    node["rfi_mask"]["sigma_floor"] = config.rfi_mask.sigma_floor;
    node["rfi_mask"]["max_flagged_fraction"] =
        config.rfi_mask.max_flagged_fraction;
    node["detector_weighting"]["mode"] = std::string{
        citlali::config::to_string(config.detector_weighting_mode)};
    node["fitting"]["fit_radius_fwhm"] = config.fitting.fit_radius_fwhm;
    node["scan_band_mask"]["enabled"] = config.scan_band_mask.enabled;
    node["scan_band_mask"]["edge_rows"] = config.scan_band_mask.edge_rows;
    node["scan_band_mask"]["min_row_pixels"] =
        config.scan_band_mask.min_row_pixels;
    node["scan_band_mask"]["min_contiguous_rows"] =
        config.scan_band_mask.min_contiguous_rows;
    node["scan_band_mask"]["row_median_sigma_threshold"] =
        config.scan_band_mask.row_median_sigma_threshold;
    node["scan_band_mask"]["row_sigma_ratio_threshold"] =
        config.scan_band_mask.row_sigma_ratio_threshold;
    node["scan_band_mask"]["max_flagged_fraction"] =
        config.scan_band_mask.max_flagged_fraction;
    node["split_fits_by_flag"]["enabled"] =
        config.split_fits_by_flag.enabled;
    node["split_fits_by_flag"]["flag_values"] =
        beammap_sequence_node(config.split_fits_by_flag.flag_values);
    node["priors"]["enabled"] = config.priors.enabled;
    node["priors"]["filepath"] = config.priors.filepath;
    node["priors"]["candidate_top_n"] = config.priors.candidate_top_n;
    node["priors"]["min_snr"] = config.priors.min_snr;
    node["priors"]["max_d2"] = config.priors.max_d2;
    node["priors"]["max_d2_iter0"] = config.priors.max_d2_iter0;
    node["priors"]["max_d2_after_iter0"] =
        config.priors.max_d2_after_iter0;
    node["priors"]["score_lambda"] = config.priors.score_lambda;
    node["priors"]["score_lambda_iter0"] =
        config.priors.score_lambda_iter0;
    node["priors"]["score_lambda_after_iter0"] =
        config.priors.score_lambda_after_iter0;
    node["priors"]["fallback_blind"] = config.priors.fallback_blind;
    node["priors"]["align_after_iter0"] = config.priors.align_after_iter0;
    node["priors"]["alignment_scope"] = std::string{
        citlali::config::to_string(config.priors.alignment_scope)};
    node["priors"]["alignment_common_support"] = std::string{
        citlali::config::to_string(config.priors.alignment_common_support)};
    node["priors"]["alignment_common_support_quantile"] =
        config.priors.alignment_common_support_quantile;
    node["priors"]["alignment_min_matches"] =
        config.priors.alignment_min_matches;
    node["priors"]["alignment_max_d2"] = config.priors.alignment_max_d2;
    node["priors"]["alignment_fit_rotation"] =
        config.priors.alignment_fit_rotation;
    node["priors"]["alignment_max_rotation_deg"] =
        config.priors.alignment_max_rotation_deg;
    node["detector_tod_output"]["enabled"] =
        config.detector_tod_output.enabled;
    node["detector_tod_output"]["subdir_name"] =
        config.detector_tod_output.subdir_name;
    node["detector_tod_output"]["n_uniform"] =
        config.detector_tod_output.n_uniform;
    node["detector_tod_output"]["n_source_dense"] =
        config.detector_tod_output.n_source_dense;
    node["flagging"]["array_lower_fwhm_arcsec"] =
        beammap_sequence_node(config.flagging.array_lower_fwhm_arcsec);
    node["flagging"]["array_upper_fwhm_arcsec"] =
        beammap_sequence_node(config.flagging.array_upper_fwhm_arcsec);
    node["flagging"]["array_lower_sig2noise"] =
        beammap_sequence_node(config.flagging.array_lower_sig2noise);
    node["flagging"]["array_upper_sig2noise"] =
        beammap_sequence_node(config.flagging.array_upper_sig2noise);
    node["flagging"]["array_max_dist_arcsec"] =
        beammap_sequence_node(config.flagging.array_max_dist_arcsec);
    node["flagging"]["array_network_robust_z"] =
        beammap_sequence_node(config.flagging.array_network_robust_z);
    node["flagging"]["max_prior_d2"] = config.flagging.max_prior_d2;
    node["flagging"]["sens_factors"] =
        beammap_sequence_node(config.flagging.sens_factors);
    node["sens_psd_limits_Hz"] =
        beammap_sequence_node(config.flagging.sens_psd_limits_hz);
    return node;
}

}  // namespace citlali::pipeline
