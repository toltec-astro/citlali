#pragma once

#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/pipeline/post_processing_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <string>

namespace citlali::pipeline {

inline YAML::Node post_processing_edge_guard_config_node(
    const citlali::config::MapFilterEdgeGuardConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["weight_threshold_mode"] = config.weight_threshold_mode;
    node["hits_threshold_mode"] = config.hits_threshold_mode;
    node["hits_core_fraction"] = config.hits_core_fraction;
    node["guard_radius_fwhm"] = config.guard_radius_fwhm;
    node["fill_mode"] = config.fill_mode;
    node["taper_mode"] =
        std::string(citlali::config::to_string(config.taper_mode));
    node["taper_min_fraction"] = config.taper_min_fraction;
    return node;
}

inline YAML::Node post_processing_map_filter_config_node(
    const citlali::config::MapFilterConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["type"] = std::string(citlali::config::to_string(config.type));
    node["template_type"] =
        std::string(citlali::config::to_string(config.template_type));
    node["kernel_template_tail_mode"] = std::string(
        citlali::config::to_string(config.kernel_template_tail_mode));
    node["lowpass_only"] = config.lowpass_only;
    node["normalize_errors"] = config.normalize_errors;
    node["edge_guard"] =
        post_processing_edge_guard_config_node(config.edge_guard);
    node["denom_rel_tol"] = config.denom_rel_tol;
    node["tail_frac_tol"] = config.tail_frac_tol;
    node["max_loops"] = config.max_loops;
    node["denom_check_iters"] = config.denom_check_iters;
    node["max_denom_iters"] = config.max_denom_iters;
    for (const auto &[array_name, fwhm_arcsec] :
         config.template_fwhm_arcsec) {
        node["template_fwhm_arcsec"][array_name] = fwhm_arcsec;
    }
    return node;
}

inline YAML::Node post_processing_source_finding_config_node(
    const citlali::config::SourceFindingConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["source_sigma"] = config.source_sigma;
    node["source_window_arcsec"] = config.source_window_arcsec;
    node["mode"] = config.mode;
    return node;
}

inline YAML::Node post_processing_source_fitting_config_node(
    const citlali::config::SourceFittingConfig &config) {
    YAML::Node node;
    node["active"] = config.active;
    node["model"] = std::string(citlali::config::to_string(config.model));
    node["bounding_box_arcsec"] = config.bounding_box_arcsec;
    node["fitting_radius_arcsec"] = config.fitting_radius_arcsec;
    node["fit_rotation_angle"] = config.fit_rotation_angle;
    for (const auto value : config.amp_limit_factors) {
        node["amp_limit_factors"].push_back(value);
    }
    for (const auto value : config.fwhm_limit_factors) {
        node["fwhm_limit_factors"].push_back(value);
    }
    return node;
}

inline YAML::Node post_processing_config_node(
    const citlali::config::PostProcessingConfig &config) {
    YAML::Node node;
    node["map_filtering"] =
        post_processing_map_filter_config_node(config.map_filtering);
    node["map_histogram_n_bins"] = config.map_histogram_n_bins;
    node["source_finding"] =
        post_processing_source_finding_config_node(config.source_finding);
    node["source_fitting"] =
        post_processing_source_fitting_config_node(config.source_fitting);
    return node;
}

inline YAML::Node post_processing_effective_resolution_node(
    const PostProcessingEffectiveResolutionRecord &resolution) {
    YAML::Node node;
    node["reduction_type"] =
        std::string(citlali::config::to_string(resolution.reduction_type));
    node["mapmaking_enabled"] = resolution.mapmaking_enabled;
    node["coadd_enabled"] = resolution.coadd_enabled;
    node["map_filtering_requested"] = resolution.map_filtering_requested;
    node["map_filtering_effective"] = resolution.map_filtering_effective;
    node["map_filtering_disabled_by_mapmaking"] =
        resolution.map_filtering_disabled_by_mapmaking;
    node["source_finding_requested"] = resolution.source_finding_requested;
    node["source_finding_effective"] = resolution.source_finding_effective;
    node["source_finding_disabled_by_mapmaking"] =
        resolution.source_finding_disabled_by_mapmaking;
    node["source_fitting_required_by_reduction"] =
        resolution.source_fitting_required_by_reduction;
    node["source_fitting_required_by_map_filtering"] =
        resolution.source_fitting_required_by_map_filtering;
    node["source_fitting_required_by_source_finding"] =
        resolution.source_fitting_required_by_source_finding;
    node["source_fitting_effective"] =
        resolution.source_fitting_effective;
    node["source_fitting_disabled_by_mapmaking"] =
        resolution.source_fitting_disabled_by_mapmaking;
    return node;
}

inline YAML::Node post_processing_fit_cardinality_node(
    const PostProcessingFitCardinality &cardinality) {
    YAML::Node node;
    node["context_count"] = cardinality.context_count;
    node["attempt_count"] = cardinality.attempt_count;
    node["valid_count"] = cardinality.valid_count;
    return node;
}

inline YAML::Node post_processing_map_context_realized_node(
    const PostProcessingMapContextRealizedState &state) {
    YAML::Node node;
    node["filter_context_count"] = state.filter_context_count;
    node["filtered_map_count"] = state.filtered_map_count;
    node["source_finding_context_count"] =
        state.source_finding_context_count;
    node["detected_source_count"] = state.detected_source_count;
    node["source_table_write_count"] = state.source_table_write_count;
    node["source_table_row_count"] = state.source_table_row_count;
    node["catalog_fits"] =
        post_processing_fit_cardinality_node(state.catalog_fits);
    return node;
}

inline YAML::Node post_processing_realized_state_node(
    const PostProcessingRealizedState &realized) {
    YAML::Node node;
    node["reduction_completed"] = realized.reduction_completed;
    node["observation"] =
        post_processing_map_context_realized_node(realized.observation);
    node["coadd"] =
        post_processing_map_context_realized_node(realized.coadd);
    node["pointing_fits"]["raw"] =
        post_processing_fit_cardinality_node(realized.pointing_raw_fits);
    node["pointing_fits"]["filtered"] =
        post_processing_fit_cardinality_node(
            realized.pointing_filtered_fits);
    node["beammap_fits"] =
        post_processing_fit_cardinality_node(realized.beammap_fits);
    node["outputs_completed"] = realized.outputs_completed;
    return node;
}

}  // namespace citlali::pipeline
