#pragma once

#include <citlali/core/pipeline/pointing_execution_plan.h>

#include <yaml-cpp/yaml.h>

namespace citlali::pipeline {

inline YAML::Node pointing_config_node(
    const citlali::config::PointingConfig &config) {
    YAML::Node node;
    node["source_strategy"] =
        std::string(citlali::config::to_string(config.source_strategy));
    node["fit_gaussian"] = config.fit_gaussian;
    node["fruitloops_center_mode"] =
        std::string(citlali::config::to_string(
            config.fruitloops_center_mode));
    node["header_max_radius_arcsec"] =
        config.header_max_radius_arcsec;
    node["header_require_coverage"] =
        config.header_require_coverage;
    return node;
}

inline YAML::Node pointing_request_presence_node(
    const PointingRequestPresence &presence) {
    YAML::Node node;
    node["source_strategy"] = presence.source_strategy;
    node["fit_gaussian"] = presence.fit_gaussian;
    node["fruitloops_center_mode"] =
        presence.fruitloops_center_mode;
    node["header_max_radius_arcsec"] =
        presence.header_max_radius_arcsec;
    node["header_require_coverage"] =
        presence.header_require_coverage;
    return node;
}

inline YAML::Node pointing_effective_resolution_node(
    const PointingEffectiveResolutionRecord &resolution) {
    YAML::Node node;
    node["mapmaking_enabled"] = resolution.mapmaking_enabled;
    node["map_filter_enabled"] = resolution.map_filter_enabled;
    node["coadd_enabled"] = resolution.coadd_enabled;
    node["fit_output_path_available"] =
        resolution.fit_output_path_available;
    node["explicit_request"] =
        pointing_request_presence_node(resolution.explicit_request);
    node["fit_disabled_by_mapmaking"] =
        resolution.fit_disabled_by_mapmaking;
    node["fit_disabled_by_output_policy"] =
        resolution.fit_disabled_by_output_policy;
    node["default_header_max_radius_arcsec"] =
        resolution.default_header_max_radius_arcsec;
    node["header_max_radius_defaulted"] =
        resolution.header_max_radius_defaulted;
    return node;
}

inline YAML::Node pointing_observation_node(
    const PointingObservationState &observation) {
    YAML::Node node;
    node["observation_index"] = observation.observation_index;
    node["obsnum"] = observation.obsnum;
    node["map_count"] = observation.map_count;
    node["raw_fit_attempt_count"] = observation.raw_fit.attempt_count;
    node["raw_valid_fit_count"] = observation.raw_fit.valid_count;
    node["raw_fit_results_recorded"] = observation.raw_fit.recorded;
    node["filtered_fit_attempt_count"] =
        observation.filtered_fit.attempt_count;
    node["filtered_valid_fit_count"] =
        observation.filtered_fit.valid_count;
    node["filtered_fit_results_recorded"] =
        observation.filtered_fit.recorded;
    node["outputs_completed"] = observation.outputs_completed;
    return node;
}

inline YAML::Node pointing_realized_state_node(
    const PointingRealizedState &realized) {
    YAML::Node node;
    node["reduction_completed"] = realized.reduction_completed;
    node["pointing_executed"] = realized.pointing_executed;
    node["completed_observation_count"] =
        realized.completed_observation_count;
    node["scientific_map_count"] = realized.scientific_map_count;
    node["raw_fit_attempt_count"] = realized.raw_fit_attempt_count;
    node["raw_valid_fit_count"] = realized.raw_valid_fit_count;
    node["filtered_fit_attempt_count"] =
        realized.filtered_fit_attempt_count;
    node["filtered_valid_fit_count"] =
        realized.filtered_valid_fit_count;
    node["outputs_completed"] = realized.outputs_completed;
    return node;
}

inline YAML::Node pointing_observations_node(
    const std::vector<PointingObservationState> &observations) {
    YAML::Node node{YAML::NodeType::Sequence};
    for (const auto &observation : observations) {
        node.push_back(pointing_observation_node(observation));
    }
    return node;
}

}  // namespace citlali::pipeline
