#pragma once

#include <citlali/core/pipeline/astrometry_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

namespace citlali::pipeline {

inline YAML::Node astrometry_double_sequence(
    const std::vector<double> &values) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto value : values) {
        node.push_back(value);
    }
    return node;
}

inline YAML::Node astrometry_config_node(
    const citlali::config::AstrometryConfig &config) {
    YAML::Node node;
    const auto &offsets = config.pointing_offsets;
    node["pointing_offsets"]["enabled"] = offsets.enabled;
    node["pointing_offsets"]["az_arcsec"] =
        astrometry_double_sequence(offsets.az_arcsec);
    node["pointing_offsets"]["alt_arcsec"] =
        astrometry_double_sequence(offsets.alt_arcsec);
    node["pointing_offsets"]["modified_julian_date"] =
        astrometry_double_sequence(offsets.modified_julian_date);
    return node;
}

inline YAML::Node astrometry_effective_resolution_node(
    const AstrometryEffectiveResolution &resolution) {
    YAML::Node node;
    node["application_mode"] =
        std::string{to_string(resolution.application_mode)};
    node["explicit_mjd_support"] = resolution.explicit_mjd_support;
    return node;
}

inline YAML::Node astrometry_realized_state_node(
    const AstrometryRealizedState &realized) {
    YAML::Node node;
    node["installation_count"] = realized.installation_count;
    node["application_count"] = realized.application_count;
    node["telescope_sample_count"] = realized.telescope_sample_count;
    return node;
}

inline YAML::Node astrometry_observation_plan_node(
    const AstrometryObservationPlan &observation) {
    YAML::Node node;
    node["observation_index"] = observation.observation_index;
    node["obsnum"] = observation.obsnum;
    node["requested"] = astrometry_config_node(observation.requested);
    node["effective"]["config"] =
        astrometry_config_node(observation.effective);
    node["effective"]["resolution"] =
        astrometry_effective_resolution_node(observation.resolution);
    node["realized"] = astrometry_realized_state_node(observation.realized);
    return node;
}

inline YAML::Node astrometry_observation_plans_node(
    const std::vector<AstrometryObservationPlan> &observations) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto &observation : observations) {
        node.push_back(astrometry_observation_plan_node(observation));
    }
    return node;
}

}  // namespace citlali::pipeline
