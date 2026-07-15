#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/polarimetry_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <string>

namespace citlali::pipeline {

inline YAML::Node polarimetry_config_node(
    const citlali::config::TimestreamPolarimetryConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["grouping"] =
        std::string{citlali::config::to_string(config.grouping)};
    node["ignore_hwpr"] =
        std::string{citlali::config::to_string(config.hwpr_policy)};
    return node;
}

inline YAML::Node polarimetry_capability_resolution_node(
    const PolarimetryCapabilityResolution &resolution) {
    YAML::Node node;
    node["enabled_capability_available"] =
        resolution.enabled_capability_available;
    node["requested_enabled"] = resolution.requested_enabled;
    node["request_accepted"] = resolution.request_accepted;
    node["disabled_by_capability"] =
        resolution.disabled_by_capability;
    return node;
}

inline YAML::Node polarimetry_realized_state_node(
    const PolarimetryRealizedState &state) {
    YAML::Node node;
    node["reduction_completed"] = state.reduction_completed;
    node["polarimetry_executed"] = state.polarimetry_executed;
    node["hwpr_loaded"] = state.hwpr_loaded;
    return node;
}

}  // namespace citlali::pipeline
