#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/mapmaking_config_serialization.h>
#include <citlali/core/pipeline/science_map_provenance_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *mapmaking_provenance_schema_version =
    "citlali-mapmaking-provenance-v3";
inline constexpr const char *mapmaking_provenance_filename =
    "mapmaking_provenance.yaml";

inline YAML::Node mapmaking_observation_provenance_node(
    const MapmakingObservationState &observation) {
    auto node = mapmaking_observation_state_node(observation);
    const bool science_state_available =
        observation.bundle_identity.has_value() &&
        observation.realized_maps.size() == observation.map_count &&
        observation.bundle_identity->ordered_slots.size() ==
            observation.map_count;
    node["science_state"]["available"] = science_state_available;
    node["science_state"]["bundle_identity"] =
        science_map_optional_bundle_identity_node(
            observation.bundle_identity,
            observation.science_state_absence_reason);
    node["science_state"]["realized_maps"] =
        science_map_realized_maps_node(observation.realized_maps);
    if (!science_state_available) {
        node["science_state"]["absence_reason"] =
            observation.science_state_absence_reason;
    }
    return node;
}

inline YAML::Node mapmaking_observations_provenance_node(
    const std::vector<MapmakingObservationState> &observations) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto &observation : observations) {
        node.push_back(mapmaking_observation_provenance_node(observation));
    }
    return node;
}

inline YAML::Node mapmaking_provenance_node(
    const MapmakingExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = mapmaking_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["requested"] = mapmaking_config_node(plan.requested);
    root["effective"]["config"] = mapmaking_config_node(plan.effective);
    root["effective"]["resolution"] =
        mapmaking_effective_resolution_node(plan.effective_resolution);
    root["science_contract"] = science_map_policy_contract_node();
    root["science_contract"]["cuts"]["requested"] =
        science_map_exact_double_node(plan.requested.coverage_cut);
    root["science_contract"]["cuts"]["effective"] =
        science_map_exact_double_node(plan.effective.coverage_cut);
    root["observations"] =
        mapmaking_observations_provenance_node(plan.observations);
    root["coadd"] = mapmaking_coadd_state_node(plan.coadd);
    root["realized"] = mapmaking_realized_state_node(plan.realized);
    return root;
}

inline std::filesystem::path mapmaking_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / mapmaking_provenance_filename;
}

inline void write_mapmaking_provenance_file(
    const std::filesystem::path &reduction_dir,
    const MapmakingExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error(
            "cannot write uninitialized mapmaking provenance");
    }
    write_yaml_file_atomic(
        mapmaking_provenance_path(reduction_dir),
        mapmaking_provenance_node(plan));
}

}  // namespace citlali::pipeline
