#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/mapmaking_config_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *mapmaking_provenance_schema_version =
    "citlali-mapmaking-provenance-v2";
inline constexpr const char *mapmaking_provenance_filename =
    "mapmaking_provenance.yaml";

inline YAML::Node mapmaking_provenance_node(
    const MapmakingExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = mapmaking_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["requested"] = mapmaking_config_node(plan.requested);
    root["effective"]["config"] = mapmaking_config_node(plan.effective);
    root["effective"]["resolution"] =
        mapmaking_effective_resolution_node(plan.effective_resolution);
    root["observations"] =
        mapmaking_observations_node(plan.observations);
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
