#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/pointing_config_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *pointing_provenance_schema_version =
    "citlali-pointing-provenance-v1";
inline constexpr const char *pointing_provenance_filename =
    "pointing_provenance.yaml";

inline YAML::Node pointing_provenance_node(
    const PointingExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = pointing_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["requested"] = pointing_config_node(plan.requested);
    root["effective"]["config"] = pointing_config_node(plan.effective);
    root["effective"]["resolution"] =
        pointing_effective_resolution_node(plan.effective_resolution);
    root["observations"] = pointing_observations_node(plan.observations);
    root["realized"] = pointing_realized_state_node(plan.realized);
    return root;
}

inline std::filesystem::path pointing_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / pointing_provenance_filename;
}

inline void write_pointing_provenance_file(
    const std::filesystem::path &reduction_dir,
    const PointingExecutionPlan &plan) {
    if (!plan.initialized || !plan.realized.reduction_completed) {
        throw std::logic_error(
            "cannot write incomplete pointing provenance");
    }
    write_yaml_file_atomic(
        pointing_provenance_path(reduction_dir),
        pointing_provenance_node(plan));
}

}  // namespace citlali::pipeline
