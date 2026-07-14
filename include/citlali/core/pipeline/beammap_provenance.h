#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/beammap_config_serialization.h>
#include <citlali/core/pipeline/beammap_provenance_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *beammap_provenance_schema_version =
    "citlali-beammap-provenance-v1";
inline constexpr const char *beammap_provenance_filename =
    "beammap_provenance.yaml";

inline YAML::Node beammap_provenance_node(
    const BeammapExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = beammap_provenance_schema_version;
    root["initialized"] = plan.initialized();
    root["requested"] = beammap_config_node(plan.requested());
    root["effective"]["config"] =
        beammap_config_node(plan.effective());
    root["effective"]["resolution"] =
        beammap_effective_resolution_node(plan.resolution());
    root["observations"] =
        beammap_observation_states_node(plan.observations());
    root["realized"] = beammap_realized_state_node(plan.realized());
    return root;
}

inline std::filesystem::path beammap_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / beammap_provenance_filename;
}

inline void write_beammap_provenance_file(
    const std::filesystem::path &reduction_dir,
    const BeammapExecutionPlan &plan) {
    if (!plan.initialized() || !plan.realized().reduction_completed ||
        !plan.realized().outputs_completed) {
        throw std::logic_error(
            "cannot write incomplete beammap provenance");
    }
    write_yaml_file_atomic(
        beammap_provenance_path(reduction_dir),
        beammap_provenance_node(plan));
}

}  // namespace citlali::pipeline
