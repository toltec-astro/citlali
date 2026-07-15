#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/polarimetry_config_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *polarimetry_provenance_schema_version =
    "citlali-polarimetry-provenance-v1";
inline constexpr const char *polarimetry_provenance_filename =
    "polarimetry_provenance.yaml";

inline YAML::Node polarimetry_provenance_node(
    const PolarimetryExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = polarimetry_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["capability"]["status"] =
        std::string{polarimetry_capability_status};
    root["capability"]["enabled_supported"] =
        enabled_polarimetry_available;
    root["capability"]["reason"] =
        std::string{polarimetry_capability_reason};
    root["capability"]["exit_condition"] =
        std::string{polarimetry_capability_exit_condition};
    root["requested"] = polarimetry_config_node(plan.requested);
    root["effective"]["config"] =
        polarimetry_config_node(plan.effective);
    root["effective"]["capability_resolution"] =
        polarimetry_capability_resolution_node(plan.capability);
    root["realized"] =
        polarimetry_realized_state_node(plan.realized);
    return root;
}

inline std::filesystem::path polarimetry_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / polarimetry_provenance_filename;
}

inline void write_polarimetry_provenance_file(
    const std::filesystem::path &reduction_dir,
    const PolarimetryExecutionPlan &plan) {
    if (!plan.initialized || !plan.realized.reduction_completed) {
        throw std::logic_error(
            "cannot write incomplete polarimetry provenance");
    }
    write_yaml_file_atomic(
        polarimetry_provenance_path(reduction_dir),
        polarimetry_provenance_node(plan));
}

}  // namespace citlali::pipeline
