#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/noise_config_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *noise_provenance_schema_version =
    "citlali-noise-products-provenance-v1";
inline constexpr const char *noise_provenance_filename =
    "noise_products_provenance.yaml";

inline YAML::Node noise_provenance_node(
    const NoiseExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = noise_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["requested"] = noise_config_node(plan.requested);
    root["effective"]["config"] = noise_config_node(plan.effective);
    root["effective"]["resolution"] =
        noise_effective_resolution_node(plan.effective_resolution);
    root["realized"] = noise_realized_state_node(plan.realized);
    return root;
}

inline std::filesystem::path noise_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / noise_provenance_filename;
}

inline void write_noise_provenance_file(
    const std::filesystem::path &reduction_dir,
    const NoiseExecutionPlan &plan) {
    if (!plan.initialized || !plan.realized.reduction_completed) {
        throw std::logic_error(
            "cannot write incomplete noise-products provenance");
    }
    write_yaml_file_atomic(
        noise_provenance_path(reduction_dir),
        noise_provenance_node(plan));
}

}  // namespace citlali::pipeline
