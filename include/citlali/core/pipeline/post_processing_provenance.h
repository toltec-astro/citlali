#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/post_processing_config_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *post_processing_provenance_schema_version =
    "citlali-post-processing-provenance-v1";
inline constexpr const char *post_processing_provenance_filename =
    "post_processing_provenance.yaml";

inline YAML::Node post_processing_provenance_node(
    const PostProcessingExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = post_processing_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["requested"] = post_processing_config_node(plan.requested);
    root["effective"]["values"] =
        post_processing_config_node(plan.effective);
    root["effective"]["resolution"] =
        post_processing_effective_resolution_node(
            plan.effective_resolution);
    root["realized"] =
        post_processing_realized_state_node(plan.realized);
    return root;
}

inline std::filesystem::path post_processing_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / post_processing_provenance_filename;
}

inline void write_post_processing_provenance_file(
    const std::filesystem::path &reduction_dir,
    const PostProcessingExecutionPlan &plan) {
    if (!plan.initialized || !plan.realized.reduction_completed ||
        !plan.realized.outputs_completed) {
        throw std::logic_error(
            "cannot write incomplete post-processing provenance");
    }
    write_yaml_file_atomic(
        post_processing_provenance_path(reduction_dir),
        post_processing_provenance_node(plan));
}

}  // namespace citlali::pipeline
