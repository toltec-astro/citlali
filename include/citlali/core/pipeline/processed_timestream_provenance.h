#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/processed_timestream_config_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *processed_timestream_provenance_schema_version =
    "citlali-processed-timestream-provenance-v2";
inline constexpr const char *processed_timestream_provenance_filename =
    "processed_timestream_provenance.yaml";

inline YAML::Node processed_timestream_provenance_node(
    const ProcessedTimestreamExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] =
        processed_timestream_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["requested"] =
        processed_timestream_config_snapshot_node(plan.requested);
    root["effective"]["config"] =
        processed_timestream_config_snapshot_node(plan.effective);
    root["effective"]["resolutions"] =
        processed_timestream_effective_resolutions_node(
            plan.effective_resolutions);
    root["realized"] =
        processed_timestream_realized_state_node(plan.realized);
    return root;
}

inline std::filesystem::path processed_timestream_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / processed_timestream_provenance_filename;
}

inline void write_processed_timestream_provenance_file(
    const std::filesystem::path &reduction_dir,
    const ProcessedTimestreamExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error(
            "cannot write uninitialized processed timestream provenance");
    }
    const auto output_path =
        processed_timestream_provenance_path(reduction_dir);
    write_yaml_file_atomic(output_path,
                           processed_timestream_provenance_node(plan));
}

}  // namespace citlali::pipeline
