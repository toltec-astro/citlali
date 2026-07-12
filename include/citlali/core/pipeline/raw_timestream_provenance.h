#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/raw_timestream_config_serialization.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *raw_timestream_provenance_schema_version =
    "citlali-raw-timestream-provenance-v1";
inline constexpr const char *raw_timestream_provenance_filename =
    "raw_timestream_provenance.yaml";

template <class Value>
YAML::Node raw_optional_scalar_node(const std::optional<Value> &value) {
    YAML::Node node;
    node["available"] = value.has_value();
    if (value) {
        node["value"] = *value;
    }
    return node;
}

inline const char *raw_downsample_request_kind_name(
    RawDownsampleRequestKind kind) {
    switch (kind) {
        case RawDownsampleRequestKind::disabled:
            return "disabled";
        case RawDownsampleRequestKind::explicit_factor:
            return "explicit_factor";
        case RawDownsampleRequestKind::target_frequency:
            return "target_frequency";
    }
    return "unknown";
}

inline YAML::Node raw_timestream_effective_resolutions_node(
    const RawTimestreamEffectiveResolutions &resolutions) {
    YAML::Node node;
    const auto &filtering = resolutions.filtering;
    auto filtering_node = node["filtering"];
    filtering_node["fir_requested"] = filtering.fir_requested;
    filtering_node["fixed_notch_requested"] =
        filtering.fixed_notch_requested;
    filtering_node["fixed_notch_effective"] =
        filtering.fixed_notch_effective;
    filtering_node["iir_highpass_requested"] =
        filtering.iir_highpass_requested;
    filtering_node["edge_guard_requested"] =
        filtering.edge_guard_requested;
    filtering_node["downsample_requested"] =
        filtering.downsample_requested;
    filtering_node["downsample_filter_dependency_satisfied"] =
        filtering.downsample_filter_dependency_satisfied;

    const auto &downsampling = resolutions.downsampling;
    auto downsampling_node = node["downsampling"];
    downsampling_node["kind"] =
        raw_downsample_request_kind_name(downsampling.kind);
    downsampling_node["requested_factor"] =
        downsampling.requested_factor;
    downsampling_node["requested_frequency_hz"] =
        downsampling.requested_frequency_hz;

    const auto &source = resolutions.source_protection;
    auto source_node = node["source_protection"];
    source_node["despike_requested"] = source.despike_requested;
    source_node["source_protection_requested"] =
        source.source_protection_requested;

    const auto &corrections = resolutions.corrections;
    auto corrections_node = node["corrections"];
    corrections_node["flux_calibration_requested"] =
        corrections.flux_calibration_requested;
    corrections_node["extinction_correction_requested"] =
        corrections.extinction_correction_requested;
    return node;
}

inline YAML::Node raw_timestream_observation_state_node(
    const std::optional<RawTimestreamObservationState> &observation) {
    YAML::Node node;
    node["available"] = observation.has_value();
    if (!observation) {
        return node;
    }

    auto value = node["value"];
    value["native_sample_rate_hz"] =
        raw_optional_scalar_node(observation->native_sample_rate_hz);
    value["effective_sample_rate_hz"] =
        raw_optional_scalar_node(observation->effective_sample_rate_hz);
    value["downsample_factor"] =
        raw_optional_scalar_node(observation->downsample_factor);
    value["filter_edge_guard_samples"] =
        raw_optional_scalar_node(observation->filter_edge_guard_samples);
    value["filter_outer_context_samples"] =
        raw_optional_scalar_node(
            observation->filter_outer_context_samples);
    value["filter_edge_guard_parity_deferred"] =
        observation->filter_edge_guard_parity_deferred;
    value["source_protection_active"] =
        raw_optional_scalar_node(observation->source_protection_active);
    value["extinction_active"] =
        raw_optional_scalar_node(observation->extinction_active);
    value["extinction_model"] =
        raw_optional_scalar_node(observation->extinction_model);
    return node;
}

inline YAML::Node raw_timestream_realized_state_node(
    const RawTimestreamRealizedState &realized) {
    YAML::Node node;
    node["execution_completed"] = realized.execution_completed;
    node["completed_scan_count"] = realized.completed_scan_count;
    node["flagged_sample_count"] = realized.flagged_sample_count;
    node["dynamic_notch_count"] = realized.dynamic_notch_count;
    node["required_output_count"] = realized.required_output_count;
    return node;
}

inline YAML::Node raw_timestream_provenance_node(
    const RawTimestreamExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = raw_timestream_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["requested"] = raw_timestream_request_node(plan.requested);
    root["effective"]["config"] =
        raw_timestream_request_node(plan.effective);
    root["effective"]["resolutions"] =
        raw_timestream_effective_resolutions_node(
            plan.effective_resolutions);
    root["observation"] =
        raw_timestream_observation_state_node(plan.observation);
    root["realized"] =
        raw_timestream_realized_state_node(plan.realized);
    return root;
}

inline std::filesystem::path raw_timestream_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / raw_timestream_provenance_filename;
}

inline void write_raw_timestream_provenance_file(
    const std::filesystem::path &reduction_dir,
    const RawTimestreamExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error(
            "cannot write uninitialized raw timestream provenance");
    }
    write_yaml_file_atomic(
        raw_timestream_provenance_path(reduction_dir),
        raw_timestream_provenance_node(plan));
}

}  // namespace citlali::pipeline
