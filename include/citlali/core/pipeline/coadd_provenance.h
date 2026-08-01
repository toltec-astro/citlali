#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/coadd_execution_plan.h>
#include <citlali/core/pipeline/science_map_provenance_serialization.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <optional>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *coadd_provenance_schema_version =
    "citlali-coadd-provenance-v2";
inline constexpr const char *coadd_provenance_filename =
    "coadd_provenance.yaml";

template <class Value>
YAML::Node coadd_optional_value_node(const std::optional<Value> &value) {
    YAML::Node node;
    node["available"] = value.has_value();
    if (value) {
        node["value"] = *value;
    }
    return node;
}

inline YAML::Node coadd_provenance_node(
    const CoaddExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = coadd_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["requested"]["enabled"] = plan.requested.enabled;
    root["effective"]["config"]["enabled"] = plan.effective.enabled;
    root["effective"]["resolution"]["mapmaking_enabled"] =
        plan.effective_resolution.mapmaking_enabled;
    root["effective"]["resolution"]["requested_enabled"] =
        plan.effective_resolution.requested_enabled;
    root["effective"]["resolution"]["effective_enabled"] =
        plan.effective_resolution.effective_enabled;
    root["effective"]["resolution"]["disabled_by_mapmaking"] =
        plan.effective_resolution.disabled_by_mapmaking;
    root["science_contract"] = science_map_policy_contract_node();
    root["science_contract"]["cuts"]["requested"] =
        science_map_optional_exact_double_node(
            plan.science.requested_coverage_cut);
    root["science_contract"]["cuts"]["effective"] =
        science_map_optional_exact_double_node(
            plan.science.effective_coverage_cut);
    const bool science_state_available =
        plan.science.common_identity.has_value() &&
        !plan.science.realized_maps.empty() &&
        plan.science.common_identity->ordered_slots.size() ==
            plan.science.realized_maps.size();
    root["observation_resolved"]["available"] =
        science_state_available;
    root["observation_resolved"]["common_identity"] =
        science_map_optional_bundle_identity_node(
            plan.science.common_identity,
            plan.science.absence_reason);
    root["observation_resolved"]["realized_maps"] =
        science_map_realized_maps_node(plan.science.realized_maps);
    root["observation_resolved"]["admissions"] =
        science_map_coadd_admissions_node(plan.science.admissions);
    root["observation_resolved"]["admitted_observation_count"] =
        plan.science.admissions.size();
    if (!science_state_available) {
        root["observation_resolved"]["absence_reason"] =
            plan.science.absence_reason;
    }
    root["realized"]["reduction_completed"] =
        plan.realized.reduction_completed;
    root["realized"]["coadd_executed"] =
        plan.realized.coadd_executed;
    root["realized"]["map_count"] =
        coadd_optional_value_node(plan.realized.map_count);
    root["realized"]["required_map_write_count"] =
        coadd_optional_value_node(
            plan.realized.required_map_write_count);
    root["realized"]["outputs_completed"] =
        plan.realized.outputs_completed;
    return root;
}

inline std::filesystem::path coadd_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / coadd_provenance_filename;
}

inline void write_coadd_provenance_file(
    const std::filesystem::path &reduction_dir,
    const CoaddExecutionPlan &plan) {
    if (!plan.initialized || !plan.realized.reduction_completed) {
        throw std::logic_error(
            "cannot write incomplete coadd provenance");
    }
    write_yaml_file_atomic(
        coadd_provenance_path(reduction_dir),
        coadd_provenance_node(plan));
}

}  // namespace citlali::pipeline
