#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/kids_external_config.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace citlali::pipeline {

inline constexpr std::string_view kids_external_provenance_schema{
    "citlali-kids-external-provenance-v1"};
inline constexpr std::string_view kids_external_provenance_filename{
    "kids_external_provenance.yaml"};

inline YAML::Node kids_external_config_identity_node(
    const KidsExternalConfigIdentity &identity) {
    YAML::Node node;
    node["fitter"]["modelspec"] = identity.fitter.modelspec;
    node["fitter"]["weight_window"]["type"] =
        identity.fitter.weight_window_type;
    node["fitter"]["weight_window"]["fwhm_Hz"] =
        identity.fitter.weight_window_fwhm_hz;
    node["solver"]["fitreportdir"] =
        identity.solver.fit_report_directory;
    node["solver"]["parallel_policy"] =
        identity.solver.parallel_policy;
    node["solver"]["extra_output"] = identity.solver.extra_output;
    return node;
}

inline YAML::Node kids_external_provenance_node(
    const KidsExternalConfigPlan &plan) {
    require_valid_kids_external_config_plan(plan);
    YAML::Node root;
    root["schema_version"] = std::string{kids_external_provenance_schema};
    root["initialized"] = plan.initialized;
    root["authority"] = "kidscpp";
    root["config_schema"] = std::string{kids_external_config_schema};
    root["data_schema"] = plan.data_schema;
    root["dependency"]["name"] = "kidscpp";
    root["dependency"]["version"] = plan.dependency_version;
    for (const auto type : supported_kids_tod_types) {
        root["supported_tod_types"].push_back(
            std::string{citlali::config::to_string(type)});
    }
    root["selected_tod_type"] =
        std::string{citlali::config::to_string(plan.selected_tod_type)};
    root["requested"]["values"] =
        kids_external_config_identity_node(plan.requested.values);
    root["requested"]["solver_extra_output_present"] =
        plan.requested.solver_extra_output_present;
    root["effective"]["values"] =
        kids_external_config_identity_node(plan.effective.values);
    root["effective"]["resolution"]
        ["solver_extra_output_forced_disabled"] =
        plan.effective.solver_extra_output_forced_disabled;
    return root;
}

inline std::filesystem::path kids_external_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / kids_external_provenance_filename;
}

inline void write_kids_external_provenance_file(
    const std::filesystem::path &reduction_dir,
    const KidsExternalConfigPlan &plan) {
    write_yaml_file_atomic(kids_external_provenance_path(reduction_dir),
                           kids_external_provenance_node(plan));
}

}  // namespace citlali::pipeline
