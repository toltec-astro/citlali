#pragma once

#include <citlali/core/pipeline/astrometry_config_serialization.h>
#include <citlali/core/pipeline/atomic_yaml_output.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace citlali::pipeline {

inline constexpr const char *astrometry_provenance_schema_version =
    "citlali-astrometry-provenance-v1";
inline constexpr const char *astrometry_provenance_filename =
    "astrometry_provenance.yaml";

inline YAML::Node astrometry_provenance_node(
    const AstrometryExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = astrometry_provenance_schema_version;
    root["initialized"] = plan.initialized;
    root["authority"]["calibration_selection"] = "tolteca";
    root["authority"]["application"] = "citlali";
    root["authority"]["support_origin_metadata_available"] = false;
    root["authority"]["configured_values_origin"] = "upstream-unspecified";
    root["identity"]["axes"].push_back("az");
    root["identity"]["axes"].push_back("alt");
    root["identity"]["offset_unit"] = "arcsec";
    root["identity"]["time_support"] = "modified-julian-date";
    root["identity"]["algorithm"] =
        "legacy-citlali-constant-or-linear-v1";
    root["contract"]["upstream_selection_owner"] = "tolteca";
    root["contract"]["one_configured_value"] = "constant";
    root["contract"]["two_values_without_positive_mjd_pair"] =
        "observation-span-linear";
    root["contract"]["two_values_with_positive_mjd_pair"] =
        "explicit-mjd-linear";
    root["contract"]["explicit_mjd_requires_observation_bracketing"] = true;
    root["contract"]["extrapolation"] = "forbidden";
    root["expected_observation_count"] = plan.expected_observation_count;
    root["observations"] =
        astrometry_observation_plans_node(plan.observations);
    root["reduction_completed"] = plan.reduction_completed;
    return root;
}

inline std::filesystem::path astrometry_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / astrometry_provenance_filename;
}

inline void write_astrometry_provenance_file(
    const std::filesystem::path &reduction_dir,
    const AstrometryExecutionPlan &plan) {
    if (!plan.initialized || !plan.reduction_completed) {
        throw std::logic_error(
            "cannot write incomplete astrometry provenance");
    }
    write_yaml_file_atomic(
        astrometry_provenance_path(reduction_dir),
        astrometry_provenance_node(plan));
}

}  // namespace citlali::pipeline
