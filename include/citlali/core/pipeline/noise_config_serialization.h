#pragma once

#include <citlali/core/config/noise_config.h>
#include <citlali/core/pipeline/noise_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <optional>

namespace citlali::pipeline {

inline YAML::Node noise_config_node(
    const citlali::config::NoiseConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["n_noise_maps"] = config.n_noise_maps;
    node["randomize_dets"] = config.randomize_dets;
    node["write_realizations"] = config.write_realizations;
    node["products"]["enabled"] = config.products_enabled;
    node["products"]["apply_empirical_weights"] =
        config.apply_empirical_weights;
    return node;
}

inline YAML::Node noise_effective_resolution_node(
    const NoiseEffectiveResolutionRecord &resolution) {
    YAML::Node node;
    node["mapmaking_enabled"] = resolution.mapmaking_enabled;
    node["requested_enabled"] = resolution.requested_enabled;
    node["effective_enabled"] = resolution.effective_enabled;
    node["disabled_by_mapmaking"] = resolution.disabled_by_mapmaking;
    node["requested_n_noise_maps"] =
        resolution.requested_n_noise_maps;
    node["effective_n_noise_maps"] =
        resolution.effective_n_noise_maps;
    node["count_zeroed_while_disabled"] =
        resolution.count_zeroed_while_disabled;
    node["randomization"]["engine"] = resolution.random_engine;
    node["randomization"]["seed"] = resolution.random_seed;
    node["randomization"]["seed_policy"] = resolution.seed_policy;
    node["randomization"]["generator_scope"] =
        resolution.generator_scope;
    node["randomization"]["joint_assignment_design"] =
        resolution.joint_assignment_design;
    node["randomization"]["dependence_status"] =
        noise_dependence_status;
    return node;
}

template <class Value>
YAML::Node noise_optional_value_node(const std::optional<Value> &value) {
    YAML::Node node;
    node["available"] = value.has_value();
    if (value) {
        node["value"] = *value;
    }
    return node;
}

inline YAML::Node noise_realized_state_node(
    const NoiseRealizedState &realized) {
    YAML::Node node;
    node["reduction_completed"] = realized.reduction_completed;
    node["generation_executed"] = realized.generation_executed;
    node["noise_maps_per_scientific_map"] =
        noise_optional_value_node(realized.noise_maps_per_scientific_map);
    node["observation_scientific_map_count"] =
        noise_optional_value_node(realized.observation_scientific_map_count);
    node["observation_noise_realization_count"] =
        noise_optional_value_node(
            realized.observation_noise_realization_count);
    node["coadd_scientific_map_count"] =
        noise_optional_value_node(realized.coadd_scientific_map_count);
    node["coadd_noise_realization_count"] =
        noise_optional_value_node(realized.coadd_noise_realization_count);
    node["total_noise_realization_count"] =
        noise_optional_value_node(realized.total_noise_realization_count);
    node["empirical_product_map_count"] =
        noise_optional_value_node(realized.empirical_product_map_count);
    node["realization_image_write_count"] =
        noise_optional_value_node(realized.realization_image_write_count);
    node["actual_completion_valid"] = realized.actual_completion_valid;
    node["completed_count_matches_effective"] =
        realized.completed_count_matches_effective;
    node["uncertainty_use_valid"] = realized.uncertainty_use_valid;
    node["completion_basis"] = realized.completion_basis;
    node["outputs_completed"] = realized.outputs_completed;
    return node;
}

}  // namespace citlali::pipeline
