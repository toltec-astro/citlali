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
    node["outputs_suppressed_while_disabled"] =
        resolution.outputs_suppressed_while_disabled;
    node["randomization"]["engine"] = resolution.random_engine;
    node["randomization"]["seed"] = resolution.random_seed;
    node["randomization"]["seed_policy"] = resolution.seed_policy;
    node["randomization"]["generator_scope"] =
        resolution.generator_scope;
    node["randomization"]["key_policy_version"] =
        noise_realization_key_policy_version;
    node["randomization"]["generator_version"] =
        noise_realization_generator_version;
    node["randomization"]["ensemble_mode"] =
        noise_ensemble_mode_source_imprinted_current;
    return node;
}

inline YAML::Node noise_assignment_policy_node() {
    YAML::Node node;
    node["key_policy_version"] = noise_realization_key_policy_version;
    node["generator_version"] = noise_realization_generator_version;
    node["master_seed"] = noise_random_seed;
    node["seed_policy"] = noise_seed_policy_name;
    node["generator_scope"] = noise_generator_scope_name;
    node["ensemble_mode"] =
        noise_ensemble_mode_source_imprinted_current;
    node["interpretation"] = "restricted_diagnostic_only";
    node["signal_content"] = "deterministic_signal_may_remain";
    node["negative_source_realizations"] = "permitted";
    node["coherence_unit_identity_policy"] =
        noise_coherence_unit_identity_policy;
    node["channel_identity_policy"] = noise_channel_identity_policy;
    node["ordering_policy"] = noise_assignment_ordering_policy;
    return node;
}

inline YAML::Node noise_assignment_record_node(
    const NoiseAssignmentRecord &record) {
    YAML::Node node;
    node["key_policy_version"] = record.key_policy_version;
    node["generator_version"] = record.generator_version;
    node["observation_id"] = record.observation_id;
    node["ensemble_mode"] = record.ensemble_mode;
    node["conditioning_iteration"] = record.conditioning_iteration;
    node["pass_id"] = record.pass_id;
    node["pass_ordinal"] = record.pass_ordinal;
    node["randomize_channels"] = record.randomize_channels;
    node["coherence_unit_identity_policy"] =
        record.coherence_unit_identity_policy;
    node["channel_identity_policy"] = record.channel_identity_policy;
    node["ordering_policy"] = record.ordering_policy;
    node["partition"]["coherence_unit_count"] =
        record.coherence_unit_count;
    node["partition"]["channel_count"] = record.channel_count;
    node["completed_realization_ids"] = record.completed_realization_ids;
    node["digests"]["namespace"] = record.namespace_digest;
    node["digests"]["partition"] = record.partition_digest;
    node["digests"]["reconstruction"] =
        record.reconstruction_digest;
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
    node["zero_work"] = realized.zero_work;
    node["outputs_promised"] = realized.outputs_promised;
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
    node["outputs_completed"] = realized.outputs_completed;
    return node;
}

}  // namespace citlali::pipeline
