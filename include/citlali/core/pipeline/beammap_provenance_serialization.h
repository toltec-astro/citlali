#pragma once

#include <citlali/core/pipeline/beammap_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <optional>
#include <string>

namespace citlali::pipeline {

inline YAML::Node beammap_request_presence_node(
    const BeammapRequestPresence &presence) {
    YAML::Node node;
    node["max_d2_iter0"] = presence.max_d2_iter0;
    node["max_d2_after_iter0"] = presence.max_d2_after_iter0;
    node["score_lambda_iter0"] = presence.score_lambda_iter0;
    node["score_lambda_after_iter0"] =
        presence.score_lambda_after_iter0;
    node["split_flag_values"] = presence.split_flag_values;
    return node;
}

inline YAML::Node beammap_effective_resolution_node(
    const BeammapEffectiveResolutionRecord &resolution) {
    YAML::Node node;
    node["explicit_request"] =
        beammap_request_presence_node(resolution.explicit_request);
    node["mapmaking_enabled"] = resolution.mapmaking_enabled;
    node["requested_max_iterations"] =
        resolution.requested_max_iterations;
    node["effective_max_iterations"] =
        resolution.effective_max_iterations;
    node["max_iterations_forced_without_mapmaking"] =
        resolution.max_iterations_forced_without_mapmaking;
    node["requested_locator_iter"] = resolution.requested_locator_iter;
    node["effective_locator_iter"] = resolution.effective_locator_iter;
    node["locator_iter_forced_zero"] =
        resolution.locator_iter_forced_zero;
    node["requested_measurement_start_iter"] =
        resolution.requested_measurement_start_iter;
    node["effective_measurement_start_iter"] =
        resolution.effective_measurement_start_iter;
    node["measurement_start_iter_adjusted"] =
        resolution.measurement_start_iter_adjusted;
    node["legacy_phase_behavior"] = resolution.legacy_phase_behavior;
    node["measurement_pass_available"] =
        resolution.measurement_pass_available;
    node["convergence_check_available"] =
        resolution.convergence_check_available;
    node["convergence_active"] = resolution.convergence_active;
    node["prior_path_available"] = resolution.prior_path_available;
    node["priors_disabled_by_missing_path"] =
        resolution.priors_disabled_by_missing_path;
    node["max_d2_iter0_inherited"] =
        resolution.max_d2_iter0_inherited;
    node["max_d2_after_iter0_inherited"] =
        resolution.max_d2_after_iter0_inherited;
    node["score_lambda_iter0_inherited"] =
        resolution.score_lambda_iter0_inherited;
    node["score_lambda_after_iter0_inherited"] =
        resolution.score_lambda_after_iter0_inherited;
    node["split_flag_values_defaulted"] =
        resolution.split_flag_values_defaulted;
    node["split_flag_values_sorted"] =
        resolution.split_flag_values_sorted;
    node["split_flag_values_deduplicated"] =
        resolution.split_flag_values_deduplicated;
    node["requested_split_flag_count"] =
        resolution.requested_split_flag_count;
    node["effective_split_flag_count"] =
        resolution.effective_split_flag_count;
    return node;
}

template <class Value>
YAML::Node beammap_optional_value_node(
    const std::optional<Value> &value) {
    YAML::Node node;
    node["available"] = value.has_value();
    if (value) {
        node["value"] = *value;
    }
    return node;
}

inline YAML::Node beammap_iteration_state_node(
    const BeammapIterationState &iteration) {
    YAML::Node node;
    node["iteration_index"] = iteration.iteration_index;
    node["phase"] = std::string{
        beammap_iteration_phase_name(iteration.phase)};
    node["active_map_count"] = iteration.active_map_count;
    node["mapmaking_pass_count"] = iteration.mapmaking_pass_count;
    node["source_aware_rtc_rerun"] =
        beammap_optional_value_node(iteration.source_aware_rtc_rerun);
    node["fitting_completed"] = iteration.fitting_completed;
    node["newly_converged_map_count"] =
        iteration.newly_converged_map_count;
    node["total_converged_map_count"] =
        iteration.total_converged_map_count;
    node["termination_reason"] = std::string{
        beammap_termination_reason_name(iteration.termination_reason)};
    node["completed"] = iteration.completed;
    return node;
}

inline YAML::Node beammap_iteration_states_node(
    const std::vector<BeammapIterationState> &iterations) {
    YAML::Node node{YAML::NodeType::Sequence};
    for (const auto &iteration : iterations) {
        node.push_back(beammap_iteration_state_node(iteration));
    }
    return node;
}

inline YAML::Node beammap_detector_tod_realized_state_node(
    const BeammapDetectorTodRealizedState &detector_tod) {
    YAML::Node node;
    node["required"] = detector_tod.required;
    node["completed_write_count"] =
        detector_tod.completed_write_count;
    node["output_iteration"] =
        beammap_optional_value_node(detector_tod.output_iteration);
    node["detector_count"] =
        beammap_optional_value_node(detector_tod.detector_count);
    node["slot_count"] =
        beammap_optional_value_node(detector_tod.slot_count);
    node["maximum_sample_count"] =
        beammap_optional_value_node(detector_tod.maximum_sample_count);
    return node;
}

inline YAML::Node beammap_observation_state_node(
    const BeammapObservationState &observation) {
    YAML::Node node;
    node["observation_index"] = observation.observation_index;
    node["obsnum"] = observation.obsnum;
    node["detector_count"] = observation.detector_count;
    node["map_count"] = observation.map_count;
    node["scan_count"] = observation.scan_count;
    node["iterations"] =
        beammap_iteration_states_node(observation.iterations);
    node["terminal_iteration"] =
        beammap_optional_value_node(observation.terminal_iteration);
    node["termination_reason"] = std::string{
        beammap_termination_reason_name(observation.termination_reason)};
    node["detector_tod"] =
        beammap_detector_tod_realized_state_node(
            observation.detector_tod);
    node["outputs_completed"] = observation.outputs_completed;
    return node;
}

inline YAML::Node beammap_observation_states_node(
    const std::vector<BeammapObservationState> &observations) {
    YAML::Node node{YAML::NodeType::Sequence};
    for (const auto &observation : observations) {
        node.push_back(beammap_observation_state_node(observation));
    }
    return node;
}

inline YAML::Node beammap_realized_state_node(
    const BeammapRealizedState &realized) {
    YAML::Node node;
    node["reduction_completed"] = realized.reduction_completed;
    node["beammap_executed"] = realized.beammap_executed;
    node["completed_observation_count"] =
        beammap_optional_value_node(realized.completed_observation_count);
    node["completed_iteration_count"] =
        realized.completed_iteration_count;
    node["outputs_completed"] = realized.outputs_completed;
    return node;
}

}  // namespace citlali::pipeline
