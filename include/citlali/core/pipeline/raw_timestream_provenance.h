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
    "citlali-raw-timestream-provenance-v2";
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

inline YAML::Node raw_rtc_stage_realization_node(
    const RawRtcStageRealization &stage) {
    YAML::Node node;
    node["stage_identity"] = stage.stage_identity;
    node["parent_identity"] = stage.parent_identity;
    node["process_identity"] = stage.process_identity;
    node["observation_scope"] = stage.observation_scope;
    node["process_label"] = stage.process_label;
    node["stage_view"] = stage.stage_view;
    node["assigned_grid_identity"] = stage.assigned_grid_identity;
    node["physical_event_semantics"] =
        stage.physical_event_semantics;
    node["assigned_time_semantics"] = stage.assigned_time_semantics;
    node["lattice_label"] = stage.lattice_label;
    node["phase_label"] = stage.phase_label;
    node["representative_assigned_time"]["rule"] =
        stage.representative_assigned_time_rule;
    node["representative_assigned_time"]["value_hex"] =
        stage.representative_assigned_time_hex;
    node["assigned_time_values_digest"] =
        stage.assigned_time_values_digest;
    node["edge_rule"] = stage.edge_rule;
    node["influence_support_policy"] =
        stage.influence_support_policy;
    node["operator_ordering"] = stage.operator_ordering;
    node["detector_ordering"] = stage.detector_ordering;
    node["source_mask"]["identity"] = stage.source_mask_identity;
    node["source_mask"]["frame"] = stage.source_mask_frame;
    node["source_mask"]["admission"] = stage.source_mask_admission;
    node["source_mask"]["reason"] = stage.source_mask_reason;
    node["source_mask"]["timing_sensitive_accuracy"] =
        stage.source_mask_timing_accuracy;
    node["source_mask"]["admitted"] = stage.source_mask_admitted;
    node["scan_id"] = stage.scan_id;
    node["absolute_assigned_start"] = stage.absolute_assigned_start;
    node["input_sample_count"] = stage.input_sample_count;
    node["output_sample_count"] = stage.output_sample_count;
    node["detector_count"] = stage.detector_count;
    node["inner_start"] = stage.inner_start;
    node["inner_sample_count"] = stage.inner_sample_count;
    node["filter_guard_samples"] = stage.filter_guard_samples;
    node["filter_context_samples"] = stage.filter_context_samples;
    node["native_sample_rate_hz"] = stage.native_sample_rate_hz;
    node["effective_sample_rate_hz"] = stage.effective_sample_rate_hz;
    node["downsample_factor"] = stage.downsample_factor;
    node["simulated"] = stage.simulated;
    node["response"]["complete_available"] =
        stage.complete_response_available;
    node["response"]["signal_stage_bits"] = stage.signal_stage_bits;
    node["response"]["response_stage_bits"] = stage.response_stage_bits;
    node["response"]["unavailable_cause_bits"] =
        stage.response_unavailable_cause_bits;
    node["influence"]["sample_count"] = stage.influenced_sample_count;
    node["influence"]["intervals"] = YAML::Node(YAML::NodeType::Sequence);
    for (const auto &interval : stage.influence_intervals) {
        YAML::Node value;
        value["detector"] = interval.detector;
        value["first_assigned_sample"] =
            interval.first_assigned_sample;
        value["last_assigned_sample"] =
            interval.last_assigned_sample;
        value["cause_bits"] = interval.cause_bits;
        node["influence"]["intervals"].push_back(value);
    }
    auto coefficients = node["operator_coefficients"];
    coefficients["fir_hex"] = stage.fir_coefficients_hex;
    coefficients["notch_a_hex"] = stage.notch_a_coefficients_hex;
    coefficients["notch_b_hex"] = stage.notch_b_coefficients_hex;
    coefficients["iir_highpass_alpha_hex"] =
        stage.iir_highpass_alpha_hex;
    coefficients["iir_highpass_order"] = stage.iir_highpass_order;
    coefficients["notch_zero_phase"] = stage.notch_zero_phase;
    coefficients["iir_highpass_zero_phase"] =
        stage.iir_highpass_zero_phase;
    coefficients["fir_normalization"] = stage.fir_normalization;
    coefficients["downsample_normalization"] =
        stage.downsample_normalization;
    coefficients["fir_state_reset"] = stage.fir_state_reset;
    coefficients["notch_state_reset"] = stage.notch_state_reset;
    coefficients["iir_highpass_state_reset"] =
        stage.iir_highpass_state_reset;
    coefficients["notch_section_layout"] =
        stage.notch_section_layout;
    return node;
}

inline YAML::Node raw_rtc_product_realization_node(
    const RawRtcProductRealization &product) {
    YAML::Node node;
    node["product_identity"] = product.product_identity;
    node["stage_identity"] = product.stage_identity;
    node["parent_identity"] = product.parent_identity;
    node["process_identity"] = product.process_identity;
    node["completion_identity"] = product.completion_identity;
    node["assigned_grid_identity"] = product.assigned_grid_identity;
    node["physical_event_semantics"] =
        product.physical_event_semantics;
    node["product_kind"] = product.product_kind;
    node["filepath"] = product.filepath;
    node["scan_id"] = product.scan_id;
    node["output_row"] = product.output_row;
    node["mini_output"] = product.mini_output;
    node["outer_output"] = product.outer_output;
    node["simulated"] = product.simulated;
    node["complete"] = product.complete;
    return node;
}

inline YAML::Node interface_sync_offset_config_node(
    const citlali::config::InterfaceSyncOffsetConfig &config) {
    YAML::Node node;
    node["unit"] = "s";
    for (std::size_t index = 0;
         index < citlali::config::toltec_interface_count; ++index) {
        node["offsets"]["toltec" + std::to_string(index)] =
            config.toltec_offset_sec[index];
    }
    node["offsets"]["hwpr"] = config.hwpr_offset_sec;
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
    value["rtc_contract"] = raw_rtc_contract_node(
        observation->rtc_contract);
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
    node["completed_scan_count"] =
        raw_optional_scalar_node(realized.completed_scan_count);
    node["flagged_sample_count"] =
        raw_optional_scalar_node(realized.flagged_sample_count);
    node["dynamic_notch_count"] =
        raw_optional_scalar_node(realized.dynamic_notch_count);
    node["required_timestream_write_count"] =
        raw_optional_scalar_node(
            realized.required_timestream_write_count);
    node["rtc"]["observation_scope"] = realized.rtc_observation_scope;
    node["rtc"]["bundle_identity"] = realized.rtc_bundle_identity;
    node["rtc"]["bundle_complete"] = realized.rtc_bundle_complete;
    node["rtc"]["stages"] = YAML::Node(YAML::NodeType::Sequence);
    for (const auto &stage : realized.rtc_stages) {
        node["rtc"]["stages"].push_back(
            raw_rtc_stage_realization_node(stage));
    }
    node["rtc"]["products"] = YAML::Node(YAML::NodeType::Sequence);
    for (const auto &product : realized.rtc_products) {
        node["rtc"]["products"].push_back(
            raw_rtc_product_realization_node(product));
    }
    return node;
}

inline YAML::Node raw_timestream_provenance_node(
    const RawTimestreamExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = raw_timestream_provenance_schema_version;
    root["initialized"] = plan.initialized;
    auto requested = raw_timestream_request_node(plan.requested);
    requested["interface_sync_offset"] =
        interface_sync_offset_config_node(plan.interface_sync_requested);
    requested["rtc_contract"] =
        raw_rtc_contract_node(plan.requested_rtc_contract);
    root["requested"] = requested;
    auto effective = raw_timestream_request_node(plan.effective);
    effective["interface_sync_offset"] =
        interface_sync_offset_config_node(plan.interface_sync_effective);
    effective["rtc_contract"] =
        raw_rtc_contract_node(plan.effective_rtc_contract);
    root["effective"]["config"] = effective;
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
    if (!plan.observation.has_value()) {
        throw std::logic_error(
            "cannot write raw timestream provenance before observation begins");
    }
    if (!plan.realized.execution_completed) {
        throw std::logic_error(
            "cannot write incomplete raw timestream provenance");
    }
    if (!plan.realized.completed_scan_count.has_value()
        || !plan.realized.required_timestream_write_count.has_value()) {
        throw std::logic_error(
            "cannot write raw timestream provenance without realized counts");
    }
    write_yaml_file_atomic(
        raw_timestream_provenance_path(reduction_dir),
        raw_timestream_provenance_node(plan));
}

}  // namespace citlali::pipeline
