#pragma once

#include <citlali/core/pipeline/mapmaking_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

namespace citlali::pipeline {

inline YAML::Node mapmaking_config_node(
    const citlali::config::MapmakingConfig &config) {
    YAML::Node node;
    node["enabled"] = config.enabled;
    node["crpix1"] = config.crpix1;
    node["crpix2"] = config.crpix2;
    node["crval1_J2000"] = config.crval1_j2000;
    node["crval2_J2000"] = config.crval2_j2000;
    node["tan_ra"] = config.tan_ra;
    node["tan_dec"] = config.tan_dec;
    node["cunit"] = config.unit;
    node["grouping"] =
        std::string{citlali::config::to_string(config.grouping)};
    node["method"] =
        std::string{citlali::config::to_string(config.method)};
    node["pixel_axes"] =
        std::string{citlali::config::to_string(config.pixel_axes_frame)};
    node["source_map_regime"] =
        std::string{citlali::config::to_string(config.source_map_regime)};
    node["pixel_size_arcsec"] = config.pixel_size_arcsec;
    node["x_size_pix"] = config.x_size_pix;
    node["y_size_pix"] = config.y_size_pix;
    node["coverage_cut"] = config.coverage_cut;
    node["jinc_filter"]["r_max"] = config.jinc_filter.r_max;
    node["jinc_filter"]["subpixel_n"] =
        config.jinc_filter.subpixel_n;
    node["jinc_filter"]["shape_params"] =
        YAML::Node(YAML::NodeType::Map);
    for (const auto &[array_name, shape] :
         config.jinc_filter.shape_params) {
        YAML::Node values(YAML::NodeType::Sequence);
        for (const auto value : shape) {
            values.push_back(value);
        }
        node["jinc_filter"]["shape_params"][array_name] = values;
    }
    node["maximum_likelihood"]["max_iterations"] =
        config.maximum_likelihood.max_iterations;
    node["maximum_likelihood"]["tolerance"] =
        config.maximum_likelihood.tolerance;
    return node;
}

inline YAML::Node mapmaking_effective_resolution_node(
    const MapmakingEffectiveResolutionRecord &resolution) {
    YAML::Node node;
    node["reduction_type"] = std::string{
        citlali::config::to_string(resolution.reduction_type)};
    node["requested_grouping"] = std::string{
        citlali::config::to_string(resolution.requested_grouping)};
    node["effective_grouping"] = std::string{
        citlali::config::to_string(resolution.effective_grouping)};
    node["automatic_grouping_resolved"] =
        resolution.automatic_grouping_resolved;
    node["detector_grouping_fell_back_to_array"] =
        resolution.detector_grouping_fell_back_to_array;
    node["requested_unit"] = resolution.requested_unit;
    node["effective_unit"] = resolution.effective_unit;
    node["uncalibrated_unit_substituted"] =
        resolution.uncalibrated_unit_substituted;
    return node;
}

template <class Value>
YAML::Node mapmaking_optional_value_node(
    const std::optional<Value> &value) {
    YAML::Node node;
    node["available"] = value.has_value();
    if (value) {
        node["value"] = *value;
    }
    return node;
}

inline unsigned long long mapmaking_observation_number(
    const std::string &obsnum) {
    unsigned long long value = 0;
    const auto result = std::from_chars(
        obsnum.data(), obsnum.data() + obsnum.size(), value);
    if (result.ec != std::errc{} ||
        result.ptr != obsnum.data() + obsnum.size() || value == 0) {
        throw std::logic_error(
            "mapmaking obsnum must be a positive integer");
    }
    return value;
}

inline YAML::Node mapmaking_observation_state_node(
    const MapmakingObservationState &observation) {
    YAML::Node node;
    node["observation_index"] = observation.observation_index;
    node["obsnum"] = mapmaking_observation_number(observation.obsnum);
    node["map_count"] = observation.map_count;
    node["effective_pixel_size_rad"] =
        observation.effective_pixel_size_rad;
    node["required_map_write_count"] =
        observation.required_map_write_count;
    node["outputs_completed"] = observation.outputs_completed;
    return node;
}

inline YAML::Node mapmaking_observations_node(
    const std::vector<MapmakingObservationState> &observations) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto &observation : observations) {
        node.push_back(mapmaking_observation_state_node(observation));
    }
    return node;
}

inline YAML::Node mapmaking_coadd_state_node(
    const std::optional<MapmakingCoaddState> &coadd) {
    YAML::Node node;
    node["available"] = coadd.has_value();
    if (!coadd) {
        return node;
    }
    node["map_count"] = coadd->map_count;
    node["required_map_write_count"] =
        coadd->required_map_write_count;
    node["outputs_completed"] = coadd->outputs_completed;
    return node;
}

inline YAML::Node mapmaking_realized_state_node(
    const MapmakingRealizedState &realized) {
    YAML::Node node;
    node["reduction_completed"] = realized.reduction_completed;
    node["mapmaking_executed"] = realized.mapmaking_executed;
    node["completed_observation_count"] =
        mapmaking_optional_value_node(
            realized.completed_observation_count);
    node["completed_coadd_count"] =
        mapmaking_optional_value_node(realized.completed_coadd_count);
    return node;
}

inline YAML::Node jinc_shape_params_node(
    const std::map<std::string, std::array<double, 3>> &shape_params) {
    YAML::Node node(YAML::NodeType::Map);
    for (const auto &[array_name, shape] : shape_params) {
        YAML::Node values(YAML::NodeType::Sequence);
        for (const auto value : shape) {
            values.push_back(value);
        }
        node[array_name] = values;
    }
    return node;
}

inline YAML::Node jinc_processing_facts_node(
    const std::vector<std::pair<std::string, std::string>> &facts) {
    YAML::Node node(YAML::NodeType::Map);
    for (const auto &[name, value] : facts) {
        if (name.empty() || node[name]) {
            throw std::logic_error(
                "JINC processing facts require unique nonempty names");
        }
        node[name] = value;
    }
    return node;
}

inline YAML::Node jinc_observation_state_node(
    const std::optional<mapmaking::JincObservationProvenance> &state) {
    YAML::Node node;
    node["available"] = state.has_value() && state->available;
    if (!state || !state->available) {
        return node;
    }
    const auto &record = *state;
    node["contract_version"] =
        std::string{mapmaking::jinc_contract_version};
    node["requested"]["digest"] = record.requested_digest;
    node["requested"]["r_max"] = record.requested_r_max;
    node["requested"]["subpixel_n"] = record.requested_subpixel_n;
    node["requested"]["shape_params"] =
        jinc_shape_params_node(record.requested_shape_params);
    node["effective"]["digest"] = record.effective_digest;
    node["effective"]["r_max"] = record.effective_r_max;
    node["effective"]["subpixel_n"] = record.effective_subpixel_n;
    node["effective"]["shape_params"] =
        jinc_shape_params_node(record.effective_shape_params);
    node["resolved"]["support_convention"] = record.support_convention;
    node["resolved"]["phase_convention"] = record.phase_convention;
    node["resolved"]["estimator"] = record.estimator;
    node["resolved"]["formal_support_policy"] =
        record.formal_support_policy;
    node["resolved"]["coverage_estimator"] =
        record.coverage_estimator;
    node["resolved"]["kernel_response"] = record.kernel_response;
    node["resolved"]["arrays"] = YAML::Node(YAML::NodeType::Sequence);
    for (const auto &array : record.resolved_arrays) {
        YAML::Node array_node;
        array_node["array_id"] = array.array_id;
        array_node["array_name"] = array.array_name;
        array_node["a"] = array.a;
        array_node["b"] = array.b;
        array_node["c"] = array.c;
        array_node["r_max"] = array.r_max;
        array_node["pixel_size_rad"] = array.pixel_size_rad;
        array_node["array_scale_rad"] = array.array_scale_rad;
        array_node["cache_half_width_pixels"] =
            array.cache_half_width_pixels;
        array_node["cache_rows"] = array.cache_rows;
        array_node["cache_cols"] = array.cache_cols;
        node["resolved"]["arrays"].push_back(array_node);
    }
    node["realized"]["kernel_template_identity"] =
        record.kernel_template_identity;
    node["realized"]["processing_configuration_identity"] =
        record.processing_configuration_identity;
    node["realized"]["processing_configuration_bound"] =
        record.processing_configuration_bound;
    node["realized"]["processing_configuration_facts"] =
        jinc_processing_facts_node(
            record.processing_configuration_facts);
    node["realized"]["processing_realization_identity"] =
        record.processing_realization_identity;
    node["realized"]["processing_realization_bound"] =
        record.processing_realization_bound;
    node["realized"]["processing_realization_facts"] =
        jinc_processing_facts_node(record.processing_realization_facts);
    node["realized"]["coverage_sample_frequency_identity"] =
        record.coverage_sample_frequency_identity;
    node["realized"]["coverage_sample_frequency_hz"] =
        record.coverage_sample_frequency_hz;
    node["realized"]["coverage_sample_frequency_hz_hex"] =
        mapmaking::jinc_double_hex(record.coverage_sample_frequency_hz);
    node["realized"]["summation_method"] =
        record.realized.summation_method;
    node["realized"]["conditioning_policy"] =
        record.realized.conditioning_policy;
    node["realized"]["map_count"] = record.realized.map_count;
    node["realized"]["realized_map_count"] =
        record.realized.realized_map_count;
    node["realized"]["realization_pass_count"] =
        record.realized.realization_pass_count;
    node["realized"]["last_pass_active_map_indices"] =
        record.realized.last_pass_active_map_indices;
    node["realized"]["total_pixel_count"] =
        record.realized.total_pixel_count;
    node["realized"]["formally_supported_pixel_count"] =
        record.realized.formally_supported_pixel_count;
    node["realized"]["exact_cancellation_pixel_count"] =
        record.realized.exact_cancellation_pixel_count;
    node["realized"]["unresolved_cancellation_pixel_count"] =
        record.realized.unresolved_cancellation_pixel_count;
    node["realized"]["invalid_q_pixel_count"] =
        record.realized.invalid_q_pixel_count;
    node["realized"]["nonfinite_accumulator_pixel_count"] =
        record.realized.nonfinite_accumulator_pixel_count;
    node["realized"]["contributor_count_max"] =
        record.realized.contributor_count_max;
    node["realized"]["rho_resolution_bound_max"] =
        record.realized.rho_resolution_bound_max;
    node["realized"]["rho_resolution_bound_max_hex"] =
        mapmaking::jinc_double_hex(
            record.realized.rho_resolution_bound_max);
    node["realized"]["map_summaries"] =
        YAML::Node(YAML::NodeType::Sequence);
    for (std::size_t index = 0;
         index < record.realized.map_summaries.size(); ++index) {
        const auto &map = record.realized.map_summaries[index];
        YAML::Node map_node;
        map_node["map_index"] = index;
        map_node["realized"] = map.realized;
        map_node["realization_pass"] = map.realization_pass;
        map_node["total_pixel_count"] = map.total_pixel_count;
        map_node["formally_supported_pixel_count"] =
            map.formally_supported_pixel_count;
        map_node["exact_cancellation_pixel_count"] =
            map.exact_cancellation_pixel_count;
        map_node["unresolved_cancellation_pixel_count"] =
            map.unresolved_cancellation_pixel_count;
        map_node["invalid_q_pixel_count"] = map.invalid_q_pixel_count;
        map_node["nonfinite_accumulator_pixel_count"] =
            map.nonfinite_accumulator_pixel_count;
        map_node["contributor_count_max"] = map.contributor_count_max;
        map_node["rho_resolution_bound_max"] =
            map.rho_resolution_bound_max;
        map_node["rho_resolution_bound_max_hex"] =
            mapmaking::jinc_double_hex(map.rho_resolution_bound_max);
        node["realized"]["map_summaries"].push_back(map_node);
    }
    node["realized"]["product_joins"] =
        YAML::Node(YAML::NodeType::Sequence);
    for (const auto &join : record.realized.product_joins) {
        YAML::Node join_node;
        join_node["product_identity"] = join.product_identity;
        join_node["product_scope"] = join.product_scope;
        join_node["output_file"] = join.output_file;
        join_node["hdu_name"] = join.hdu_name;
        join_node["content_digest"] = join.content_digest;
        node["realized"]["product_joins"].push_back(join_node);
    }
    return node;
}

}  // namespace citlali::pipeline
