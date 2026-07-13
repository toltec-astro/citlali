#pragma once

#include <citlali/core/pipeline/mapmaking_execution_plan.h>

#include <yaml-cpp/yaml.h>

#include <optional>
#include <string>

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

inline YAML::Node mapmaking_observation_state_node(
    const std::optional<MapmakingObservationState> &observation) {
    YAML::Node node;
    node["available"] = observation.has_value();
    if (!observation) {
        return node;
    }
    node["map_count"] =
        mapmaking_optional_value_node(observation->map_count);
    node["effective_pixel_size_rad"] =
        mapmaking_optional_value_node(
            observation->effective_pixel_size_rad);
    node["required_map_write_count"] =
        mapmaking_optional_value_node(
            observation->required_map_write_count);
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

}  // namespace citlali::pipeline
