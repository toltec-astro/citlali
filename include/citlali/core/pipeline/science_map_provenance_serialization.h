#pragma once

#include <citlali/core/mapmaking/science_map_contract.h>

#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *science_map_coefficient_product = "weight_I";
inline constexpr const char *science_map_coefficient_lifecycle_stage =
    "realized-per-map-record";
inline constexpr const char *science_map_empirical_coefficient_lifecycle_stage =
    mapmaking::science_map_observation_empirical_coefficient_stage;
inline constexpr const char *science_map_coefficient_interpretation =
    "nonprecision-gridding-coefficient";
inline constexpr const char *science_map_precision_status =
    "conditional-unavailable-pending-SCI-PTC-001";
inline constexpr const char *science_map_covariance_status =
    "unavailable-pending-SCI-PTC-001";

inline std::string science_map_double_decimal(double value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return stream.str();
}

inline YAML::Node science_map_exact_double_node(double value) {
    YAML::Node node;
    node["numeric"] = science_map_double_decimal(value);
    node["hex"] = mapmaking::science_map_double_hex(value);
    node["encoding"] = "binary64-max-digits10-and-c99-hexfloat";
    return node;
}

inline double science_map_exact_double_value(const YAML::Node &node) {
    if (!node || !node["numeric"] || !node["hex"] ||
        !node["encoding"] ||
        node["encoding"].as<std::string>() !=
            "binary64-max-digits10-and-c99-hexfloat") {
        throw std::logic_error(
            "science-map exact double record is incomplete");
    }
    const auto parse = [](const std::string &text) {
        char *end = nullptr;
        const double value = std::strtod(text.c_str(), &end);
        if (end == text.c_str() || *end != '\0') {
            throw std::logic_error(
                "invalid science-map binary64 scalar value");
        }
        return value;
    };
    const auto hex = node["hex"].as<std::string>();
    const double decoded = parse(hex);
    const double numeric = parse(node["numeric"].as<std::string>());
    if (!mapmaking::science_map_exact_double_equal(numeric, decoded)) {
        throw std::logic_error(
            "science-map numeric and binary64 hex values disagree");
    }
    return decoded;
}

inline YAML::Node science_map_optional_exact_double_node(
    const std::optional<double> &value) {
    YAML::Node node;
    node["available"] = value.has_value();
    if (value) {
        node["value"] = science_map_exact_double_node(*value);
    }
    return node;
}

inline YAML::Node science_map_string_sequence_node(
    const std::vector<std::string> &values) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto &value : values) {
        node.push_back(value);
    }
    return node;
}

inline YAML::Node science_map_exact_double_sequence_node(
    const std::vector<double> &values) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto value : values) {
        node.push_back(science_map_exact_double_node(value));
    }
    return node;
}

inline YAML::Node science_map_wcs_identity_node(
    const mapmaking::ScienceMapWcsIdentity &identity) {
    YAML::Node node;
    node["coordinate_frame"] = identity.coordinate_frame;
    node["projection"] = identity.projection;
    node["axis_types"] =
        science_map_string_sequence_node(identity.axis_types);
    node["axis_units"] =
        science_map_string_sequence_node(identity.axis_units);
    node["pixel_scale"] =
        science_map_exact_double_sequence_node(identity.pixel_scale);
    node["reference_world"] =
        science_map_exact_double_sequence_node(identity.reference_world);
    node["reference_pixel"] =
        science_map_exact_double_sequence_node(identity.reference_pixel);
    node["source_epoch"] =
        science_map_exact_double_node(identity.source_epoch);
    node["orientation_rad"] =
        science_map_exact_double_node(identity.orientation_rad);
    return node;
}

inline YAML::Node science_map_slot_identity_node(
    const mapmaking::ScienceMapSlotIdentity &identity) {
    YAML::Node node;
    node["ordered_slot"] = identity.ordered_slot;
    node["grouping"] = identity.grouping;
    node["group_identity"] = identity.group_identity;
    node["array_identity"] = identity.array_identity;
    node["stokes_identity"] = identity.stokes_identity;
    node["frequency_hz"] =
        science_map_exact_double_node(identity.frequency_hz);
    return node;
}

inline YAML::Node science_map_slot_identities_node(
    const std::vector<mapmaking::ScienceMapSlotIdentity> &identities) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (const auto &identity : identities) {
        node.push_back(science_map_slot_identity_node(identity));
    }
    return node;
}

inline YAML::Node science_map_bundle_identity_node(
    const mapmaking::ScienceMapBundleIdentity &identity) {
    YAML::Node node;
    node["contract_version"] = identity.contract_version;
    node["identity_digest"] =
        mapmaking::science_map_bundle_identity_digest(identity);
    node["grouping"] = identity.grouping;
    node["signal_unit"] = identity.signal_unit;
    node["estimator_identity"] = identity.estimator_identity;
    node["response_identity"] = identity.response_identity;
    node["parallel_equivalence_policy"] =
        identity.parallel_equivalence_policy;
    node["required_companions"] =
        science_map_string_sequence_node(identity.required_companions);
    node["policies"]["validity"] = identity.validity_policy;
    node["policies"]["coefficient"] = identity.coefficient_policy;
    node["policies"]["normalization_support"] =
        identity.normalization_support_policy;
    node["policies"]["science_policy_support"] =
        identity.science_policy_support_policy;
    node["policies"]["nonfinite"] = identity.nonfinite_policy;
    node["wcs"] = science_map_wcs_identity_node(identity.wcs);
    node["shape"]["rows"] = identity.rows;
    node["shape"]["cols"] = identity.cols;
    node["ordered_slots"] =
        science_map_slot_identities_node(identity.ordered_slots);
    return node;
}

inline YAML::Node science_map_optional_bundle_identity_node(
    const std::optional<mapmaking::ScienceMapBundleIdentity> &identity,
    const std::string &absence_reason) {
    YAML::Node node;
    node["available"] = identity.has_value();
    if (identity) {
        node["value"] = science_map_bundle_identity_node(*identity);
    }
    else {
        node["absence_reason"] = absence_reason;
    }
    return node;
}

inline YAML::Node science_map_threshold_realization_node(
    const mapmaking::ScienceMapThresholdRealization &realization) {
    YAML::Node node;
    node["order_statistic_algorithm"] =
        realization.order_statistic_algorithm;
    node["support_algorithm"] = realization.support_algorithm;
    node["coefficient_product"] = realization.coefficient_product;
    node["coefficient_stage"] = realization.coefficient_stage;
    node["requested_cut"] =
        science_map_exact_double_node(realization.requested_cut);
    node["realized_cut"] =
        science_map_exact_double_node(realization.realized_cut);
    node["realized_threshold"] =
        science_map_exact_double_node(realization.realized_threshold);
    node["selected_positive_value"] =
        science_map_exact_double_node(realization.selected_positive_value);
    node["positive_value_count"] = realization.positive_value_count;
    node["selected_zero_based_index"]["available"] =
        realization.selected_index_available;
    if (realization.selected_index_available) {
        node["selected_zero_based_index"]["value"] =
            realization.selected_zero_based_index;
    }
    node["finite_convention"] = realization.finite_convention;
    node["positivity_convention"] = realization.positivity_convention;
    node["comparison_convention"] = realization.comparison_convention;
    return node;
}

inline YAML::Node science_map_product_inventory_node(
    const mapmaking::ScienceMapRealizedMap &realized) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(mapmaking::ScienceMapProduct::count);
         ++index) {
        const auto product =
            static_cast<mapmaking::ScienceMapProduct>(index);
        YAML::Node entry;
        entry["identity"] = mapmaking::science_map_product_name(product);
        entry["unit"] = mapmaking::science_map_product_unit(product);
        entry["available"] = realized.product_available.at(index);
        entry["nonzero_count"] = realized.product_nonzero_count.at(index);
        entry["value_sum"] = realized.product_value_sum.at(index);
        entry["value_sum_encoding"] =
            product == mapmaking::ScienceMapProduct::upstream_eligible_exposure ||
                    product == mapmaking::ScienceMapProduct::retained_exposure
                ? "binary64-c99-hexfloat"
                : "base10-integer";
        entry["absence_reason"] =
            realized.product_absence_reason.at(index);
        node.push_back(entry);
    }
    return node;
}

inline YAML::Node science_map_realized_map_node(
    const mapmaking::ScienceMapRealizedMap &realized,
    std::size_t ordered_slot) {
    YAML::Node node;
    node["ordered_slot"] = ordered_slot;
    node["initialized"] = realized.initialized;
    node["thresholds"]["normalization"] =
        science_map_threshold_realization_node(realized.normalization);
    node["thresholds"]["science_policy"] =
        science_map_threshold_realization_node(realized.science_policy);
    node["products"] = science_map_product_inventory_node(realized);
    node["required_companions"] =
        science_map_string_sequence_node(realized.required_companions);
    node["admitted_bundle_identity"] =
        realized.admitted_bundle_identity;
    node["raw_parent_digest"] = realized.raw_parent_digest;
    return node;
}

inline YAML::Node science_map_realized_maps_node(
    const std::vector<mapmaking::ScienceMapRealizedMap> &realized_maps) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (std::size_t index = 0; index < realized_maps.size(); ++index) {
        node.push_back(science_map_realized_map_node(
            realized_maps.at(index), index));
    }
    return node;
}

inline YAML::Node science_map_coadd_admission_node(
    const mapmaking::ScienceMapCoaddAdmission &admission,
    std::size_t admission_index) {
    YAML::Node node;
    node["admission_index"] = admission_index;
    node["observation_id"] = admission.observation_id;
    node["embedding"]["delta_row"] = admission.delta_row;
    node["embedding"]["delta_col"] = admission.delta_col;
    node["embedding"]["registration_identity"] =
        admission.registration_identity;
    node["embedding"]["centering_identity"] =
        admission.centering_identity;
    node["observation_shape"]["rows"] = admission.observation_rows;
    node["observation_shape"]["cols"] = admission.observation_cols;
    node["coadd_shape"]["rows"] = admission.coadd_rows;
    node["coadd_shape"]["cols"] = admission.coadd_cols;
    node["ordered_map_count"] = admission.ordered_map_count;
    node["admitted_bundle_identity"] =
        admission.admitted_bundle_identity;
    node["response_identity"] = admission.response_identity;
    node["coefficient_stage"] = admission.coefficient_stage;
    node["policies"]["normalization_support"] =
        admission.normalization_support_policy;
    node["policies"]["science_policy_support"] =
        admission.science_policy_support_policy;
    node["policies"]["validity"] = admission.validity_policy;
    node["policies"]["nonfinite"] = admission.nonfinite_policy;
    node["observation_exposure_seconds"] =
        science_map_exact_double_node(
            admission.observation_exposure_seconds);
    node["numerically_contributing_pixel_count"] =
        YAML::Node(YAML::NodeType::Sequence);
    for (const auto count :
         admission.numerically_contributing_pixel_count) {
        node["numerically_contributing_pixel_count"].push_back(count);
    }
    node["observation_raw_parent_digests"] =
        science_map_string_sequence_node(
            admission.observation_raw_parent_digests);
    return node;
}

inline YAML::Node science_map_coadd_admissions_node(
    const std::vector<mapmaking::ScienceMapCoaddAdmission> &admissions) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (std::size_t index = 0; index < admissions.size(); ++index) {
        node.push_back(science_map_coadd_admission_node(
            admissions.at(index), index));
    }
    return node;
}

inline YAML::Node science_map_coefficient_contract_node() {
    YAML::Node node;
    node["product"] = science_map_coefficient_product;
    node["lifecycle_stage"] = science_map_coefficient_lifecycle_stage;
    YAML::Node allowed_stages(YAML::NodeType::Sequence);
    allowed_stages.push_back(
        mapmaking::science_map_observation_unscaled_coefficient_stage);
    allowed_stages.push_back(
        mapmaking::science_map_observation_empirical_coefficient_stage);
    allowed_stages.push_back(
        mapmaking::science_map_coadd_unscaled_coefficient_stage);
    allowed_stages.push_back(
        mapmaking::science_map_coadd_empirical_coefficient_stage);
    node["allowed_realized_stages"] = allowed_stages;
    node["policy"] = mapmaking::science_map_coefficient_policy_version;
    node["interpretation"] = science_map_coefficient_interpretation;
    node["precision_status"] = science_map_precision_status;
    node["covariance_status"] = science_map_covariance_status;
    return node;
}

inline YAML::Node science_map_policy_contract_node() {
    YAML::Node node;
    node["contract_version"] = mapmaking::science_map_contract_version;
    node["order_statistic_algorithm"] =
        mapmaking::science_map_order_statistic_version;
    node["normalization_support_algorithm"] =
        mapmaking::science_map_normalization_support_version;
    node["science_policy_support_algorithm"] =
        mapmaking::science_map_policy_support_version;
    node["validity_algorithm"] = mapmaking::science_map_validity_version;
    node["contribution_algorithm"] =
        mapmaking::science_map_ordinary_contribution_version;
    node["coadd_estimator"] =
        mapmaking::science_map_coadd_estimator_version;
    node["nonfinite_policy"] =
        mapmaking::science_map_nonfinite_policy_version;
    node["coefficient"] = science_map_coefficient_contract_node();
    return node;
}

}  // namespace citlali::pipeline
