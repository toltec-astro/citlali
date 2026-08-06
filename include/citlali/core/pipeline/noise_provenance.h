#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/noise_config_serialization.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *noise_provenance_schema_version =
    "citlali-noise-products-provenance-v1";
inline constexpr const char *noise_provenance_filename =
    noise_provenance_join_id;

inline YAML::Node noise_provenance_node(
    const NoiseExecutionPlan &plan) {
    YAML::Node root;
    root["schema_version"] = noise_provenance_schema_version;
    root["package"]["package_id"] = noise_package_id;
    root["package"]["provenance_id"] = noise_provenance_join_id;
    root["package"]["product_contract_version"] =
        noise_product_contract_version;
    root["package"]["authority"] = "package_sidecar";
    root["package"]["detached_product_status"] =
        "unverified_out_of_contract";

    YAML::Node inventory{YAML::NodeType::Sequence};
    const auto add_product = [&](const char *identity, const char *scope,
                                 const char *restriction) {
        YAML::Node product;
        product["product_identity"] = identity;
        product["product_version"] = noise_product_contract_version;
        product["semantic_digest"] =
            noise_product_semantic_digest(identity);
        product["digest_kind"] = "semantic_contract_sha256";
        product["scope"] = scope;
        product["restriction"] = restriction;
        inventory.push_back(product);
    };
    add_product(
        noise_conditional_stack_scatter_product_id, "map_pixel",
        "conditional_completed_stack_descriptive_not_physical_noise_variance_or_covariance");
    add_product(
        noise_formal_coefficient_product_id, "map_pixel",
        "pre_scale_nonprecision_coefficient_not_inverse_variance_or_precision");
    add_product(
        noise_scaled_coefficient_product_id, "map_pixel",
        "existing_use_only_nonprecision_not_inverse_variance_or_precision");
    add_product(
        noise_coefficient_standardized_signal_product_id, "map_pixel",
        "engineering_standardization_not_significance");
    add_product(
        noise_filtered_pixel_stack_scatter_product_id, "filtered_map_pixel",
        "conditional_diagnostic_strict_operator_edge_parity_pending_FLT");
    add_product(
        noise_stack_scatter_ratio_product_id, "filtered_map_pixel",
        "descriptive_ratio_not_significance_positive_finite_denominator_required");
    add_product(
        noise_realization_product_id, "realization_map",
        "source_imprinted_current_conditional_design_member");
    add_product(
        noise_pooled_stack_scale_product_id, "map_summary",
        "engineering_scale_diagnostic_not_significance");
    add_product(
        noise_source_finder_score_product_id, "source_finder",
        "existing_quicklook_engineering_score_not_significance");
    add_product(
        noise_fitted_amplitude_rms_ratio_product_id, "source_table",
        "fitted_amplitude_over_full_map_rms_not_significance");
    add_product(
        noise_fixed_projection_scatter_product_id, "fixed_linear_projection",
        "conditional_finite_stack_diagnostic_not_aperture_uncertainty");
    root["package"]["product_contract_inventory"] = inventory;
    root["initialized"] = plan.initialized;
    root["requested"] = noise_config_node(plan.requested);
    root["effective"]["config"] = noise_config_node(plan.effective);
    root["effective"]["resolution"] =
        noise_effective_resolution_node(plan.effective_resolution);
    root["realized"] = noise_realized_state_node(plan.realized);
    root["semantics"]["ensemble_identity"] = noise_ensemble_identity;
    root["semantics"]["estimator_identity"] = noise_estimator_identity;
    root["semantics"]["target"] = noise_estimator_target;
    root["semantics"]["centering"] = noise_centering_identity;
    root["semantics"]["normalization"] = noise_normalization_identity;
    root["semantics"]["support"] = noise_support_identity;
    root["semantics"]["missingness"] = noise_missingness_identity;
    root["semantics"]["realization_identity"] =
        "zero_based_ordinal_within_completed_stack";
    root["semantics"]["dependence_correction"] = "none";
    root["semantics"]["covariance_domain"] =
        "no_covariance_product_pixelwise_scatter_and_fixed_scalar_projections_only";
    root["semantics"]["rank_status"] = "not_estimated";
    root["semantics"]["support_count_policy"] =
        "uniform_completed_R_at_package_scope_no_per_pixel_counts";
    root["semantics"]["uncertainty_use_requires_completed_R_at_least"] = 2;
    root["semantics"]["prohibited_interpretations"] =
        "physical_noise_variance_covariance_precision_significance_aperture_uncertainty_calibration";
    root["global_nonprecision_scale_diagnostic"]["identity"] =
        noise_scale_diagnostic_identity;
    root["global_nonprecision_scale_diagnostic"]["formula"] =
        noise_scale_diagnostic_formula;
    root["global_nonprecision_scale_diagnostic"]["calibration_region"] =
        "finite_positive_q_p_and_V_p_at_or_above_realized_weight_support_threshold";
    root["global_nonprecision_scale_diagnostic"]["overlap"] =
        "same_completed_stack";
    root["global_nonprecision_scale_diagnostic"]["application"] =
        "existing_use_only";
    root["global_nonprecision_scale_diagnostic"]["precision_status"] =
        "not_established";
    root["filter_operator_parity_gate"]["status"] =
        noise_filter_parity_gate_status;
    root["filter_operator_parity_gate"]["reason"] =
        noise_filter_parity_gate_reason;
    root["filter_operator_parity_gate"]["finding_disposition"] =
        "F005_open_conditioned_on_FLT_repair_and_reaudit";
    return root;
}

inline std::filesystem::path noise_provenance_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / noise_provenance_filename;
}

inline void write_noise_provenance_file(
    const std::filesystem::path &reduction_dir,
    const NoiseExecutionPlan &plan) {
    if (!plan.initialized || !plan.realized.reduction_completed) {
        throw std::logic_error(
            "cannot write incomplete noise-products provenance");
    }
    auto node = noise_provenance_node(plan);
    YAML::Node members{YAML::NodeType::Sequence};
    std::vector<std::filesystem::path> paths;
    if (std::filesystem::is_directory(reduction_dir)) {
        for (const auto &entry :
             std::filesystem::recursive_directory_iterator(reduction_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const auto extension = entry.path().extension().string();
            if (extension == ".fits" || extension == ".ecsv" ||
                extension == ".nc") {
                paths.push_back(entry.path());
            }
        }
    }
    std::sort(paths.begin(), paths.end());

    std::string canonical_inventory;
    for (const auto &path : paths) {
        const auto relative =
            std::filesystem::relative(path, reduction_dir).generic_string();
        const auto digest = citlali::utils::sha256_file(path);
        const auto size = std::filesystem::file_size(path);
        YAML::Node member;
        member["member_product_identity"] = relative;
        member["sha256"] = digest;
        member["size_bytes"] = size;
        member["digest_kind"] = "file_sha256";
        member["detached_status"] =
            "unverified_out_of_contract_without_package";
        members.push_back(member);
        canonical_inventory += relative + "\n" + digest + "\n" +
                               std::to_string(size) + "\n";
    }
    node["package"]["member_files"] = members;
    const auto member_inventory_digest =
        "sha256:" + citlali::utils::sha256(canonical_inventory);
    node["package"]["member_inventory_digest"] =
        member_inventory_digest;
    node["package"]["member_inventory_digest_kind"] =
        "canonical_relative_path_file_sha256_size_v1";
    write_yaml_file_atomic(noise_provenance_path(reduction_dir), node);
}

}  // namespace citlali::pipeline
