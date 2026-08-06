#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/noise_config_serialization.h>

#include <fitsio.h>
#include <netcdf>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <ios>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *noise_provenance_schema_version =
    "citlali-noise-products-provenance-v1";
inline constexpr const char *noise_provenance_filename =
    noise_provenance_join_id;

struct NoiseProductContractDefinition {
    const char *identity;
    const char *scope;
    const char *restriction;
};

inline constexpr std::array<NoiseProductContractDefinition, 12>
    noise_product_contract_definitions{{
        {noise_conditional_stack_scatter_product_id, "map_pixel",
         "conditional_completed_stack_descriptive_not_physical_noise_variance_or_covariance"},
        {noise_formal_coefficient_product_id, "map_pixel",
         "pre_scale_nonprecision_coefficient_not_inverse_variance_or_precision"},
        {noise_scaled_coefficient_product_id, "map_pixel",
         "existing_use_only_nonprecision_not_inverse_variance_or_precision"},
        {noise_coefficient_standardized_signal_product_id, "map_pixel",
         "engineering_standardization_not_significance"},
        {noise_filtered_pixel_stack_scatter_product_id,
         "filtered_map_pixel",
         "conditional_diagnostic_strict_operator_edge_parity_pending_FLT"},
        {noise_stack_scatter_ratio_product_id, "filtered_map_pixel",
         "descriptive_ratio_not_significance_positive_finite_denominator_required"},
        {noise_realization_product_id, "realization_map",
         "source_imprinted_current_conditional_design_member"},
        {noise_pooled_stack_scale_product_id, "map_summary",
         "engineering_scale_diagnostic_not_significance"},
        {noise_scale_diagnostic_identity, "map_summary",
         "engineering_scale_diagnostic_not_precision_or_significance"},
        {noise_source_finder_score_product_id, "source_finder",
         "existing_quicklook_engineering_score_not_significance"},
        {noise_fitted_amplitude_rms_ratio_product_id, "source_table",
         "fitted_amplitude_over_full_map_rms_not_significance"},
        {noise_fixed_projection_scatter_product_id,
         "fixed_linear_projection",
         "conditional_finite_stack_diagnostic_not_aperture_uncertainty"},
    }};

inline bool noise_product_identity_is_declared(
    std::string_view identity) {
    return std::any_of(
        noise_product_contract_definitions.begin(),
        noise_product_contract_definitions.end(),
        [&](const auto &definition) {
            return identity == definition.identity;
        });
}

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
    for (const auto &definition : noise_product_contract_definitions) {
        YAML::Node product;
        product["product_identity"] = definition.identity;
        product["product_version"] = noise_product_contract_version;
        product["semantic_digest"] =
            noise_product_semantic_digest(definition.identity);
        product["digest_kind"] = "semantic_contract_sha256";
        product["scope"] = definition.scope;
        product["restriction"] = definition.restriction;
        inventory.push_back(product);
    }
    root["package"]["product_contract_inventory"] = inventory;
    root["initialized"] = plan.initialized;
    root["requested"] = noise_config_node(plan.requested);
    root["effective"]["config"] = noise_config_node(plan.effective);
    root["effective"]["resolution"] =
        noise_effective_resolution_node(plan.effective_resolution);
    root["expected"] = noise_expected_counts_node(plan.expected);
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

inline void begin_noise_product_publication(
    const std::filesystem::path &reduction_dir,
    NoiseExecutionPlan &plan) {
    if (!std::filesystem::is_directory(reduction_dir)) {
        throw std::ios_base::failure(
            "noise publication root is not an existing directory: " +
            reduction_dir.string());
    }
    const auto final_path = noise_provenance_path(reduction_dir);
    const std::vector<std::filesystem::path> stale_authorities{
        final_path,
        std::filesystem::path{final_path.string() + ".tmp"},
        std::filesystem::path{final_path.string() + ".pending"},
        std::filesystem::path{final_path.string() + ".pending.tmp"}};
    for (const auto &path : stale_authorities) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        if (ec) {
            throw std::ios_base::failure(
                "failed to invalidate stale noise package authority " +
                path.string() + ": " + ec.message());
        }
    }
    begin_noise_run_publication(plan);
}

inline std::optional<std::string> read_noise_fits_string_key(
    fitsfile *fits, const char *key) {
    char *raw_value = nullptr;
    int status = 0;
    fits_read_key_longstr(fits, key, &raw_value, nullptr, &status);
    if (status == KEY_NO_EXIST) {
        return std::nullopt;
    }
    if (status != 0) {
        char message[FLEN_STATUS]{};
        fits_get_errstatus(status, message);
        throw std::runtime_error(
            std::string{"failed to read FITS noise join key "} + key +
            ": " + message);
    }
    const std::string value = raw_value == nullptr ? "" : raw_value;
    int free_status = 0;
    if (raw_value != nullptr) {
        fits_free_memory(raw_value, &free_status);
    }
    if (free_status != 0) {
        throw std::runtime_error(
            std::string{"failed to release FITS key storage for "} + key);
    }
    return value;
}

struct NoiseMemberJoinValidation {
    std::vector<std::string> product_identities;
    std::size_t realization_image_count = 0;
};

inline NoiseMemberJoinValidation validate_noise_fits_joins(
    const std::filesystem::path &path) {
    fitsfile *fits = nullptr;
    int status = 0;
    fits_open_file(&fits, path.string().c_str(), READONLY, &status);
    if (status != 0) {
        char message[FLEN_STATUS]{};
        fits_get_errstatus(status, message);
        throw std::runtime_error(
            "failed to open admitted FITS noise member " + path.string() +
            ": " + message);
    }

    NoiseMemberJoinValidation validation;
    try {
        int hdu_count = 0;
        fits_get_num_hdus(fits, &hdu_count, &status);
        if (status != 0) {
            throw std::runtime_error(
                "failed to enumerate admitted FITS noise member " +
                path.string());
        }
        std::set<std::string> identities;
        for (int hdu_index = 1; hdu_index <= hdu_count; ++hdu_index) {
            int hdu_type = 0;
            fits_movabs_hdu(fits, hdu_index, &hdu_type, &status);
            if (status != 0) {
                throw std::runtime_error(
                    "failed to select FITS HDU in admitted noise member " +
                    path.string());
            }
            const auto package = read_noise_fits_string_key(fits, "NOIPKG");
            const auto provenance =
                read_noise_fits_string_key(fits, "NOIPROV");
            const auto identity =
                read_noise_fits_string_key(fits, "NOIPRID");
            const auto version =
                read_noise_fits_string_key(fits, "NOIPVER");
            const auto digest =
                read_noise_fits_string_key(fits, "NOIDGST");
            const auto scope =
                read_noise_fits_string_key(fits, "NOISCOPE");
            const auto validity =
                read_noise_fits_string_key(fits, "NOIVALID");
            const auto restriction =
                read_noise_fits_string_key(fits, "NOIRESTR");
            const bool any_join = package || provenance || identity ||
                version || digest || scope || validity || restriction;
            if (!any_join) {
                continue;
            }
            if (!package || !provenance || !identity || !version ||
                !digest || !scope || !validity || !restriction) {
                throw std::runtime_error(
                    "partial FITS noise-product join in " + path.string());
            }
            if (*package != noise_package_id ||
                *provenance != noise_provenance_join_id ||
                *version != noise_product_contract_version ||
                *digest != noise_product_semantic_digest(*identity) ||
                !noise_product_identity_is_declared(*identity) ||
                scope->empty() || validity->empty() ||
                restriction->empty()) {
                throw std::runtime_error(
                    "invalid FITS noise-product join in " + path.string());
            }
            identities.insert(*identity);
            if (*identity == noise_realization_product_id) {
                ++validation.realization_image_count;
            }
        }
        if (identities.empty()) {
            throw std::runtime_error(
                "admitted FITS member has no noise-product join: " +
                path.string());
        }
        validation.product_identities.assign(
            identities.begin(), identities.end());
    }
    catch (...) {
        int close_status = 0;
        fits_close_file(fits, &close_status);
        throw;
    }
    status = 0;
    fits_close_file(fits, &status);
    if (status != 0) {
        throw std::runtime_error(
            "failed to close admitted FITS noise member " + path.string());
    }
    return validation;
}

inline NoiseMemberJoinValidation validate_noise_ecsv_join(
    const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::ios_base::failure(
            "failed to open admitted ECSV noise member " + path.string());
    }
    std::ostringstream header;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.front() != '#') {
            break;
        }
        header << line << '\n';
    }
    const std::string text = header.str();
    const std::vector<std::string> required{
        "noise_product_contract", noise_package_id,
        noise_provenance_join_id,
        noise_fitted_amplitude_rms_ratio_product_id,
        noise_product_contract_version,
        noise_product_semantic_digest(
            noise_fitted_amplitude_rms_ratio_product_id),
        "source_table_row",
        "finite_amplitude_and_finite_positive_full_map_rms",
        "legacy_alias_deprecated_not_significance"};
    for (const auto &token : required) {
        if (text.find(token) == std::string::npos) {
            throw std::runtime_error(
                "invalid ECSV noise-product join in " + path.string() +
                ": missing " + token);
        }
    }
    return {{noise_fitted_amplitude_rms_ratio_product_id}, 0};
}

inline NoiseMemberJoinValidation validate_noise_netcdf_joins(
    const std::filesystem::path &path) {
    struct VariableContract {
        const char *name;
        const char *identity;
        const char *validity;
        const char *restriction;
    };
    const std::vector<VariableContract> variables{
        {"map_noise_weight_median_ratio",
         noise_scale_diagnostic_identity,
         "available_when_finite_positive_calibration_support_exists",
         "engineering_scale_diagnostic_not_precision_or_significance"},
        {"map_noise_weight_scale", noise_scale_diagnostic_identity,
         "available_when_finite_positive_median_ratio_exists",
         "nonprecision_scale_not_inverse_variance_or_precision"},
        {"map_noise_products_s2n_sigma",
         noise_pooled_stack_scale_product_id,
         "available_when_finite_pooled_stack_scale_exists",
         "engineering_scale_diagnostic_not_calibrated_significance"}};
    std::set<std::string> identities;
    try {
        netCDF::NcFile file(path.string(), netCDF::NcFile::read);
        for (const auto &contract : variables) {
            const auto variable = file.getVar(contract.name);
            if (variable.isNull()) {
                throw std::runtime_error(
                    "missing NetCDF noise-contract variable " +
                    std::string{contract.name});
            }
            const auto comment_attribute = variable.getAtt("comment");
            if (comment_attribute.isNull()) {
                throw std::runtime_error(
                    "missing NetCDF noise-contract comment for " +
                    std::string{contract.name});
            }
            std::string comment;
            comment_attribute.getValues(comment);
            const std::vector<std::string> required{
                std::string{"package_id="} + noise_package_id,
                std::string{"provenance_id="} + noise_provenance_join_id,
                std::string{"product_identity="} + contract.identity,
                std::string{"product_version="} +
                    noise_product_contract_version,
                "scope=map_summary",
                std::string{"validity="} + contract.validity,
                std::string{"restriction="} + contract.restriction};
            for (const auto &token : required) {
                if (comment.find(token) == std::string::npos) {
                    throw std::runtime_error(
                        "invalid NetCDF noise-product join for " +
                        std::string{contract.name} + " in " +
                        path.string() + ": missing " + token);
                }
            }
            identities.insert(contract.identity);
        }
    }
    catch (const netCDF::exceptions::NcException &error) {
        throw std::runtime_error(
            "failed to validate admitted NetCDF noise member " +
            path.string() + ": " + error.what());
    }
    NoiseMemberJoinValidation validation;
    validation.product_identities.assign(
        identities.begin(), identities.end());
    return validation;
}

struct ValidatedNoiseMember {
    std::filesystem::path path;
    std::string relative_path;
    NoisePublishedMemberKind kind = NoisePublishedMemberKind::fits;
    std::vector<std::string> product_identities;
    std::string sha256;
    std::uintmax_t size_bytes = 0;
    std::size_t realization_image_count = 0;
};

inline std::vector<ValidatedNoiseMember> validate_noise_member_inventory(
    const std::filesystem::path &reduction_dir,
    const std::vector<NoisePublishedMember> &members) {
    std::error_code ec;
    const auto root = std::filesystem::canonical(reduction_dir, ec);
    if (ec || !std::filesystem::is_directory(root)) {
        throw std::ios_base::failure(
            "noise publication root is unavailable: " +
            reduction_dir.string());
    }

    std::set<std::string> unique_paths;
    std::vector<ValidatedNoiseMember> validated;
    validated.reserve(members.size());
    for (const auto &member : members) {
        ec.clear();
        const auto candidate = member.path.is_absolute()
            ? member.path.lexically_normal()
            : std::filesystem::absolute(member.path, ec).lexically_normal();
        if (ec) {
            throw std::runtime_error(
                "cannot resolve noise package member path: " +
                member.path.string());
        }
        ec.clear();
        const auto status = std::filesystem::symlink_status(candidate, ec);
        if (ec || status.type() == std::filesystem::file_type::not_found ||
            std::filesystem::is_symlink(status) ||
            !std::filesystem::is_regular_file(status)) {
            throw std::runtime_error(
                "noise package member is missing, non-regular, or a symlink: " +
                candidate.string());
        }
        ec.clear();
        const auto canonical = std::filesystem::canonical(candidate, ec);
        if (ec) {
            throw std::runtime_error(
                "cannot canonicalize noise package member: " +
                candidate.string());
        }
        const auto relative = canonical.lexically_relative(root);
        bool outside_root = relative.empty() || relative.is_absolute();
        for (const auto &component : relative) {
            if (component == ".." || component == ".") {
                outside_root = true;
            }
        }
        if (outside_root) {
            throw std::runtime_error(
                "noise package member is outside the reduction root: " +
                candidate.string());
        }
        const std::string relative_name = relative.generic_string();
        if (!unique_paths.insert(relative_name).second) {
            throw std::runtime_error(
                "duplicate noise package member: " + relative_name);
        }
        const auto extension = canonical.extension().string();
        const char *expected_extension = nullptr;
        switch (member.kind) {
            case NoisePublishedMemberKind::fits:
                expected_extension = ".fits";
                break;
            case NoisePublishedMemberKind::ecsv:
                expected_extension = ".ecsv";
                break;
            case NoisePublishedMemberKind::netcdf:
                expected_extension = ".nc";
                break;
        }
        if (extension != expected_extension) {
            throw std::runtime_error(
                "noise package member kind/extension mismatch: " +
                relative_name);
        }

        NoiseMemberJoinValidation joins;
        switch (member.kind) {
            case NoisePublishedMemberKind::fits:
                joins = validate_noise_fits_joins(canonical);
                break;
            case NoisePublishedMemberKind::ecsv:
                joins = validate_noise_ecsv_join(canonical);
                break;
            case NoisePublishedMemberKind::netcdf:
                joins = validate_noise_netcdf_joins(canonical);
                break;
        }
        validated.push_back(
            {canonical, relative_name, member.kind,
             std::move(joins.product_identities),
             citlali::utils::sha256_file(canonical),
             std::filesystem::file_size(canonical),
             joins.realization_image_count});
    }
    std::sort(
        validated.begin(), validated.end(),
        [](const auto &left, const auto &right) {
            return left.relative_path < right.relative_path;
        });
    return validated;
}

inline bool validated_noise_members_match(
    const std::vector<ValidatedNoiseMember> &left,
    const std::vector<ValidatedNoiseMember> &right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].relative_path != right[index].relative_path ||
            left[index].kind != right[index].kind ||
            left[index].product_identities !=
                right[index].product_identities ||
            left[index].sha256 != right[index].sha256 ||
            left[index].size_bytes != right[index].size_bytes ||
            left[index].realization_image_count !=
                right[index].realization_image_count) {
            return false;
        }
    }
    return true;
}

inline void write_noise_provenance_file(
    const std::filesystem::path &reduction_dir,
    const NoiseExecutionPlan &plan) {
    if (!plan.initialized || !plan.publication_started ||
        !plan.expected.initialized ||
        !plan.realized.reduction_completed ||
        !plan.realized.actual_completion_valid ||
        !plan.realized.completed_count_matches_effective ||
        !plan.realized.outputs_completed) {
        throw std::logic_error(
            "cannot write incomplete noise-products provenance");
    }
    const auto validated = validate_noise_member_inventory(
        reduction_dir, plan.published_members);
    std::size_t realization_image_count = 0;
    for (const auto &member : validated) {
        if (realization_image_count >
            std::numeric_limits<std::size_t>::max() -
                member.realization_image_count) {
            throw std::overflow_error(
                "noise realization-image inventory count overflow");
        }
        realization_image_count += member.realization_image_count;
    }
    if (!plan.realized.realization_image_write_count ||
        realization_image_count !=
            *plan.realized.realization_image_write_count) {
        throw std::logic_error(
            "noise realization FITS inventory does not match observed writes");
    }

    auto node = noise_provenance_node(plan);
    YAML::Node members{YAML::NodeType::Sequence};
    std::string canonical_inventory;
    for (const auto &validated_member : validated) {
        YAML::Node member;
        member["member_product_identity"] =
            validated_member.relative_path;
        member["member_kind"] = noise_published_member_kind_name(
            validated_member.kind);
        YAML::Node identities{YAML::NodeType::Sequence};
        for (const auto &identity :
             validated_member.product_identities) {
            identities.push_back(identity);
        }
        member["joined_product_identities"] = identities;
        member["sha256"] = validated_member.sha256;
        member["size_bytes"] = validated_member.size_bytes;
        member["digest_kind"] = "file_sha256";
        member["detached_status"] =
            "unverified_out_of_contract_without_package";
        members.push_back(member);
        canonical_inventory += validated_member.relative_path + "\n" +
            validated_member.sha256 + "\n" +
            std::to_string(validated_member.size_bytes) + "\n";
    }
    node["package"]["member_files"] = members;
    node["package"]["member_count"] = validated.size();
    const auto member_inventory_digest =
        "sha256:" + citlali::utils::sha256(canonical_inventory);
    node["package"]["member_inventory_digest"] =
        member_inventory_digest;
    node["package"]["member_inventory_digest_kind"] =
        "canonical_relative_path_file_sha256_size_v1";
    node["package"]["publication_state"] = "complete";
    node["package"]["complete"] = true;

    const auto final_path = noise_provenance_path(reduction_dir);
    const std::filesystem::path pending_path{
        final_path.string() + ".pending"};
    try {
        write_yaml_file_atomic(pending_path, node);
        const auto revalidated = validate_noise_member_inventory(
            reduction_dir, plan.published_members);
        if (!validated_noise_members_match(validated, revalidated)) {
            throw std::runtime_error(
                "noise package member changed during final publication");
        }
        std::error_code ec;
        std::filesystem::rename(pending_path, final_path, ec);
        if (ec) {
            throw std::ios_base::failure(
                "failed to atomically publish noise package authority: " +
                ec.message());
        }
    }
    catch (...) {
        std::error_code ec;
        std::filesystem::remove(pending_path, ec);
        ec.clear();
        std::filesystem::remove(
            std::filesystem::path{pending_path.string() + ".tmp"}, ec);
        throw;
    }
}

}  // namespace citlali::pipeline
