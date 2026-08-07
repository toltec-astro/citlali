#pragma once

#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/noise_config_serialization.h>

#include <fitsio.h>
#include <netcdf>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <limits>
#include <map>
#include <optional>
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
inline constexpr const char *noise_semantic_digest_kind =
    "semantic_contract_sha256";
inline constexpr const char *noise_compact_missingness =
    "nonfinite_unavailable";
inline constexpr const char *noise_netcdf_join_schema =
    "citlali_noise_product_join_v1";
inline constexpr const char *noise_member_inventory_digest_kind =
    "sha256";
inline constexpr const char *noise_member_inventory_preimage_encoding =
    "canonical_length_prefixed_member_records_v2";

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

inline bool noise_join_value_is_one_of(
    std::string_view value,
    std::initializer_list<std::string_view> allowed) {
    return std::find(allowed.begin(), allowed.end(), value) !=
        allowed.end();
}

inline bool noise_realization_scope_is_canonical(
    std::string_view scope) {
    constexpr std::string_view prefix{"realization_map_index_"};
    if (!scope.starts_with(prefix)) {
        return false;
    }
    const auto ordinal = scope.substr(prefix.size());
    if (ordinal.empty() ||
        (ordinal.size() > 1 && ordinal.front() == '0')) {
        return false;
    }
    return std::all_of(
        ordinal.begin(), ordinal.end(), [](unsigned char value) {
            return std::isdigit(value) != 0;
        });
}

inline bool noise_fits_join_matches_contract(
    std::string_view identity, std::string_view scope,
    std::string_view validity, std::string_view restriction,
    std::string_view missingness) {
    if (missingness != noise_compact_missingness) {
        return false;
    }
    const bool map_scope = noise_join_value_is_one_of(
        scope, {"raw_map_pixel", "filtered_map_pixel"});
    if (identity == noise_formal_coefficient_product_id) {
        return map_scope && validity == "available" &&
            restriction ==
                "nonprecision_snapshot_not_inverse_variance";
    }
    if (identity == noise_scaled_coefficient_product_id) {
        return map_scope &&
            noise_join_value_is_one_of(
                validity, {"available", "unavailable"}) &&
            restriction ==
                "existing_use_only_nonprecision_not_precision";
    }
    if (identity == noise_conditional_stack_scatter_product_id) {
        if (!noise_join_value_is_one_of(
                validity, {"conditional_descriptive", "unavailable"})) {
            return false;
        }
        return (scope == "raw_map_pixel" &&
                restriction ==
                    "retained_legacy_name_not_physical_noise_variance_or_covariance") ||
            (scope == "filtered_map_pixel" &&
             restriction ==
                 "retained_legacy_name_not_physical_noise_variance_strict_parity_pending_FLT");
    }
    if (identity == noise_coefficient_standardized_signal_product_id) {
        return map_scope &&
            noise_join_value_is_one_of(
                validity, {"available_where_finite", "unavailable"}) &&
            restriction ==
                "retained_legacy_name_engineering_standardization_not_significance";
    }
    if (identity == noise_filtered_pixel_stack_scatter_product_id) {
        return scope == "filtered_map_pixel" &&
            noise_join_value_is_one_of(
                validity,
                {"available_where_finite_on_valid_support", "R_lt_2",
                 "scatter_unavailable_or_nonfinite", "response_invalid",
                 "support_invalid"}) &&
            restriction ==
                "retained_legacy_name_not_aperture_uncertainty_strict_parity_pending_FLT";
    }
    if (identity == noise_stack_scatter_ratio_product_id) {
        return scope == "filtered_map_pixel" &&
            noise_join_value_is_one_of(
                validity,
                {"available_where_finite_positive_denominator_on_valid_support",
                 "R_lt_2", "scatter_unavailable_or_nonfinite",
                 "response_invalid", "support_invalid"}) &&
            restriction ==
                "retained_legacy_name_conditional_descriptive_ratio_not_significance";
    }
    if (identity == noise_realization_product_id) {
        return noise_realization_scope_is_canonical(scope) &&
            validity == "conditional_design_member" &&
            restriction ==
                "source_imprinted_current_not_physical_noise_repeat";
    }
    return false;
}

inline bool noise_fits_identity_is_empirical_map_product(
    std::string_view identity) {
    return identity == noise_conditional_stack_scatter_product_id ||
        identity == noise_formal_coefficient_product_id ||
        identity == noise_scaled_coefficient_product_id ||
        identity == noise_coefficient_standardized_signal_product_id ||
        identity == noise_filtered_pixel_stack_scatter_product_id ||
        identity == noise_stack_scatter_ratio_product_id;
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
        product["digest_kind"] = noise_semantic_digest_kind;
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
    std::size_t empirical_map_product_count = 0;
    bool contains_stack_derived_product = false;
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
        std::map<std::string, std::size_t> identity_counts;
        std::optional<std::string> empirical_map_scope;
        std::set<std::string> realization_scopes;
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
            const auto digest_kind =
                read_noise_fits_string_key(fits, "NOIDGKND");
            const auto scope =
                read_noise_fits_string_key(fits, "NOISCOPE");
            const auto validity =
                read_noise_fits_string_key(fits, "NOIVALID");
            const auto restriction =
                read_noise_fits_string_key(fits, "NOIRESTR");
            const auto missingness =
                read_noise_fits_string_key(fits, "NOIMISS");
            const bool any_join = package || provenance || identity ||
                version || digest || digest_kind || scope || validity ||
                restriction || missingness;
            if (!any_join) {
                continue;
            }
            if (!package || !provenance || !identity || !version ||
                !digest || !digest_kind || !scope || !validity ||
                !restriction || !missingness) {
                throw std::runtime_error(
                    "partial FITS noise-product join in " + path.string());
            }
            if (*package != noise_package_id ||
                *provenance != noise_provenance_join_id ||
                *version != noise_product_contract_version ||
                *digest != noise_product_semantic_digest(*identity) ||
                *digest_kind != noise_semantic_digest_kind ||
                !noise_product_identity_is_declared(*identity) ||
                !noise_fits_join_matches_contract(
                    *identity, *scope, *validity, *restriction,
                    *missingness)) {
                throw std::runtime_error(
                    "invalid FITS noise-product join in " + path.string());
            }
            auto &identity_count = identity_counts[*identity];
            ++identity_count;
            if (*identity != noise_realization_product_id &&
                identity_count > 1) {
                throw std::runtime_error(
                    "duplicate non-realization FITS noise-product identity " +
                    *identity + " in " + path.string());
            }
            if (noise_fits_identity_is_empirical_map_product(*identity)) {
                if (!empirical_map_scope) {
                    empirical_map_scope = *scope;
                }
                else if (*empirical_map_scope != *scope) {
                    throw std::runtime_error(
                        "mixed empirical FITS noise-product scopes in " +
                        path.string());
                }
            }
            if (*identity == noise_realization_product_id) {
                if (!realization_scopes.insert(*scope).second) {
                    throw std::runtime_error(
                        "duplicate FITS noise-realization scope " +
                        *scope + " in " + path.string());
                }
                ++validation.realization_image_count;
            }
        }
        if (identity_counts.empty()) {
            throw std::runtime_error(
                "admitted FITS member has no noise-product join: " +
                path.string());
        }
        for (std::size_t ordinal = 0;
             ordinal < validation.realization_image_count; ++ordinal) {
            if (!realization_scopes.contains(
                    "realization_map_index_" +
                    std::to_string(ordinal))) {
                throw std::runtime_error(
                    "noncanonical FITS noise-realization scope sequence in " +
                    path.string());
            }
        }
        const bool empirical_member = std::any_of(
            identity_counts.begin(), identity_counts.end(),
            [](const auto &entry) {
                return noise_fits_identity_is_empirical_map_product(
                    entry.first);
            });
        if (empirical_member) {
            const auto identity_count = [&](const char *identity_name) {
                const auto item = identity_counts.find(identity_name);
                return item == identity_counts.end()
                    ? std::size_t{0}
                    : item->second;
            };
            if (validation.realization_image_count != 0 ||
                identity_count(noise_formal_coefficient_product_id) != 1 ||
                identity_count(
                    noise_conditional_stack_scatter_product_id) != 1 ||
                identity_count(
                    noise_filtered_pixel_stack_scatter_product_id) !=
                    identity_count(noise_stack_scatter_ratio_product_id)) {
                throw std::runtime_error(
                    "incomplete or mixed empirical FITS noise-product bundle in " +
                    path.string());
            }
            validation.empirical_map_product_count = 1;
        }
        validation.contains_stack_derived_product = true;
        validation.product_identities.reserve(identity_counts.size());
        for (const auto &[identity_name, count] : identity_counts) {
            static_cast<void>(count);
            validation.product_identities.push_back(identity_name);
        }
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
    std::ostringstream yaml_header;
    std::string line;
    std::string data_header;
    bool first_line = true;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.front() != '#') {
            data_header = line;
            break;
        }
        if (first_line) {
            if (line != "# %ECSV 1.0") {
                throw std::runtime_error(
                    "invalid ECSV noise-product header in " +
                    path.string());
            }
            first_line = false;
            continue;
        }
        if (line.starts_with("# ")) {
            yaml_header << line.substr(2) << '\n';
        }
        else if (line == "#") {
            yaml_header << '\n';
        }
        else {
            throw std::runtime_error(
                "invalid ECSV noise-product header line in " +
                path.string());
        }
    }
    if (first_line) {
        throw std::runtime_error(
            "empty ECSV noise-product member " + path.string());
    }

    YAML::Node document;
    try {
        document = YAML::Load(yaml_header.str());
    }
    catch (const YAML::Exception &error) {
        throw std::runtime_error(
            "invalid structured ECSV noise-product header in " +
            path.string() + ": " + error.what());
    }
    const auto datatype = document["datatype"];
    if (!datatype.IsSequence()) {
        throw std::runtime_error(
            "invalid ECSV noise-product datatype binding in " +
            path.string());
    }
    std::size_t bound_column_count = 0;
    for (const auto &column : datatype) {
        if (column.IsMap() && column["name"] &&
            column["name"].as<std::string>() == "sig2noise") {
            ++bound_column_count;
        }
    }
    std::istringstream data_columns{data_header};
    std::size_t data_column_count = 0;
    std::string data_column;
    while (data_columns >> data_column) {
        if (data_column == "sig2noise") {
            ++data_column_count;
        }
    }
    if (bound_column_count != 1 || data_column_count != 1) {
        throw std::runtime_error(
            "invalid ECSV noise-product column binding in " +
            path.string());
    }

    const auto meta = document["meta"];
    if (!meta.IsSequence()) {
        throw std::runtime_error(
            "admitted ECSV member has no structured noise-product join: " +
            path.string());
    }
    YAML::Node contract;
    std::size_t contract_count = 0;
    for (const auto &entry : meta) {
        if (entry.IsMap() && entry["noise_product_contract"]) {
            contract = entry["noise_product_contract"];
            ++contract_count;
        }
    }
    const std::map<std::string, std::string> expected{
        {"package_id", noise_package_id},
        {"provenance_id", noise_provenance_join_id},
        {"column", "sig2noise"},
        {"product_identity",
         noise_fitted_amplitude_rms_ratio_product_id},
        {"product_version", noise_product_contract_version},
        {"semantic_digest",
         noise_product_semantic_digest(
             noise_fitted_amplitude_rms_ratio_product_id)},
        {"digest_kind", noise_semantic_digest_kind},
        {"scope", "source_table_row"},
        {"validity",
         "finite_amplitude_and_finite_positive_full_map_rms"},
        {"restriction",
         "legacy_alias_deprecated_not_significance"}};
    if (contract_count != 1 || !contract.IsMap() ||
        contract.size() != expected.size()) {
        throw std::runtime_error(
            "invalid ECSV noise-product join in " + path.string());
    }
    for (const auto &[key, value] : expected) {
        if (!contract[key] || !contract[key].IsScalar() ||
            contract[key].as<std::string>() != value) {
            throw std::runtime_error(
                "invalid ECSV noise-product join in " + path.string() +
                ": inconsistent " + key);
        }
    }
    return {{noise_fitted_amplitude_rms_ratio_product_id}, 0, 0, false};
}

inline void validate_noise_netcdf_join_comment(
    const std::string &comment, std::string_view variable,
    std::string_view identity, std::string_view semantic_digest,
    std::string_view validity, std::string_view restriction) {
    const std::string marker =
        std::string{"; "} + noise_netcdf_join_schema + "|";
    const auto marker_pos = comment.find(marker);
    if (marker_pos == std::string::npos || marker_pos == 0 ||
        comment.find(marker, marker_pos + marker.size()) !=
            std::string::npos) {
        throw std::runtime_error(
            "missing or duplicate structured NetCDF noise-product join");
    }
    const auto record = comment.substr(marker_pos + 2);
    std::vector<std::string> tokens;
    std::size_t begin = 0;
    while (begin <= record.size()) {
        const auto end = record.find('|', begin);
        tokens.push_back(record.substr(begin, end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    const std::vector<std::pair<std::string, std::string>> expected{
        {"variable", std::string{variable}},
        {"package_id", noise_package_id},
        {"provenance_id", noise_provenance_join_id},
        {"product_identity", std::string{identity}},
        {"product_version", noise_product_contract_version},
        {"semantic_digest", std::string{semantic_digest}},
        {"digest_kind", noise_semantic_digest_kind},
        {"scope", "map_summary"},
        {"validity", std::string{validity}},
        {"restriction", std::string{restriction}}};
    if (tokens.size() != expected.size() + 1 ||
        tokens.front() != noise_netcdf_join_schema) {
        throw std::runtime_error(
            "invalid structured NetCDF noise-product join shape");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const auto &token = tokens[index + 1];
        const auto separator = token.find('=');
        if (separator == std::string::npos ||
            token.find('=', separator + 1) != std::string::npos ||
            token.substr(0, separator) != expected[index].first ||
            token.substr(separator + 1) != expected[index].second) {
            throw std::runtime_error(
                "invalid structured NetCDF noise-product join field " +
                expected[index].first);
        }
    }
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
            try {
                validate_noise_netcdf_join_comment(
                    comment, contract.name, contract.identity,
                    noise_product_semantic_digest(contract.identity),
                    contract.validity,
                    contract.restriction);
            }
            catch (const std::runtime_error &error) {
                throw std::runtime_error(
                    "invalid NetCDF noise-product join for " +
                    std::string{contract.name} + " in " +
                    path.string() + ": " + error.what());
            }
            identities.insert(contract.identity);
        }
        for (const auto &[name, variable] : file.getVars()) {
            const bool expected = std::any_of(
                variables.begin(), variables.end(),
                [&](const auto &contract) {
                    return name == contract.name;
                });
            if (expected) {
                continue;
            }
            const auto comment_attribute = variable.getAtt("comment");
            if (comment_attribute.isNull()) {
                continue;
            }
            std::string comment;
            comment_attribute.getValues(comment);
            if (comment.find(noise_netcdf_join_schema) !=
                std::string::npos) {
                throw std::runtime_error(
                    "unexpected NetCDF noise-contract variable " + name);
            }
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
    validation.contains_stack_derived_product = true;
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
    std::size_t empirical_map_product_count = 0;
    bool contains_stack_derived_product = false;
};

inline std::vector<ValidatedNoiseMember> validate_noise_member_inventory(
    const std::filesystem::path &reduction_dir,
    const std::vector<NoisePublishedMember> &members) {
    std::error_code ec;
    const auto lexical_root =
        std::filesystem::absolute(reduction_dir, ec).lexically_normal();
    if (ec) {
        throw std::ios_base::failure(
            "noise publication root is unavailable: " +
            reduction_dir.string());
    }
    ec.clear();
    const auto root_status =
        std::filesystem::symlink_status(lexical_root, ec);
    if (ec || std::filesystem::is_symlink(root_status) ||
        !std::filesystem::is_directory(root_status)) {
        throw std::ios_base::failure(
            "noise publication root is missing, non-directory, or a symlink: " +
            lexical_root.string());
    }
    ec.clear();
    const auto root = std::filesystem::canonical(lexical_root, ec);
    if (ec) {
        throw std::ios_base::failure(
            "cannot canonicalize noise publication root: " +
            lexical_root.string());
    }

    std::set<std::string> unique_paths;
    std::set<std::string> unique_resolved_paths;
    std::vector<ValidatedNoiseMember> validated;
    validated.reserve(members.size());
    for (const auto &member : members) {
        ec.clear();
        const auto candidate = member.path.is_absolute()
            ? member.path
            : std::filesystem::absolute(member.path, ec);
        if (ec) {
            throw std::runtime_error(
                "cannot resolve noise package member path: " +
                member.path.string());
        }
        if (candidate.lexically_normal() != candidate) {
            throw std::runtime_error(
                "noise package member path is not lexically normalized: " +
                candidate.string());
        }
        const auto relative = candidate.lexically_relative(lexical_root);
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
        if (relative_name.find('\\') != std::string::npos ||
            (lexical_root / relative) != candidate) {
            throw std::runtime_error(
                "noise package member path is not a canonical relative identity: " +
                candidate.string());
        }
        if (!unique_paths.insert(relative_name).second) {
            throw std::runtime_error(
                "duplicate noise package member: " + relative_name);
        }

        auto lexical_component = lexical_root;
        for (const auto &component : relative) {
            lexical_component /= component;
            ec.clear();
            const auto status =
                std::filesystem::symlink_status(lexical_component, ec);
            const bool is_leaf = lexical_component == candidate;
            if (ec || std::filesystem::is_symlink(status) ||
                (is_leaf && !std::filesystem::is_regular_file(status)) ||
                (!is_leaf && !std::filesystem::is_directory(status))) {
                throw std::runtime_error(
                    "noise package member has a missing, non-regular, or symlink path component: " +
                    lexical_component.string());
            }
        }
        ec.clear();
        const auto canonical = std::filesystem::canonical(candidate, ec);
        if (ec) {
            throw std::runtime_error(
                "cannot canonicalize noise package member: " +
                candidate.string());
        }
        const auto canonical_relative = canonical.lexically_relative(root);
        bool canonical_outside = canonical_relative.empty() ||
            canonical_relative.is_absolute();
        for (const auto &component : canonical_relative) {
            if (component == ".." || component == ".") {
                canonical_outside = true;
            }
        }
        if (canonical_outside) {
            throw std::runtime_error(
                "noise package member resolves outside the reduction root: " +
                candidate.string());
        }
        if (!unique_resolved_paths.insert(canonical.generic_string()).second) {
            throw std::runtime_error(
                "duplicate resolved noise package member: " +
                canonical.string());
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
             joins.realization_image_count,
             joins.empirical_map_product_count,
             joins.contains_stack_derived_product});
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
                right[index].realization_image_count ||
            left[index].empirical_map_product_count !=
                right[index].empirical_map_product_count ||
            left[index].contains_stack_derived_product !=
                right[index].contains_stack_derived_product) {
            return false;
        }
    }
    return true;
}

inline void validate_noise_package_member_semantics(
    const NoiseExecutionPlan &plan,
    const std::vector<ValidatedNoiseMember> &members) {
    std::size_t empirical_map_product_count = 0;
    bool contains_stack_derived_product = false;
    bool contains_empirical_netcdf = false;
    for (const auto &member : members) {
        if (empirical_map_product_count >
            std::numeric_limits<std::size_t>::max() -
                member.empirical_map_product_count) {
            throw std::overflow_error(
                "noise empirical-member inventory count overflow");
        }
        empirical_map_product_count +=
            member.empirical_map_product_count;
        contains_stack_derived_product =
            contains_stack_derived_product ||
            member.contains_stack_derived_product;
        contains_empirical_netcdf = contains_empirical_netcdf ||
            member.kind == NoisePublishedMemberKind::netcdf;
    }

    if (!plan.effective.enabled) {
        if (contains_stack_derived_product) {
            throw std::logic_error(
                "disabled noise package contains a stack-derived member");
        }
        return;
    }
    if (!plan.realized.empirical_product_map_count) {
        throw std::logic_error(
            "noise package lacks an observed empirical-product count");
    }
    if (!plan.effective.products_enabled) {
        if (empirical_map_product_count != 0 ||
            contains_empirical_netcdf ||
            *plan.realized.empirical_product_map_count != 0) {
            throw std::logic_error(
                "noise package contains empirical members while products are disabled");
        }
        return;
    }
    if (empirical_map_product_count !=
        *plan.realized.empirical_product_map_count) {
        throw std::logic_error(
            "noise package empirical FITS inventory does not match observed empirical product maps");
    }
    if (contains_empirical_netcdf && empirical_map_product_count == 0) {
        throw std::logic_error(
            "noise package contains empirical NetCDF diagnostics without empirical map products");
    }
}

inline void append_noise_member_inventory_field(
    std::string &preimage, std::string_view value) {
    preimage += std::to_string(value.size());
    preimage.push_back(':');
    preimage.append(value.data(), value.size());
}

inline std::string noise_member_inventory_preimage_v2(
    const std::vector<ValidatedNoiseMember> &members) {
    for (std::size_t index = 1; index < members.size(); ++index) {
        if (members[index - 1].relative_path >=
            members[index].relative_path) {
            throw std::logic_error(
                "noise package member inventory is not in canonical lexical order");
        }
    }
    std::string preimage{"citlali-noise-member-inventory-v2|"};
    append_noise_member_inventory_field(
        preimage, std::to_string(members.size()));
    for (const auto &member : members) {
        append_noise_member_inventory_field(
            preimage, member.relative_path);
        append_noise_member_inventory_field(preimage, member.sha256);
        append_noise_member_inventory_field(
            preimage, std::to_string(member.size_bytes));
    }
    return preimage;
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
    validate_noise_package_member_semantics(plan, validated);
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
    }
    node["package"]["member_files"] = members;
    node["package"]["member_count"] = validated.size();
    const auto member_inventory_preimage =
        noise_member_inventory_preimage_v2(validated);
    const auto member_inventory_digest =
        "sha256:" + citlali::utils::sha256(member_inventory_preimage);
    node["package"]["member_inventory_digest"] =
        member_inventory_digest;
    node["package"]["member_inventory_digest_kind"] =
        noise_member_inventory_digest_kind;
    node["package"]["member_inventory_preimage_encoding"] =
        noise_member_inventory_preimage_encoding;
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
