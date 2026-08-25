#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>
#include <citlali/core/utils/sha256.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

inline constexpr std::uint32_t noise_random_seed = 5489U;
inline constexpr const char *noise_random_engine_name =
    "boost::random::mt19937";
inline constexpr const char *noise_seed_policy_name =
    "fixed_internal_default";
inline constexpr const char *noise_generator_scope_name =
    "reduction_pipeline_invocation";
inline constexpr const char *noise_package_id =
    "citlali-noise-products";
inline constexpr const char *noise_provenance_join_id =
    "noise_products_provenance.yaml";
inline constexpr const char *noise_product_contract_version =
    "SCI-NOI-002-v1";
inline constexpr const char *noise_ensemble_identity =
    "source_imprinted_current";
inline constexpr const char *noise_estimator_identity =
    "conditional_finite_stack_scatter";
inline constexpr const char *noise_estimator_target =
    "completed_source_imprinted_current_stack";
inline constexpr const char *noise_centering_identity =
    "empirical_completed_stack_mean";
inline constexpr const char *noise_normalization_identity =
    "centered_sum_of_squares_divided_by_completed_R";
inline constexpr const char *noise_support_identity =
    "product_specific_common_map_support_and_response";
inline constexpr const char *noise_missingness_identity =
    "nonfinite_or_invalid_values_unavailable";
inline constexpr const char *noise_dependence_status =
    "joint_design_class_recorded_assignment_census_not_persisted";
inline constexpr const char *noise_scale_diagnostic_identity =
    "global_nonprecision_scale_diagnostic";
inline constexpr const char *noise_scale_diagnostic_formula =
    "alpha=1/median_D(q_p*V_p)";
inline constexpr const char *noise_filter_parity_gate_status =
    "scope_blocked_not_applicable_pending_FLT";
inline constexpr const char *noise_filter_parity_gate_reason =
    "signal filter_maps uses signal-background affine edge handling and kernel response while realization filter_noise paths are zero-centered; strict operator-edge parity is not established and FLT mathematics is excluded";

inline constexpr const char *noise_conditional_stack_scatter_product_id =
    "conditional_finite_stack_scatter";
inline constexpr const char *noise_formal_coefficient_product_id =
    "formal_nonprecision_coefficient_snapshot";
inline constexpr const char *noise_scaled_coefficient_product_id =
    "global_nonprecision_scaled_coefficient";
inline constexpr const char *noise_coefficient_standardized_signal_product_id =
    "coefficient_standardized_signal";
inline constexpr const char *noise_filtered_pixel_stack_scatter_product_id =
    "filtered_pixel_stack_scatter";
inline constexpr const char *noise_stack_scatter_ratio_product_id =
    "conditional_stack_scatter_ratio";
inline constexpr const char *noise_realization_product_id =
    "source_imprinted_current_realization";
inline constexpr const char *noise_pooled_stack_scale_product_id =
    "pooled_stack_scale_diagnostic";
inline constexpr const char *noise_source_finder_score_product_id =
    "source_finder_engineering_score";
inline constexpr const char *noise_fitted_amplitude_rms_ratio_product_id =
    "fitted_amplitude_over_full_map_rms_ratio";
inline constexpr const char *noise_fixed_projection_scatter_product_id =
    "fixed_projection_stack_scatter";

inline const char *noise_joint_assignment_design_identity(
    bool randomize_dets) {
    return randomize_dets
        ? "sequential_mt19937_per_scan_per_detector_sign_assignments"
        : "sequential_mt19937_per_scan_detector_shared_sign_assignments";
}

inline std::string noise_product_semantic_digest(
    std::string_view product_identity) {
    const std::string canonical =
        std::string{noise_package_id} + "|" +
        noise_product_contract_version + "|" +
        std::string{product_identity};
    return "sha256:" + citlali::utils::sha256(canonical);
}

struct NoiseEffectiveResolutionRecord {
    bool mapmaking_enabled = false;
    bool requested_enabled = false;
    bool effective_enabled = false;
    bool disabled_by_mapmaking = false;
    int requested_n_noise_maps = 0;
    int effective_n_noise_maps = 0;
    bool count_zeroed_while_disabled = false;
    std::uint32_t random_seed = noise_random_seed;
    std::string random_engine = noise_random_engine_name;
    std::string seed_policy = noise_seed_policy_name;
    std::string generator_scope = noise_generator_scope_name;
    std::string joint_assignment_design;
};

struct NoiseRealizedState {
    bool reduction_completed = false;
    bool generation_executed = false;
    std::optional<std::size_t> noise_maps_per_scientific_map;
    std::optional<std::size_t> observation_scientific_map_count;
    std::optional<std::size_t> observation_noise_realization_count;
    std::optional<std::size_t> coadd_scientific_map_count;
    std::optional<std::size_t> coadd_noise_realization_count;
    std::optional<std::size_t> total_noise_realization_count;
    std::optional<std::size_t> empirical_product_map_count;
    std::optional<std::size_t> realization_image_write_count;
    bool actual_completion_valid = false;
    bool completed_count_matches_effective = false;
    bool uncertainty_use_valid = false;
    std::string completion_basis;
    bool outputs_completed = false;
};

struct NoiseExpectedCounts {
    bool initialized = false;
    std::size_t noise_maps_per_scientific_map = 0;
    std::size_t observation_scientific_map_count = 0;
    std::size_t observation_noise_realization_count = 0;
    std::size_t coadd_scientific_map_count = 0;
    std::size_t coadd_noise_realization_count = 0;
    std::size_t total_noise_realization_count = 0;
    std::size_t empirical_product_map_count = 0;
    std::size_t realization_image_write_count = 0;
};

enum class NoisePublishedMemberKind {
    fits,
    ecsv,
    netcdf,
};

inline const char *noise_published_member_kind_name(
    NoisePublishedMemberKind kind) {
    switch (kind) {
        case NoisePublishedMemberKind::fits:
            return "fits";
        case NoisePublishedMemberKind::ecsv:
            return "ecsv";
        case NoisePublishedMemberKind::netcdf:
            return "netcdf";
    }
    throw std::logic_error("unknown noise published-member kind");
}

struct NoisePublishedMember {
    std::filesystem::path path;
    NoisePublishedMemberKind kind = NoisePublishedMemberKind::fits;
};

struct NoiseExecutionPlan {
    bool initialized = false;
    citlali::config::NoiseConfig requested;
    citlali::config::NoiseConfig effective;
    NoiseEffectiveResolutionRecord effective_resolution;
    NoiseExpectedCounts expected;
    NoiseRealizedState realized;
    bool publication_started = false;
    std::vector<NoisePublishedMember> published_members;

    void reset_from_request(
        const citlali::config::NoiseConfig &request,
        bool mapmaking_enabled) {
        if (request.enabled && request.n_noise_maps <= 0) {
            throw std::invalid_argument(
                "enabled noise request requires a positive realization count");
        }
        initialized = true;
        requested = request;
        effective = request;
        if (!mapmaking_enabled || !request.enabled) {
            effective.enabled = false;
            effective.n_noise_maps = 0;
        }
        effective_resolution = {};
        effective_resolution.mapmaking_enabled = mapmaking_enabled;
        effective_resolution.requested_enabled = request.enabled;
        effective_resolution.effective_enabled = effective.enabled;
        effective_resolution.disabled_by_mapmaking =
            request.enabled && !mapmaking_enabled;
        effective_resolution.requested_n_noise_maps = request.n_noise_maps;
        effective_resolution.effective_n_noise_maps = effective.n_noise_maps;
        effective_resolution.count_zeroed_while_disabled =
            request.n_noise_maps != effective.n_noise_maps;
        effective_resolution.random_seed = noise_random_seed;
        effective_resolution.random_engine = noise_random_engine_name;
        effective_resolution.seed_policy = noise_seed_policy_name;
        effective_resolution.generator_scope = noise_generator_scope_name;
        effective_resolution.joint_assignment_design =
            noise_joint_assignment_design_identity(effective.randomize_dets);
        expected = {};
        realized = {};
        publication_started = false;
        published_members.clear();
    }
};

inline std::size_t checked_noise_count_product(
    std::size_t lhs, std::size_t rhs, const char *label) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error(
            std::string{"noise cardinality overflow: "} + label);
    }
    return lhs * rhs;
}

inline void begin_noise_run_publication(NoiseExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error("noise plan is not initialized");
    }
    plan.expected = {};
    plan.realized = {};
    plan.publication_started = true;
    plan.published_members.clear();

    const std::size_t zero = 0;
    if (!plan.effective.enabled) {
        plan.realized.noise_maps_per_scientific_map = zero;
    }
    plan.realized.observation_scientific_map_count = zero;
    plan.realized.observation_noise_realization_count = zero;
    plan.realized.coadd_scientific_map_count = zero;
    plan.realized.coadd_noise_realization_count = zero;
    plan.realized.total_noise_realization_count = zero;
    plan.realized.empirical_product_map_count = zero;
    plan.realized.realization_image_write_count = zero;
    plan.realized.completion_basis = "publication_incomplete";
}

inline void record_noise_published_member(
    NoiseExecutionPlan &plan, const std::filesystem::path &path,
    NoisePublishedMemberKind kind) {
    if (!plan.publication_started) {
        throw std::logic_error(
            "noise member publication was not initialized");
    }
    if (path.empty()) {
        throw std::invalid_argument("noise published-member path is empty");
    }
    plan.published_members.push_back({path, kind});
}

template <class FitsOutputs>
std::vector<std::filesystem::path> noise_fits_output_paths(
    const FitsOutputs &outputs) {
    std::vector<std::filesystem::path> paths;
    paths.reserve(outputs.size());
    for (const auto &output : outputs) {
        paths.emplace_back(output.filepath + ".fits");
    }
    return paths;
}

inline void record_noise_fits_members(
    NoiseExecutionPlan &plan,
    const std::vector<std::filesystem::path> &data_paths,
    const std::vector<std::filesystem::path> &noise_paths,
    bool data_members_have_package_join) {
    if (plan.effective.enabled && plan.effective.products_enabled &&
        data_members_have_package_join) {
        for (const auto &path : data_paths) {
            record_noise_published_member(
                plan, path, NoisePublishedMemberKind::fits);
        }
    }
    if (plan.effective.enabled && plan.effective.write_realizations) {
        for (const auto &path : noise_paths) {
            record_noise_published_member(
                plan, path, NoisePublishedMemberKind::fits);
        }
    }
}

template <class MapBuffer>
bool noise_data_fits_have_package_join(
    const NoiseExecutionPlan &plan, bool is_filtered,
    const MapBuffer &map_buffer) {
    if (!plan.effective.enabled || !plan.effective.products_enabled) {
        return false;
    }
    const auto &products =
        is_filtered && map_buffer.raw_science_parent
            ? *map_buffer.raw_science_parent
            : map_buffer.science_products;
    // Package membership follows output-family identity, not availability of
    // the optional F009/F010 successor profile.  Legacy/JINC coadds remain
    // coadds and may carry only the configured standalone scaled coefficient;
    // full empirical bundles belong to observation products.
    const bool coadd_output = products.initialized && products.is_coadd;
    return !coadd_output || plan.effective.apply_empirical_weights;
}

inline void add_noise_observed_count(
    std::optional<std::size_t> &value, std::size_t increment,
    const char *label) {
    if (!value) {
        throw std::logic_error(
            std::string{"noise observed counter is unavailable: "} + label);
    }
    if (*value > std::numeric_limits<std::size_t>::max() - increment) {
        throw std::overflow_error(
            std::string{"noise observed-counter overflow: "} + label);
    }
    *value += increment;
}

inline void record_noise_map_output_stage(
    NoiseExecutionPlan &plan, bool is_coadd, bool is_filtered,
    std::size_t scientific_map_count,
    std::size_t noise_maps_per_scientific_map,
    std::size_t empirical_product_map_count,
    std::size_t realization_image_write_count) {
    if (!plan.publication_started) {
        throw std::logic_error(
            "noise map-output observation was not initialized");
    }
    if (!plan.effective.enabled) {
        if (noise_maps_per_scientific_map != 0 ||
            empirical_product_map_count != 0 ||
            realization_image_write_count != 0) {
            throw std::logic_error(
                "disabled noise execution observed nonzero work");
        }
        return;
    }
    if (is_coadd && empirical_product_map_count != 0) {
        throw std::logic_error(
            "successor coadd observed an empirical companion product");
    }
    if (scientific_map_count == 0 ||
        noise_maps_per_scientific_map == 0) {
        throw std::logic_error(
            "enabled noise map-output stage observed zero cardinality");
    }
    if (!plan.realized.noise_maps_per_scientific_map) {
        plan.realized.noise_maps_per_scientific_map =
            noise_maps_per_scientific_map;
    }
    else if (*plan.realized.noise_maps_per_scientific_map !=
             noise_maps_per_scientific_map) {
        throw std::logic_error(
            "noise map-output stages observed inconsistent stack sizes");
    }

    const auto realization_count = checked_noise_count_product(
        scientific_map_count, noise_maps_per_scientific_map,
        "observed realization count");
    if (!is_filtered) {
        if (is_coadd) {
            add_noise_observed_count(
                plan.realized.coadd_scientific_map_count,
                scientific_map_count, "coadd scientific maps");
            add_noise_observed_count(
                plan.realized.coadd_noise_realization_count,
                realization_count, "coadd realizations");
        }
        else {
            add_noise_observed_count(
                plan.realized.observation_scientific_map_count,
                scientific_map_count, "observation scientific maps");
            add_noise_observed_count(
                plan.realized.observation_noise_realization_count,
                realization_count, "observation realizations");
        }
        add_noise_observed_count(
            plan.realized.total_noise_realization_count,
            realization_count, "total realizations");
        plan.realized.generation_executed = true;
    }
    add_noise_observed_count(
        plan.realized.empirical_product_map_count,
        empirical_product_map_count, "empirical product maps");
    add_noise_observed_count(
        plan.realized.realization_image_write_count,
        realization_image_write_count, "realization image writes");
}

template <class MapBuffer>
void record_noise_map_output_stage(
    NoiseExecutionPlan &plan, bool is_coadd, bool is_filtered,
    const MapBuffer &map_buffer) {
    if (!plan.effective.enabled) {
        record_noise_map_output_stage(
            plan, is_coadd, is_filtered, 0, 0, 0, 0);
        return;
    }
    if (map_buffer.n_noise <= 0) {
        throw std::logic_error(
            "enabled noise map-output buffer has no realizations");
    }
    const auto scientific_map_count =
        static_cast<std::size_t>(map_buffer.signal.size());
    const auto noise_count =
        static_cast<std::size_t>(map_buffer.n_noise);
    const auto empirical_count = plan.effective.products_enabled && !is_coadd
        ? static_cast<std::size_t>(map_buffer.noise_variance.size())
        : std::size_t{0};
    const auto realization_write_count = plan.effective.write_realizations
        ? checked_noise_count_product(
              static_cast<std::size_t>(map_buffer.noise.size()),
              noise_count, "observed realization image writes")
        : std::size_t{0};
    record_noise_map_output_stage(
        plan, is_coadd, is_filtered, scientific_map_count, noise_count,
        empirical_count, realization_write_count);
}

template <class MapBuffer>
void record_noise_selected_map_output_stage(
    NoiseExecutionPlan &plan, bool is_coadd, bool is_filtered,
    const MapBuffer &map_buffer,
    std::size_t published_scientific_map_count) {
    if (!plan.effective.enabled) {
        record_noise_map_output_stage(
            plan, is_coadd, is_filtered, 0, 0, 0, 0);
        return;
    }
    if (published_scientific_map_count == 0 ||
        published_scientific_map_count > map_buffer.signal.size()) {
        throw std::logic_error(
            "selected noise map-output count is outside the map buffer");
    }
    if (map_buffer.n_noise <= 0) {
        throw std::logic_error(
            "enabled selected noise map-output buffer has no realizations");
    }
    if (plan.effective.products_enabled &&
        published_scientific_map_count > map_buffer.noise_variance.size()) {
        throw std::logic_error(
            "selected empirical noise products are outside the map buffer");
    }
    if (plan.effective.write_realizations &&
        published_scientific_map_count > map_buffer.noise.size()) {
        throw std::logic_error(
            "selected noise realizations are outside the map buffer");
    }
    const auto noise_count =
        static_cast<std::size_t>(map_buffer.n_noise);
    const auto empirical_count = plan.effective.products_enabled && !is_coadd
        ? published_scientific_map_count
        : std::size_t{0};
    const auto realization_write_count =
        plan.effective.write_realizations
            ? checked_noise_count_product(
                  published_scientific_map_count, noise_count,
                  "selected realization image writes")
            : std::size_t{0};
    record_noise_map_output_stage(
        plan, is_coadd, is_filtered, published_scientific_map_count,
        noise_count, empirical_count, realization_write_count);
}

template <class MapBuffer>
void record_noise_map_output_publication(
    NoiseExecutionPlan &plan, bool is_coadd, bool is_filtered,
    const MapBuffer &map_buffer,
    const std::vector<std::filesystem::path> &data_paths,
    const std::vector<std::filesystem::path> &noise_paths) {
    if (data_paths.empty()) {
        return;
    }
    record_noise_fits_members(
        plan, data_paths, noise_paths,
        noise_data_fits_have_package_join(
            plan, is_filtered, map_buffer));
    record_noise_map_output_stage(
        plan, is_coadd, is_filtered, map_buffer);
}

inline NoiseExpectedCounts expected_noise_counts(
    const NoiseExecutionPlan &plan,
    const MapmakingExecutionPlan &mapmaking_plan,
    bool filtered_maps_enabled) {
    NoiseExpectedCounts expected;
    expected.initialized = true;
    if (!plan.effective.enabled) {
        return expected;
    }
    if (plan.effective.n_noise_maps <= 0) {
        throw std::logic_error(
            "enabled effective noise-map count is not positive");
    }

    for (const auto &observation : mapmaking_plan.observations) {
        if (!observation.outputs_completed) {
            throw std::logic_error(
                "noise completion found incomplete observation maps");
        }
        if (expected.observation_scientific_map_count >
            std::numeric_limits<std::size_t>::max() -
                observation.map_count) {
            throw std::overflow_error(
                "noise cardinality overflow: observation maps");
        }
        expected.observation_scientific_map_count +=
            observation.map_count;
    }
    const bool coadd_available = mapmaking_plan.coadd.has_value();
    expected.coadd_scientific_map_count = coadd_available
        ? mapmaking_plan.coadd->map_count
        : std::size_t{0};
    expected.noise_maps_per_scientific_map =
        static_cast<std::size_t>(plan.effective.n_noise_maps);
    expected.observation_noise_realization_count =
        checked_noise_count_product(
            expected.observation_scientific_map_count,
            expected.noise_maps_per_scientific_map,
            "observation realizations");
    expected.coadd_noise_realization_count =
        checked_noise_count_product(
            expected.coadd_scientific_map_count,
            expected.noise_maps_per_scientific_map,
            "coadd realizations");
    if (expected.observation_noise_realization_count >
        std::numeric_limits<std::size_t>::max() -
            expected.coadd_noise_realization_count) {
        throw std::overflow_error(
            "noise cardinality overflow: total realizations");
    }
    expected.total_noise_realization_count =
        expected.observation_noise_realization_count +
        expected.coadd_noise_realization_count;

    const std::size_t observation_output_stage_count =
        filtered_maps_enabled && !coadd_available ? std::size_t{2}
                                                  : std::size_t{1};
    if (plan.effective.products_enabled) {
        expected.empirical_product_map_count =
            checked_noise_count_product(
                expected.observation_scientific_map_count,
                observation_output_stage_count,
                "observation empirical product maps");
    }
    if (plan.effective.write_realizations) {
        const std::size_t coadd_output_stage_count = coadd_available
            ? (filtered_maps_enabled ? std::size_t{2} : std::size_t{1})
            : std::size_t{0};
        expected.realization_image_write_count =
            checked_noise_count_product(
                expected.observation_noise_realization_count,
                observation_output_stage_count,
                "observation realization image writes") +
            checked_noise_count_product(
                expected.coadd_noise_realization_count,
                coadd_output_stage_count,
                "coadd realization image writes");
    }
    return expected;
}

inline bool observed_noise_count_matches(
    const std::optional<std::size_t> &observed,
    std::size_t expected) {
    return observed && *observed == expected;
}

inline void record_noise_run_completed(
    NoiseExecutionPlan &plan,
    const MapmakingExecutionPlan &mapmaking_plan,
    bool filtered_maps_enabled) {
    if (!plan.initialized) {
        throw std::logic_error("noise plan is not initialized");
    }
    if (!mapmaking_plan.initialized ||
        !mapmaking_plan.realized.reduction_completed) {
        throw std::logic_error(
            "noise completion requires completed mapmaking provenance");
    }
    if (plan.effective.enabled &&
        !mapmaking_plan.realized.mapmaking_executed) {
        throw std::logic_error(
            "enabled noise maps require completed mapmaking");
    }

    if (!plan.publication_started) {
        throw std::logic_error(
            "noise completion requires initialized publication state");
    }
    plan.expected = expected_noise_counts(
        plan, mapmaking_plan, filtered_maps_enabled);
    if (!plan.effective.enabled) {
        const bool zero_work =
            !plan.realized.generation_executed &&
            observed_noise_count_matches(
                plan.realized.noise_maps_per_scientific_map, 0) &&
            observed_noise_count_matches(
                plan.realized.observation_scientific_map_count, 0) &&
            observed_noise_count_matches(
                plan.realized.observation_noise_realization_count, 0) &&
            observed_noise_count_matches(
                plan.realized.coadd_scientific_map_count, 0) &&
            observed_noise_count_matches(
                plan.realized.coadd_noise_realization_count, 0) &&
            observed_noise_count_matches(
                plan.realized.total_noise_realization_count, 0) &&
            observed_noise_count_matches(
                plan.realized.empirical_product_map_count, 0) &&
            observed_noise_count_matches(
                plan.realized.realization_image_write_count, 0);
        if (!zero_work) {
            throw std::logic_error(
                "disabled noise completion observed nonzero or unavailable work");
        }
        plan.realized.reduction_completed = true;
        plan.realized.actual_completion_valid = true;
        plan.realized.completed_count_matches_effective = true;
        plan.realized.uncertainty_use_valid = false;
        plan.realized.completion_basis = "effective_disabled_zero_work";
        plan.realized.outputs_completed = true;
        return;
    }
    const bool counts_match =
        plan.realized.generation_executed &&
        observed_noise_count_matches(
            plan.realized.noise_maps_per_scientific_map,
            plan.expected.noise_maps_per_scientific_map) &&
        observed_noise_count_matches(
            plan.realized.observation_scientific_map_count,
            plan.expected.observation_scientific_map_count) &&
        observed_noise_count_matches(
            plan.realized.observation_noise_realization_count,
            plan.expected.observation_noise_realization_count) &&
        observed_noise_count_matches(
            plan.realized.coadd_scientific_map_count,
            plan.expected.coadd_scientific_map_count) &&
        observed_noise_count_matches(
            plan.realized.coadd_noise_realization_count,
            plan.expected.coadd_noise_realization_count) &&
        observed_noise_count_matches(
            plan.realized.total_noise_realization_count,
            plan.expected.total_noise_realization_count) &&
        observed_noise_count_matches(
            plan.realized.empirical_product_map_count,
            plan.expected.empirical_product_map_count) &&
        observed_noise_count_matches(
            plan.realized.realization_image_write_count,
            plan.expected.realization_image_write_count);
    if (!counts_match) {
        plan.realized.actual_completion_valid = false;
        plan.realized.completed_count_matches_effective = false;
        plan.realized.outputs_completed = false;
        plan.realized.completion_basis =
            "observed_lifecycle_counts_incomplete_or_mismatched";
        throw std::logic_error(
            "noise completion observed counts do not match the effective plan");
    }

    plan.realized.reduction_completed = true;
    plan.realized.actual_completion_valid = true;
    plan.realized.completed_count_matches_effective = true;
    plan.realized.uncertainty_use_valid =
        plan.expected.noise_maps_per_scientific_map >= 2;
    plan.realized.completion_basis =
        "observed_successful_publication_lifecycle";
    plan.realized.outputs_completed = true;
}

}  // namespace citlali::pipeline
