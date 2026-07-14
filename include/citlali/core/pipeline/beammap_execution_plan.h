#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/config_value.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace citlali::pipeline {

struct BeammapRequestPresence {
    bool max_d2_iter0 = false;
    bool max_d2_after_iter0 = false;
    bool score_lambda_iter0 = false;
    bool score_lambda_after_iter0 = false;
    bool split_flag_values = false;
};

struct BeammapEffectiveResolutionRecord {
    bool mapmaking_enabled = false;
    int requested_max_iterations = 1;
    int effective_max_iterations = 1;
    bool max_iterations_forced_without_mapmaking = false;
    int requested_locator_iter = 0;
    int effective_locator_iter = 0;
    bool locator_iter_forced_zero = false;
    int requested_measurement_start_iter = 1;
    int effective_measurement_start_iter = 1;
    bool measurement_start_iter_adjusted = false;
    bool legacy_phase_behavior = false;
    bool measurement_pass_available = false;
    bool convergence_check_available = false;
    bool convergence_active = false;
    bool prior_path_available = false;
    bool priors_disabled_by_missing_path = false;
    bool max_d2_iter0_inherited = false;
    bool max_d2_after_iter0_inherited = false;
    bool score_lambda_iter0_inherited = false;
    bool score_lambda_after_iter0_inherited = false;
    bool split_flag_values_defaulted = false;
    bool split_flag_values_sorted = false;
    bool split_flag_values_deduplicated = false;
    std::size_t requested_split_flag_count = 0;
    std::size_t effective_split_flag_count = 0;
};

inline std::vector<int> resolve_beammap_split_flag_values(
    const std::vector<int> &requested,
    BeammapEffectiveResolutionRecord &resolution) {
    resolution.requested_split_flag_count = requested.size();
    if (requested.empty()) {
        resolution.split_flag_values_defaulted = true;
        resolution.effective_split_flag_count = 2;
        return {0, 1};
    }

    auto effective = requested;
    resolution.split_flag_values_sorted =
        !std::is_sorted(effective.begin(), effective.end());
    std::sort(effective.begin(), effective.end());
    const auto unique_end = std::unique(effective.begin(), effective.end());
    resolution.split_flag_values_deduplicated =
        unique_end != effective.end();
    effective.erase(unique_end, effective.end());
    resolution.effective_split_flag_count = effective.size();
    return effective;
}

class BeammapExecutionPlan {
public:
    [[nodiscard]] bool initialized() const { return initialized_; }

    [[nodiscard]] const citlali::config::BeammapConfig &requested() const {
        return requested_;
    }

    [[nodiscard]] const citlali::config::BeammapConfig &effective() const {
        return effective_;
    }

    [[nodiscard]] const BeammapEffectiveResolutionRecord &resolution() const {
        return resolution_;
    }

    void reset_from_request(
        const citlali::config::BeammapConfig &request,
        const BeammapRequestPresence &presence,
        bool mapmaking_enabled) {
        requested_ = request;
        effective_ = request;
        resolution_ = {};
        resolution_.mapmaking_enabled = mapmaking_enabled;
        resolution_.requested_max_iterations =
            request.iteration.max_iterations;
        if (!mapmaking_enabled) {
            effective_.iteration.max_iterations = 1;
            resolution_.max_iterations_forced_without_mapmaking =
                request.iteration.max_iterations != 1;
        }
        resolution_.effective_max_iterations =
            effective_.iteration.max_iterations;

        resolution_.requested_locator_iter =
            request.phase_strategy.locator_iter;
        if (effective_.phase_strategy.locator_iter != 0) {
            effective_.phase_strategy.locator_iter = 0;
            resolution_.locator_iter_forced_zero = true;
        }
        resolution_.effective_locator_iter =
            effective_.phase_strategy.locator_iter;
        resolution_.requested_measurement_start_iter =
            request.phase_strategy.measurement_start_iter;
        if (effective_.phase_strategy.measurement_start_iter <=
            effective_.phase_strategy.locator_iter) {
            effective_.phase_strategy.measurement_start_iter =
                effective_.phase_strategy.locator_iter + 1;
            resolution_.measurement_start_iter_adjusted = true;
        }
        resolution_.effective_measurement_start_iter =
            effective_.phase_strategy.measurement_start_iter;
        resolution_.legacy_phase_behavior =
            !effective_.phase_strategy.enabled;
        const int first_measurement_iteration =
            effective_.phase_strategy.enabled
                ? effective_.phase_strategy.measurement_start_iter
                : 1;
        resolution_.measurement_pass_available =
            effective_.iteration.max_iterations > first_measurement_iteration;
        resolution_.convergence_check_available =
            effective_.iteration.max_iterations >
            first_measurement_iteration + 1;
        resolution_.convergence_active =
            mapmaking_enabled && effective_.iteration.tolerance > 0.0 &&
            resolution_.convergence_check_available;

        resolution_.max_d2_iter0_inherited = !presence.max_d2_iter0;
        if (resolution_.max_d2_iter0_inherited) {
            effective_.priors.max_d2_iter0 = effective_.priors.max_d2;
        }
        resolution_.max_d2_after_iter0_inherited =
            !presence.max_d2_after_iter0;
        if (resolution_.max_d2_after_iter0_inherited) {
            effective_.priors.max_d2_after_iter0 = effective_.priors.max_d2;
        }
        resolution_.score_lambda_iter0_inherited =
            !presence.score_lambda_iter0;
        if (resolution_.score_lambda_iter0_inherited) {
            effective_.priors.score_lambda_iter0 =
                effective_.priors.score_lambda;
        }
        resolution_.score_lambda_after_iter0_inherited =
            !presence.score_lambda_after_iter0;
        if (resolution_.score_lambda_after_iter0_inherited) {
            effective_.priors.score_lambda_after_iter0 =
                effective_.priors.score_lambda;
        }
        resolution_.prior_path_available =
            citlali::config::has_config_value(effective_.priors.filepath);
        resolution_.priors_disabled_by_missing_path =
            effective_.priors.enabled && !resolution_.prior_path_available;
        if (resolution_.priors_disabled_by_missing_path) {
            effective_.priors.enabled = false;
        }

        effective_.split_fits_by_flag.flag_values =
            resolve_beammap_split_flag_values(
                presence.split_flag_values
                    ? request.split_fits_by_flag.flag_values
                    : std::vector<int>{},
                resolution_);
        initialized_ = true;
    }

private:
    bool initialized_ = false;
    citlali::config::BeammapConfig requested_;
    citlali::config::BeammapConfig effective_;
    BeammapEffectiveResolutionRecord resolution_;
};

}  // namespace citlali::pipeline
