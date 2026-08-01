#pragma once

#include <citlali/core/config/coadd_config.h>
#include <citlali/core/mapmaking/science_map_contract.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct CoaddEffectiveResolutionRecord {
    bool mapmaking_enabled = false;
    bool requested_enabled = false;
    bool effective_enabled = false;
    bool disabled_by_mapmaking = false;
};

struct CoaddRealizedState {
    bool reduction_completed = false;
    bool coadd_executed = false;
    std::optional<std::size_t> map_count;
    std::optional<std::size_t> required_map_write_count;
    bool outputs_completed = false;
};

struct CoaddScienceState {
    std::optional<double> requested_coverage_cut;
    std::optional<double> effective_coverage_cut;
    std::optional<mapmaking::ScienceMapBundleIdentity> common_identity;
    std::vector<mapmaking::ScienceMapRealizedMap> realized_maps;
    std::vector<mapmaking::ScienceMapCoaddAdmission> admissions;
    std::string absence_reason = "science-map coadd state not recorded";

    void clear_iteration() {
        requested_coverage_cut.reset();
        effective_coverage_cut.reset();
        common_identity.reset();
        realized_maps.clear();
        admissions.clear();
        absence_reason = "science-map coadd state not recorded";
    }
};

struct CoaddExecutionPlan {
    bool initialized = false;
    citlali::config::CoaddConfig requested;
    citlali::config::CoaddConfig effective;
    CoaddEffectiveResolutionRecord effective_resolution;
    CoaddScienceState science;
    CoaddRealizedState realized;

    void reset_from_request(
        const citlali::config::CoaddConfig &request,
        bool mapmaking_enabled) {
        initialized = true;
        requested = request;
        effective = request;
        if (!mapmaking_enabled) {
            effective.enabled = false;
        }
        effective_resolution = CoaddEffectiveResolutionRecord{
            mapmaking_enabled,
            request.enabled,
            effective.enabled,
            request.enabled && !mapmaking_enabled,
        };
        science = {};
        realized = {};
    }

    void begin_iteration() {
        if (!initialized) {
            throw std::logic_error("coadd plan is not initialized");
        }
        science.clear_iteration();
        realized = {};
    }

    void resolve_common_identity(
        mapmaking::ScienceMapBundleIdentity common_identity) {
        if (!initialized || !effective.enabled) {
            throw std::logic_error(
                "cannot record science-map state for unavailable coadd");
        }
        if (science.common_identity) {
            if (mapmaking::science_map_bundle_identity_digest(
                    *science.common_identity) !=
                mapmaking::science_map_bundle_identity_digest(
                    common_identity)) {
                throw std::logic_error(
                    "coadd common science-map identity cannot change");
            }
            return;
        }
        if (!science.realized_maps.empty() || !science.admissions.empty()) {
            throw std::logic_error(
                "coadd common identity must precede realized state and admissions");
        }
        if (common_identity.ordered_slots.empty()) {
            throw std::logic_error(
                "coadd common identity has no ordered map slots");
        }
        auto next = science;
        next.common_identity = std::move(common_identity);
        next.absence_reason.clear();
        science = std::move(next);
    }

    void record_realized_maps(
        std::vector<mapmaking::ScienceMapRealizedMap> realized_maps) {
        if (!science.common_identity) {
            throw std::logic_error(
                "cannot record coadd realized maps before common identity");
        }
        if (!science.realized_maps.empty()) {
            throw std::logic_error(
                "coadd realized science-map state is already recorded");
        }
        if (realized_maps.empty() ||
            science.common_identity->ordered_slots.size() !=
                realized_maps.size()) {
            throw std::logic_error(
                "coadd science-map identity/record cardinality differs");
        }
        science.realized_maps = std::move(realized_maps);
    }

    void record_unavailable_realized_maps(
        std::vector<mapmaking::ScienceMapRealizedMap> realized_maps,
        std::string absence_reason) {
        if (!initialized || !effective.enabled || science.common_identity ||
            !science.realized_maps.empty() || !science.admissions.empty()) {
            throw std::logic_error(
                "cannot record unavailable coadd product inventory in the current state");
        }
        if (absence_reason.empty() || realized_maps.empty() ||
            std::any_of(
                realized_maps.begin(), realized_maps.end(),
                [](const auto &record) {
                    return !mapmaking::
                        science_map_realized_map_has_explicit_product_absence(
                            record);
                })) {
            throw std::logic_error(
                "science-map coadd absence inventory is incomplete");
        }
        science.realized_maps = std::move(realized_maps);
        science.absence_reason = std::move(absence_reason);
    }

    void record_science_state(
        mapmaking::ScienceMapBundleIdentity common_identity,
        std::vector<mapmaking::ScienceMapRealizedMap> realized_maps) {
        resolve_common_identity(std::move(common_identity));
        record_realized_maps(std::move(realized_maps));
    }

    void record_admission(
        mapmaking::ScienceMapCoaddAdmission admission) {
        validate_admission(admission);
        science.admissions.push_back(std::move(admission));
    }

    void validate_admission(
        const mapmaking::ScienceMapCoaddAdmission &admission) const {
        if (!science.common_identity) {
            throw std::logic_error(
                "cannot record coadd admission before common identity");
        }
        const auto &common_identity = *science.common_identity;
        if (admission.observation_id.empty() ||
            admission.admitted_bundle_identity.empty() ||
            admission.response_identity.empty() ||
            admission.registration_identity !=
                "centered-integer-common-grid-embedding-v1" ||
            admission.centering_identity != "L-identity-v1" ||
            admission.coefficient_stage.empty() ||
            admission.normalization_support_policy !=
                common_identity.normalization_support_policy ||
            admission.science_policy_support_policy !=
                common_identity.science_policy_support_policy ||
            admission.validity_policy != common_identity.validity_policy ||
            admission.nonfinite_policy !=
                common_identity.nonfinite_policy ||
            admission.response_identity !=
                common_identity.response_identity ||
            admission.delta_row < 0 || admission.delta_col < 0 ||
            admission.observation_rows <= 0 ||
            admission.observation_cols <= 0 ||
            admission.coadd_rows != common_identity.rows ||
            admission.coadd_cols != common_identity.cols ||
            admission.coadd_rows < admission.observation_rows ||
            admission.coadd_cols < admission.observation_cols ||
            (admission.coadd_rows - admission.observation_rows) % 2 != 0 ||
            (admission.coadd_cols - admission.observation_cols) % 2 != 0 ||
            admission.delta_row !=
                (admission.coadd_rows - admission.observation_rows) / 2 ||
            admission.delta_col !=
                (admission.coadd_cols - admission.observation_cols) / 2 ||
            admission.ordered_map_count !=
                common_identity.ordered_slots.size() ||
            admission.numerically_contributing_pixel_count.size() !=
                admission.ordered_map_count ||
            admission.observation_raw_parent_digests.size() !=
                admission.ordered_map_count ||
            !std::isfinite(admission.observation_exposure_seconds) ||
            admission.observation_exposure_seconds < 0.0) {
            throw std::logic_error(
                "coadd admission provenance is incomplete");
        }
        if (std::any_of(
                admission.observation_raw_parent_digests.begin(),
                admission.observation_raw_parent_digests.end(),
                [](const auto &digest) { return digest.empty(); })) {
            throw std::logic_error(
                "coadd admission raw-parent digest is incomplete");
        }
        if (admission.coefficient_stage !=
                mapmaking::science_map_observation_unscaled_coefficient_stage &&
            admission.coefficient_stage !=
                mapmaking::science_map_observation_empirical_coefficient_stage) {
            throw std::logic_error(
                "coadd admission coefficient stage is not an allowed realized stage");
        }
        if (common_identity.wcs.reference_pixel.size() != 2) {
            throw std::logic_error(
                "coadd common identity lacks a two-axis reference pixel");
        }
        auto observation_identity = common_identity;
        observation_identity.rows = admission.observation_rows;
        observation_identity.cols = admission.observation_cols;
        observation_identity.wcs.reference_pixel[0] -=
            static_cast<double>(admission.delta_col);
        observation_identity.wcs.reference_pixel[1] -=
            static_cast<double>(admission.delta_row);
        if (admission.admitted_bundle_identity !=
            mapmaking::science_map_bundle_identity_digest(
                observation_identity)) {
            throw std::logic_error(
                "coadd admission observation identity digest is inconsistent");
        }
        if (std::any_of(
                science.admissions.begin(), science.admissions.end(),
                [&admission](const auto &record) {
                    return record.observation_id ==
                           admission.observation_id;
                })) {
            throw std::logic_error(
                "coadd observation admission is already recorded");
        }
    }
};

static_assert(std::is_copy_constructible_v<CoaddExecutionPlan>,
              "coadd admission stages a copy of the execution plan");
static_assert(std::is_nothrow_move_assignable_v<CoaddExecutionPlan>,
              "coadd admission commits the staged plan without failure");

inline void record_coadd_run_completed(
    CoaddExecutionPlan &plan,
    const MapmakingExecutionPlan &mapmaking_plan) {
    if (!plan.initialized) {
        throw std::logic_error("coadd plan is not initialized");
    }
    if (!mapmaking_plan.initialized ||
        !mapmaking_plan.realized.reduction_completed) {
        throw std::logic_error(
            "coadd completion requires completed mapmaking provenance");
    }

    const bool coadd_available = mapmaking_plan.coadd.has_value();
    if (plan.effective.enabled != coadd_available) {
        throw std::logic_error(
            "effective coadd policy differs from realized output state");
    }

    CoaddRealizedState completed;
    if (coadd_available) {
        const auto &coadd = *mapmaking_plan.coadd;
        if (!coadd.outputs_completed || coadd.map_count == 0 ||
            coadd.required_map_write_count < coadd.map_count) {
            throw std::logic_error(
                "coadd output cardinality is incomplete");
        }
        if (!plan.science.realized_maps.empty() &&
            plan.science.realized_maps.size() != coadd.map_count) {
            throw std::logic_error(
                "coadd science-map product inventory cardinality is incomplete");
        }
        completed.coadd_executed = true;
        completed.map_count = coadd.map_count;
        completed.required_map_write_count =
            coadd.required_map_write_count;
        completed.outputs_completed = true;
    }
    completed.reduction_completed = true;
    plan.science.requested_coverage_cut =
        mapmaking_plan.requested.coverage_cut;
    plan.science.effective_coverage_cut =
        mapmaking_plan.effective.coverage_cut;
    plan.realized = std::move(completed);
}

}  // namespace citlali::pipeline
