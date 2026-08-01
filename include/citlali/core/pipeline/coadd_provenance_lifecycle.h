#pragma once

#include <citlali/core/pipeline/coadd_execution_plan.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::pipeline {

inline bool science_map_coadd_admission_provenance_equal(
    const mapmaking::ScienceMapCoaddAdmission &lhs,
    const mapmaking::ScienceMapCoaddAdmission &rhs) {
    return lhs.observation_id == rhs.observation_id &&
           lhs.delta_row == rhs.delta_row &&
           lhs.delta_col == rhs.delta_col &&
           lhs.observation_rows == rhs.observation_rows &&
           lhs.observation_cols == rhs.observation_cols &&
           lhs.coadd_rows == rhs.coadd_rows &&
           lhs.coadd_cols == rhs.coadd_cols &&
           lhs.ordered_map_count == rhs.ordered_map_count &&
           lhs.admitted_bundle_identity == rhs.admitted_bundle_identity &&
           lhs.response_identity == rhs.response_identity &&
           lhs.registration_identity == rhs.registration_identity &&
           lhs.centering_identity == rhs.centering_identity &&
           lhs.coefficient_stage == rhs.coefficient_stage &&
           lhs.normalization_support_policy ==
               rhs.normalization_support_policy &&
           lhs.science_policy_support_policy ==
               rhs.science_policy_support_policy &&
           lhs.validity_policy == rhs.validity_policy &&
           lhs.nonfinite_policy == rhs.nonfinite_policy &&
           mapmaking::science_map_exact_double_equal(
               lhs.observation_exposure_seconds,
               rhs.observation_exposure_seconds) &&
           lhs.numerically_contributing_pixel_count ==
               rhs.numerically_contributing_pixel_count &&
           lhs.observation_raw_parent_digests ==
               rhs.observation_raw_parent_digests;
}

template <class Engine>
void begin_coadd_iteration_if_available(Engine &engine) {
    if constexpr (has_coadd_plan_v<Engine>) {
        auto &plan = coadd_plan(engine);
        if (plan.initialized) {
            plan.begin_iteration();
        }
    }
}

template <class Engine>
void record_coadd_realized_maps_if_available(Engine &engine) {
    if constexpr (has_coadd_plan_v<Engine> &&
                  requires { engine.cmb.science_products; }) {
        auto &plan = coadd_plan(engine);
        if (!plan.initialized || !plan.effective.enabled) {
            return;
        }
        const auto &products = engine.cmb.science_products;
        if (!plan.science.common_identity) {
            if (!products.initialized) {
                plan.science.absence_reason =
                    "science-map coadd products were not initialized";
                return;
            }
            plan.record_unavailable_realized_maps(
                products.realized,
                "coadd common identity was not resolved for the effective profile");
            return;
        }
        if (!products.initialized || !products.identity_admitted ||
            !products.bundle_identity) {
            throw std::logic_error(
                "coadd realized products lack the admitted common identity");
        }
        if (mapmaking::science_map_bundle_identity_digest(
                *plan.science.common_identity) !=
            mapmaking::science_map_bundle_identity_digest(
                *products.bundle_identity)) {
            throw std::logic_error(
                "coadd realized identity differs from admitted common identity");
        }
        if (plan.science.admissions.size() !=
            products.coadd_admissions.size()) {
            throw std::logic_error(
                "coadd realized membership cardinality differs from admitted plan");
        }
        for (std::size_t index = 0;
             index < plan.science.admissions.size(); ++index) {
            if (!science_map_coadd_admission_provenance_equal(
                    plan.science.admissions.at(index),
                    products.coadd_admissions.at(index))) {
                throw std::logic_error(
                    "coadd realized membership differs from admitted plan");
            }
        }
        plan.record_realized_maps(products.realized);
    }
}

}  // namespace citlali::pipeline
