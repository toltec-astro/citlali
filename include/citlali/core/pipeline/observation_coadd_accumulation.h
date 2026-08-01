#pragma once

#include <citlali/core/mapmaking/science_map_contract.h>
#include <citlali/core/pipeline/stage_profile.h>
#include <citlali/core/utils/utils.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

// Profiles without the approved ordinary SCI-MAP-001 contribution predicate
// retain the pre-repair arithmetic and publish an explicit product absence.
// This compatibility lane is intentionally not presented as F009 admission.
inline std::tuple<Eigen::Index, Eigen::Index> legacy_centered_coadd_offsets(
    Eigen::Index coadd_rows, Eigen::Index coadd_cols, Eigen::Index obs_rows,
    Eigen::Index obs_cols) {
    return {
        static_cast<Eigen::Index>(0.5 * (coadd_rows - obs_rows)),
        static_cast<Eigen::Index>(0.5 * (coadd_cols - obs_cols))};
}

template <class CoaddMapBuffer, class ObservationMapBuffer>
void accumulate_legacy_observation_into_coadd(
    CoaddMapBuffer &cmb, const ObservationMapBuffer &omb,
    Eigen::Index n_maps, bool run_kernel) {
    const auto [delta_row, delta_col] = legacy_centered_coadd_offsets(
        cmb.n_rows, cmb.n_cols, omb.n_rows, omb.n_cols);

    for (Eigen::Index map_index = 0; map_index < n_maps; ++map_index) {
        auto cmb_weight_block = cmb.weight.at(map_index).block(
            delta_row, delta_col, omb.n_rows, omb.n_cols);
        auto cmb_signal_block = cmb.signal.at(map_index).block(
            delta_row, delta_col, omb.n_rows, omb.n_cols);
        cmb_weight_block += omb.weight.at(map_index);
        cmb_signal_block +=
            (omb.signal.at(map_index).array() *
             omb.weight.at(map_index).array())
                .matrix();

        if (run_kernel) {
            auto cmb_kernel_block = cmb.kernel.at(map_index).block(
                delta_row, delta_col, omb.n_rows, omb.n_cols);
            cmb_kernel_block +=
                (omb.kernel.at(map_index).array() *
                 omb.weight.at(map_index).array())
                    .matrix();
        }
        if (!cmb.coverage.empty()) {
            auto cmb_coverage_block = cmb.coverage.at(map_index).block(
                delta_row, delta_col, omb.n_rows, omb.n_cols);
            cmb_coverage_block += omb.coverage.at(map_index);
        }
        if (!cmb.noise.empty() && !omb.noise.empty()) {
            for (Eigen::Index realization = 0;
                 realization < cmb.n_noise; ++realization) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                         Eigen::Dynamic>> cmb_noise(
                    cmb.noise.at(map_index).data() +
                        realization * cmb.n_rows * cmb.n_cols,
                    cmb.n_rows, cmb.n_cols);
                Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                               Eigen::Dynamic>> omb_noise(
                    omb.noise.at(map_index).data() +
                        realization * omb.n_rows * omb.n_cols,
                    omb.n_rows, omb.n_cols);
                auto cmb_noise_block = cmb_noise.block(
                    delta_row, delta_col, omb.n_rows, omb.n_cols);
                cmb_noise_block +=
                    (omb_noise.array() * omb.weight.at(map_index).array())
                        .matrix();
            }
        }
    }
}

inline std::tuple<Eigen::Index, Eigen::Index> centered_coadd_offsets(
    Eigen::Index coadd_rows, Eigen::Index coadd_cols, Eigen::Index obs_rows,
    Eigen::Index obs_cols) {
    if (coadd_rows <= 0 || coadd_cols <= 0 || obs_rows <= 0 || obs_cols <= 0) {
        throw std::runtime_error(
            "coadd admission requires positive observation and coadd shapes");
    }
    if (coadd_rows < obs_rows || coadd_cols < obs_cols) {
        throw std::runtime_error(
            "coadd admission rejected a negative centered shape offset");
    }
    const Eigen::Index row_difference = coadd_rows - obs_rows;
    const Eigen::Index col_difference = coadd_cols - obs_cols;
    if ((row_difference % 2) != 0 || (col_difference % 2) != 0) {
        throw std::runtime_error(
            "coadd admission rejected an odd centered shape offset");
    }
    const Eigen::Index delta_row = row_difference / 2;
    const Eigen::Index delta_col = col_difference / 2;
    if (delta_row < 0 || delta_col < 0 ||
        delta_row + obs_rows > coadd_rows ||
        delta_col + obs_cols > coadd_cols) {
        throw std::runtime_error(
            "coadd admission produced an out-of-bounds centered embedding");
    }
    return {delta_row, delta_col};
}

inline bool science_map_exact_double_vector_equal(
    const std::vector<double> &lhs, const std::vector<double> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!mapmaking::science_map_exact_double_equal(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

inline bool science_map_slot_identity_equal(
    const mapmaking::ScienceMapSlotIdentity &lhs,
    const mapmaking::ScienceMapSlotIdentity &rhs) {
    return lhs.ordered_slot == rhs.ordered_slot &&
           lhs.grouping == rhs.grouping &&
           lhs.group_identity == rhs.group_identity &&
           lhs.array_identity == rhs.array_identity &&
           lhs.stokes_identity == rhs.stokes_identity &&
           mapmaking::science_map_exact_double_equal(lhs.frequency_hz,
                                                     rhs.frequency_hz);
}

inline bool science_map_bundle_identity_equal(
    const mapmaking::ScienceMapBundleIdentity &lhs,
    const mapmaking::ScienceMapBundleIdentity &rhs) {
    if (lhs.contract_version != rhs.contract_version ||
        lhs.grouping != rhs.grouping || lhs.signal_unit != rhs.signal_unit ||
        lhs.estimator_identity != rhs.estimator_identity ||
        lhs.response_identity != rhs.response_identity ||
        lhs.parallel_equivalence_policy !=
            rhs.parallel_equivalence_policy ||
        lhs.required_companions != rhs.required_companions ||
        lhs.validity_policy != rhs.validity_policy ||
        lhs.coefficient_policy != rhs.coefficient_policy ||
        lhs.normalization_support_policy !=
            rhs.normalization_support_policy ||
        lhs.science_policy_support_policy !=
            rhs.science_policy_support_policy ||
        lhs.nonfinite_policy != rhs.nonfinite_policy ||
        lhs.rows != rhs.rows || lhs.cols != rhs.cols ||
        lhs.wcs.coordinate_frame != rhs.wcs.coordinate_frame ||
        lhs.wcs.projection != rhs.wcs.projection ||
        lhs.wcs.axis_types != rhs.wcs.axis_types ||
        lhs.wcs.axis_units != rhs.wcs.axis_units ||
        !science_map_exact_double_vector_equal(lhs.wcs.pixel_scale,
                                               rhs.wcs.pixel_scale) ||
        !science_map_exact_double_vector_equal(lhs.wcs.reference_world,
                                               rhs.wcs.reference_world) ||
        !science_map_exact_double_vector_equal(lhs.wcs.reference_pixel,
                                               rhs.wcs.reference_pixel) ||
        !mapmaking::science_map_exact_double_equal(lhs.wcs.source_epoch,
                                                   rhs.wcs.source_epoch) ||
        !mapmaking::science_map_exact_double_equal(lhs.wcs.orientation_rad,
                                                   rhs.wcs.orientation_rad) ||
        lhs.ordered_slots.size() != rhs.ordered_slots.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.ordered_slots.size(); ++i) {
        if (!science_map_slot_identity_equal(lhs.ordered_slots[i],
                                             rhs.ordered_slots[i])) {
            return false;
        }
    }
    return true;
}

inline mapmaking::ScienceMapBundleIdentity coadd_bundle_identity_for_embedding(
    const mapmaking::ScienceMapBundleIdentity &observation,
    Eigen::Index coadd_rows, Eigen::Index coadd_cols,
    Eigen::Index delta_row, Eigen::Index delta_col) {
    if (observation.wcs.reference_pixel.size() != 2) {
        throw std::runtime_error(
            "coadd admission requires a two-axis reference-pixel identity");
    }
    auto result = observation;
    result.rows = coadd_rows;
    result.cols = coadd_cols;
    result.wcs.reference_pixel[0] += static_cast<double>(delta_col);
    result.wcs.reference_pixel[1] += static_cast<double>(delta_row);
    return result;
}

template <class Matrix>
bool science_map_matrix_has_shape(const Matrix &matrix, Eigen::Index rows,
                                  Eigen::Index cols) {
    return matrix.rows() == rows && matrix.cols() == cols;
}

template <class Matrix>
bool science_map_double_matrix_exact_equal(const Matrix &lhs,
                                           const Matrix &rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        return false;
    }
    for (Eigen::Index col = 0; col < lhs.cols(); ++col) {
        for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
            if (!mapmaking::science_map_exact_double_equal(lhs(row, col),
                                                           rhs(row, col))) {
                return false;
            }
        }
    }
    return true;
}

inline std::int64_t science_map_checked_count_add(std::int64_t lhs,
                                                  std::int64_t rhs) {
    if (rhs < 0 || lhs < 0 ||
        lhs > std::numeric_limits<std::int64_t>::max() - rhs) {
        throw std::runtime_error(
            "coadd admission rejected an invalid or overflowing count");
    }
    return lhs + rhs;
}

inline double science_map_checked_nonnegative_add(double lhs, double rhs,
                                                  const char *label) {
    if (!std::isfinite(lhs) || lhs < 0.0 || !std::isfinite(rhs) || rhs < 0.0) {
        throw std::runtime_error(std::string("coadd admission rejected ") +
                                 label +
                                 " because it is non-finite or negative");
    }
    const double result = lhs + rhs;
    if (!std::isfinite(result)) {
        throw std::runtime_error(std::string("coadd admission rejected ") +
                                 label + " overflow");
    }
    return result;
}

inline bool science_map_has_companion(
    const mapmaking::ScienceMapBundleIdentity &identity,
    const std::string &name) {
    return std::find(identity.required_companions.begin(),
                     identity.required_companions.end(),
                     name) != identity.required_companions.end();
}

template <class MapBuffer>
void require_science_map_plane_inventory(const MapBuffer &buffer,
                                         Eigen::Index n_maps,
                                         bool coadd) {
    const auto count = static_cast<std::size_t>(n_maps);
    const auto &products = buffer.science_products;
    if (!products.initialized || products.is_coadd != coadd ||
        products.geometric_hits.size() != count ||
        products.contributing_hits.size() != count ||
        products.coadd_observation_count.size() != count ||
        products.upstream_eligible_exposure.size() != count ||
        products.retained_exposure.size() != count ||
        products.normalization_support.size() != count ||
        products.science_policy_support.size() != count ||
        products.science_valid.size() != count ||
        products.realized.size() != count) {
        throw std::runtime_error(
            "coadd admission rejected an incomplete F010 product inventory");
    }
}

template <class MapBuffer>
void require_explicit_unavailable_science_map_inventory(
    const MapBuffer &buffer, Eigen::Index n_maps, bool coadd) {
    const auto count = static_cast<std::size_t>(n_maps);
    const auto &products = buffer.science_products;
    const bool has_plane_inventory =
        !products.geometric_hits.empty() ||
        !products.contributing_hits.empty() ||
        !products.coadd_observation_count.empty() ||
        !products.upstream_eligible_exposure.empty() ||
        !products.retained_exposure.empty() ||
        !products.normalization_support.empty() ||
        !products.science_policy_support.empty() ||
        !products.science_valid.empty();
    if (!products.initialized || products.is_coadd != coadd ||
        products.realized.size() != count || has_plane_inventory) {
        throw std::runtime_error(
            "coadd dispatch rejected an incomplete unavailable-profile inventory");
    }
    for (const auto &record : products.realized) {
        for (std::size_t product = 0;
             product < static_cast<std::size_t>(
                           mapmaking::ScienceMapProduct::count);
             ++product) {
            if (record.product_available[product] ||
                record.product_absence_reason[product].empty()) {
                throw std::runtime_error(
                    "coadd dispatch requires explicit successor-product absence");
            }
        }
    }
}

template <class CoaddMapBuffer, class ObservationMapBuffer>
bool science_map_v1_coadd_profile_enabled(
    const CoaddMapBuffer &cmb, const ObservationMapBuffer &omb,
    Eigen::Index n_maps) {
    if (n_maps <= 0) {
        throw std::runtime_error(
            "coadd dispatch requires a positive map count");
    }
    const auto &observation_products = omb.science_products;
    const auto &coadd_products = cmb.science_products;
    if (!observation_products.initialized || !coadd_products.initialized) {
        throw std::runtime_error(
            "coadd dispatch requires initialized science-map profile authority");
    }
    if (observation_products.ordinary_contribution_predicate_available !=
        coadd_products.ordinary_contribution_predicate_available) {
        throw std::runtime_error(
            "observation/coadd science-map profile authority diverged");
    }
    if (!observation_products.ordinary_contribution_predicate_available) {
        require_explicit_unavailable_science_map_inventory(
            cmb, n_maps, true);
        require_explicit_unavailable_science_map_inventory(
            omb, n_maps, false);
        return false;
    }
    require_science_map_plane_inventory(cmb, n_maps, true);
    require_science_map_plane_inventory(omb, n_maps, false);
    return true;
}

template <class CoaddMapBuffer, class ObservationMapBuffer>
mapmaking::ScienceMapCoaddAdmission preflight_observation_for_coadd(
    const CoaddMapBuffer &cmb, const ObservationMapBuffer &omb,
    Eigen::Index n_maps, bool run_kernel, const std::string &observation_id,
    double observation_exposure_seconds) {
    if (n_maps <= 0 || observation_id.empty()) {
        throw std::runtime_error(
            "coadd admission requires a nonempty observation bundle identity");
    }
    if (!std::isfinite(observation_exposure_seconds) ||
        observation_exposure_seconds < 0.0) {
        throw std::runtime_error(
            "coadd admission requires finite nonnegative observation exposure");
    }
    if (!std::isfinite(omb.cov_cut) || omb.cov_cut < 0.0) {
        throw std::runtime_error(
            "coadd admission requires a finite nonnegative coverage cut");
    }

    const auto [delta_row, delta_col] = centered_coadd_offsets(
        cmb.n_rows, cmb.n_cols, omb.n_rows, omb.n_cols);
    const auto count = static_cast<std::size_t>(n_maps);
    if (cmb.signal.size() != count || cmb.weight.size() != count ||
        omb.signal.size() != count || omb.weight.size() != count) {
        throw std::runtime_error(
            "coadd admission rejected a signal/coefficient map cardinality mismatch");
    }
    if ((run_kernel &&
         (cmb.kernel.size() != count || omb.kernel.size() != count)) ||
        (!run_kernel && (!cmb.kernel.empty() || !omb.kernel.empty()))) {
        throw std::runtime_error(
            "coadd admission rejected a response/kernel inventory mismatch");
    }
    if (cmb.coverage.size() != count || omb.coverage.size() != count) {
        throw std::runtime_error(
            "coadd admission requires the retained-exposure compatibility alias");
    }
    if (cmb.noise.empty() != omb.noise.empty() ||
        cmb.n_noise != omb.n_noise ||
        (!cmb.noise.empty() &&
         (cmb.noise.size() != count || omb.noise.size() != count))) {
        throw std::runtime_error(
            "coadd admission rejected a noise-realization inventory mismatch");
    }

    require_science_map_plane_inventory(cmb, n_maps, true);
    require_science_map_plane_inventory(omb, n_maps, false);
    const auto &observation_products = omb.science_products;
    const auto &coadd_products = cmb.science_products;
    if (!observation_products.identity_admitted ||
        !observation_products.bundle_identity.has_value()) {
        throw std::runtime_error(
            "coadd admission rejected a missing full-precision observation identity");
    }
    const auto &observation_identity =
        *observation_products.bundle_identity;
    if (observation_identity.rows != omb.n_rows ||
        observation_identity.cols != omb.n_cols ||
        observation_identity.ordered_slots.size() != count ||
        observation_identity.grouping != omb.map_grouping ||
        observation_identity.signal_unit != omb.sig_unit) {
        throw std::runtime_error(
            "coadd admission rejected observation state/identity divergence");
    }
    if (!mapmaking::science_map_exact_double_equal(cmb.pixel_size_rad,
                                                   omb.pixel_size_rad) ||
        (!cmb.sig_unit.empty() && cmb.sig_unit != observation_identity.signal_unit) ||
        (!cmb.map_grouping.empty() &&
         cmb.map_grouping != observation_identity.grouping)) {
        throw std::runtime_error(
            "coadd admission rejected coadd geometry, unit, or grouping identity");
    }
    if (run_kernel !=
        science_map_has_companion(observation_identity, "kernel_I")) {
        throw std::runtime_error(
            "coadd admission rejected a kernel required-companion mismatch");
    }
    for (Eigen::Index realization = 0; realization < omb.n_noise;
         ++realization) {
        const auto name = "noise_realization_" +
                          std::to_string(realization) + "_I";
        if (!science_map_has_companion(observation_identity, name)) {
            throw std::runtime_error(
                "coadd admission rejected an undeclared noise companion");
        }
    }

    const auto expected_coadd_identity = coadd_bundle_identity_for_embedding(
        observation_identity, cmb.n_rows, cmb.n_cols, delta_row, delta_col);
    if (coadd_products.bundle_identity.has_value() &&
        (!coadd_products.identity_admitted ||
         !science_map_bundle_identity_equal(
             *coadd_products.bundle_identity, expected_coadd_identity))) {
        throw std::runtime_error(
            "coadd admission rejected a bundle/WCS/response identity mismatch");
    }
    if (!coadd_products.bundle_identity.has_value() &&
        coadd_products.identity_admitted) {
        throw std::runtime_error(
            "coadd admission rejected inconsistent coadd identity state");
    }

    mapmaking::ScienceMapCoaddAdmission admission;
    admission.observation_id = observation_id;
    admission.delta_row = delta_row;
    admission.delta_col = delta_col;
    admission.observation_rows = omb.n_rows;
    admission.observation_cols = omb.n_cols;
    admission.coadd_rows = cmb.n_rows;
    admission.coadd_cols = cmb.n_cols;
    admission.ordered_map_count = count;
    admission.admitted_bundle_identity =
        mapmaking::science_map_bundle_identity_digest(observation_identity);
    admission.response_identity = observation_identity.response_identity;
    admission.coefficient_stage = observation_products.coefficient_stage;
    admission.normalization_support_policy =
        observation_identity.normalization_support_policy;
    admission.science_policy_support_policy =
        observation_identity.science_policy_support_policy;
    admission.validity_policy = observation_identity.validity_policy;
    admission.nonfinite_policy = observation_identity.nonfinite_policy;
    admission.observation_exposure_seconds = observation_exposure_seconds;
    admission.numerically_contributing_pixel_count.assign(count, 0U);
    admission.observation_raw_parent_digests.reserve(count);
    const std::string observation_identity_digest =
        mapmaking::science_map_bundle_identity_digest(observation_identity);

    for (Eigen::Index map_index = 0; map_index < n_maps; ++map_index) {
        const auto slot = static_cast<std::size_t>(map_index);
        if (!science_map_matrix_has_shape(omb.signal[slot], omb.n_rows,
                                          omb.n_cols) ||
            !science_map_matrix_has_shape(omb.weight[slot], omb.n_rows,
                                          omb.n_cols) ||
            !science_map_matrix_has_shape(cmb.signal[slot], cmb.n_rows,
                                          cmb.n_cols) ||
            !science_map_matrix_has_shape(cmb.weight[slot], cmb.n_rows,
                                          cmb.n_cols)) {
            throw std::runtime_error(
                "coadd admission rejected a numerical plane shape mismatch");
        }
        if (run_kernel &&
            (!science_map_matrix_has_shape(omb.kernel[slot], omb.n_rows,
                                            omb.n_cols) ||
             !science_map_matrix_has_shape(cmb.kernel[slot], cmb.n_rows,
                                            cmb.n_cols))) {
            throw std::runtime_error(
                "coadd admission rejected a kernel plane shape mismatch");
        }
        if (!science_map_matrix_has_shape(omb.coverage[slot], omb.n_rows,
                                          omb.n_cols) ||
            !science_map_matrix_has_shape(cmb.coverage[slot], cmb.n_rows,
                                          cmb.n_cols)) {
            throw std::runtime_error(
                "coadd admission rejected a coverage-alias shape mismatch");
        }

        const auto require_obs_product_shape = [&](const auto &plane) {
            return science_map_matrix_has_shape(plane, omb.n_rows, omb.n_cols);
        };
        const auto require_coadd_product_shape = [&](const auto &plane) {
            return science_map_matrix_has_shape(plane, cmb.n_rows, cmb.n_cols);
        };
        if (!require_obs_product_shape(
                observation_products.geometric_hits[slot]) ||
            !require_obs_product_shape(
                observation_products.contributing_hits[slot]) ||
            !require_obs_product_shape(
                observation_products.coadd_observation_count[slot]) ||
            !require_obs_product_shape(
                observation_products.upstream_eligible_exposure[slot]) ||
            !require_obs_product_shape(
                observation_products.retained_exposure[slot]) ||
            !require_obs_product_shape(
                observation_products.normalization_support[slot]) ||
            !require_obs_product_shape(
                observation_products.science_policy_support[slot]) ||
            !require_obs_product_shape(observation_products.science_valid[slot]) ||
            !require_coadd_product_shape(coadd_products.geometric_hits[slot]) ||
            !require_coadd_product_shape(coadd_products.contributing_hits[slot]) ||
            !require_coadd_product_shape(coadd_products.coadd_observation_count[slot]) ||
            !require_coadd_product_shape(coadd_products.upstream_eligible_exposure[slot]) ||
            !require_coadd_product_shape(coadd_products.retained_exposure[slot]) ||
            !require_coadd_product_shape(coadd_products.normalization_support[slot]) ||
            !require_coadd_product_shape(coadd_products.science_policy_support[slot]) ||
            !require_coadd_product_shape(coadd_products.science_valid[slot])) {
            throw std::runtime_error(
                "coadd admission rejected an F010 plane shape mismatch");
        }
        if (!science_map_double_matrix_exact_equal(
                omb.coverage[slot],
                observation_products.retained_exposure[slot]) ||
            !science_map_double_matrix_exact_equal(
                cmb.coverage[slot], coadd_products.retained_exposure[slot])) {
            throw std::runtime_error(
                "coadd admission rejected a non-bitwise coverage alias");
        }
        const auto &realized = observation_products.realized[slot];
        const auto threshold_record_valid = [](const auto &threshold,
                                               double expected_cut) {
            if (!mapmaking::science_map_exact_double_equal(
                    threshold.requested_cut, expected_cut) ||
                !mapmaking::science_map_exact_double_equal(
                    threshold.realized_cut, expected_cut) ||
                threshold.order_statistic_algorithm !=
                    mapmaking::science_map_order_statistic_version ||
                !std::isfinite(threshold.realized_threshold) ||
                threshold.realized_threshold < 0.0 ||
                !std::isfinite(threshold.selected_positive_value) ||
                threshold.selected_positive_value < 0.0) {
                return false;
            }
            if (threshold.positive_value_count == 0U) {
                return !threshold.selected_index_available &&
                    threshold.selected_zero_based_index == 0U &&
                    mapmaking::science_map_exact_double_equal(
                        threshold.selected_positive_value, 0.0) &&
                    mapmaking::science_map_exact_double_equal(
                        threshold.realized_threshold, 0.0);
            }
            const auto lower = static_cast<std::size_t>(std::floor(
                0.75 * static_cast<double>(threshold.positive_value_count)));
            const auto expected_index =
                (lower + threshold.positive_value_count) / 2U;
            return threshold.selected_index_available &&
                threshold.selected_zero_based_index == expected_index &&
                expected_index < threshold.positive_value_count &&
                threshold.selected_positive_value > 0.0 &&
                mapmaking::science_map_exact_double_equal(
                    threshold.realized_threshold,
                    threshold.selected_positive_value * expected_cut);
        };
        const bool product_facts_match =
            mapmaking::science_map_realized_product_facts_match(omb, slot);
        if (!realized.initialized || realized.raw_parent_digest.empty() ||
            realized.admitted_bundle_identity != observation_identity_digest ||
            realized.required_companions !=
                observation_identity.required_companions ||
            realized.normalization.support_algorithm !=
                observation_identity.normalization_support_policy ||
            realized.science_policy.support_algorithm !=
                observation_identity.science_policy_support_policy ||
            realized.normalization.order_statistic_algorithm !=
                mapmaking::science_map_order_statistic_version ||
            realized.science_policy.order_statistic_algorithm !=
                mapmaking::science_map_order_statistic_version ||
            realized.normalization.coefficient_product != "weight_I" ||
            realized.science_policy.coefficient_product != "weight_I" ||
            realized.normalization.coefficient_stage !=
                mapmaking::science_map_observation_normalization_coefficient_stage ||
            realized.science_policy.coefficient_stage !=
                observation_products.coefficient_stage ||
            realized.normalization.finite_convention !=
                "coefficient must be finite" ||
            realized.science_policy.finite_convention !=
                "coefficient must be finite" ||
            realized.normalization.positivity_convention !=
                "coefficient > 0" ||
            realized.science_policy.positivity_convention !=
                "coefficient > 0" ||
            realized.normalization.comparison_convention != ">=" ||
            realized.science_policy.comparison_convention != ">=" ||
            !threshold_record_valid(realized.normalization,
                                    omb.cov_cut / 10.0) ||
            !threshold_record_valid(realized.science_policy, omb.cov_cut) ||
            !product_facts_match) {
            throw std::runtime_error(
                "coadd admission rejected incomplete or tampered observation realized provenance");
        }
        if (observation_products.coefficient_stage !=
                mapmaking::science_map_observation_unscaled_coefficient_stage &&
            observation_products.coefficient_stage !=
                mapmaking::science_map_observation_empirical_coefficient_stage) {
            throw std::runtime_error(
                "coadd admission rejected an unknown coefficient lifecycle stage");
        }
        if (realized.raw_parent_digest !=
            mapmaking::science_map_raw_parent_digest(omb, slot)) {
            throw std::runtime_error(
                "coadd admission rejected a stale or tampered raw-parent/product digest");
        }
        admission.observation_raw_parent_digests.push_back(
            realized.raw_parent_digest);

        if (!cmb.noise.empty()) {
            if (omb.noise[slot].dimension(0) != omb.n_rows ||
                omb.noise[slot].dimension(1) != omb.n_cols ||
                omb.noise[slot].dimension(2) != omb.n_noise ||
                cmb.noise[slot].dimension(0) != cmb.n_rows ||
                cmb.noise[slot].dimension(1) != cmb.n_cols ||
                cmb.noise[slot].dimension(2) != cmb.n_noise) {
                throw std::runtime_error(
                    "coadd admission rejected a realization cube shape mismatch");
            }
        }

        for (Eigen::Index col = 0; col < omb.n_cols; ++col) {
            for (Eigen::Index row = 0; row < omb.n_rows; ++row) {
                const Eigen::Index coadd_row = row + delta_row;
                const Eigen::Index coadd_col = col + delta_col;
                const auto normalization =
                    observation_products.normalization_support[slot](row, col);
                const auto science_policy =
                    observation_products.science_policy_support[slot](row, col);
                const auto science_valid =
                    observation_products.science_valid[slot](row, col);
                if (normalization > 1 || science_policy > 1 ||
                    science_valid > 1 ||
                    (science_valid != 0 &&
                     (normalization == 0 || science_policy == 0))) {
                    throw std::runtime_error(
                        "coadd admission rejected a nonbinary or inconsistent validity state");
                }

                const auto geometric =
                    observation_products.geometric_hits[slot](row, col);
                const auto contributing =
                    observation_products.contributing_hits[slot](row, col);
                const double eligible =
                    observation_products.upstream_eligible_exposure[slot](row,
                                                                           col);
                (void)science_map_checked_count_add(
                    coadd_products.geometric_hits[slot](coadd_row, coadd_col),
                    geometric);
                (void)science_map_checked_nonnegative_add(
                    coadd_products.upstream_eligible_exposure[slot](
                        coadd_row, coadd_col),
                    eligible, "upstream-eligible exposure");

                // The authoritative normalization-support state is the
                // numerical membership input. A false state is skipped before
                // signal, coefficient, kernel, noise, or retained exposure is
                // evaluated. It is deliberately not reconstructed from u>0.
                if (normalization == 0) {
                    if (science_valid != 0) {
                        throw std::runtime_error(
                            "coadd admission rejected validity outside normalization support");
                    }
                    continue;
                }

                const double coefficient = omb.weight[slot](row, col);
                const double signal = omb.signal[slot](row, col);
                if (!std::isfinite(coefficient) || coefficient <= 0.0 ||
                    !std::isfinite(signal)) {
                    throw std::runtime_error(
                        "coadd admission rejected a non-finite signal or nonpositive coefficient on declared support");
                }
                const double weighted_signal = coefficient * signal;
                const double next_weight =
                    cmb.weight[slot](coadd_row, coadd_col) + coefficient;
                const double next_signal =
                    cmb.signal[slot](coadd_row, coadd_col) + weighted_signal;
                if (!std::isfinite(weighted_signal) ||
                    !std::isfinite(next_weight) || next_weight <= 0.0 ||
                    !std::isfinite(next_signal)) {
                    throw std::runtime_error(
                        "coadd admission rejected numerical accumulator overflow");
                }
                if (run_kernel) {
                    const double kernel = omb.kernel[slot](row, col);
                    const double weighted_kernel = coefficient * kernel;
                    const double next_kernel =
                        cmb.kernel[slot](coadd_row, coadd_col) +
                        weighted_kernel;
                    if (!std::isfinite(kernel) ||
                        !std::isfinite(weighted_kernel) ||
                        !std::isfinite(next_kernel)) {
                        throw std::runtime_error(
                            "coadd admission rejected a non-finite required kernel companion");
                    }
                }
                for (Eigen::Index realization = 0;
                     realization < omb.n_noise; ++realization) {
                    if (cmb.noise.empty()) {
                        break;
                    }
                    const double noise =
                        omb.noise[slot](row, col, realization);
                    const double weighted_noise = coefficient * noise;
                    const double next_noise =
                        cmb.noise[slot](coadd_row, coadd_col, realization) +
                        weighted_noise;
                    if (!std::isfinite(noise) ||
                        !std::isfinite(weighted_noise) ||
                        !std::isfinite(next_noise)) {
                        throw std::runtime_error(
                            "coadd admission rejected a non-finite required realization companion");
                    }
                }

                const double retained =
                    observation_products.retained_exposure[slot](row, col);
                (void)science_map_checked_nonnegative_add(
                    coadd_products.retained_exposure[slot](coadd_row,
                                                           coadd_col),
                    retained, "retained exposure");
                (void)science_map_checked_count_add(
                    coadd_products.contributing_hits[slot](coadd_row,
                                                           coadd_col),
                    contributing);
                (void)science_map_checked_count_add(
                    coadd_products.coadd_observation_count[slot](
                        coadd_row, coadd_col),
                    1);
                const bool expected_valid = science_policy != 0;
                if ((science_valid != 0) != expected_valid) {
                    throw std::runtime_error(
                        "coadd admission rejected observation validity/companion divergence");
                }
                ++admission.numerically_contributing_pixel_count[slot];
            }
        }
    }

    (void)science_map_checked_nonnegative_add(
        cmb.exposure_time, observation_exposure_seconds,
        "coadd observation exposure");
    return admission;
}

template <class CoaddMapBuffer, class ObservationMapBuffer>
void commit_observation_to_coadd(
    CoaddMapBuffer &cmb, const ObservationMapBuffer &omb,
    Eigen::Index n_maps, bool run_kernel,
    mapmaking::ScienceMapCoaddAdmission admission) {
    const auto count = static_cast<std::size_t>(n_maps);
    auto next_obsnums = cmb.obsnums;
    next_obsnums.reserve(next_obsnums.size() + 1);
    next_obsnums.push_back(admission.observation_id);
    auto next_admissions = cmb.science_products.coadd_admissions;
    next_admissions.reserve(next_admissions.size() + 1);
    next_admissions.push_back(admission);

    const auto &observation_identity =
        *omb.science_products.bundle_identity;
    const auto coadd_identity = coadd_bundle_identity_for_embedding(
        observation_identity, cmb.n_rows, cmb.n_cols, admission.delta_row,
        admission.delta_col);
    auto next_identity = cmb.science_products.bundle_identity;
    bool next_identity_admitted = cmb.science_products.identity_admitted;
    auto next_map_grouping = cmb.map_grouping;
    auto next_signal_unit = cmb.sig_unit;
    auto next_wcs = cmb.wcs;
    if (!next_identity.has_value()) {
        next_identity = coadd_identity;
        next_identity_admitted = true;
        next_map_grouping = coadd_identity.grouping;
        next_signal_unit = coadd_identity.signal_unit;
        if (next_wcs.crval.size() >= 2 &&
            coadd_identity.wcs.reference_world.size() == 2) {
            next_wcs.crval[0] = static_cast<float>(
                coadd_identity.wcs.reference_world[0]);
            next_wcs.crval[1] = static_cast<float>(
                coadd_identity.wcs.reference_world[1]);
        }
        if (next_wcs.crpix.size() >= 2 &&
            coadd_identity.wcs.reference_pixel.size() == 2) {
            next_wcs.crpix[0] = static_cast<float>(
                coadd_identity.wcs.reference_pixel[0]);
            next_wcs.crpix[1] = static_cast<float>(
                coadd_identity.wcs.reference_pixel[1]);
        }
    }
    static_assert(std::is_nothrow_move_assignable_v<decltype(next_identity)>);
    static_assert(std::is_nothrow_move_assignable_v<decltype(next_wcs)>);

    // No operation below performs a fallible compatibility check. All map
    // slots and all pixels have passed the immutable-bundle preflight.
    for (Eigen::Index map_index = 0; map_index < n_maps; ++map_index) {
        const auto slot = static_cast<std::size_t>(map_index);
        for (Eigen::Index col = 0; col < omb.n_cols; ++col) {
            for (Eigen::Index row = 0; row < omb.n_rows; ++row) {
                const Eigen::Index coadd_row = row + admission.delta_row;
                const Eigen::Index coadd_col = col + admission.delta_col;

                cmb.science_products.geometric_hits[slot](coadd_row,
                                                           coadd_col) +=
                    omb.science_products.geometric_hits[slot](row, col);
                cmb.science_products.upstream_eligible_exposure[slot](
                    coadd_row, coadd_col) +=
                    omb.science_products.upstream_eligible_exposure[slot](row,
                                                                          col);

                if (omb.science_products.normalization_support[slot](row,
                                                                      col) ==
                    0) {
                    continue;
                }
                const double coefficient = omb.weight[slot](row, col);
                cmb.weight[slot](coadd_row, coadd_col) += coefficient;
                cmb.signal[slot](coadd_row, coadd_col) +=
                    coefficient * omb.signal[slot](row, col);
                if (run_kernel) {
                    cmb.kernel[slot](coadd_row, coadd_col) +=
                        coefficient * omb.kernel[slot](row, col);
                }
                for (Eigen::Index realization = 0;
                     realization < omb.n_noise; ++realization) {
                    if (cmb.noise.empty()) {
                        break;
                    }
                    cmb.noise[slot](coadd_row, coadd_col, realization) +=
                        coefficient *
                        omb.noise[slot](row, col, realization);
                }
                cmb.science_products.contributing_hits[slot](coadd_row,
                                                              coadd_col) +=
                    omb.science_products.contributing_hits[slot](row, col);
                cmb.science_products.coadd_observation_count[slot](
                    coadd_row, coadd_col) += 1;
                cmb.science_products.retained_exposure[slot](coadd_row,
                                                              coadd_col) +=
                    omb.science_products.retained_exposure[slot](row, col);
                cmb.coverage[slot](coadd_row, coadd_col) =
                    cmb.science_products.retained_exposure[slot](coadd_row,
                                                                 coadd_col);
            }
        }
    }

    cmb.exposure_time += admission.observation_exposure_seconds;
    cmb.science_products.bundle_identity = std::move(next_identity);
    cmb.science_products.identity_admitted = next_identity_admitted;
    cmb.map_grouping.swap(next_map_grouping);
    cmb.sig_unit.swap(next_signal_unit);
    cmb.wcs = std::move(next_wcs);
    cmb.obsnums.swap(next_obsnums);
    cmb.science_products.coadd_admissions.swap(next_admissions);
}

template <class CoaddMapBuffer, class ObservationMapBuffer>
void accumulate_observation_into_coadd(
    CoaddMapBuffer &cmb, const ObservationMapBuffer &omb,
    Eigen::Index n_maps, bool run_kernel, const std::string &observation_id,
    double observation_exposure_seconds) {
    auto admission = preflight_observation_for_coadd(
        cmb, omb, n_maps, run_kernel, observation_id,
        observation_exposure_seconds);
    commit_observation_to_coadd(cmb, omb, n_maps, run_kernel,
                                std::move(admission));
}

template <class Engine>
bool should_run_observation_coadd(const Engine &engine) {
    return !engine.rtcproc.run_polarization;
}

template <class TodProc, class Logger>
void coadd_observation(TodProc &todproc,
                       StageProfileCollector &stage_profile,
                       const Logger &logger) {
    auto &engine = todproc.engine();
    (void)stage_profile;

    logger->info("coadding");
    const auto profile_scope =
        profile_stage(stage_profile, "observation.coadd", logger);
    if (should_run_observation_coadd(engine)) {
        todproc.coadd();
    }
}

}  // namespace citlali::pipeline
