#pragma once

#include <citlali/core/mapmaking/jinc_contract.h>

#include <Eigen/Core>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline constexpr const char *native_noise_support_authority_v3 =
    "citlali.native_science_projection_v1";

// Bounded identity for one scan's established jackknife sign assignment. The
// sign matrix remains runtime state; canonical provenance carries only its
// shape, population census, and digest.
struct NativeNoiseAssignmentSummaryV3 {
    bool enabled = false;
    bool randomize_detectors = false;
    std::size_t realization_count = 0;
    std::size_t assignment_column_count = 0;
    std::size_t assignment_count = 0;
    std::size_t positive_sign_count = 0;
    std::size_t negative_sign_count = 0;
    std::string assignment_digest;
    std::string support_authority;

    void validate(std::size_t detector_count) const {
        if (!enabled) {
            if (randomize_detectors || realization_count != 0 ||
                assignment_column_count != 0 || assignment_count != 0 ||
                positive_sign_count != 0 || negative_sign_count != 0 ||
                !assignment_digest.empty() || !support_authority.empty()) {
                throw std::logic_error(
                    "disabled native noise assignment carries realized state");
            }
            return;
        }
        const auto expected_columns =
            randomize_detectors ? detector_count : std::size_t{1};
        if (detector_count == 0 || realization_count == 0 ||
            assignment_column_count != expected_columns ||
            realization_count >
                std::numeric_limits<std::size_t>::max() /
                    assignment_column_count ||
            assignment_count !=
                realization_count * assignment_column_count ||
            positive_sign_count > assignment_count ||
            negative_sign_count >
                assignment_count - positive_sign_count ||
            positive_sign_count + negative_sign_count != assignment_count ||
            assignment_digest.empty() ||
            support_authority != native_noise_support_authority_v3) {
            throw std::logic_error(
                "native noise assignment summary is incomplete");
        }
    }
};

template <class SignMatrix>
NativeNoiseAssignmentSummaryV3 make_native_noise_assignment_summary_v3(
    const SignMatrix &signs, bool enabled, bool randomize_detectors,
    std::size_t realization_count, std::size_t detector_count) {
    if (!enabled) {
        if (signs.size() != 0) {
            throw std::logic_error(
                "disabled native noise assignment has sign data");
        }
        return {};
    }
    const auto expected_columns =
        randomize_detectors ? detector_count : std::size_t{1};
    const auto eigen_index_max =
        static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
    if (realization_count == 0 || detector_count == 0 ||
        realization_count > eigen_index_max ||
        expected_columns > eigen_index_max ||
        signs.rows() != static_cast<Eigen::Index>(realization_count) ||
        signs.cols() != static_cast<Eigen::Index>(expected_columns)) {
        throw std::logic_error(
            "native noise assignment shape differs from effective policy");
    }
    NativeNoiseAssignmentSummaryV3 result;
    result.enabled = true;
    result.randomize_detectors = randomize_detectors;
    result.realization_count = realization_count;
    result.assignment_column_count = expected_columns;
    result.assignment_count = static_cast<std::size_t>(signs.size());
    for (Eigen::Index row = 0; row < signs.rows(); ++row) {
        for (Eigen::Index column = 0; column < signs.cols(); ++column) {
            const auto sign = signs(row, column);
            if (sign == 1) {
                ++result.positive_sign_count;
            }
            else if (sign == -1) {
                ++result.negative_sign_count;
            }
            else {
                throw std::logic_error(
                    "native noise assignment sign must be exactly -1 or +1");
            }
        }
    }
    result.assignment_digest = mapmaking::jinc_matrix_digest(signs);
    result.support_authority = native_noise_support_authority_v3;
    result.validate(detector_count);
    return result;
}

}  // namespace citlali::pipeline
