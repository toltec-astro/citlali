#pragma once

#include <citlali/core/timestream/rtc/response_manifest.h>

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mapmaking {

enum class DiscreteGFieldGroup {
    source_identity,
    construction_route,
    units_and_amplitude,
    wcs_and_pixelization,
    center,
    grid_shape,
    masks_support_and_validity,
    gaussian_family_and_parameters,
    truncation,
    apt_rows_detector_mixture_and_weights,
    normalization_and_denominator,
    crop,
    lifecycle_stage,
    digests,
};

inline constexpr std::array<DiscreteGFieldGroup, 14>
    discrete_g_field_groups{
        DiscreteGFieldGroup::source_identity,
        DiscreteGFieldGroup::construction_route,
        DiscreteGFieldGroup::units_and_amplitude,
        DiscreteGFieldGroup::wcs_and_pixelization,
        DiscreteGFieldGroup::center,
        DiscreteGFieldGroup::grid_shape,
        DiscreteGFieldGroup::masks_support_and_validity,
        DiscreteGFieldGroup::gaussian_family_and_parameters,
        DiscreteGFieldGroup::truncation,
        DiscreteGFieldGroup::apt_rows_detector_mixture_and_weights,
        DiscreteGFieldGroup::normalization_and_denominator,
        DiscreteGFieldGroup::crop,
        DiscreteGFieldGroup::lifecycle_stage,
        DiscreteGFieldGroup::digests,
    };

inline constexpr std::string_view discrete_g_field_group_name(
    DiscreteGFieldGroup group) {
    switch (group) {
        case DiscreteGFieldGroup::source_identity:
            return "G01_source_identity";
        case DiscreteGFieldGroup::construction_route:
            return "G02_construction_route";
        case DiscreteGFieldGroup::units_and_amplitude:
            return "G03_units_and_amplitude";
        case DiscreteGFieldGroup::wcs_and_pixelization:
            return "G04_WCS_and_pixelization";
        case DiscreteGFieldGroup::center:
            return "G05_center";
        case DiscreteGFieldGroup::grid_shape:
            return "G06_grid_shape";
        case DiscreteGFieldGroup::masks_support_and_validity:
            return "G07_masks_support_and_validity";
        case DiscreteGFieldGroup::gaussian_family_and_parameters:
            return "G08_Gaussian_family_and_parameters";
        case DiscreteGFieldGroup::truncation:
            return "G09_truncation";
        case DiscreteGFieldGroup::apt_rows_detector_mixture_and_weights:
            return "G10_APT_rows_detector_mixture_and_weights";
        case DiscreteGFieldGroup::normalization_and_denominator:
            return "G11_normalization_and_denominator";
        case DiscreteGFieldGroup::crop:
            return "G12_crop";
        case DiscreteGFieldGroup::lifecycle_stage:
            return "G13_lifecycle_stage";
        case DiscreteGFieldGroup::digests:
            return "G14_digests";
    }
    return "";
}

enum class DiscreteGFrame {
    altaz_tangent_plane,
};

enum class DiscreteGPixelization {
    pixel_center_sampled,
};

enum class DiscreteGSignalUnit {
    mjy_per_beam,
};

enum class DiscreteGState {
    available,
    unavailable,
};

struct DiscreteGFieldEvidence {
    DiscreteGFieldGroup group = DiscreteGFieldGroup::source_identity;
    std::string identity;
    std::string digest;
};

struct DiscreteGRetainedGrid {
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;
    Eigen::Matrix2d wcs_linear_arcsec_per_pixel = Eigen::Matrix2d::Zero();
    double crpix1_fits = 0.0;
    double crpix2_fits = 0.0;
    double crval1 = 0.0;
    double crval2 = 0.0;
    DiscreteGFrame frame = DiscreteGFrame::altaz_tangent_plane;
    DiscreteGPixelization pixelization =
        DiscreteGPixelization::pixel_center_sampled;
    DiscreteGSignalUnit signal_unit = DiscreteGSignalUnit::mjy_per_beam;
    std::string axis1;
    std::string axis2;
    std::string angular_unit;
    std::string epoch;
};

struct DiscreteGDetectorComponent {
    std::string apt_row_identity;
    std::string detector_identity;
    double fwhm_arcsec = 0.0;
    Eigen::MatrixXd weights;
};

struct DiscreteGInput {
    std::vector<DiscreteGFieldEvidence> field_groups;
    DiscreteGRetainedGrid grid;
    std::vector<DiscreteGDetectorComponent> detectors;
    Eigen::MatrixXd denominator;
};

struct DiscreteGResult {
    DiscreteGState state = DiscreteGState::unavailable;
    Eigen::MatrixXd plane;
    std::vector<timestream::RTCUnavailableReason> unavailable_reasons;
};

inline DiscreteGResult render_discrete_g(const DiscreteGInput &input) {
    using timestream::RTCUnavailableReasonCode;
    DiscreteGResult result;
    auto unavailable = [&](RTCUnavailableReasonCode code,
                           std::string affected_object,
                           std::string evidence) {
        result.unavailable_reasons.push_back(
            {code, std::move(affected_object), std::move(evidence)});
    };

    std::array<std::size_t, discrete_g_field_groups.size()> group_counts{};
    for (const auto &evidence : input.field_groups) {
        const auto position = std::find(discrete_g_field_groups.begin(),
                                        discrete_g_field_groups.end(),
                                        evidence.group);
        if (position == discrete_g_field_groups.end()) {
            unavailable(
                RTCUnavailableReasonCode::missing_or_mismatched_artifact_identity,
                "discrete_g_field_group", "unrecognized field group");
            continue;
        }
        const auto index = static_cast<std::size_t>(
            std::distance(discrete_g_field_groups.begin(), position));
        ++group_counts[index];
        if (evidence.identity.empty() ||
            !timestream::rtc_exact_sha256_digest(evidence.digest)) {
            unavailable(
                RTCUnavailableReasonCode::missing_or_mismatched_artifact_identity,
                std::string(discrete_g_field_group_name(evidence.group)),
                "field-group identity or exact persisted-byte digest is missing");
        }
    }
    for (std::size_t index = 0; index < group_counts.size(); ++index) {
        if (group_counts[index] != 1) {
            unavailable(
                RTCUnavailableReasonCode::missing_or_mismatched_artifact_identity,
                std::string(discrete_g_field_group_name(
                    discrete_g_field_groups[index])),
                "field group must occur exactly once");
        }
    }

    const auto &grid = input.grid;
    if (grid.rows <= 0 || grid.cols <= 0 || grid.rows % 2 == 0 ||
        grid.cols % 2 == 0) {
        unavailable(RTCUnavailableReasonCode::incompatible_grid_wcs_or_units,
                    "retained_terminal_parent_grid",
                    "retained grid must have positive odd row and column extents");
    }
    if (!grid.wcs_linear_arcsec_per_pixel.allFinite() ||
        grid.wcs_linear_arcsec_per_pixel.determinant() == 0.0 ||
        !std::isfinite(grid.crpix1_fits) ||
        !std::isfinite(grid.crpix2_fits) || !std::isfinite(grid.crval1) ||
        !std::isfinite(grid.crval2) || grid.axis1 != "AZOFFSET" ||
        grid.axis2 != "ELOFFSET" || grid.angular_unit != "arcsec" ||
        grid.epoch.empty() ||
        grid.frame != DiscreteGFrame::altaz_tangent_plane ||
        grid.pixelization != DiscreteGPixelization::pixel_center_sampled ||
        grid.signal_unit != DiscreteGSignalUnit::mjy_per_beam) {
        unavailable(RTCUnavailableReasonCode::incompatible_grid_wcs_or_units,
                    "retained_terminal_parent_WCS",
                    "full-precision AltAz tangent-plane WCS is incomplete or incompatible");
    }

    if (input.denominator.rows() != grid.rows ||
        input.denominator.cols() != grid.cols) {
        unavailable(RTCUnavailableReasonCode::invalid_support_denominator,
                    "terminal_parent_C",
                    "denominator shape does not match the retained grid");
    }
    else if (!input.denominator.allFinite() ||
             (input.denominator.array() <= 0.0).any()) {
        unavailable(RTCUnavailableReasonCode::invalid_support_denominator,
                    "terminal_parent_C",
                    "every denominator pixel must be finite and strictly positive");
    }

    if (input.detectors.empty()) {
        unavailable(RTCUnavailableReasonCode::invalid_detector_selection_or_weight,
                    "terminal_parent_detector_population",
                    "no contributing detector is identified");
    }
    std::set<std::string> apt_rows;
    std::set<std::string> detector_ids;
    for (const auto &detector : input.detectors) {
        if (detector.apt_row_identity.empty() ||
            detector.detector_identity.empty() ||
            !apt_rows.insert(detector.apt_row_identity).second ||
            !detector_ids.insert(detector.detector_identity).second) {
            unavailable(
                RTCUnavailableReasonCode::invalid_detector_selection_or_weight,
                "terminal_parent_detector_population",
                "APT row and detector identities must be nonempty and unique");
        }
        if (!std::isfinite(detector.fwhm_arcsec) ||
            detector.fwhm_arcsec <= 0.0) {
            unavailable(RTCUnavailableReasonCode::invalid_gaussian_parameter,
                        detector.detector_identity,
                        "realized per-detector FWHM must be finite and positive");
        }
        if (detector.weights.rows() != grid.rows ||
            detector.weights.cols() != grid.cols ||
            !detector.weights.allFinite()) {
            unavailable(
                RTCUnavailableReasonCode::invalid_detector_selection_or_weight,
                detector.detector_identity,
                "exact contribution-weight plane is missing, non-finite, or shape-incompatible");
        }
    }

    if (!result.unavailable_reasons.empty()) {
        return result;
    }

    Eigen::MatrixXd plane(grid.rows, grid.cols);
    constexpr double amplitude_mjy_per_beam = 1.0;
    const double fwhm_to_sigma =
        1.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    const double center_col = grid.crpix1_fits - 1.0;
    const double center_row = grid.crpix2_fits - 1.0;
    for (Eigen::Index row = 0; row < grid.rows; ++row) {
        for (Eigen::Index col = 0; col < grid.cols; ++col) {
            const Eigen::Vector2d pixel_offset(
                static_cast<double>(col) - center_col,
                static_cast<double>(row) - center_row);
            const Eigen::Vector2d tangent_offset =
                grid.wcs_linear_arcsec_per_pixel * pixel_offset;
            const double rho_squared = tangent_offset.squaredNorm();
            double numerator = 0.0;
            for (const auto &detector : input.detectors) {
                const double sigma = detector.fwhm_arcsec * fwhm_to_sigma;
                const double limit_squared = 9.0 * sigma * sigma;
                const double gaussian =
                    rho_squared <= limit_squared
                        ? std::exp(-rho_squared / (2.0 * sigma * sigma))
                        : 0.0;
                numerator += detector.weights(row, col) * gaussian;
            }
            plane(row, col) =
                amplitude_mjy_per_beam * numerator /
                input.denominator(row, col);
            if (!std::isfinite(plane(row, col))) {
                unavailable(
                    RTCUnavailableReasonCode::incomplete_rectangular_support,
                    "discrete_g_plane",
                    "rendered value is non-finite on the required rectangular grid");
                result.plane.resize(0, 0);
                return result;
            }
        }
    }

    result.state = DiscreteGState::available;
    result.plane = std::move(plane);
    return result;
}

}  // namespace mapmaking
