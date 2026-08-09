#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace timestream {

enum class CalibrationValidityCause {
    not_evaluated,
    valid_complete_product,
    calibration_not_requested,
    unsupported_target_unit,
    acquisition_identity_unavailable,
    acquisition_identity_invalid,
    missing_required_factor,
    invalid_required_factor,
    invalid_atmosphere_support,
};

inline constexpr std::string_view to_string(CalibrationValidityCause cause) {
    switch (cause) {
        case CalibrationValidityCause::not_evaluated:
            return "not_evaluated";
        case CalibrationValidityCause::valid_complete_product:
            return "valid_complete_product";
        case CalibrationValidityCause::calibration_not_requested:
            return "calibration_not_requested";
        case CalibrationValidityCause::unsupported_target_unit:
            return "unsupported_target_unit";
        case CalibrationValidityCause::acquisition_identity_unavailable:
            return "acquisition_identity_unavailable";
        case CalibrationValidityCause::acquisition_identity_invalid:
            return "acquisition_identity_invalid";
        case CalibrationValidityCause::missing_required_factor:
            return "missing_required_factor";
        case CalibrationValidityCause::invalid_required_factor:
            return "invalid_required_factor";
        case CalibrationValidityCause::invalid_atmosphere_support:
            return "invalid_atmosphere_support";
    }
    return "unknown";
}

enum class CalibrationNuisanceAvailability {
    available,
    unavailable,
    not_applicable,
};

inline constexpr std::string_view to_string(
    CalibrationNuisanceAvailability availability) {
    switch (availability) {
        case CalibrationNuisanceAvailability::available:
            return "available";
        case CalibrationNuisanceAvailability::unavailable:
            return "unavailable";
        case CalibrationNuisanceAvailability::not_applicable:
            return "not_applicable";
    }
    return "unknown";
}

struct CalibrationNuisanceState {
    std::string id;
    CalibrationNuisanceAvailability value_availability =
        CalibrationNuisanceAvailability::unavailable;
    CalibrationNuisanceAvailability availability =
        CalibrationNuisanceAvailability::unavailable;
    std::string validity{"unavailable"};
    std::string correlation_scope;
    std::string value_source;
    std::string uncertainty_source;
    std::string limitation;
};

struct CalibrationProductAdmissionInputs {
    std::string target_unit;
    bool calibration_requested = false;
    bool extinction_requested = false;
    bool responsivity_required = false;
    bool sensitivity_required = false;
    bool beam_template_required = true;
    bool acquisition_identity_available = false;
    bool acquisition_identity_valid = false;
    std::string acquisition_identity_detail;
    std::string apt_artifact_sha256;
    std::string acquisition_binding_sha256;
    std::string raw_observation_identity;
    std::string acquisition_binding_mode;
    std::string acquisition_key_schema;
    std::string response_identity;
    Eigen::VectorXd target_unit_factor;
    Eigen::VectorXd detector_flxscale;
    Eigen::VectorXd detector_responsivity;
    Eigen::VectorXd detector_sensitivity;
    Eigen::VectorXd detector_beam_major_fwhm_arcsec;
    Eigen::VectorXd detector_beam_minor_fwhm_arcsec;
    Eigen::VectorXd minimum_extinction_correction;
    Eigen::VectorXd maximum_extinction_correction;
};

struct CalibrationProduct {
    static constexpr std::string_view schema_version =
        "sci-cal-001-complete-calibration-product-v1";
    static constexpr std::string_view factor_composition =
        "signal_prime=signal*target_unit_factor*detector_flxscale*sample_extinction_correction";
    static constexpr std::string_view conditional_variance_transfer =
        "variance_prime=total_signal_multiplier^2*conditional_variance";
    static constexpr std::string_view conditional_inverse_variance_transfer =
        "inverse_variance_prime=conditional_inverse_variance/total_signal_multiplier^2";
    static constexpr std::string_view precision_limitation =
        "conditional_only;excludes_calibration_and_response_systematics;not_total_precision_or_significance";
    static constexpr std::string_view photometry_policy =
        "top_of_atmosphere_point_source_peak_mJy_per_beam";
    static constexpr std::string_view factor_provenance =
        "target_unit_factor=unity_dimensionless_for_mJy_per_beam;"
        "detector_flxscale=selected_APT_flxscale[mJy_per_beam_per_xs],multiplicative;"
        "sample_extinction=exp(line_of_sight_optical_depth),multiplicative;"
        "responsivity=despike_donor_target_relative_response_only,not_absolute_flux;"
        "sens=selected_APT_sens[mJy_per_beam_sqrt_s],already_contains_flxscale;"
        "beam_axes=selected_APT_a_fwhm_b_fwhm[arcsec],response_identity_only";
    static constexpr std::string_view compatibility_fcf_semantics =
        "fcf=target_unit_factor_times_scan_mean_extinction;"
        "excludes_detector_flxscale_because_selected_APT_sens_already_contains_flxscale;"
        "not_authoritative_total_calibration";
    static constexpr std::string_view weight_recipient_semantics =
        "approximate_or_hybrid_weight=(sqrt(sample_rate)*fcf*selected_APT_sens)^-2;"
        "full_weight=conditional_inverse_variance_from_already_calibrated_samples;"
        "constant_weight=nonprecision_coefficient;"
        "all_total_precision_and_significance_claims_fail_closed_without_nuisance_covariance";
    static constexpr std::string_view compact_covariance_state =
        "unavailable;no_nuisance_covariance_invented";

    bool admitted = false;
    CalibrationValidityCause validity_cause =
        CalibrationValidityCause::not_evaluated;
    std::string validity_detail{"calibration product has not been evaluated"};
    std::string target_unit;
    std::string apt_artifact_sha256;
    std::string acquisition_binding_sha256;
    std::string raw_observation_identity;
    std::string acquisition_binding_mode;
    std::string acquisition_key_schema;
    std::string response_identity;
    Eigen::VectorXd target_unit_factor;
    Eigen::VectorXd detector_flxscale;
    Eigen::VectorXd signal_multiplier_without_extinction;
    Eigen::VectorXd minimum_extinction_correction;
    Eigen::VectorXd maximum_extinction_correction;
    std::vector<CalibrationNuisanceState> nuisances;

    [[nodiscard]] bool valid() const {
        return admitted &&
               validity_cause ==
                   CalibrationValidityCause::valid_complete_product;
    }
};

inline bool finite_positive_vector(const Eigen::VectorXd &values) {
    return values.size() > 0 &&
           (values.array().isFinite() && (values.array() > 0.0)).all();
}

inline CalibrationProduct reject_calibration_product(
    const CalibrationProductAdmissionInputs &inputs,
    CalibrationValidityCause cause, std::string detail) {
    CalibrationProduct result;
    result.validity_cause = cause;
    result.validity_detail = std::move(detail);
    result.target_unit = inputs.target_unit;
    result.apt_artifact_sha256 = inputs.apt_artifact_sha256;
    result.acquisition_binding_sha256 = inputs.acquisition_binding_sha256;
    result.raw_observation_identity = inputs.raw_observation_identity;
    result.acquisition_binding_mode = inputs.acquisition_binding_mode;
    result.acquisition_key_schema = inputs.acquisition_key_schema;
    result.response_identity = inputs.response_identity;
    return result;
}

inline CalibrationProduct admit_calibration_product(
    const CalibrationProductAdmissionInputs &inputs) {
    if (!inputs.calibration_requested) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::calibration_not_requested,
            "detector flux calibration was not requested");
    }
    if (inputs.target_unit != "mJy/beam") {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::unsupported_target_unit,
            "SCI-CAL-001 supports only top-of-atmosphere point-source-peak mJy/beam");
    }
    if (!inputs.acquisition_identity_available) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::acquisition_identity_unavailable,
            inputs.acquisition_identity_detail);
    }
    if (!inputs.acquisition_identity_valid) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::acquisition_identity_invalid,
            inputs.acquisition_identity_detail);
    }
    if (inputs.apt_artifact_sha256.empty() ||
        inputs.acquisition_binding_sha256.empty() ||
        inputs.raw_observation_identity.empty() ||
        inputs.acquisition_binding_mode.empty() ||
        inputs.acquisition_key_schema.empty()) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::acquisition_identity_unavailable,
            "calibration artifact or acquisition-binding provenance is unavailable");
    }

    const Eigen::Index detector_count = inputs.detector_flxscale.size();
    if (detector_count <= 0 || inputs.target_unit_factor.size() != detector_count) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::missing_required_factor,
            "target-unit and detector-flxscale factor cardinalities differ");
    }
    if (!finite_positive_vector(inputs.target_unit_factor) ||
        !finite_positive_vector(inputs.detector_flxscale)) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::invalid_required_factor,
            "target-unit or detector-flxscale factor is non-finite or non-positive");
    }
    if (inputs.responsivity_required &&
        inputs.detector_responsivity.size() != detector_count) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::missing_required_factor,
            "required detector responsivity is missing or has incorrect cardinality");
    }
    if (inputs.responsivity_required &&
        !finite_positive_vector(inputs.detector_responsivity)) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::invalid_required_factor,
            "required detector responsivity is non-finite or non-positive");
    }
    if (inputs.sensitivity_required &&
        inputs.detector_sensitivity.size() != detector_count) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::missing_required_factor,
            "required detector sensitivity is missing or has incorrect cardinality");
    }
    if (inputs.sensitivity_required &&
        !finite_positive_vector(inputs.detector_sensitivity)) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::invalid_required_factor,
            "required detector sensitivity is non-finite or non-positive");
    }
    if (inputs.beam_template_required &&
        (inputs.detector_beam_major_fwhm_arcsec.size() != detector_count ||
         inputs.detector_beam_minor_fwhm_arcsec.size() != detector_count)) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::missing_required_factor,
            "originating detector beam/template axes are missing or have incorrect cardinality");
    }
    if (inputs.beam_template_required &&
        (!finite_positive_vector(inputs.detector_beam_major_fwhm_arcsec) ||
         !finite_positive_vector(inputs.detector_beam_minor_fwhm_arcsec))) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::invalid_required_factor,
            "originating detector beam/template axes are non-finite or non-positive");
    }
    if (inputs.extinction_requested &&
        (inputs.minimum_extinction_correction.size() != detector_count ||
         inputs.maximum_extinction_correction.size() != detector_count ||
         !finite_positive_vector(inputs.minimum_extinction_correction) ||
         !finite_positive_vector(inputs.maximum_extinction_correction) ||
         (inputs.minimum_extinction_correction.array() < 1.0).any() ||
         (inputs.maximum_extinction_correction.array() <
          inputs.minimum_extinction_correction.array()).any())) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::invalid_atmosphere_support,
            "observation extinction support is incomplete or invalid");
    }

    CalibrationProduct result;
    result.admitted = true;
    result.validity_cause = CalibrationValidityCause::valid_complete_product;
    result.validity_detail = inputs.extinction_requested
        ? "complete factor, acquisition, unit, and atmosphere product admitted"
        : "complete factor, acquisition, and unit product admitted; extinction not requested";
    result.target_unit = inputs.target_unit;
    result.apt_artifact_sha256 = inputs.apt_artifact_sha256;
    result.acquisition_binding_sha256 = inputs.acquisition_binding_sha256;
    result.raw_observation_identity = inputs.raw_observation_identity;
    result.acquisition_binding_mode = inputs.acquisition_binding_mode;
    result.acquisition_key_schema = inputs.acquisition_key_schema;
    result.response_identity = inputs.response_identity;
    result.target_unit_factor = inputs.target_unit_factor;
    result.detector_flxscale = inputs.detector_flxscale;
    result.signal_multiplier_without_extinction =
        inputs.target_unit_factor.array() * inputs.detector_flxscale.array();
    result.minimum_extinction_correction = inputs.extinction_requested
        ? inputs.minimum_extinction_correction
        : Eigen::VectorXd::Ones(detector_count);
    result.maximum_extinction_correction = inputs.extinction_requested
        ? inputs.maximum_extinction_correction
        : Eigen::VectorXd::Ones(detector_count);

    const auto available = CalibrationNuisanceAvailability::available;
    const auto unavailable = CalibrationNuisanceAvailability::unavailable;
    const auto not_applicable = CalibrationNuisanceAvailability::not_applicable;
    result.nuisances = {
        {"detector_flxscale", available, unavailable, "valid_applied_value",
         "detector",
         "selected_APT_flxscale", "unavailable",
         "value applied; uncertainty unavailable and never represented as zero"},
        {"common_absolute_calibrator_scale", unavailable, unavailable,
         "unavailable", "array_or_observation",
         "upstream_calibrator_model", "unavailable",
         "shared absolute-scale covariance unavailable"},
        {"tolproj_pointing_flxscale_correction", unavailable, unavailable,
         "upstream_lineage_unavailable", "observation",
         "selected_APT_lineage", "unavailable",
         "correction lineage retained by the selected artifact; uncertainty unavailable"},
        {"wvr_atmosphere_model",
         inputs.extinction_requested ? available : not_applicable,
         inputs.extinction_requested ? unavailable : not_applicable,
         inputs.extinction_requested ? "valid_applied_value" : "not_applied",
         "observation_and_array",
         "SCI-CAL-001_fixed_DJF25_operator", "unavailable",
         "operator identity is available; physical/model uncertainty unavailable"},
        {"beam_template_response", available, unavailable,
         "valid_identity_without_fidelity_claim", "detector_and_product",
         inputs.response_identity, "unavailable",
         "originating and realized response identities are retained without covariance or fidelity claim"},
        {"beammap_sensitivity_estimation",
         inputs.detector_sensitivity.size() == detector_count
             ? available : unavailable,
         unavailable,
         inputs.detector_sensitivity.size() == detector_count
             ? "valid_for_approximate_conditional_weight" : "unavailable",
         "detector",
         inputs.detector_sensitivity.size() == detector_count
             ? "selected_APT_sens" : "unavailable",
         "unavailable",
         "sensitivity can support approximate conditional weights only; not total precision"},
    };
    return result;
}

inline double transfer_conditional_variance(double variance,
                                            double multiplier) {
    if (!std::isfinite(variance) || variance < 0.0 ||
        !std::isfinite(multiplier) || multiplier <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return variance * multiplier * multiplier;
}

inline double transfer_conditional_inverse_variance(double weight,
                                                    double multiplier) {
    if (!std::isfinite(weight) || weight < 0.0 ||
        !std::isfinite(multiplier) || multiplier <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return weight / (multiplier * multiplier);
}

inline std::string calibration_nuisance_state_summary(
    const CalibrationProduct &product) {
    std::ostringstream result;
    bool first = true;
    for (const auto &nuisance : product.nuisances) {
        if (!first) {
            result << ';';
        }
        first = false;
        result << nuisance.id
               << "=value:" << to_string(nuisance.value_availability)
               << ",uncertainty:" << to_string(nuisance.availability)
               << ",validity:" << nuisance.validity
               << '@' << nuisance.correlation_scope
               << "[value=" << nuisance.value_source
               << ",uncertainty=" << nuisance.uncertainty_source
               << ",limitation=" << nuisance.limitation << ']';
    }
    return result.str();
}

inline double minimum_total_signal_multiplier(
    const CalibrationProduct &product) {
    if (!product.valid() || product.signal_multiplier_without_extinction.size() == 0 ||
        product.minimum_extinction_correction.size() !=
            product.signal_multiplier_without_extinction.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (product.signal_multiplier_without_extinction.array() *
            product.minimum_extinction_correction.array()).minCoeff();
}

inline double maximum_total_signal_multiplier(
    const CalibrationProduct &product) {
    if (!product.valid() || product.signal_multiplier_without_extinction.size() == 0 ||
        product.maximum_extinction_correction.size() !=
            product.signal_multiplier_without_extinction.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (product.signal_multiplier_without_extinction.array() *
            product.maximum_extinction_correction.array()).maxCoeff();
}

}  // namespace timestream
