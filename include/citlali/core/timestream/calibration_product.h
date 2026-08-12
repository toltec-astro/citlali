#pragma once

#include <citlali/core/utils/sha256.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
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

struct CalibrationLineageInputRecord {
    std::string path;
    std::string sha256;
    std::uint64_t bytes = 0;
    std::string mtime_utc;
};

struct CalibrationLineageRowField {
    std::string name;
    std::string ecsv_datatype;
    std::string value;
};

struct CalibrationLineageRow {
    Eigen::Index ordered_detector_index = -1;
    Eigen::Index selected_source_row_index = -1;
    int network = -1;
    Eigen::Index network_local_tone = -1;
    double absolute_tone_frequency_hz =
        std::numeric_limits<double>::quiet_NaN();
    std::string uid;
    bool eligible = false;
    std::string validity_basis;
    std::string stable_association;
    std::vector<CalibrationLineageRowField> retained_fields;
};

struct CalibrationRawArtifact {
    std::string path;
    std::string sha256;
    std::string interface;
    int network = -1;
    std::vector<double> absolute_tone_frequency_hz;
};

struct CalibrationAppliedExtinctionStateBasis {
    bool available = false;
    bool active = false;
    Eigen::VectorXd sample_elevation_rad;
    std::map<int, Eigen::VectorXd> los_tau_by_array;
};

struct CalibrationPackageLineage {
    std::string selected_apt_source_path;
    std::string selected_apt_sha256;
    std::string apt_row_association_sha256;
    std::string apt_observation_identity;
    std::string apt_matched_observation_identity;
    std::string apt_selected_source;
    bool legacy_metadata_available = false;
    bool modern_tolapt_manifest_available = false;
    std::string modern_tolapt_manifest_path;
    std::string modern_tolapt_manifest_sha256;
    std::string modern_tolapt_contract_version;
    std::string modern_tolapt_run_id;
    std::string modern_tolapt_output_key;
    std::string modern_tolapt_output_path;
    CalibrationLineageInputRecord modern_tolapt_design_input;
    CalibrationLineageInputRecord modern_tolapt_measured_input;
    std::string tolapt_manifest_association_sha256;
    std::vector<CalibrationLineageRow> ordered_rows;
    std::vector<CalibrationRawArtifact> raw_artifacts;
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
    bool apt_lineage_available = false;
    bool apt_lineage_valid = false;
    std::string apt_lineage_detail;
    std::string apt_artifact_sha256;
    std::string apt_row_association_sha256;
    std::string apt_observation_identity;
    std::string apt_matched_observation_identity;
    std::string apt_selected_source;
    std::string tolapt_manifest_association_sha256;
    std::string acquisition_binding_sha256;
    std::string raw_observation_identity;
    std::string acquisition_binding_mode;
    std::string acquisition_key_schema;
    std::string response_identity;
    std::string applied_sample_extinction_state_sha256;
    CalibrationAppliedExtinctionStateBasis applied_sample_extinction_state;
    std::string atmosphere_operator_id;
    std::string atmosphere_operator_contract_sha256;
    std::string atmosphere_node_table_sha256;
    std::string passband_set_id;
    std::string reference_profile_id;
    double reference_spectral_index_alpha = 0.0;
    bool reference_spectral_index_default_applied = true;
    double tau225 = std::numeric_limits<double>::quiet_NaN();
    CalibrationPackageLineage package_lineage;
    Eigen::VectorXd target_unit_factor;
    bool observation_flxscale_correction_applied = false;
    double applied_observation_flxscale_correction = 1.0;
    std::string observation_flxscale_correction_state{"not_applied"};
    std::string observation_flxscale_correction_source_identity{"not_applied"};
    std::string observation_flxscale_correction_recipient_identity;
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
        "signal_prime=signal*target_unit_factor*applied_observation_flxscale_correction*detector_flxscale*sample_extinction_correction";
    static constexpr std::string_view conditional_variance_transfer =
        "variance_prime=total_signal_multiplier^2*conditional_variance";
    static constexpr std::string_view conditional_inverse_variance_transfer =
        "inverse_variance_prime=conditional_inverse_variance/total_signal_multiplier^2";
    static constexpr std::string_view precision_limitation =
        "conditional_only;excludes_calibration_and_response_systematics;not_total_precision_or_significance";
    static constexpr std::string_view photometry_policy =
        "top_of_atmosphere_point_source_peak_mJy_per_beam";
    static constexpr std::string_view observation_correction_source_identity =
        "raw_observation_metadata:flxscale_correction";
    static constexpr std::string_view factor_provenance =
        "target_unit_factor=unity_dimensionless_for_mJy_per_beam;"
        "applied_observation_flxscale_correction=observation_metadata_scalar,multiplicative,source_APT_immutable;"
        "detector_flxscale=selected_APT_flxscale[mJy_per_beam_per_xs],multiplicative;"
        "sample_extinction=exp(line_of_sight_optical_depth),multiplicative;"
        "responsivity=despike_donor_target_relative_response_only,not_absolute_flux;"
        "sens=selected_APT_sens[mJy_per_beam_sqrt_s],already_contains_flxscale;"
        "beam_axes=selected_APT_a_fwhm_b_fwhm[arcsec],response_identity_only";
    static constexpr std::string_view compatibility_fcf_semantics =
        "fcf=target_unit_factor_times_applied_observation_flxscale_correction_times_scan_mean_extinction;"
        "excludes_detector_flxscale_because_selected_APT_sens_already_contains_flxscale;"
        "not_authoritative_total_calibration";
    static constexpr std::string_view weight_recipient_semantics =
        "approximate_weight:coefficient=(sqrt(sample_rate)*compatibility_fcf*selected_APT_sens)^-2,"
        "stage=PTCProc::calc_weights,units=(mJy/beam)^-2,normalization=none,"
        "support=unflagged_detector,calibration=target_unit_factor_once_and_applied_observation_flxscale_correction_once_and_selected_APT_sens_contains_flxscale_once_from_immutable_source_APT;"
        "hybrid_weight:baseline=approximate_weight,coefficient=dimensionless_residual_variance_correction,"
        "stage=PTCProc::calc_weights,units=(mJy/beam)^-2,normalization=array_median_full_over_approx_bounded,"
        "support=unflagged_detector_with_valid_approximate_weight,calibration=no_additional_factor;"
        "validated_weight:baseline=approximate_weight,coefficient=dimensionless_validation_factor_with_optional_cap,"
        "stage=PTCProc::calc_weights,units=(mJy/beam)^-2,normalization=validated_weight_policy,"
        "support=unflagged_detector_with_valid_approximate_weight,calibration=no_additional_factor;"
        "full_weight:coefficient=conditional_inverse_variance_of_already_calibrated_samples,"
        "stage=PTCProc::calc_weights,units=(mJy/beam)^-2,normalization=sample_variance,"
        "support=unflagged_samples,calibration=already_applied_once;"
        "constant_weight:coefficient=unity,stage=PTCProc::calc_weights,units=dimensionless,"
        "normalization=none,support=unflagged_detector,calibration=not_a_precision_recipient;"
        "naive_map_signal:recipient=weighted_mean_of_already_calibrated_samples,"
        "stage=NaiveMapmaker_then_MapBuffer::normalize_maps,units=mJy/beam,normalization=sum_weight,"
        "support=positive_accumulated_weight,calibration=already_applied_once;"
        "naive_map_weight:recipient=sum_of_already_scaled_inverse_variance_coefficients,"
        "stage=NaiveMapmaker_then_MapBuffer::normalize_maps,units=(mJy/beam)^-2,normalization=none,"
        "support=positive_accumulated_weight,calibration=no_second_factor;"
        "noise_variance_I:recipient=conditional_finite_stack_scatter_of_normalized_noise_realizations,"
        "stage=MapBuffer::calc_noise_products_then_FITS,units=(mJy/beam)^2,"
        "normalization=completed_realization_count,support=finite_completed_noise_realization_stack,"
        "calibration=realizations_already_calibrated_once;"
        "all_total_precision_and_significance_claims_fail_closed_without_nuisance_covariance";
    static constexpr std::string_view compact_covariance_state =
        "unavailable;no_nuisance_covariance_invented";

    bool admitted = false;
    CalibrationValidityCause validity_cause =
        CalibrationValidityCause::not_evaluated;
    std::string validity_detail{"calibration product has not been evaluated"};
    std::string target_unit;
    std::string apt_artifact_sha256;
    std::string apt_row_association_sha256;
    std::string apt_observation_identity;
    std::string apt_matched_observation_identity;
    std::string apt_selected_source;
    std::string tolapt_manifest_association_sha256;
    std::string acquisition_binding_sha256;
    std::string raw_observation_identity;
    std::string acquisition_binding_mode;
    std::string acquisition_key_schema;
    std::string response_identity;
    std::string calibration_identity;
    std::string package_identity;
    std::string factor_state_sha256;
    std::string applied_sample_extinction_state_sha256;
    CalibrationAppliedExtinctionStateBasis applied_sample_extinction_state;
    bool applied_identity_finalized = false;
    std::string atmosphere_operator_id;
    std::string atmosphere_operator_contract_sha256;
    std::string atmosphere_node_table_sha256;
    std::string passband_set_id;
    std::string reference_profile_id;
    double reference_spectral_index_alpha = 0.0;
    bool reference_spectral_index_default_applied = true;
    double tau225 = std::numeric_limits<double>::quiet_NaN();
    CalibrationPackageLineage package_lineage;
    Eigen::VectorXd identity_target_unit_factor;
    Eigen::VectorXd target_unit_factor;
    bool observation_flxscale_correction_applied = false;
    double applied_observation_flxscale_correction = 1.0;
    std::string observation_flxscale_correction_state{"not_applied"};
    std::string observation_flxscale_correction_source_identity{"not_applied"};
    std::string observation_flxscale_correction_recipient_identity;
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

inline void append_calibration_identity_field(
    std::ostringstream &stream, std::string_view name,
    std::string_view value) {
    stream << '|' << name.size() << ':' << name << '=' << value.size()
           << ':' << value;
}

inline std::string calibration_hexfloat(double value) {
    std::ostringstream stream;
    stream << std::hexfloat << value;
    return stream.str();
}

inline std::string calibration_vector_identity(
    const Eigen::VectorXd &values) {
    std::ostringstream stream;
    stream << "calibration-vector-hexfloat-v1|count=" << values.size();
    for (Eigen::Index index = 0; index < values.size(); ++index) {
        stream << '|' << index << '=' << calibration_hexfloat(values(index));
    }
    return citlali::utils::sha256(stream.str());
}

inline std::string applied_sample_extinction_state_identity(
    const CalibrationAppliedExtinctionStateBasis &basis) {
    if (!basis.active) {
        return citlali::utils::sha256(
            "sci-cal-001-applied-extinction-state-v1|active=false");
    }
    std::ostringstream stream;
    stream << "sci-cal-001-applied-extinction-state-v1|active=true";
    append_calibration_identity_field(
        stream, "sample_elevation_sha256",
        calibration_vector_identity(basis.sample_elevation_rad));
    for (const auto &[array_id, los_tau] : basis.los_tau_by_array) {
        append_calibration_identity_field(
            stream, "array_" + std::to_string(array_id) +
                        "_los_tau_sha256",
            calibration_vector_identity(los_tau));
    }
    return citlali::utils::sha256(stream.str());
}

inline std::string admitted_factor_state_identity(
    const CalibrationProductAdmissionInputs &inputs) {
    std::ostringstream stream;
    stream << "sci-cal-001-admitted-factor-state-v1";
    append_calibration_identity_field(
        stream, "target_unit_factor_sha256",
        calibration_vector_identity(inputs.target_unit_factor));
    std::ostringstream observation_correction;
    observation_correction << std::hexfloat
                           << inputs.applied_observation_flxscale_correction;
    append_calibration_identity_field(
        stream, "observation_flxscale_correction_applied",
        inputs.observation_flxscale_correction_applied ? "true" : "false");
    append_calibration_identity_field(
        stream, "applied_observation_flxscale_correction",
        observation_correction.str());
    append_calibration_identity_field(
        stream, "observation_flxscale_correction_state",
        inputs.observation_flxscale_correction_state);
    append_calibration_identity_field(
        stream, "observation_flxscale_correction_source_identity",
        inputs.observation_flxscale_correction_source_identity);
    append_calibration_identity_field(
        stream, "observation_flxscale_correction_recipient_identity",
        inputs.observation_flxscale_correction_recipient_identity);
    append_calibration_identity_field(
        stream, "detector_flxscale_sha256",
        calibration_vector_identity(inputs.detector_flxscale));
    append_calibration_identity_field(
        stream, "minimum_extinction_correction_sha256",
        calibration_vector_identity(inputs.minimum_extinction_correction));
    append_calibration_identity_field(
        stream, "maximum_extinction_correction_sha256",
        calibration_vector_identity(inputs.maximum_extinction_correction));
    append_calibration_identity_field(
        stream, "applied_sample_extinction_state_sha256",
        inputs.applied_sample_extinction_state_sha256.empty()
            ? applied_sample_extinction_state_identity(
                  inputs.applied_sample_extinction_state)
            : inputs.applied_sample_extinction_state_sha256);
    return citlali::utils::sha256(stream.str());
}

inline std::string admitted_calibration_identity(
    const CalibrationProductAdmissionInputs &inputs,
    std::string_view factor_state_sha256) {
    std::ostringstream stream;
    stream << "sci-cal-001-canonical-calibration-identity-v1";
    append_calibration_identity_field(
        stream, "selected_apt_source_path",
        inputs.package_lineage.selected_apt_source_path);
    append_calibration_identity_field(
        stream, "selected_apt_sha256", inputs.apt_artifact_sha256);
    append_calibration_identity_field(
        stream, "apt_row_association_sha256",
        inputs.apt_row_association_sha256);
    append_calibration_identity_field(
        stream, "apt_observation_identity", inputs.apt_observation_identity);
    append_calibration_identity_field(
        stream, "apt_matched_observation_identity",
        inputs.apt_matched_observation_identity);
    append_calibration_identity_field(
        stream, "apt_selected_source", inputs.apt_selected_source);
    append_calibration_identity_field(
        stream, "tolapt_manifest_association_sha256",
        inputs.tolapt_manifest_association_sha256);
    append_calibration_identity_field(
        stream, "acquisition_binding_sha256",
        inputs.acquisition_binding_sha256);
    append_calibration_identity_field(
        stream, "raw_observation_identity",
        inputs.raw_observation_identity);
    append_calibration_identity_field(
        stream, "target_unit", inputs.target_unit);
    append_calibration_identity_field(
        stream, "factor_composition", CalibrationProduct::factor_composition);
    append_calibration_identity_field(
        stream, "factor_provenance", CalibrationProduct::factor_provenance);
    append_calibration_identity_field(
        stream, "factor_state_sha256", factor_state_sha256);
    append_calibration_identity_field(
        stream, "atmosphere_operator_id", inputs.atmosphere_operator_id);
    append_calibration_identity_field(
        stream, "atmosphere_operator_contract_sha256",
        inputs.atmosphere_operator_contract_sha256);
    append_calibration_identity_field(
        stream, "atmosphere_node_table_sha256",
        inputs.atmosphere_node_table_sha256);
    append_calibration_identity_field(
        stream, "passband_set_id", inputs.passband_set_id);
    append_calibration_identity_field(
        stream, "reference_profile_id", inputs.reference_profile_id);
    std::ostringstream reference_state;
    reference_state << std::hexfloat
                    << inputs.reference_spectral_index_alpha
                    << ";default="
                    << (inputs.reference_spectral_index_default_applied
                            ? "true" : "false")
                    << ";tau225=" << inputs.tau225;
    append_calibration_identity_field(
        stream, "reference_and_tau_state", reference_state.str());
    append_calibration_identity_field(
        stream, "response_basis_provenance", inputs.response_identity);
    append_calibration_identity_field(
        stream, "validity", "valid_complete_product");
    return citlali::utils::sha256(stream.str());
}

inline std::string calibration_package_identity(
    const CalibrationProduct &product) {
    std::ostringstream preimage;
    preimage << "sci-cal-001-calibration-package-v2";
    append_calibration_identity_field(
        preimage, "calibration_identity", product.calibration_identity);
    append_calibration_identity_field(
        preimage, "package_local_apt_path", "selected_calibration_apt.ecsv");
    append_calibration_identity_field(
        preimage, "package_local_apt_sha256", product.apt_artifact_sha256);
    append_calibration_identity_field(
        preimage, "acquisition_binding_sha256",
        product.acquisition_binding_sha256);
    return citlali::utils::sha256(preimage.str());
}

inline void finalize_calibration_product_identity(
    CalibrationProduct &product, std::string response_identity) {
    if (!product.valid()) {
        throw std::logic_error(
            "cannot finalize an invalid calibration product identity");
    }
    if (product.applied_identity_finalized) {
        if (product.response_identity != response_identity) {
            throw std::logic_error(
                "repeat calibration finalization conflicts with the consumed response snapshot");
        }
        return;
    }
    product.response_identity = std::move(response_identity);
    CalibrationProductAdmissionInputs inputs;
    inputs.target_unit = product.target_unit;
    inputs.apt_artifact_sha256 = product.apt_artifact_sha256;
    inputs.apt_row_association_sha256 = product.apt_row_association_sha256;
    inputs.apt_observation_identity = product.apt_observation_identity;
    inputs.apt_matched_observation_identity =
        product.apt_matched_observation_identity;
    inputs.apt_selected_source = product.apt_selected_source;
    inputs.tolapt_manifest_association_sha256 =
        product.tolapt_manifest_association_sha256;
    inputs.acquisition_binding_sha256 = product.acquisition_binding_sha256;
    inputs.raw_observation_identity = product.raw_observation_identity;
    inputs.response_identity = product.response_identity;
    inputs.atmosphere_operator_id = product.atmosphere_operator_id;
    inputs.atmosphere_operator_contract_sha256 =
        product.atmosphere_operator_contract_sha256;
    inputs.atmosphere_node_table_sha256 =
        product.atmosphere_node_table_sha256;
    inputs.passband_set_id = product.passband_set_id;
    inputs.reference_profile_id = product.reference_profile_id;
    inputs.reference_spectral_index_alpha =
        product.reference_spectral_index_alpha;
    inputs.reference_spectral_index_default_applied =
        product.reference_spectral_index_default_applied;
    inputs.tau225 = product.tau225;
    inputs.package_lineage = product.package_lineage;
    product.calibration_identity = admitted_calibration_identity(
        inputs, product.factor_state_sha256);
    product.package_identity = calibration_package_identity(product);
    product.applied_identity_finalized = true;
}

inline void require_finalized_calibration_product_join(
    const CalibrationProduct &product) {
    if (!product.valid() || !product.applied_identity_finalized ||
        product.calibration_identity.empty() || product.package_identity.empty()) {
        throw std::logic_error(
            "dependent calibrated product requires a finalized CALID/PKGID join");
    }
}

inline CalibrationProduct reject_calibration_product(
    const CalibrationProductAdmissionInputs &inputs,
    CalibrationValidityCause cause, std::string detail) {
    CalibrationProduct result;
    result.validity_cause = cause;
    result.validity_detail = std::move(detail);
    result.target_unit = inputs.target_unit;
    result.apt_artifact_sha256 = inputs.apt_artifact_sha256;
    result.apt_row_association_sha256 =
        inputs.apt_row_association_sha256;
    result.apt_observation_identity = inputs.apt_observation_identity;
    result.apt_matched_observation_identity =
        inputs.apt_matched_observation_identity;
    result.apt_selected_source = inputs.apt_selected_source;
    result.tolapt_manifest_association_sha256 =
        inputs.tolapt_manifest_association_sha256;
    result.acquisition_binding_sha256 = inputs.acquisition_binding_sha256;
    result.raw_observation_identity = inputs.raw_observation_identity;
    result.acquisition_binding_mode = inputs.acquisition_binding_mode;
    result.acquisition_key_schema = inputs.acquisition_key_schema;
    result.response_identity = inputs.response_identity;
    result.applied_sample_extinction_state_sha256 =
        inputs.applied_sample_extinction_state_sha256;
    result.applied_sample_extinction_state =
        inputs.applied_sample_extinction_state;
    result.observation_flxscale_correction_applied =
        inputs.observation_flxscale_correction_applied;
    result.applied_observation_flxscale_correction =
        inputs.applied_observation_flxscale_correction;
    result.observation_flxscale_correction_state =
        inputs.observation_flxscale_correction_state;
    result.observation_flxscale_correction_source_identity =
        inputs.observation_flxscale_correction_source_identity;
    result.observation_flxscale_correction_recipient_identity =
        inputs.observation_flxscale_correction_recipient_identity;
    result.atmosphere_operator_id = inputs.atmosphere_operator_id;
    result.atmosphere_operator_contract_sha256 =
        inputs.atmosphere_operator_contract_sha256;
    result.atmosphere_node_table_sha256 =
        inputs.atmosphere_node_table_sha256;
    result.passband_set_id = inputs.passband_set_id;
    result.reference_profile_id = inputs.reference_profile_id;
    result.reference_spectral_index_alpha =
        inputs.reference_spectral_index_alpha;
    result.reference_spectral_index_default_applied =
        inputs.reference_spectral_index_default_applied;
    result.tau225 = inputs.tau225;
    result.package_lineage = inputs.package_lineage;
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
    if (!inputs.apt_lineage_available || !inputs.apt_lineage_valid) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::acquisition_identity_invalid,
            inputs.apt_lineage_detail.empty()
                ? "selected APT lineage is unavailable or invalid"
                : inputs.apt_lineage_detail);
    }
    if (inputs.package_lineage.selected_apt_source_path.empty() ||
        inputs.package_lineage.selected_apt_sha256 !=
            inputs.apt_artifact_sha256 ||
        inputs.apt_row_association_sha256.empty() ||
        inputs.package_lineage.apt_row_association_sha256 !=
            inputs.apt_row_association_sha256) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::acquisition_identity_invalid,
            "selected APT artifact and ordered row lineage are incomplete or inconsistent");
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
    if (inputs.response_identity.empty() ||
        inputs.atmosphere_operator_id.empty() ||
        inputs.atmosphere_operator_contract_sha256.empty() ||
        inputs.atmosphere_node_table_sha256.empty() ||
        inputs.passband_set_id.empty() ||
        inputs.reference_profile_id.empty()) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::missing_required_factor,
            "calibration response-basis or fixed-operator provenance is unavailable");
    }
    if (inputs.applied_sample_extinction_state.available &&
        inputs.extinction_requested &&
        (!inputs.applied_sample_extinction_state.active ||
         inputs.applied_sample_extinction_state.sample_elevation_rad.size() <= 0 ||
         inputs.applied_sample_extinction_state.los_tau_by_array.empty())) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::invalid_atmosphere_support,
            "complete applied sample-elevation/extinction identity is unavailable");
    }
    if (inputs.applied_sample_extinction_state.available &&
        ((!inputs.extinction_requested &&
         (inputs.applied_sample_extinction_state.active ||
          inputs.applied_sample_extinction_state.sample_elevation_rad.size() != 0 ||
          !inputs.applied_sample_extinction_state.los_tau_by_array.empty())) ||
        inputs.applied_sample_extinction_state_sha256 !=
            applied_sample_extinction_state_identity(
                inputs.applied_sample_extinction_state))) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::invalid_atmosphere_support,
            "applied sample-extinction identity basis is inconsistent");
    }
    if (inputs.package_lineage.modern_tolapt_manifest_available &&
        (inputs.package_lineage.modern_tolapt_manifest_path.empty() ||
         inputs.package_lineage.modern_tolapt_manifest_sha256.empty() ||
         inputs.package_lineage.modern_tolapt_contract_version.empty() ||
         inputs.package_lineage.modern_tolapt_run_id.empty() ||
         inputs.package_lineage.modern_tolapt_output_key.empty() ||
         inputs.package_lineage.modern_tolapt_output_path.empty() ||
         inputs.package_lineage.tolapt_manifest_association_sha256.empty() ||
         inputs.package_lineage.tolapt_manifest_association_sha256 !=
             inputs.tolapt_manifest_association_sha256 ||
         inputs.package_lineage.modern_tolapt_design_input.path.empty() ||
         inputs.package_lineage.modern_tolapt_design_input.sha256.empty() ||
         inputs.package_lineage.modern_tolapt_design_input.mtime_utc.empty() ||
         inputs.package_lineage.modern_tolapt_measured_input.path.empty() ||
         inputs.package_lineage.modern_tolapt_measured_input.sha256.empty() ||
         inputs.package_lineage.modern_tolapt_measured_input.mtime_utc.empty())) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::acquisition_identity_invalid,
            "contract-associated TolAPT manifest provenance is incomplete or inconsistent");
    }
    if (!inputs.package_lineage.modern_tolapt_manifest_available &&
        (!inputs.tolapt_manifest_association_sha256.empty() ||
         !inputs.package_lineage.tolapt_manifest_association_sha256.empty())) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::acquisition_identity_invalid,
            "TolAPT manifest association is present without an admitted modern manifest");
    }

    const Eigen::Index detector_count = inputs.detector_flxscale.size();
    if (detector_count <= 0 || inputs.target_unit_factor.size() != detector_count) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::missing_required_factor,
            "target-unit and detector-flxscale factor cardinalities differ");
    }
    std::size_t raw_tone_count = 0;
    for (const auto &artifact : inputs.package_lineage.raw_artifacts) {
        raw_tone_count += artifact.absolute_tone_frequency_hz.size();
        if (artifact.path.empty() || artifact.sha256.empty() ||
            artifact.interface.empty() || artifact.network < 0) {
            return reject_calibration_product(
                inputs, CalibrationValidityCause::acquisition_identity_invalid,
                "raw acquisition artifact provenance is incomplete");
        }
    }
    if (inputs.package_lineage.ordered_rows.size() !=
            static_cast<std::size_t>(detector_count) ||
        raw_tone_count != static_cast<std::size_t>(detector_count)) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::acquisition_identity_invalid,
            "raw acquisition and selected APT stable-join cardinalities differ");
    }
    const bool correction_state_valid =
        inputs.observation_flxscale_correction_applied
            ? inputs.observation_flxscale_correction_state == "applied_once" &&
                  inputs.observation_flxscale_correction_source_identity ==
                      CalibrationProduct::observation_correction_source_identity &&
                  !inputs.observation_flxscale_correction_recipient_identity.empty() &&
                  inputs.observation_flxscale_correction_recipient_identity ==
                      inputs.raw_observation_identity
            : inputs.applied_observation_flxscale_correction == 1.0 &&
                  inputs.observation_flxscale_correction_state == "not_applied" &&
                  inputs.observation_flxscale_correction_source_identity ==
                      "not_applied" &&
                  inputs.observation_flxscale_correction_recipient_identity.empty();
    if (!correction_state_valid ||
        !finite_positive_vector(inputs.target_unit_factor) ||
        !std::isfinite(inputs.applied_observation_flxscale_correction) ||
        inputs.applied_observation_flxscale_correction <= 0.0 ||
        !finite_positive_vector(inputs.detector_flxscale)) {
        return reject_calibration_product(
            inputs, CalibrationValidityCause::invalid_required_factor,
            "target-unit, observation correction, or detector-flxscale factor is non-finite or non-positive");
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

    Eigen::VectorXd composed_target_factor(detector_count);
    Eigen::VectorXd composed_signal_factor(detector_count);
    for (Eigen::Index detector = 0; detector < detector_count; ++detector) {
        composed_target_factor(detector) =
            inputs.target_unit_factor(detector) *
            inputs.applied_observation_flxscale_correction;
        composed_signal_factor(detector) =
            composed_target_factor(detector) *
            inputs.detector_flxscale(detector);
        const double minimum_total = composed_signal_factor(detector) *
            (inputs.extinction_requested
                 ? inputs.minimum_extinction_correction(detector) : 1.0);
        const double maximum_total = composed_signal_factor(detector) *
            (inputs.extinction_requested
                 ? inputs.maximum_extinction_correction(detector) : 1.0);
        if (!std::isfinite(composed_target_factor(detector)) ||
            composed_target_factor(detector) <= 0.0 ||
            !std::isfinite(composed_signal_factor(detector)) ||
            composed_signal_factor(detector) <= 0.0 ||
            !std::isfinite(minimum_total) || minimum_total <= 0.0 ||
            !std::isfinite(maximum_total) || maximum_total <= 0.0) {
            return reject_calibration_product(
                inputs, CalibrationValidityCause::invalid_required_factor,
                "composed calibration factor is non-finite or non-positive");
        }
    }

    CalibrationProduct result;
    result.admitted = true;
    result.validity_cause = CalibrationValidityCause::valid_complete_product;
    result.validity_detail = inputs.extinction_requested
        ? "complete factor, acquisition, unit, and atmosphere product admitted"
        : "complete factor, acquisition, and unit product admitted; extinction not requested";
    result.target_unit = inputs.target_unit;
    result.apt_artifact_sha256 = inputs.apt_artifact_sha256;
    result.apt_row_association_sha256 =
        inputs.apt_row_association_sha256;
    result.apt_observation_identity = inputs.apt_observation_identity;
    result.apt_matched_observation_identity =
        inputs.apt_matched_observation_identity;
    result.apt_selected_source = inputs.apt_selected_source;
    result.tolapt_manifest_association_sha256 =
        inputs.tolapt_manifest_association_sha256;
    result.acquisition_binding_sha256 = inputs.acquisition_binding_sha256;
    result.raw_observation_identity = inputs.raw_observation_identity;
    result.acquisition_binding_mode = inputs.acquisition_binding_mode;
    result.acquisition_key_schema = inputs.acquisition_key_schema;
    result.response_identity = inputs.response_identity;
    result.applied_sample_extinction_state_sha256 =
        inputs.applied_sample_extinction_state_sha256.empty()
            ? applied_sample_extinction_state_identity(
                  inputs.applied_sample_extinction_state)
            : inputs.applied_sample_extinction_state_sha256;
    result.applied_sample_extinction_state =
        inputs.applied_sample_extinction_state;
    result.factor_state_sha256 = admitted_factor_state_identity(inputs);
    result.calibration_identity = admitted_calibration_identity(
        inputs, result.factor_state_sha256);
    result.package_identity = calibration_package_identity(result);
    result.atmosphere_operator_id = inputs.atmosphere_operator_id;
    result.atmosphere_operator_contract_sha256 =
        inputs.atmosphere_operator_contract_sha256;
    result.atmosphere_node_table_sha256 =
        inputs.atmosphere_node_table_sha256;
    result.passband_set_id = inputs.passband_set_id;
    result.reference_profile_id = inputs.reference_profile_id;
    result.reference_spectral_index_alpha =
        inputs.reference_spectral_index_alpha;
    result.reference_spectral_index_default_applied =
        inputs.reference_spectral_index_default_applied;
    result.tau225 = inputs.tau225;
    result.package_lineage = inputs.package_lineage;
    result.observation_flxscale_correction_applied =
        inputs.observation_flxscale_correction_applied;
    result.applied_observation_flxscale_correction =
        inputs.applied_observation_flxscale_correction;
    result.observation_flxscale_correction_state =
        inputs.observation_flxscale_correction_state;
    result.observation_flxscale_correction_source_identity =
        inputs.observation_flxscale_correction_source_identity;
    result.observation_flxscale_correction_recipient_identity =
        inputs.observation_flxscale_correction_recipient_identity;
    // calibrate_tod consumes this established compatibility carrier. The
    // canonical identity above retains the conceptual target-unit and
    // observation-correction factors separately.
    result.identity_target_unit_factor = inputs.target_unit_factor;
    result.target_unit_factor = std::move(composed_target_factor);
    result.detector_flxscale = inputs.detector_flxscale;
    result.signal_multiplier_without_extinction =
        std::move(composed_signal_factor);
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
        {"tolproj_pointing_flxscale_correction",
         inputs.observation_flxscale_correction_applied
             ? available : not_applicable,
         inputs.observation_flxscale_correction_applied
             ? unavailable : not_applicable,
         inputs.observation_flxscale_correction_applied
             ? "valid_applied_value" : "not_applied",
         "observation",
         inputs.observation_flxscale_correction_applied
             ? "observation_flxscale_correction_metadata" : "not_applied",
         "unavailable",
         inputs.observation_flxscale_correction_applied
             ? "explicit correction value applied once; uncertainty unavailable"
             : "no observation correction was supplied"},
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
