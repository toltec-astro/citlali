#pragma once

#include <citlali/core/config/timestream_enums.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/timestream/calibration_product.h>

#include <Eigen/Core>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

template <class Calib>
Eigen::VectorXd apt_column_or_empty(const Calib &calib,
                                    const std::string &name) {
    const auto found = calib.apt.find(name);
    if (found == calib.apt.end()) {
        return {};
    }
    return found->second;
}

template <class Engine>
std::string calibration_response_identity(const Engine &engine) {
    const auto &raw = raw_time_chunk_config(engine);
    const auto &mapmaking = mapmaking_config(engine);
    const auto major = apt_column_or_empty(engine.calib, "a_fwhm");
    const auto minor = apt_column_or_empty(engine.calib, "b_fwhm");
    const auto angle = apt_column_or_empty(engine.calib, "angle");
    std::ostringstream beam;
    beam << "selected-apt-beam-state-v1"
         << "|a_fwhm=" << timestream::calibration_vector_identity(major)
         << "|b_fwhm=" << timestream::calibration_vector_identity(minor)
         << "|angle=" << timestream::calibration_vector_identity(angle);
    std::ostringstream identity;
    identity << std::boolalpha << std::hexfloat
        << "calibration-response-basis-provenance-v1"
        << ";originating_beam_state_sha256="
        << citlali::utils::sha256(beam.str())
        << ";originating_beam_fields=selected_APT(a_fwhm,b_fwhm,angle)"
        << ";realized_mapmaker_class="
        << citlali::config::to_string(mapmaking.method)
        << ";realized_map_grouping="
        << citlali::config::to_string(mapmaking.grouping)
        << ";realized_kernel_enabled=" << engine.rtcproc.run_kernel
        << ";realized_kernel_class="
        << (engine.rtcproc.run_kernel ? raw.kernel.type : "identity_response")
        << ";realized_fir_enabled=" << engine.rtcproc.run_tod_filter
        << ";realized_fir_low_hz=" << raw.filter.freq_low_Hz
        << ";realized_fir_high_hz=" << raw.filter.freq_high_Hz
        << ";realized_fir_terms=" << raw.filter.n_terms
        << ";realized_fixed_notch_enabled=" << engine.rtcproc.run_tod_notch
        << ";realized_iir_highpass_enabled="
        << engine.rtcproc.run_tod_iir_highpass
        << ";realized_iir_highpass_hz=" << raw.iir_filter.freq_Hz
        << ";realized_iir_highpass_order=" << raw.iir_filter.order
        << ";realized_iir_zero_phase=" << raw.iir_filter.zero_phase
        << ";realized_downsample_enabled=" << engine.rtcproc.run_downsample
        << ";realized_downsample_factor=" << engine.rtcproc.downsampler.factor
        << ";normalization_contract=top_of_atmosphere_point_source_peak_mJy_per_beam"
        << ";semantics=provenance_only_no_empirical_response_fidelity_or_covariance_claim";
    return identity.str();
}

template <class Calib>
timestream::CalibrationPackageLineage calibration_package_lineage(
    const Calib &calib) {
    timestream::CalibrationPackageLineage result;
    const auto &lineage = calib.apt_lineage;
    result.selected_apt_source_path = lineage.selected_apt_path;
    result.selected_apt_sha256 = lineage.selected_apt_sha256;
    result.apt_row_association_sha256 = lineage.row_association_sha256;
    result.apt_observation_identity = lineage.observation_identity;
    result.apt_matched_observation_identity =
        lineage.matched_observation_identity;
    result.apt_selected_source = lineage.selected_source;
    result.legacy_metadata_available = lineage.legacy_metadata_available;
    result.modern_tolapt_manifest_available =
        lineage.modern_tolapt_manifest_available;
    result.modern_tolapt_manifest_path = lineage.modern_tolapt_manifest_path;
    result.modern_tolapt_manifest_sha256 =
        lineage.modern_tolapt_manifest_sha256;
    result.modern_tolapt_contract_version =
        lineage.modern_tolapt_contract_version;
    result.modern_tolapt_run_id = lineage.modern_tolapt_run_id;
    result.modern_tolapt_output_key = lineage.modern_tolapt_output_key;
    result.modern_tolapt_output_path = lineage.modern_tolapt_output_path;
    const auto copy_input = [](const auto &input) {
        return timestream::CalibrationLineageInputRecord{
            input.path, input.sha256, input.bytes, input.mtime_utc};
    };
    result.modern_tolapt_design_input =
        copy_input(lineage.modern_tolapt_design_input);
    result.modern_tolapt_measured_input =
        copy_input(lineage.modern_tolapt_measured_input);
    result.tolapt_manifest_association_sha256 =
        lineage.modern_tolapt_association_sha256;
    result.ordered_rows.reserve(lineage.ordered_rows.size());
    for (const auto &source : lineage.ordered_rows) {
        timestream::CalibrationLineageRow row;
        row.ordered_detector_index = source.ordered_detector_index;
        row.selected_source_row_index = source.selected_source_row_index;
        const auto detector = source.ordered_detector_index;
        if (detector >= 0 && detector < calib.apt.at("nw").size()) {
            row.network = static_cast<int>(calib.apt.at("nw")(detector));
            row.network_local_tone = static_cast<Eigen::Index>(
                calib.apt.at("kids_tone")(detector));
            row.absolute_tone_frequency_hz =
                calib.apt.at("tone_freq")(detector);
        }
        row.uid = source.uid;
        row.eligible = source.eligible;
        row.validity_basis = source.validity_basis;
        row.stable_association = source.stable_association;
        row.retained_fields.reserve(source.retained_fields.size());
        for (const auto &field : source.retained_fields) {
            row.retained_fields.push_back(
                {field.name, field.ecsv_datatype, field.value});
        }
        result.ordered_rows.push_back(std::move(row));
    }
    result.raw_artifacts.reserve(
        calib.apt_acquisition_binding.raw_artifacts.size());
    for (const auto &source :
         calib.apt_acquisition_binding.raw_artifacts) {
        result.raw_artifacts.push_back(
            {source.path, source.sha256, source.interface, source.network,
             source.absolute_tone_frequency_hz});
    }
    return result;
}

template <class Engine>
timestream::CalibrationProductAdmissionInputs
make_calibration_product_admission_inputs(const Engine &engine) {
    timestream::CalibrationProductAdmissionInputs inputs;
    const auto &raw_config = raw_time_chunk_config(engine);
    inputs.target_unit = engine.omb.sig_unit;
    inputs.calibration_requested = raw_config.flux_calibration_enabled;
    inputs.extinction_requested = raw_config.extinction_correction_enabled;
    inputs.responsivity_required = raw_config.despike.enabled;
    inputs.sensitivity_required =
        citlali::config::is_approximate_processed_weighting_type(
            engine.ptcproc.weighting_type) ||
        citlali::config::is_hybrid_processed_weighting_type(
            engine.ptcproc.weighting_type) ||
        citlali::config::is_validated_processed_weighting_type(
            engine.ptcproc.weighting_type);
    inputs.beam_template_required = true;
    inputs.acquisition_identity_available =
        engine.calib.apt_acquisition_binding.available;
    inputs.acquisition_identity_valid =
        engine.calib.apt_acquisition_binding.valid;
    inputs.acquisition_identity_detail =
        engine.calib.apt_acquisition_binding.detail;
    inputs.apt_lineage_available = engine.calib.apt_lineage.available;
    inputs.apt_lineage_valid = engine.calib.apt_lineage.valid;
    inputs.apt_lineage_detail = engine.calib.apt_lineage.detail;
    inputs.apt_artifact_sha256 =
        engine.calib.apt_acquisition_binding.artifact_sha256;
    inputs.apt_row_association_sha256 =
        engine.calib.apt_lineage.row_association_sha256;
    inputs.apt_observation_identity =
        engine.calib.apt_lineage.observation_identity;
    inputs.apt_matched_observation_identity =
        engine.calib.apt_lineage.matched_observation_identity;
    inputs.apt_selected_source = engine.calib.apt_lineage.selected_source;
    inputs.tolapt_manifest_association_sha256 =
        engine.calib.apt_lineage.modern_tolapt_association_sha256;
    inputs.acquisition_binding_sha256 =
        engine.calib.apt_acquisition_binding.binding_sha256;
    inputs.raw_observation_identity =
        engine.calib.apt_acquisition_binding.raw_observation_identity;
    inputs.acquisition_binding_mode =
        engine.calib.apt_acquisition_binding.mode;
    inputs.acquisition_key_schema =
        engine.calib.apt_acquisition_binding.key_schema;
    inputs.response_identity = calibration_response_identity(engine);
    inputs.atmosphere_operator_id =
        std::string{engine.rtcproc.calibration.operator_id()};
    inputs.atmosphere_operator_contract_sha256 =
        std::string{engine.rtcproc.calibration.operator_contract_sha256()};
    inputs.atmosphere_node_table_sha256 =
        std::string{engine.rtcproc.calibration.operator_nodes_sha256()};
    inputs.passband_set_id =
        std::string{engine.rtcproc.calibration.passband_set_id()};
    inputs.reference_profile_id =
        std::string{engine.rtcproc.calibration.reference_profile_id()};
    inputs.reference_spectral_index_alpha =
        engine.rtcproc.calibration.effective_reference_spectral_index_alpha();
    inputs.reference_spectral_index_default_applied =
        engine.rtcproc.calibration.reference_spectral_index_default_applied();
    inputs.tau225 = engine.telescope.tau_225_GHz;
    inputs.package_lineage = calibration_package_lineage(engine.calib);
    inputs.target_unit_factor = engine.calib.flux_conversion_factor;
    inputs.detector_flxscale = apt_column_or_empty(engine.calib, "flxscale");
    inputs.detector_responsivity =
        apt_column_or_empty(engine.calib, "responsivity");
    inputs.detector_sensitivity = apt_column_or_empty(engine.calib, "sens");
    inputs.detector_beam_major_fwhm_arcsec =
        apt_column_or_empty(engine.calib, "a_fwhm");
    inputs.detector_beam_minor_fwhm_arcsec =
        apt_column_or_empty(engine.calib, "b_fwhm");

    const Eigen::Index detector_count = engine.calib.n_dets;
    inputs.minimum_extinction_correction =
        Eigen::VectorXd::Ones(detector_count);
    inputs.maximum_extinction_correction =
        Eigen::VectorXd::Ones(detector_count);
    if (!inputs.extinction_requested) {
        return inputs;
    }

    const auto elevation = engine.telescope.tel_data.find("TelElAct");
    if (elevation == engine.telescope.tel_data.end() ||
        elevation->second.size() <= 0) {
        throw std::domain_error(
            "complete calibration product requires observation sample elevations");
    }
    const auto los_by_array = engine.rtcproc.calibration.calc_tau(
        elevation->second, engine.telescope.tau_225_GHz);
    const auto array = apt_column_or_empty(engine.calib, "array");
    if (array.size() != detector_count) {
        inputs.minimum_extinction_correction.resize(0);
        inputs.maximum_extinction_correction.resize(0);
        return inputs;
    }
    for (Eigen::Index detector = 0; detector < detector_count; ++detector) {
        const double raw_array = array(detector);
        if (!std::isfinite(raw_array) || raw_array != std::floor(raw_array) ||
            raw_array < 0.0 || raw_array > 2.0) {
            inputs.minimum_extinction_correction(detector) =
                std::numeric_limits<double>::quiet_NaN();
            inputs.maximum_extinction_correction(detector) =
                std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const auto found = los_by_array.find(static_cast<int>(raw_array));
        if (found == los_by_array.end() || found->second.size() <= 0) {
            inputs.minimum_extinction_correction(detector) =
                std::numeric_limits<double>::quiet_NaN();
            inputs.maximum_extinction_correction(detector) =
                std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const auto correction = found->second.array().exp();
        inputs.minimum_extinction_correction(detector) = correction.minCoeff();
        inputs.maximum_extinction_correction(detector) = correction.maxCoeff();
    }
    return inputs;
}

template <class Engine>
void admit_complete_calibration_product(Engine &engine) {
    auto inputs = make_calibration_product_admission_inputs(engine);
    engine.rtcproc.calibration.admit_product(inputs);
}

}  // namespace citlali::pipeline
