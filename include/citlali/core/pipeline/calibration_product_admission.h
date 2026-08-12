#pragma once

#include <citlali/core/config/timestream_enums.h>
#include <citlali/core/pipeline/flxscale_correction.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/raw_timestream_config_serialization.h>
#include <citlali/core/timestream/calibration_product.h>

#include <Eigen/Core>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace citlali::pipeline {

template <class Engine, class = void>
struct has_complete_calibration_product : std::false_type {};

template <class Engine>
struct has_complete_calibration_product<
    Engine, std::void_t<decltype(
                std::declval<Engine &>().rtcproc.calibration.product)>>
    : std::true_type {};

template <class Calib>
Eigen::VectorXd apt_column_or_empty(const Calib &calib,
                                    const std::string &name) {
    const auto found = calib.apt.find(name);
    if (found == calib.apt.end()) {
        return {};
    }
    return found->second;
}

inline double applied_observation_flxscale_correction(
    const Eigen::VectorXd &legacy_common_factor,
    Eigen::Index detector_count, double recorded_factor) {
    if (detector_count <= 0 ||
        legacy_common_factor.size() != detector_count) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!std::isfinite(recorded_factor) || recorded_factor <= 0.0 ||
        !(legacy_common_factor.array() == recorded_factor).all()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return recorded_factor;
}

template <class Engine, class AppliedResponseNotchHistory>
std::string calibration_response_identity(
    const Engine &engine,
    const AppliedResponseNotchHistory &applied_response_notches) {
    const auto &raw = raw_time_chunk_config(engine);
    const auto *requested_raw = &raw;
    std::string_view requested_state_source =
        "effective_compatibility_fallback";
    if constexpr (has_raw_timestream_plan_v<Engine>) {
        const auto &plan = raw_timestream_plan(engine);
        if (plan.initialized) {
            requested_raw = &plan.requested;
            requested_state_source = "raw_timestream_plan.requested";
        }
    }
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
    const auto requested_state_sha256 = citlali::utils::sha256(
        YAML::Dump(raw_timestream_request_node(*requested_raw)));
    identity << std::boolalpha << std::hexfloat
        << "calibration-response-basis-provenance-v3"
        << ";requested_state_sha256=" << requested_state_sha256
        << ";requested_state_source=" << requested_state_source
        << ";fruit_iteration=" << engine.iteration.fruit_iter
        << ";originating_beam_state_sha256="
        << citlali::utils::sha256(beam.str())
        << ";originating_beam_fields=selected_APT(a_fwhm,b_fwhm,angle)"
        << ";effective_mapmaker_class="
        << citlali::config::to_string(mapmaking.method)
        << ";effective_map_grouping="
        << citlali::config::to_string(mapmaking.grouping)
        << ";effective_kernel_enabled=" << engine.rtcproc.run_kernel
        << ";effective_kernel_class="
        << (engine.rtcproc.run_kernel ? raw.kernel.type : "identity_response")
        << ";requested_fir_enabled=" << requested_raw->filter.enabled
        << ";effective_fir_enabled=" << engine.rtcproc.run_tod_filter
        << ";effective_fir_state="
        << (engine.rtcproc.run_tod_filter ? "scheduled" : "inactive");
    if (engine.rtcproc.run_tod_filter) {
        identity << ";effective_fir_low_hz=" << engine.rtcproc.filter.freq_low_Hz
                 << ";effective_fir_high_hz=" << engine.rtcproc.filter.freq_high_Hz
                 << ";effective_fir_terms=" << engine.rtcproc.filter.n_terms
                 << ";effective_fir_a_gibbs=" << engine.rtcproc.filter.a_gibbs;
    }
    identity
        << ";requested_fixed_notch_enabled="
        << requested_raw->filter.notch.enabled
        << ";effective_fixed_notch_enabled=" << engine.rtcproc.run_tod_notch
        << ";effective_fixed_notch_state="
        << (engine.rtcproc.run_tod_notch ? "scheduled" : "inactive");
    if (engine.rtcproc.run_tod_notch) {
        identity << ";effective_fixed_notch_zero_phase="
                 << engine.rtcproc.filter.notch_zero_phase
                 << ";effective_fixed_notch_sample_rate_hz="
                 << engine.telescope.fsmp;
        for (std::size_t index = 0;
             index < engine.rtcproc.filter.w0s.size(); ++index) {
            const double center = engine.rtcproc.filter.w0s[index];
            const double width = index < engine.rtcproc.filter.qs.size() &&
                    engine.rtcproc.filter.qs[index] != 0.0
                ? center / engine.rtcproc.filter.qs[index]
                : std::numeric_limits<double>::quiet_NaN();
            identity << ";effective_fixed_notch[" << index << "]="
                     << center << ',' << width;
        }
    }
    const auto &line_audit = engine.rtcproc.line_audit;
    if (line_audit.enabled && line_audit.pre_filter_enabled &&
        line_audit.fixed_notch_enabled) {
        std::size_t applied_index = 0;
        for (std::size_t index = 0;
             index < line_audit.fixed_notch_freqs_hz.size(); ++index) {
            const double center = line_audit.fixed_notch_freqs_hz[index];
            const double width = line_audit.fixed_notch_widths_hz.empty()
                ? std::numeric_limits<double>::quiet_NaN()
                : line_audit.fixed_notch_widths_hz[
                      std::min(index,
                               line_audit.fixed_notch_widths_hz.size() - 1)];
            if (!std::isfinite(center) || center <= 0.0 ||
                center >= 0.5 * engine.telescope.fsmp ||
                !std::isfinite(width) || width <= 0.0) {
                continue;
            }
            identity << ";effective_line_audit_fixed_notch["
                     << applied_index++ << "]=" << center << ',' << width
                     << ",zero_phase=true";
        }
    }
    identity
        << ";effective_iir_highpass_enabled="
        << engine.rtcproc.run_tod_iir_highpass
        << ";effective_downsample_enabled=" << engine.rtcproc.run_downsample
        << ";normalization_contract=top_of_atmosphere_point_source_peak_mJy_per_beam";
    if (engine.rtcproc.run_tod_iir_highpass) {
        identity << ";effective_iir_highpass_hz="
                 << engine.rtcproc.filter.iir_highpass_freq_Hz
                 << ";effective_iir_highpass_order="
                 << engine.rtcproc.filter.iir_highpass_order
                 << ";effective_iir_zero_phase="
                 << engine.rtcproc.filter.iir_highpass_zero_phase;
    }
    if (engine.rtcproc.run_downsample) {
        identity << ";effective_downsample_factor="
                 << engine.rtcproc.downsampler.factor;
    }
    std::vector<std::string> applied_notches;
    for (const auto &[scan, notches] : applied_response_notches) {
        for (const auto &notch : notches) {
            std::ostringstream value;
            value << std::hexfloat << "phase=" << notch.phase
                  << ",stage=" << notch.stage
                  << ",fruit_iteration=" << notch.fruit_iteration
                  << ",scan=" << (notch.scan >= 0 ? notch.scan : scan)
                  << ",ptc_iteration=" << notch.ptc_iteration
                  << ",model_subtracted=" << std::boolalpha
                  << notch.model_subtracted
                  << ",scope=" << notch.scope
                  << ",det=" << notch.detector
                  << ",ordinal=" << notch.ordinal
                  << ",geometry=" << notch.geometry
                  << ",center_hz=" << notch.center_hz
                  << ",width_hz=" << notch.width_hz
                  << ",sample_rate_hz=" << notch.sample_rate_hz
                  << ",phase_convention=" << notch.phase_convention;
            applied_notches.push_back(value.str());
        }
    }
    identity << ";actual_applied_response_history_available="
             << engine.rtcproc.applied_response_history_available()
             << ";actual_applied_notch_count=" << applied_notches.size();
    for (std::size_t index = 0; index < applied_notches.size(); ++index) {
        identity << ";actual_applied_notch[" << index << "]="
                 << applied_notches[index];
    }
    identity << ";semantics=provenance_only_no_empirical_response_fidelity_or_covariance_claim";
    return identity.str();
}

template <class Engine>
std::string calibration_response_identity(const Engine &engine) {
    return calibration_response_identity(
        engine, engine.rtcproc.snapshot_applied_response_notches());
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
    const Eigen::Index detector_count = engine.calib.n_dets;
    inputs.target_unit_factor = Eigen::VectorXd::Ones(detector_count);
    const auto correction = engine.calib.mean_flux_conversion_factor.find(
        std::string{observation_flxscale_correction_state_key});
    inputs.observation_flxscale_correction_applied =
        correction != engine.calib.mean_flux_conversion_factor.end();
    const double recorded_correction =
        inputs.observation_flxscale_correction_applied
            ? correction->second : 1.0;
    inputs.applied_observation_flxscale_correction =
        applied_observation_flxscale_correction(
            engine.calib.flux_conversion_factor, detector_count,
            recorded_correction);
    inputs.observation_flxscale_correction_state =
        inputs.observation_flxscale_correction_applied
            ? "applied_once" : "not_applied";
    inputs.observation_flxscale_correction_source_identity =
        inputs.observation_flxscale_correction_applied
            ? std::string{
                  timestream::CalibrationProduct::
                      observation_correction_source_identity}
            : "not_applied";
    inputs.observation_flxscale_correction_recipient_identity =
        inputs.observation_flxscale_correction_applied
            ? inputs.raw_observation_identity : std::string{};
    inputs.detector_flxscale = apt_column_or_empty(engine.calib, "flxscale");
    inputs.detector_responsivity =
        apt_column_or_empty(engine.calib, "responsivity");
    inputs.detector_sensitivity = apt_column_or_empty(engine.calib, "sens");
    inputs.detector_beam_major_fwhm_arcsec =
        apt_column_or_empty(engine.calib, "a_fwhm");
    inputs.detector_beam_minor_fwhm_arcsec =
        apt_column_or_empty(engine.calib, "b_fwhm");

    inputs.minimum_extinction_correction =
        Eigen::VectorXd::Ones(detector_count);
    inputs.maximum_extinction_correction =
        Eigen::VectorXd::Ones(detector_count);
    inputs.applied_sample_extinction_state.available = true;
    inputs.applied_sample_extinction_state_sha256 = citlali::utils::sha256(
        "sci-cal-001-applied-extinction-state-v1|active=false");
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
    inputs.applied_sample_extinction_state.active = true;
    inputs.applied_sample_extinction_state.sample_elevation_rad =
        elevation->second;
    inputs.applied_sample_extinction_state.los_tau_by_array = los_by_array;
    inputs.applied_sample_extinction_state_sha256 =
        timestream::applied_sample_extinction_state_identity(
            inputs.applied_sample_extinction_state);
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

template <class Engine>
void finalize_complete_calibration_product_identity(Engine &engine) {
    auto &product = engine.rtcproc.calibration.product;
    const auto applied_response_notches =
        engine.rtcproc.consume_applied_response_notches();
    timestream::finalize_calibration_product_identity(
        product,
        calibration_response_identity(engine, applied_response_notches));
    engine.rtcproc.record_finalized_calibration_join(
        engine.observation_identity.obsnum,
        product.calibration_identity, product.package_identity);
    engine.calib.apt_meta["calibration_identity"] =
        product.calibration_identity;
    engine.calib.apt_meta["package_identity"] = product.package_identity;
    engine.calib.apt_meta["calibration_response_identity"] =
        product.response_identity;
    if constexpr (has_raw_timestream_plan_v<Engine>) {
        auto &plan = raw_timestream_plan(engine);
        if (plan.observation) {
            plan.observation->calibration_identity =
                product.calibration_identity;
            plan.observation->calibration_package_identity =
                product.package_identity;
            plan.observation->calibration_response_identity =
                product.response_identity;
            plan.observation->canonical_calibration_product = product;
        }
    }
}

template <class Engine>
void finalize_complete_calibration_product_identity_if_available(
    Engine &engine) {
    if constexpr (has_complete_calibration_product<Engine>::value) {
        if (engine.rtcproc.calibration.product.valid()) {
            finalize_complete_calibration_product_identity(engine);
        }
    }
}

}  // namespace citlali::pipeline
