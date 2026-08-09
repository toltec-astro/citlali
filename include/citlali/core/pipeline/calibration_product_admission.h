#pragma once

#include <citlali/core/config/timestream_enums.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/timestream/calibration_product.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>
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
    const std::string realized = raw_kernel_enabled(engine)
        ? raw_time_chunk_config(engine).kernel.type
        : "identity_response";
    return "originating_response=selected_APT_elliptical_beam(a_fwhm,b_fwhm,angle)"
           ";realized_response=" + realized +
           ";normalization_contract=top_of_atmosphere_point_source_peak_mJy_per_beam"
           ";response_fidelity_and_covariance=unavailable";
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
    inputs.apt_artifact_sha256 =
        engine.calib.apt_acquisition_binding.artifact_sha256;
    inputs.acquisition_binding_sha256 =
        engine.calib.apt_acquisition_binding.binding_sha256;
    inputs.raw_observation_identity =
        engine.calib.apt_acquisition_binding.raw_observation_identity;
    inputs.acquisition_binding_mode =
        engine.calib.apt_acquisition_binding.mode;
    inputs.acquisition_key_schema =
        engine.calib.apt_acquisition_binding.key_schema;
    inputs.response_identity = calibration_response_identity(engine);
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
