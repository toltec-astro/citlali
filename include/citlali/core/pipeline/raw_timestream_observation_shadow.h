#pragma once

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/raw_timestream_observation_resolution.h>
#include <citlali/core/timestream/calibration_product.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct RawTimestreamObservationShadowReport {
    bool exact = true;
    bool edge_guard_deferred = false;
    std::vector<std::string> mismatches;

    void add_mismatch(std::string mismatch) {
        exact = false;
        mismatches.push_back(std::move(mismatch));
    }

    std::string diagnostic() const {
        std::ostringstream stream;
        for (std::size_t index = 0; index < mismatches.size(); ++index) {
            if (index != 0) {
                stream << "; ";
            }
            stream << mismatches[index];
        }
        return stream.str();
    }
};

inline bool raw_shadow_double_equal(double expected, double actual) {
    if (expected == actual) {
        return true;
    }
    const double scale = std::max({1.0, std::abs(expected),
                                   std::abs(actual)});
    return std::abs(expected - actual) <=
           8.0 * std::numeric_limits<double>::epsilon() * scale;
}

template <class Expected, class Actual>
void compare_raw_observation_shadow_value(
    RawTimestreamObservationShadowReport &report, const char *field,
    const Expected &expected, const Actual &actual) {
    if (expected != actual) {
        std::ostringstream stream;
        stream << field << " expected=" << expected << " actual=" << actual;
        report.add_mismatch(stream.str());
    }
}

inline void compare_raw_observation_shadow_value(
    RawTimestreamObservationShadowReport &report, const char *field,
    double expected, double actual) {
    if (!raw_shadow_double_equal(expected, actual)) {
        std::ostringstream stream;
        stream << field << " expected=" << expected << " actual=" << actual;
        report.add_mismatch(stream.str());
    }
}

template <class RtcProc>
RawTimestreamObservationShadowReport begin_raw_timestream_observation_shadow(
    RawTimestreamExecutionPlan &plan,
    citlali::config::ReductionType reduction_type,
    double native_sample_rate_hz, double actual_effective_sample_rate_hz,
    const RtcProc &rtcproc) {
    RawTimestreamObservationShadowReport report;
    if (!plan.initialized) {
        report.add_mismatch("raw timestream plan is not initialized");
        return report;
    }

    const auto sample_rate =
        resolve_raw_sample_rate(plan.requested, native_sample_rate_hz);
    if (!sample_rate.valid()) {
        report.add_mismatch(
            "sample_rate_resolution=" +
            std::string(to_string(sample_rate.error)));
        return report;
    }

    const auto edge_guard =
        resolve_raw_filter_edge_guard(plan.requested, sample_rate);
    const auto source_protection =
        resolve_raw_source_protection_observation(
            reduction_type, plan.requested.despike);

    auto &observation = plan.begin_observation();
    observation.native_sample_rate_hz = sample_rate.native_sample_rate_hz;
    observation.effective_sample_rate_hz =
        sample_rate.effective_sample_rate_hz;
    observation.downsample_factor = sample_rate.downsample_factor;
    observation.filter_edge_guard_samples = edge_guard.guard_samples;
    observation.filter_outer_context_samples = edge_guard.context_samples;
    observation.source_protection_active = source_protection.active;

    compare_raw_observation_shadow_value(
        report, "effective_sample_rate_hz",
        sample_rate.effective_sample_rate_hz,
        actual_effective_sample_rate_hz);
    compare_raw_observation_shadow_value(
        report, "downsample.enabled", plan.requested.downsample.enabled,
        rtcproc.run_downsample);
    if (plan.requested.downsample.enabled) {
        compare_raw_observation_shadow_value(
            report, "downsample.factor", sample_rate.downsample_factor,
            rtcproc.downsampler.factor);
    }
    compare_raw_observation_shadow_value(
        report, "source_protection.active", source_protection.active,
        rtcproc.despiker.source_protection_enabled);

    report.edge_guard_deferred =
        plan.requested.downsample.enabled &&
        plan.requested.downsample.factor <= 0 &&
        sample_rate.downsample_factor > 1 &&
        plan.requested.filter.edge_guard.enabled &&
        !citlali::config::is_none_raw_filter_edge_guard_mode(
            plan.requested.filter.edge_guard.mode) &&
        plan.requested.filter.edge_guard.apply_downsample;
    observation.filter_edge_guard_parity_deferred =
        report.edge_guard_deferred;
    if (!report.edge_guard_deferred) {
        compare_raw_observation_shadow_value(
            report, "filter_edge_guard.guard_samples",
            edge_guard.guard_samples,
            rtcproc.filter_edge_guard.guard_samples);
        compare_raw_observation_shadow_value(
            report, "filter_edge_guard.context_samples",
            edge_guard.context_samples,
            rtcproc.filter_edge_guard.context_samples);
    }
    return report;
}

template <class Calibration>
RawTimestreamObservationShadowReport
complete_raw_timestream_extinction_shadow(
    RawTimestreamExecutionPlan &plan, double tau_225_ghz,
    bool actual_active, const Calibration &calibration) {
    RawTimestreamObservationShadowReport report;
    if (!plan.initialized) {
        report.add_mismatch("raw timestream plan is not initialized");
        return report;
    }
    if (!plan.observation.has_value()) {
        report.add_mismatch("raw observation shadow has not begun");
        return report;
    }

    const auto extinction = resolve_raw_extinction_observation(
        plan.requested.extinction_correction_enabled, tau_225_ghz);
    plan.observation->extinction_active = extinction.active;
    plan.observation->extinction_model = extinction.model;
    plan.observation->tau225 = tau_225_ghz;
    plan.observation->reference_spectral_index_alpha =
        calibration.effective_reference_spectral_index_alpha();
    plan.observation->reference_spectral_index_default_applied =
        calibration.reference_spectral_index_default_applied();
    plan.observation->atmosphere_operator_id =
        std::string{calibration.operator_id()};
    plan.observation->atmosphere_operator_contract_sha256 =
        std::string{calibration.operator_contract_sha256()};
    plan.observation->atmosphere_node_table_sha256 =
        std::string{calibration.operator_nodes_sha256()};
    plan.observation->passband_set_id =
        std::string{calibration.passband_set_id()};
    plan.observation->reference_profile_id =
        std::string{calibration.reference_profile_id()};
    plan.observation->calibration_quality_regime =
        calibration.calibration_quality_regime;
    plan.observation->calibration_valid = calibration.calibration_valid;
    plan.observation->calibration_validity_reason =
        calibration.calibration_validity_reason;
    plan.observation->calibration_validity_detail =
        calibration.product.validity_detail;
    plan.observation->calibration_product_schema =
        std::string{calibration.product.schema_version};
    plan.observation->calibration_target_unit =
        calibration.product.target_unit;
    plan.observation->calibration_photometry_policy =
        std::string{calibration.product.photometry_policy};
    plan.observation->calibration_factor_composition =
        std::string{calibration.product.factor_composition};
    plan.observation->calibration_factor_provenance =
        std::string{calibration.product.factor_provenance};
    plan.observation->calibration_compatibility_fcf_semantics =
        std::string{calibration.product.compatibility_fcf_semantics};
    plan.observation->calibration_weight_recipient_semantics =
        std::string{calibration.product.weight_recipient_semantics};
    plan.observation->calibration_compact_covariance_state =
        std::string{calibration.product.compact_covariance_state};
    plan.observation->calibration_apt_artifact_sha256 =
        calibration.product.apt_artifact_sha256;
    plan.observation->calibration_acquisition_binding_sha256 =
        calibration.product.acquisition_binding_sha256;
    plan.observation->calibration_identity =
        calibration.product.calibration_identity;
    plan.observation->calibration_package_identity =
        calibration.product.package_identity;
    plan.observation->calibration_factor_state_sha256 =
        calibration.product.factor_state_sha256;
    plan.observation->calibration_raw_observation_identity =
        calibration.product.raw_observation_identity;
    plan.observation->calibration_acquisition_binding_mode =
        calibration.product.acquisition_binding_mode;
    plan.observation->calibration_acquisition_key_schema =
        calibration.product.acquisition_key_schema;
    plan.observation->calibration_response_identity =
        calibration.product.response_identity;
    plan.observation->calibration_conditional_variance_transfer =
        std::string{calibration.product.conditional_variance_transfer};
    plan.observation->calibration_conditional_inverse_variance_transfer =
        std::string{calibration.product.conditional_inverse_variance_transfer};
    plan.observation->calibration_precision_limitation =
        std::string{calibration.product.precision_limitation};
    plan.observation->calibration_nuisance_states =
        timestream::calibration_nuisance_state_summary(calibration.product);
    if (calibration.product.valid()) {
        plan.observation->canonical_calibration_product =
            calibration.product;
    }
    const auto minimum_total_multiplier =
        timestream::minimum_total_signal_multiplier(calibration.product);
    const auto maximum_total_multiplier =
        timestream::maximum_total_signal_multiplier(calibration.product);
    if (std::isfinite(minimum_total_multiplier) &&
        std::isfinite(maximum_total_multiplier)) {
        plan.observation->calibration_minimum_total_multiplier =
            minimum_total_multiplier;
        plan.observation->calibration_maximum_total_multiplier =
            maximum_total_multiplier;
    }

    compare_raw_observation_shadow_value(
        report, "extinction.active", extinction.active, actual_active);
    compare_raw_observation_shadow_value(
        report, "extinction.model", extinction.model,
        calibration.extinction_model);
    compare_raw_observation_shadow_value(
        report, "calibration.reference_spectral_index_alpha",
        plan.calibration_effective.spectral_index_alpha,
        calibration.effective_reference_spectral_index_alpha());
    compare_raw_observation_shadow_value(
        report, "calibration.reference_spectral_index_default_applied",
        plan.calibration_effective.default_applied,
        calibration.reference_spectral_index_default_applied());
    return report;
}

}  // namespace citlali::pipeline
