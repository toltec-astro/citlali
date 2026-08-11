#include <gtest/gtest.h>

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/rtc_learned_sampling_metrics.h>

// rtcdiag_netcdf.h transitively includes legacy header-defined PTC helpers.
// test_config_scaffold.cpp already emits those helpers in the combined test
// binary, so keep this test translation unit's copies private by name.
#define add_weight_selection_config_vars                                    \
    rtc_sampling_test_add_weight_selection_config_vars
#define add_cleaner_mode_config_vars                                        \
    rtc_sampling_test_add_cleaner_mode_config_vars
#define add_fruit_loop_iteration_config_vars                                \
    rtc_sampling_test_add_fruit_loop_iteration_config_vars
#define add_ptcdiag_compact_config_vars                                     \
    rtc_sampling_test_add_ptcdiag_compact_config_vars
#define add_fruit_loops_config_vars                                         \
    rtc_sampling_test_add_fruit_loops_config_vars
#include <citlali/core/pipeline/rtcdiag_netcdf.h>
#undef add_fruit_loops_config_vars
#undef add_ptcdiag_compact_config_vars
#undef add_fruit_loop_iteration_config_vars
#undef add_cleaner_mode_config_vars
#undef add_weight_selection_config_vars

#include <citlali/core/pipeline/timestream_alignment_helpers.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double rad_to_arcsec = 206264.80624709636;

std::map<std::string, Eigen::VectorXd> source_rows(
    const std::vector<double> &time, const std::vector<double> &az_arcsec) {
    const auto make = [](const std::vector<double> &values) {
        return Eigen::Map<const Eigen::VectorXd>(values.data(), values.size());
    };
    std::vector<double> az_rad(az_arcsec.size());
    std::transform(az_arcsec.begin(), az_arcsec.end(), az_rad.begin(),
                   [](double value) { return value / rad_to_arcsec; });
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(time.size());
    return {{"TelTime", make(time)}, {"TelAzAct", make(az_rad)},
            {"SourceAz", zeros}, {"TelElAct", zeros},
            {"SourceEl", zeros}, {"TelAzCor", zeros},
            {"TelElCor", zeros}};
}

struct FakeFilterConfig {
    bool enabled = false;
    double a_gibbs = 0.0;
    double freq_low_Hz = 0.0;
    double freq_high_Hz = 0.0;
    int n_terms = 0;
};

struct FakePlanState {
    FakeFilterConfig filter;
};

struct FakeRawPlan {
    FakePlanState requested;
    FakePlanState effective;
};

struct FakeRealizedFilter : FakeFilterConfig {
    Eigen::VectorXd filter;
};

struct FakeRtcProc {
    bool run_tod_filter = false;
    FakeRealizedFilter filter;
    bool run_downsample = false;
    struct {
        int factor = 1;
    } downsampler;
};

citlali::pipeline::RtcSamplingFilterState sampling_filter_state(
    const FakeRawPlan &plan, const FakeRtcProc &rtc) {
    citlali::pipeline::RtcSamplingFilterState state;
    state.requested_enabled = plan.requested.filter.enabled;
    state.effective_enabled = plan.effective.filter.enabled;
    state.realized_enabled = rtc.run_tod_filter;
    state.requested_a_gibbs = plan.requested.filter.a_gibbs;
    state.effective_a_gibbs = plan.effective.filter.a_gibbs;
    state.requested_low_hz = plan.requested.filter.freq_low_Hz;
    state.effective_low_hz = plan.effective.filter.freq_low_Hz;
    state.requested_high_hz = plan.requested.filter.freq_high_Hz;
    state.effective_high_hz = plan.effective.filter.freq_high_Hz;
    state.requested_n_terms = plan.requested.filter.n_terms;
    state.effective_n_terms = plan.effective.filter.n_terms;
    if (state.realized_enabled) {
        state.realized_a_gibbs = rtc.filter.a_gibbs;
        state.realized_low_hz = rtc.filter.freq_low_Hz;
        state.realized_high_hz = rtc.filter.freq_high_Hz;
        state.realized_n_terms = rtc.filter.n_terms;
        state.realized_coefficients.assign(
            rtc.filter.filter.data(),
            rtc.filter.filter.data() + rtc.filter.filter.size());
    }
    return state;
}

struct FakeCalib {
    Eigen::Index n_arrays = 1;
    Eigen::VectorXi arrays = Eigen::VectorXi::Zero(1);
};

struct FakeTelescope {
    double fsmp = 100.0;
    Eigen::MatrixXi scan_indices = Eigen::MatrixXi::Zero(4, 1);
};

citlali::pipeline::RtcDiagScanSummaryData available_scan_summary(
    std::size_t grid_size = 12) {
    citlali::pipeline::RtcDiagScanSummaryData summary;
    summary.scan_motion.resize(1);
    summary.scan_motion[0].status =
        citlali::pipeline::RtcSamplingStatusCode::prerequisite_available;
    summary.scan_motion[0].reason =
        citlali::pipeline::RtcSamplingReasonCode::none;
    summary.scan_motion[0].speed_p95_arcsec_s = 10.0;
    summary.eligible_grid_by_scan = {
        std::vector<unsigned char>(grid_size, 1)};
    return summary;
}

FakeTelescope one_scan_telescope() {
    FakeTelescope telescope;
    telescope.scan_indices.col(0) << 1, 10, 0, 11;
    return telescope;
}

TEST(RtcLearnedSamplingMetrics, CapturesPreInterpolationSupportAndThresholds) {
    const auto rows = source_rows(
        {0.0, 0.02, 0.04, 0.06, 0.30, 0.32},
        {0.0, 0.02, 0.06, 72.08, 72.10, 72.12});
    const auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        rows, rad_to_arcsec);

    ASSERT_EQ(support.interval_count, 5u);
    EXPECT_EQ(support.valid_interval_count, 3u);
    EXPECT_EQ(support.rejected_interval_count, 2u);
    // Exactly 1 arcsec/s is eligible; the 2 arcsec/s interval is too.
    EXPECT_EQ(support.eligible_interval_count, 3u);
    EXPECT_EQ(support.low_velocity_excluded_count, 0u);
    EXPECT_EQ(support.intervals[3].reason, "invalid_source_gap");
    EXPECT_EQ(support.intervals[2].reason,
              "invalid_source_speed_above_bound");
}

TEST(RtcLearnedSamplingMetrics,
     SourceIntervalLookupHasStableSharedEndpointAuthority) {
    const std::vector<double> starts{0.0, 1.0};
    const std::vector<double> stops{1.0, 2.0};
    EXPECT_EQ(citlali::pipeline::rtc_sampling_source_interval_at_time(
                  starts, stops, 1.0),
              std::optional<std::size_t>{0});
    EXPECT_EQ(citlali::pipeline::rtc_sampling_source_interval_at_time(
                  starts, stops, 1.5),
              std::optional<std::size_t>{1});
    EXPECT_FALSE(citlali::pipeline::rtc_sampling_source_interval_at_time(
                     starts, stops,
                     std::numeric_limits<double>::quiet_NaN())
                     .has_value());
    EXPECT_FALSE(citlali::pipeline::rtc_sampling_source_interval_at_time(
                     starts, std::vector<double>{1.0}, 0.5)
                     .has_value());
}

TEST(RtcLearnedSamplingMetrics, UnequalSourceColumnsFailClosedWithoutTruncation) {
    auto rows = source_rows({0.0, 0.02, 0.04}, {0.0, 0.04, 0.08});
    rows["SourceEl"].conservativeResize(2);
    const auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        rows, rad_to_arcsec);
    EXPECT_EQ(support.source_row_count, 3u);
    EXPECT_EQ(support.interval_count, 0u);
    EXPECT_TRUE(support.intervals.empty());
    EXPECT_EQ(support.reason, "unequal_source_column_lengths");
    EXPECT_EQ(citlali::pipeline::rtc_sampling_source_support_reason_code(
                  support.reason),
              citlali::pipeline::RtcSamplingReasonCode::unequal_source_column_lengths);
}

TEST(RtcLearnedSamplingMetrics, NegativeAzimuthWrapUsesShortestDifference) {
    const double almost_minus_turn =
        -(2.0 * citlali::pipeline::rtc_sampling_pi - 0.0001) * rad_to_arcsec;
    const auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        source_rows({0.0, 0.02}, {almost_minus_turn, 0.0}),
        rad_to_arcsec);
    ASSERT_EQ(support.intervals.size(), 1u);
    EXPECT_TRUE(support.intervals[0].valid);
    EXPECT_NE(support.intervals[0].reason, "invalid_source_pointing_step");
    EXPECT_LT(support.intervals[0].speed_arcsec_s, 3600.0);
}

TEST(RtcLearnedSamplingMetrics, NativeRowBoundaryGuardTracksPartialSupport) {
    const auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        source_rows({0.00, 0.02, 0.04, 0.06, 0.08, 0.10},
                    {0.00, 0.04, 0.08, 0.12, 0.16, 0.20}),
        rad_to_arcsec);
    const auto motion = citlali::pipeline::calculate_rtc_sampling_scan_motion(
        support, 0.015, 0.085);
    ASSERT_EQ(motion.status,
              citlali::pipeline::RtcSamplingStatusCode::prerequisite_available);
    EXPECT_EQ(motion.guarded_first_row_index, 2u);
    EXPECT_EQ(motion.guarded_last_row_index, 3u);
    EXPECT_EQ(motion.overlapping_interval_count, 5u);
    EXPECT_EQ(motion.boundary_guard_excluded_count, 4u);
    EXPECT_EQ(motion.partial_overlap_count, 2u);
    EXPECT_EQ(motion.source_interval_count, 1u);
    EXPECT_NEAR(motion.partial_overlap_duration_s, 0.01, 1e-15);
}

TEST(RtcLearnedSamplingMetrics, ConsecutiveScansHaveIndependentGuardedRows) {
    std::vector<double> time;
    std::vector<double> az;
    for (int i = 0; i <= 10; ++i) {
        time.push_back(0.02 * i);
        az.push_back(0.04 * i);
    }
    const auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        source_rows(time, az), rad_to_arcsec);
    const auto first = citlali::pipeline::calculate_rtc_sampling_scan_motion(
        support, 0.0, 0.10);
    const auto second = citlali::pipeline::calculate_rtc_sampling_scan_motion(
        support, 0.10, 0.20);
    ASSERT_EQ(first.status,
              citlali::pipeline::RtcSamplingStatusCode::prerequisite_available);
    ASSERT_EQ(second.status,
              citlali::pipeline::RtcSamplingStatusCode::prerequisite_available);
    EXPECT_EQ(first.guarded_first_row_index, 1u);
    EXPECT_EQ(first.guarded_last_row_index, 4u);
    EXPECT_EQ(second.guarded_first_row_index, 6u);
    EXPECT_EQ(second.guarded_last_row_index, 9u);
    EXPECT_EQ(first.source_interval_count, 3u);
    EXPECT_EQ(second.source_interval_count, 3u);
}

TEST(RtcLearnedSamplingMetrics, GuardPreservesInternalGapIdentity) {
    const auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        source_rows({0.00, 0.02, 0.04, 0.20, 0.22, 0.24, 0.26},
                    {0.00, 0.04, 0.08, 0.10, 0.14, 0.18, 0.22}),
        rad_to_arcsec);
    const auto motion = citlali::pipeline::calculate_rtc_sampling_scan_motion(
        support, 0.0, 0.26);
    EXPECT_EQ(motion.rejected_interval_count, 1u);
    EXPECT_EQ(motion.valid_interval_count, 3u);
    EXPECT_EQ(motion.status,
              citlali::pipeline::RtcSamplingStatusCode::prerequisite_available);
}

TEST(RtcLearnedSamplingMetrics, AllLowVelocityIsDeterministicallyUnavailable) {
    const auto rows = source_rows({0.0, 0.02, 0.04, 0.06, 0.08},
                                  {0.0, 0.01, 0.02, 0.03, 0.04});
    const auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        rows, rad_to_arcsec);
    EXPECT_EQ(support.valid_interval_count, 4u);
    EXPECT_EQ(support.eligible_interval_count, 0u);
    EXPECT_EQ(support.reason, "unavailable_low_velocity");

    const auto motion = citlali::pipeline::calculate_rtc_sampling_scan_motion(
        support, 0.0, 0.08);
    EXPECT_EQ(motion.status,
              citlali::pipeline::RtcSamplingStatusCode::prerequisite_unavailable);
    EXPECT_EQ(motion.reason,
              citlali::pipeline::RtcSamplingReasonCode::unavailable_low_velocity);
}

TEST(RtcLearnedSamplingMetrics, MotionUsesV95AndPreservesDiagnostics) {
    auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        source_rows({0.0, 0.02, 0.04, 0.06, 0.08},
                    {0.0, 0.02, 0.06, 0.12, 0.20}),
        rad_to_arcsec);
    const auto motion = citlali::pipeline::calculate_rtc_sampling_scan_motion(
        support, 0.0, 0.08);
    EXPECT_EQ(motion.status,
              citlali::pipeline::RtcSamplingStatusCode::prerequisite_available);
    EXPECT_NEAR(motion.speed_p95_arcsec_s, 2.95, 1e-14);
    EXPECT_NEAR(motion.speed_p99_arcsec_s, 2.99, 1e-14);
    EXPECT_DOUBLE_EQ(motion.speed_max_arcsec_s, 3.0);
}

TEST(RtcLearnedSamplingMetrics, UsesOnlyFixedDiffractionBeamAuthority) {
    const auto a1100 = citlali::pipeline::rtc_sampling_beam_authority(0);
    const auto a1400 = citlali::pipeline::rtc_sampling_beam_authority(1);
    const auto a2000 = citlali::pipeline::rtc_sampling_beam_authority(2);
    EXPECT_DOUBLE_EQ(a1100.fwhm_arcsec, 4.66);
    EXPECT_DOUBLE_EQ(a1400.fwhm_arcsec, 5.94);
    EXPECT_DOUBLE_EQ(a2000.fwhm_arcsec, 8.48);
    EXPECT_FALSE(citlali::pipeline::rtc_sampling_beam_authority(99).available);
    EXPECT_STREQ(citlali::pipeline::rtc_sampling_beam_model,
                 "circular-gaussian-temporal-intensity-v1");
}

TEST(RtcLearnedSamplingMetrics, GaussianTransferHasOwnerNormalization) {
    constexpr double theta = 4.66;
    constexpr double v95 = 100.0;
    const double sigma = citlali::pipeline::rtc_sampling_temporal_sigma_s(
        theta, v95);
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::rtc_sampling_gaussian_beam_amplitude(0.0, sigma),
        1.0);
    const double f = 7.0;
    EXPECT_NEAR(
        citlali::pipeline::rtc_sampling_gaussian_beam_amplitude(f, sigma),
        std::exp(-2.0 * citlali::pipeline::rtc_sampling_pi *
                 citlali::pipeline::rtc_sampling_pi * sigma * sigma * f * f),
        1e-15);
}

TEST(RtcLearnedSamplingMetrics, CoherentFoldUsesExactlyMUnitAmplitudeImages) {
    const std::vector<double> identity{1.0};
    constexpr double fs = 100.0;
    constexpr double sigma = 0.002;
    constexpr double f = 10.0;
    const auto response =
        citlali::pipeline::rtc_sampling_phase_zero_coherent_response_at(
            identity, f, fs, 2, sigma);
    const auto base = citlali::pipeline::rtc_sampling_composed_transfer(
        identity, 10.0, fs, sigma);
    const auto image = citlali::pipeline::rtc_sampling_composed_transfer(
        identity, -40.0, fs, sigma);
    EXPECT_NEAR(std::abs(response.unaliased - base), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(response.alias - image), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(response.folded - (base + image)), 0.0, 1e-15);
    // A forbidden 1/M tone normalization would be half this value.
    EXPECT_GT(std::abs(response.folded), 0.5 * std::abs(base + image));
}

TEST(RtcLearnedSamplingMetrics, OddAndEvenHalfOpenFoldsAreDeterministic) {
    const std::vector<double> coefficients{0.25, 0.5, 0.25};
    for (int factor : {2, 3, 4, 5}) {
        const double high = 50.0 / factor;
        const auto at_low =
            citlali::pipeline::rtc_sampling_phase_zero_coherent_response_at(
                coefficients, -high, 100.0, factor, 0.01);
        const auto below_high =
            citlali::pipeline::rtc_sampling_phase_zero_coherent_response_at(
                coefficients, std::nextafter(high, -high), 100.0, factor,
                0.01);
        EXPECT_TRUE(at_low.relative_valid);
        EXPECT_TRUE(below_high.relative_valid);
    }
}

TEST(RtcLearnedSamplingMetrics, FactorOneAliasIsExactlyZeroAndStopbandNApplicable) {
    const auto metrics =
        citlali::pipeline::calculate_rtc_sampling_candidate_metrics(
            1, 100.0, {0.25, 0.5, 0.25}, 0.02, 32);
    EXPECT_TRUE(metrics.alias_valid);
    EXPECT_DOUBLE_EQ(metrics.alias_amplitude_max_lower, 0.0);
    EXPECT_DOUBLE_EQ(metrics.alias_amplitude_max_upper, 0.0);
    EXPECT_DOUBLE_EQ(metrics.relative_distortion_max_upper, 0.0);
    EXPECT_EQ(metrics.stopband_status,
              citlali::pipeline::RtcSamplingStatusCode::not_applicable_no_decimation);
    EXPECT_EQ(metrics.stopband_reason,
              citlali::pipeline::RtcSamplingReasonCode::not_applicable_no_decimation);
}

TEST(RtcLearnedSamplingMetrics, CandidateRangeIgnoresFilterEdge) {
    const auto factors = citlali::pipeline::rtc_sampling_supported_factors(
        4.66, 488.28125, 100.0);
    ASSERT_EQ(factors.size(), 22u);
    EXPECT_EQ(factors.front(), 1);
    EXPECT_EQ(factors.back(), 22);
    EXPECT_EQ(citlali::pipeline::rtc_sampling_candidate_mmax(
                  8.48, 488.28125, 1.0),
              4140);
    EXPECT_EQ(citlali::pipeline::rtc_sampling_candidate_mmax(
                  0.1, 1.0, 100.0),
              0);
    EXPECT_EQ(citlali::pipeline::rtc_sampling_supported_factors(
                  0.1, 1.0, 100.0),
              std::vector<int>{1});
}

TEST(RtcLearnedSamplingMetrics, BoundedCharacterizationEnclosesDenseAdversary) {
    const auto analytic = citlali::pipeline::rtc_sampling_bounded_maximum(
        [](double) { return 3.0; }, -2.0, 2.0, 0.0, 4);
    ASSERT_TRUE(analytic.valid);
    EXPECT_DOUBLE_EQ(analytic.lower, 3.0);
    EXPECT_DOUBLE_EQ(analytic.upper, 3.0);
    EXPECT_DOUBLE_EQ(analytic.error_enclosure, 0.0);

    const auto bounded = citlali::pipeline::rtc_sampling_bounded_maximum(
        [](double f) { return std::abs(std::sin(101.0 * f)); },
        0.0, 1.0, 101.0, 8);
    ASSERT_TRUE(bounded.valid);
    EXPECT_LE(bounded.lower, 1.0);
    EXPECT_GE(bounded.upper, 1.0);
    EXPECT_GT(bounded.error_enclosure, 0.0);

    const auto singular_reference =
        citlali::pipeline::calculate_rtc_sampling_candidate_metrics(
            2, 100.0, {1.0, -1.0}, 0.01, 32);
    EXPECT_EQ(singular_reference.alias_status,
              citlali::pipeline::RtcSamplingStatusCode::numerical_bounded_not_converged);
    EXPECT_EQ(singular_reference.amplitude_status,
              citlali::pipeline::RtcSamplingStatusCode::numerical_failed);
    EXPECT_EQ(singular_reference.amplitude_reason,
              citlali::pipeline::RtcSamplingReasonCode::numerical_singular_reference);

    std::vector<double> long_fir(1025, 1.0 / 1025.0);
    const auto long_filter =
        citlali::pipeline::calculate_rtc_sampling_candidate_metrics(
            3, 100.0, long_fir, 0.01, 16);
    EXPECT_EQ(long_filter.alias_status,
              citlali::pipeline::RtcSamplingStatusCode::numerical_bounded_not_converged);
    EXPECT_TRUE(std::isfinite(long_filter.alias_lipschitz_bound));
    EXPECT_GT(long_filter.alias_evaluations, 0u);

    const auto deliberately_broad =
        citlali::pipeline::rtc_sampling_bounded_maximum(
            [](double f) { return std::cos(f); }, 0.0, 1.0, 1000.0, 1);
    ASSERT_TRUE(deliberately_broad.valid);
    EXPECT_GT(deliberately_broad.error_enclosure, 100.0);
    EXPECT_EQ(citlali::pipeline::rtc_sampling_bounded_status(
                  deliberately_broad.error_enclosure),
              citlali::pipeline::RtcSamplingStatusCode::numerical_bounded_not_converged);
}

TEST(RtcLearnedSamplingMetrics, ResourcePreflightIsCheckedAndNeverTruncates) {
    const auto isolated = citlali::pipeline::rtc_sampling_resource_preflight(
        {9000, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    EXPECT_EQ(isolated.range_status[0],
              citlali::pipeline::RtcSamplingStatusCode::candidate_range_resource_limit);
    EXPECT_EQ(isolated.range_status[1],
              citlali::pipeline::RtcSamplingStatusCode::candidate_range_available);
    EXPECT_TRUE(isolated.table_available);
    EXPECT_EQ(isolated.candidate_axis_size, 3u);
    EXPECT_EQ(isolated.logical_candidate_rows, 3u);

    const auto overflow = citlali::pipeline::rtc_sampling_resource_preflight(
        {2}, {1}, {1}, {std::numeric_limits<std::size_t>::max()}, {2});
    EXPECT_FALSE(overflow.table_available);
    EXPECT_EQ(overflow.table_reason,
              citlali::pipeline::RtcSamplingReasonCode::arithmetic_overflow);

    const std::vector<int> too_many_rows(1001, 8000);
    const std::vector<unsigned char> admitted(1001, 1);
    const auto row_limited =
        citlali::pipeline::rtc_sampling_resource_preflight(
            too_many_rows, admitted);
    EXPECT_FALSE(row_limited.table_available);
    EXPECT_GT(row_limited.logical_candidate_rows,
              citlali::pipeline::rtc_sampling_max_candidate_rows);
    EXPECT_EQ(row_limited.table_reason,
              citlali::pipeline::RtcSamplingReasonCode::candidate_range_resource_limit);

    const auto evaluation_limited =
        citlali::pipeline::rtc_sampling_resource_preflight(
            {2}, {1},
            {citlali::pipeline::rtc_sampling_max_actual_work_units},
            {1}, {1});
    EXPECT_FALSE(evaluation_limited.table_available);
    EXPECT_EQ(evaluation_limited.candidate_axis_size, 2u);
    EXPECT_EQ(evaluation_limited.table_reason,
              citlali::pipeline::RtcSamplingReasonCode::numerical_resource_limit);

    const auto byte_limited =
        citlali::pipeline::rtc_sampling_resource_preflight(
            {10}, {1}, {1}, {1}, {1},
            citlali::pipeline::rtc_sampling_max_estimated_rtcdiag_bytes / 10 + 1);
    EXPECT_FALSE(byte_limited.table_available);
    EXPECT_EQ(byte_limited.candidate_axis_size, 10u);
    EXPECT_EQ(byte_limited.table_reason,
              citlali::pipeline::RtcSamplingReasonCode::candidate_table_storage_limit);
}

TEST(RtcLearnedSamplingMetrics,
     ProductionResourcePreflightSeparatesRangeAndWorkLimits) {
    FakeCalib calib;
    calib.n_arrays = 2;
    calib.arrays.resize(2);
    calib.arrays << 0, 2;
    FakeRawPlan plan;
    FakeRtcProc rtc;
    auto telescope = one_scan_telescope();
    telescope.fsmp = 1000.0;
    auto summary = available_scan_summary();
    summary.scan_motion[0].speed_p95_arcsec_s = 1.0;
    citlali::pipeline::RtcSamplingCadenceState cadence;
    cadence.realized_valid = true;
    cadence.realized_reason = citlali::pipeline::RtcSamplingReasonCode::none;
    cadence.requested_effective_consistency = "consistent";
    cadence.effective_realized_consistency = "consistent";
    cadence.effective_native_hz = 1000.0;
    cadence.realized_native_hz = 1000.0;
    cadence.effective_output_hz = 1000.0;
    cadence.realized_output_hz = 1000.0;
    const auto values =
        citlali::pipeline::calculate_rtcdiag_scan_array_summary(
            calib, sampling_filter_state(plan, rtc), telescope, summary, {}, cadence,
            1, 2, 2, -999.0, -999);
    ASSERT_EQ(values.candidate_mmax, (std::vector<int>{4660, 8480}));
    EXPECT_EQ(values.candidate_range_status[0], static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::candidate_range_available));
    EXPECT_EQ(values.candidate_range_status[1], static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::candidate_range_resource_limit));
    EXPECT_FALSE(values.candidate_table_available);
    EXPECT_TRUE(values.candidate_factors.empty());
    EXPECT_TRUE(values.candidate_status.empty());
    EXPECT_EQ(values.estimated_candidate_rows, 4660u);
    EXPECT_EQ(values.candidate_table_reason,
              citlali::pipeline::RtcSamplingReasonCode::numerical_resource_limit);
    EXPECT_GT(values.estimated_actual_work_units,
              citlali::pipeline::rtc_sampling_max_actual_work_units);
}

TEST(RtcLearnedSamplingMetrics, FirDigestIsOrderSensitiveAndRepeatable) {
    const auto first = citlali::pipeline::rtc_sampling_fir_digest({0.25, 0.75});
    const auto repeat = citlali::pipeline::rtc_sampling_fir_digest({0.25, 0.75});
    const auto reversed = citlali::pipeline::rtc_sampling_fir_digest({0.75, 0.25});
    EXPECT_EQ(first, repeat);
    EXPECT_NE(first, reversed);
    EXPECT_EQ(first.rfind("sha256:", 0), 0u);
    EXPECT_EQ(first.size(), 71u);
    EXPECT_EQ(first,
              "sha256:926a87b0ac0b131a6e100f3c8d2e426433844827e7560939f34d79bcd67efa33");
    EXPECT_STREQ(
        citlali::pipeline::rtc_sampling_fir_digest_convention,
        "sha256-u64le-count-then-ieee754-binary64le-realized-order-v1");
}

TEST(RtcLearnedSamplingMetrics, CompleteContextUsesActualPhaseAndGaps) {
    const std::vector<unsigned char> support{1, 1, 1, 1, 0, 1, 1, 1, 1};
    const auto phase_zero =
        citlali::pipeline::calculate_rtc_sampling_complete_context(
            support, 0, 8, 0, 8, 2, 0, 3, 100.0);
    EXPECT_EQ(phase_zero.candidate_output_count, 5u);
    EXPECT_EQ(phase_zero.full_output_count, 2u);
    EXPECT_EQ(phase_zero.incomplete_boundary_count, 2u);
    EXPECT_EQ(phase_zero.incomplete_gap_count, 1u);

    const auto phase_one =
        citlali::pipeline::calculate_rtc_sampling_complete_context(
            support, 0, 8, 0, 8, 2, 1, 3, 100.0);
    EXPECT_EQ(phase_one.candidate_output_count, 4u);
    EXPECT_EQ(phase_one.full_output_count, 2u);
}

TEST(RtcLearnedSamplingMetrics, CompleteContextZeroIsSoleHardBoundary) {
    for (std::size_t n : {0u, 1u, 2u, 3u, 4u}) {
        std::vector<unsigned char> support(n, 1);
        const auto result =
            citlali::pipeline::calculate_rtc_sampling_complete_context(
                support, 0, n == 0 ? -1 : static_cast<Eigen::Index>(n - 1),
                0, n == 0 ? -1 : static_cast<Eigen::Index>(n - 1),
                1, 0, 3, 100.0);
        if (n < 3) {
            EXPECT_EQ(result.full_output_count, 0u);
            EXPECT_EQ(result.candidate_status,
                      citlali::pipeline::RtcSamplingStatusCode::candidate_unusable_no_complete_context);
            EXPECT_EQ(result.candidate_reason,
                      citlali::pipeline::RtcSamplingReasonCode::no_complete_context);
        }
        else {
            EXPECT_GT(result.full_output_count, 0u);
            EXPECT_EQ(result.candidate_status,
                      citlali::pipeline::RtcSamplingStatusCode::candidate_evaluable);
        }
    }
}

TEST(RtcLearnedSamplingMetrics,
     DetectorContextAccountingSeparatesGuardFromResidualScienceFlag) {
    using Category = citlali::pipeline::RtcSamplingContextCategory;
    citlali::pipeline::RtcSamplingContextDomain domain;
    domain.n_times = 3;
    domain.n_detectors = 2;
    domain.categories = {
        Category::fully_supported, Category::fully_supported,
        Category::science_flag, Category::realized_filter_guard,
        Category::fully_supported, Category::fully_supported};
    const auto result =
        citlali::pipeline::calculate_rtc_sampling_complete_context(
            domain, 0, 2, 0, 2, 1, 0, 1, 10.0);
    EXPECT_EQ(result.detector_output_cell_count, 6u);
    EXPECT_EQ(result.detector_output_category_count[static_cast<std::size_t>(
                  Category::science_flag)], 1u);
    EXPECT_EQ(result.detector_output_category_count[static_cast<std::size_t>(
                  Category::realized_filter_guard)], 1u);
    EXPECT_EQ(std::accumulate(
                  result.detector_output_category_count.begin(),
                  result.detector_output_category_count.end(),
                  std::size_t{0}),
              result.detector_output_cell_count);
    EXPECT_EQ(result.full_output_count, 2u);
}

TEST(RtcLearnedSamplingMetrics, RealizedCadenceComesFromAssignedTimeGrid) {
    citlali::pipeline::RtcSamplingCadenceState cadence;
    cadence.requested_output_hz = 10.0;
    cadence.effective_output_hz = 10.0;
    cadence.requested_factor = 1;
    cadence.effective_factor = 1;
    const Eigen::Vector4d assigned{0.0, 0.1, 0.2, 0.3};
    citlali::pipeline::measure_rtc_sampling_realized_cadence(
        cadence, assigned);
    EXPECT_TRUE(cadence.realized_valid);
    EXPECT_DOUBLE_EQ(cadence.realized_native_hz, 10.0);
    EXPECT_DOUBLE_EQ(cadence.realized_output_hz, 10.0);
    EXPECT_EQ(cadence.effective_realized_consistency, "consistent");

    const Eigen::Vector4d irregular{0.0, 0.1, 0.21, 0.3};
    citlali::pipeline::measure_rtc_sampling_realized_cadence(
        cadence, irregular);
    EXPECT_FALSE(cadence.realized_valid);
    EXPECT_EQ(cadence.realized_reason,
              citlali::pipeline::RtcSamplingReasonCode::irregular_realized_cadence);
}

TEST(RtcLearnedSamplingMetrics, TypedCarrierResetsBetweenObservations) {
    citlali::pipeline::TimestreamAlignmentState state;
    state.rtc_sampling_source_motion =
        citlali::pipeline::capture_rtc_sampling_source_motion(
            source_rows({0.0, 0.02}, {0.0, 0.04}), rad_to_arcsec);
    ASSERT_EQ(state.rtc_sampling_source_motion.interval_count, 1u);
    ASSERT_EQ(state.rtc_sampling_source_motion.eligible_interval_count, 1u);
    citlali::pipeline::bind_rtc_sampling_source_observation_identity(
        state.rtc_sampling_source_motion, 4, "obs-4", "tel-4.nc");
    EXPECT_TRUE(citlali::pipeline::rtc_sampling_source_observation_matches(
        state.rtc_sampling_source_motion, 4, "obs-4", "tel-4.nc"));
    EXPECT_FALSE(citlali::pipeline::rtc_sampling_source_observation_matches(
        state.rtc_sampling_source_motion, 5, "obs-5", "tel-5.nc"));
    citlali::pipeline::reset_rtc_sampling_source_motion(state);
    EXPECT_EQ(state.rtc_sampling_source_motion.interval_count, 0u);
    EXPECT_TRUE(state.rtc_sampling_source_motion.intervals.empty());
    EXPECT_FALSE(state.rtc_sampling_source_motion.observation_identity_available);
    EXPECT_EQ(state.rtc_sampling_source_motion.reason,
              "missing_source_motion_columns");
    state.rtc_sampling_source_motion =
        citlali::pipeline::capture_rtc_sampling_source_motion(
            source_rows({10.0, 10.02}, {0.0, 0.01}), rad_to_arcsec);
    EXPECT_EQ(state.rtc_sampling_source_motion.interval_count, 1u);
    EXPECT_EQ(state.rtc_sampling_source_motion.eligible_interval_count, 0u);
    EXPECT_EQ(state.rtc_sampling_source_motion.reason,
              "unavailable_low_velocity");
}

TEST(RtcLearnedSamplingMetrics, ProductionSummarySeparatesHwprAndCadenceStates) {
    FakeCalib calib;
    FakeRawPlan plan;
    FakeRtcProc rtc;
    const auto telescope = one_scan_telescope();
    const auto scan_summary = available_scan_summary();
    citlali::pipeline::RtcSamplingCadenceState cadence;
    cadence.requested_output_hz = 50.0;
    cadence.effective_native_hz = 100.0;
    cadence.effective_output_hz = 50.0;
    cadence.realized_native_hz = 100.0;
    cadence.realized_output_hz = 50.0;
    cadence.requested_factor = 2;
    cadence.effective_factor = 2;
    cadence.realized_factor = 2;
    cadence.realized_valid = true;
    cadence.realized_reason = citlali::pipeline::RtcSamplingReasonCode::none;
    cadence.requested_effective_consistency = "consistent";
    cadence.effective_realized_consistency = "consistent";

    citlali::pipeline::RtcSamplingHwprState file_present_disabled;
    const auto disabled =
        citlali::pipeline::calculate_rtcdiag_scan_array_summary(
            calib, sampling_filter_state(plan, rtc), telescope, scan_summary,
            file_present_disabled, cadence, 1, 1, 1, -999.0, -999);
    EXPECT_EQ(disabled.prerequisite_status[0], static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::prerequisite_available));
    EXPECT_EQ(disabled.applied_scan_status[0], static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::scan_usable_for_applied_rtc_operator));

    auto ignored = file_present_disabled;
    const auto ignored_values =
        citlali::pipeline::calculate_rtcdiag_scan_array_summary(
            calib, sampling_filter_state(plan, rtc), telescope, scan_summary,
            ignored, cadence,
            1, 1, 1, -999.0, -999);
    EXPECT_EQ(ignored_values.prerequisite_status[0], static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::prerequisite_available));

    auto enabled = ignored;
    enabled.analysis_mode =
        citlali::pipeline::RtcSamplingHwprState::AnalysisMode::hwpr_dependent;
    const auto enabled_values =
        citlali::pipeline::calculate_rtcdiag_scan_array_summary(
            calib, sampling_filter_state(plan, rtc), telescope, scan_summary,
            enabled, cadence,
            1, 1, 1, -999.0, -999);
    EXPECT_EQ(enabled_values.prerequisite_status[0], static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::prerequisite_unavailable));
    EXPECT_EQ(enabled_values.prerequisite_reason[0], static_cast<int>(
        citlali::pipeline::RtcSamplingReasonCode::unsupported_hwpr));
    EXPECT_EQ(enabled_values.candidate_mmax[0], 1);
    EXPECT_FALSE(enabled_values.candidate_table_available);
    EXPECT_TRUE(enabled_values.candidate_factors.empty());
    EXPECT_TRUE(enabled_values.candidate_status.empty());
    EXPECT_TRUE(enabled_values.candidate_plan_transfer_status.empty());
    EXPECT_EQ(enabled_values.applied_scan_status[0], static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::applied_operator_not_applicable));

    auto inconsistent = cadence;
    inconsistent.realized_valid = false;
    inconsistent.realized_reason =
        citlali::pipeline::RtcSamplingReasonCode::cadence_state_mismatch;
    const auto mismatch =
        citlali::pipeline::calculate_rtcdiag_scan_array_summary(
            calib, sampling_filter_state(plan, rtc), telescope, scan_summary,
            file_present_disabled, inconsistent, 1, 1, 1, -999.0, -999);
    EXPECT_EQ(mismatch.prerequisite_reason[0], static_cast<int>(
        citlali::pipeline::RtcSamplingReasonCode::cadence_state_mismatch));
    EXPECT_EQ(mismatch.applied_scan_status[0], static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::applied_operator_not_applicable));

    auto filtered_plan = plan;
    filtered_plan.requested.filter = {true, 40.0, 1.0, 20.0, 31};
    filtered_plan.effective.filter = {true, 50.0, 2.0, 18.0, 33};
    auto filtered_rtc = rtc;
    filtered_rtc.run_tod_filter = true;
    filtered_rtc.filter.enabled = true;
    filtered_rtc.filter.a_gibbs = 60.0;
    filtered_rtc.filter.freq_low_Hz = 3.0;
    filtered_rtc.filter.freq_high_Hz = 16.0;
    filtered_rtc.filter.n_terms = 3;
    filtered_rtc.filter.filter = Eigen::Vector3d{0.25, 0.5, 0.25};
    const auto filtered =
        citlali::pipeline::calculate_rtcdiag_scan_array_summary(
            calib, sampling_filter_state(filtered_plan, filtered_rtc),
            telescope, scan_summary,
            file_present_disabled, cadence, 1, 1, 1, -999.0, -999);
    EXPECT_EQ(filtered.fir_coefficients,
              (std::vector<double>{0.25, 0.5, 0.25}));
    EXPECT_EQ(filtered.fir_digest,
              citlali::pipeline::rtc_sampling_fir_digest(
                  {0.25, 0.5, 0.25}));
    EXPECT_TRUE(filtered.filter_requested_enabled);
    EXPECT_TRUE(filtered.filter_effective_enabled);
    EXPECT_TRUE(filtered.filter_realized_enabled);
    EXPECT_DOUBLE_EQ(filtered.filter_requested_a_gibbs, 40.0);
    EXPECT_DOUBLE_EQ(filtered.filter_effective_a_gibbs, 50.0);
    EXPECT_DOUBLE_EQ(filtered.filter_realized_a_gibbs, 60.0);
}

TEST(RtcLearnedSamplingMetrics, ExactBaseABDiagnosticCaptureIsNonInterfering) {
    auto telescope_columns = source_rows(
        {0.0, 0.02, 0.04, 0.06, 0.08},
        {0.0, 0.04, 0.08, 0.12, 0.16});
    const auto before_columns = telescope_columns;
    const Eigen::VectorXd science_samples =
        (Eigen::VectorXd(4) << 1.0, 2.0, 3.0, 4.0).finished();
    const Eigen::VectorXi science_flags =
        (Eigen::VectorXi(4) << 0, 1, 0, 1).finished();
    const Eigen::VectorXd assigned_time =
        (Eigen::VectorXd(4) << 10.0, 10.1, 10.2, 10.3).finished();
    const Eigen::VectorXd rtc_state = Eigen::VectorXd::Constant(3, 11.0);
    const Eigen::VectorXd ptc_state = Eigen::VectorXd::Constant(3, 12.0);
    const Eigen::VectorXd map_state = Eigen::VectorXd::Constant(3, 13.0);
    const auto samples_before = science_samples;
    const auto flags_before = science_flags;
    const auto time_before = assigned_time;
    const auto rtc_before = rtc_state;
    const auto ptc_before = ptc_state;
    const auto map_before = map_state;

    const auto support = citlali::pipeline::capture_rtc_sampling_source_motion(
        telescope_columns, rad_to_arcsec);
    ASSERT_GT(support.interval_count, 0u);
    for (const auto &[name, values] : before_columns) {
        ASSERT_TRUE(telescope_columns.contains(name));
        EXPECT_TRUE((telescope_columns.at(name).array() == values.array()).all());
    }
    EXPECT_TRUE((science_samples.array() == samples_before.array()).all());
    EXPECT_TRUE((science_flags.array() == flags_before.array()).all());
    EXPECT_TRUE((assigned_time.array() == time_before.array()).all());
    EXPECT_TRUE((rtc_state.array() == rtc_before.array()).all());
    EXPECT_TRUE((ptc_state.array() == ptc_before.array()).all());
    EXPECT_TRUE((map_state.array() == map_before.array()).all());
}

TEST(RtcLearnedSamplingMetrics, AtomicRtcdiagFailuresLeaveNoPartialArtifact) {
    const auto root = std::filesystem::temp_directory_path() /
                      "citlali_rtcdiag_atomic_stage_a";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    ASSERT_TRUE(std::filesystem::create_directories(root));

    const auto assert_absent = [](const std::filesystem::path &path) {
        EXPECT_FALSE(std::filesystem::exists(path));
        if (std::filesystem::exists(path.parent_path())) {
            for (const auto &entry :
                 std::filesystem::directory_iterator(path.parent_path())) {
                EXPECT_EQ(entry.path().filename().string().find(
                              path.filename().string() +
                              netcdf_atomic_staging_marker),
                          std::string::npos);
            }
        }
    };

    const auto create_failure = root / "missing" / "create.nc";
    EXPECT_THROW(write_netcdf_atomic(create_failure.string(), [](auto &) {}),
                 std::exception);
    assert_absent(create_failure);

    const auto write_failure = root / "write.nc";
    write_netcdf_atomic(write_failure.string(), [](netCDF::NcFile &fo) {
        add_netcdf_var(fo, "PRIOR_GOOD", 1);
    });
    EXPECT_THROW(
        write_netcdf_atomic(write_failure.string(), [](netCDF::NcFile &fo) {
            add_netcdf_var(fo, "BEFORE_FAILURE", 1);
            throw std::runtime_error("injected write failure");
        }),
        std::runtime_error);
    ASSERT_TRUE(std::filesystem::is_regular_file(write_failure));
    {
        netCDF::NcFile prior(write_failure.string(), netCDF::NcFile::read);
        EXPECT_FALSE(prior.getVar("PRIOR_GOOD").isNull());
        EXPECT_TRUE(prior.getVar("BEFORE_FAILURE").isNull());
    }

    const auto sync_failure = root / "sync.nc";
    EXPECT_THROW(
        write_netcdf_atomic(sync_failure.string(), [](netCDF::NcFile &fo) {
            add_netcdf_var(fo, "BEFORE_SYNC_FAILURE", 1);
            fo.close();
        }),
        std::exception);
    assert_absent(sync_failure);

    const auto rename_failure = root / "rename.nc";
    ASSERT_TRUE(std::filesystem::create_directories(rename_failure));
    const auto marker = rename_failure / "preserved";
    ASSERT_TRUE(std::filesystem::create_directory(marker));
    EXPECT_THROW(
        write_netcdf_atomic(rename_failure.string(), [](netCDF::NcFile &fo) {
            add_netcdf_var(fo, "COMPLETE", 1);
        }),
        DataIOError);
    EXPECT_TRUE(std::filesystem::is_directory(rename_failure));
    EXPECT_TRUE(std::filesystem::is_directory(marker));
    for (const auto &entry : std::filesystem::directory_iterator(root)) {
        EXPECT_EQ(entry.path().filename().string().find(
                      rename_failure.filename().string() +
                      netcdf_atomic_staging_marker),
                  std::string::npos);
    }
    std::filesystem::remove_all(root, ec);
}

TEST(RtcLearnedSamplingMetrics, PersistsSuccessorSchemaAndNoSelection) {
    citlali::pipeline::RtcDiagScanArraySummaryData values;
    values.candidate_factors = {1};
    values.candidate_phases = {0};
    values.fir_coefficients = {1.0};
    values.fir_digest = citlali::pipeline::rtc_sampling_fir_digest({1.0});
    values.fir_status =
        citlali::pipeline::RtcSamplingStatusCode::plan_transfer_available;
    values.fir_reason = citlali::pipeline::RtcSamplingReasonCode::none;
    auto one_i = std::vector<int>{0};
    auto one_d = std::vector<double>{0.0};
    values.prerequisite_status = one_i; values.prerequisite_reason = one_i;
    values.candidate_mmax = std::vector<int>{1}; values.beam_fwhm_arcsec = {4.66};
    values.candidate_range_status = std::vector<int>{static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::candidate_range_available)};
    values.candidate_range_reason = one_i;
    values.applied_scan_status = one_i; values.applied_scan_reason = one_i;
    values.temporal_sigma_s = {0.02}; values.candidate_status = one_i;
    values.candidate_reason = one_i; values.candidate_alias_status = one_i;
    values.candidate_plan_transfer_status = one_i;
    values.candidate_plan_transfer_reason = one_i;
    values.candidate_alias_reason = one_i;
    values.candidate_amplitude_status = one_i; values.candidate_amplitude_reason = one_i;
    values.candidate_phase_status = one_i; values.candidate_phase_reason = one_i;
    values.candidate_power_status = one_i; values.candidate_power_reason = one_i;
    values.candidate_distortion_status = one_i; values.candidate_distortion_reason = one_i;
    values.candidate_stopband_status = one_i;
    values.candidate_stopband_reason = one_i; values.output_sample_rate_hz = {100.0};
    values.output_nyquist_hz = {50.0}; values.samples_per_fwhm = {4.66};
    values.relative_amplitude_at_dc = {1.0}; values.relative_phase_at_dc_rad = one_d;
    values.relative_power_at_dc = {1.0}; values.relative_distortion_at_dc = one_d;
    values.alias_amplitude_max_lower = one_d; values.alias_amplitude_max_upper = one_d;
    values.alias_lipschitz_bound = one_d; values.alias_evaluations = one_i;
    values.relative_amplitude_max_lower = {1.0};
    values.relative_amplitude_max_upper = {1.0}; values.relative_phase_abs_max_upper_rad = one_d;
    values.relative_amplitude_error_enclosure = one_d;
    values.relative_amplitude_lipschitz_bound = one_d;
    values.relative_amplitude_evaluations = one_i;
    values.relative_phase_abs_max_lower_rad = one_d;
    values.relative_phase_error_enclosure_rad = one_d;
    values.relative_phase_lipschitz_bound = one_d;
    values.relative_phase_evaluations = one_i;
    values.relative_power_max_lower = {1.0};
    values.relative_power_max_upper = {1.0}; values.relative_distortion_max_upper = one_d;
    values.relative_power_error_enclosure = one_d;
    values.relative_power_lipschitz_bound = one_d;
    values.relative_power_evaluations = one_i;
    values.relative_distortion_max_lower = one_d;
    values.relative_distortion_error_enclosure = one_d;
    values.relative_distortion_lipschitz_bound = one_d;
    values.relative_distortion_evaluations = one_i;
    values.alias_error_enclosure = one_d; values.stopband_amplitude_max_lower = one_d;
    values.stopband_amplitude_max_upper = one_d; values.stopband_rejection_db_lower = one_d;
    values.stopband_rejection_db_upper = one_d; values.stopband_error_enclosure = one_d;
    values.stopband_lipschitz_bound = one_d; values.stopband_evaluations = one_i;
    values.numerical_evaluations = one_i; values.tap_count = std::vector<int>{1};
    values.left_context = one_i; values.right_context = one_i;
    values.eligible_input_support = std::vector<int>{10};
    values.candidate_output_count = std::vector<int>{10};
    values.full_output_count = std::vector<int>{10};
    values.incomplete_boundary_count = one_i; values.incomplete_gap_count = one_i;
    values.incomplete_other_count = one_i; values.longest_full_run = std::vector<int>{10};
    values.full_duration_s = {0.1}; values.full_fraction = {1.0};
    values.candidate_table_status =
        citlali::pipeline::RtcSamplingStatusCode::candidate_table_available;
    values.candidate_table_reason = citlali::pipeline::RtcSamplingReasonCode::none;
    values.candidate_table_available = true;
    values.estimated_candidate_rows = 1;
    values.estimated_rectangular_storage_cells = 1;

    const auto path = std::filesystem::temp_directory_path() /
                      "citlali_rtc_sampling_stage_a_v3.nc";
    const auto manifest_path = std::filesystem::temp_directory_path() /
                               "citlali_rtc_sampling_raw_manifest.yaml";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(manifest_path, ec);
    {
        std::ofstream manifest(manifest_path);
        manifest << "schema_version: citlali-raw-timestream-provenance-v1\n";
    }
    const auto write_stage = [&]() {
      return write_netcdf_staging(path.string(), [&](netCDF::NcFile &file) {
        const auto scan = file.addDim("n_scans", 1);
        const auto array = file.addDim("n_arrays", 1);
        citlali::pipeline::RtcSamplingHwprState hwpr;
        citlali::pipeline::RtcSamplingCadenceState cadence;
        cadence.requested_output_hz = 50.0;
        cadence.effective_native_hz = 100.0;
        cadence.effective_output_hz = 50.0;
        cadence.realized_native_hz = 100.0;
        cadence.realized_output_hz = 50.0;
        cadence.requested_factor = 2;
        cadence.effective_factor = 2;
        cadence.realized_factor = 2;
        cadence.realized_valid = true;
        cadence.realized_reason =
            citlali::pipeline::RtcSamplingReasonCode::none;
        cadence.requested_effective_consistency = "consistent";
        cadence.effective_realized_consistency = "consistent";
        auto support =
            citlali::pipeline::capture_rtc_sampling_source_motion(
                source_rows({0.0, 0.02}, {0.0, 0.04}), rad_to_arcsec);
        citlali::pipeline::bind_rtc_sampling_source_observation_identity(
            support, 0, "000001", "telescope.nc");
        citlali::pipeline::add_rtcdiag_scan_array_summary_outputs(
            file, {scan, array}, {1, 1}, values, hwpr, cadence, support,
            "raw_timestream_provenance.yaml",
            "0123456789abcdef0123456789abcdef01234567");
      });
    };
    const auto staging_path = write_stage();
    EXPECT_EQ(citlali::pipeline::finalize_rtcdiag_successor_staging(
                  staging_path, manifest_path),
              path.string());
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::read);
        const auto get_string = [&](const char *name) {
            char *raw = nullptr;
            file.getVar(name).getVar(&raw);
            const std::string value = raw == nullptr ? "" : raw;
            if (raw != nullptr) {
                nc_free_string(1, &raw);
            }
            return value;
        };
        EXPECT_EQ(get_string("RTC_DIAG_SCHEMA_VERSION"), "rtcdiag-v3");
        EXPECT_EQ(get_string("RTC_SAMPLING_ALGORITHM_VERSION"),
                  "rtc-learned-sampling-stage-a-v3");
        EXPECT_EQ(get_string("RTC_SAMPLING_FIR_DIGEST_CONVENTION"),
                  "sha256-u64le-count-then-ieee754-binary64le-realized-order-v1");
        EXPECT_EQ(get_string("RTC_SAMPLING_SOURCE_GUARD_VERSION"),
                  "native-source-row-gap-jump-and-one-row-scan-boundary-v1");
        EXPECT_EQ(get_string("RTC_SAMPLING_COUNTERFACTUAL_BINDING"),
                  "(M,phase=0,H_RTC_realized); same exact realized FIR for every unranked factor; no factor-specific synthesis");
        EXPECT_NE(get_string("RTC_SAMPLING_NUMERICAL_DOMAIN").find(
                      "[-fs/(2M),fs/(2M))"),
                  std::string::npos);
        const auto vocabulary =
            get_string("RTC_SAMPLING_STATUS_REASON_VOCABULARY");
        for (const std::string required : {
                 "candidate_not_evaluated_prerequisite",
                 "plan_transfer_available", "plan_transfer_unavailable",
                 "applied_operator_not_applicable", "numerical_converged",
                 "numerical_bounded_not_converged", "numerical_failed",
                 "not_applicable_no_decimation"}) {
            EXPECT_NE(vocabulary.find(required), std::string::npos)
                << required;
        }
        EXPECT_FALSE(file.getVar("RTC_DIAG_SCHEMA_VERSION").isNull());
        EXPECT_FALSE(file.getVar("RTC_SAMPLING_BEAM_MODEL").isNull());
        EXPECT_FALSE(file.getVar("RTC_SAMPLING_BEAM_FWHM_AUTHORITY").isNull());
        EXPECT_FALSE(file.getVar("RTC_SAMPLING_ALIAS_CONVENTION").isNull());
        EXPECT_FALSE(file.getVar("RTC_SAMPLING_STATUS_REASON_VOCABULARY").isNull());
        EXPECT_FALSE(file.getDim("n_rtc_sampling_source_intervals").isNull());
        EXPECT_FALSE(file.getVar("rtc_sampling_source_interval_reason").isNull());
        EXPECT_FALSE(file.getVar("rtc_sampling_source_interval_start_row").isNull());
        EXPECT_FALSE(file.getVar("rtc_sampling_full_output_count").isNull());
        EXPECT_FALSE(file.getVar(
            "rtc_sampling_detector_output_cell_count").isNull());
        EXPECT_FALSE(file.getVar(
            "rtc_sampling_detector_output_science_flag_count").isNull());
        EXPECT_FALSE(file.getVar(
            "rtc_sampling_detector_output_realized_filter_guard_count").isNull());
        int cadence_realized_valid = 0;
        int requested_factor = 0;
        int effective_factor = 0;
        int realized_factor = 0;
        int table_status = -1;
        int boundary_guard_rows = 0;
        double max_interval_s = 0.0;
        double max_pointing_step_rad = 0.0;
        file.getVar("RTC_SAMPLING_CADENCE_REALIZED_VALID").getVar(
            &cadence_realized_valid);
        file.getVar("RTC_SAMPLING_CADENCE_REQUESTED_FACTOR").getVar(&requested_factor);
        file.getVar("RTC_SAMPLING_CADENCE_EFFECTIVE_FACTOR").getVar(&effective_factor);
        file.getVar("RTC_SAMPLING_CADENCE_REALIZED_FACTOR").getVar(&realized_factor);
        file.getVar("RTC_SAMPLING_CANDIDATE_TABLE_STATUS").getVar(&table_status);
        file.getVar("RTC_SAMPLING_SOURCE_BOUNDARY_GUARD_ROWS").getVar(
            &boundary_guard_rows);
        file.getVar("RTC_SAMPLING_SOURCE_MAX_INTERVAL_S").getVar(
            &max_interval_s);
        file.getVar("RTC_SAMPLING_SOURCE_MAX_POINTING_STEP_RAD").getVar(
            &max_pointing_step_rad);
        EXPECT_EQ(get_string("RTC_SAMPLING_ANALYSIS_MODE"),
                  "total_intensity");
        EXPECT_EQ(get_string("RTC_SAMPLING_HWPR_REASON"), "none");
        EXPECT_EQ(cadence_realized_valid, 1);
        EXPECT_EQ(requested_factor, 2);
        EXPECT_EQ(effective_factor, 2);
        EXPECT_EQ(realized_factor, 2);
        EXPECT_EQ(table_status, static_cast<int>(
            citlali::pipeline::RtcSamplingStatusCode::candidate_table_available));
        EXPECT_EQ(boundary_guard_rows, 1);
        EXPECT_DOUBLE_EQ(max_interval_s, 0.1);
        EXPECT_DOUBLE_EQ(max_pointing_step_rad, 0.01);
        EXPECT_TRUE(file.getVar("rtc_sampling_candidate_rank").isNull());
        EXPECT_TRUE(file.getVar("rtc_sampling_candidate_selected").isNull());
        EXPECT_TRUE(file.getVar("scan_array_beam_major_fwhm_arcsec").isNull());
    }
    for (const auto stage : {
             citlali::pipeline::RtcdiagFinalizeFailureStage::manifest,
             citlali::pipeline::RtcdiagFinalizeFailureStage::provenance,
             citlali::pipeline::RtcdiagFinalizeFailureStage::validation,
             citlali::pipeline::RtcdiagFinalizeFailureStage::sync,
             citlali::pipeline::RtcdiagFinalizeFailureStage::close,
             citlali::pipeline::RtcdiagFinalizeFailureStage::publish}) {
        const auto failed_staging = write_stage();
        EXPECT_THROW(
            citlali::pipeline::finalize_rtcdiag_successor_staging(
                failed_staging, manifest_path, stage),
            DataIOError);
        EXPECT_FALSE(std::filesystem::exists(failed_staging));
        netCDF::NcFile prior(path.string(), netCDF::NcFile::read);
        EXPECT_FALSE(prior.getVar("RTC_SAMPLING_PRODUCT_CONTRACT_ID").isNull());
    }
    const auto refused_stage = write_stage();
    cleanup_netcdf_atomic_staging(refused_stage);
    EXPECT_THROW(
        citlali::pipeline::finalize_rtcdiag_successor_staging(
            path.string(), manifest_path),
        DataIOError);
    std::filesystem::remove(path, ec);
    std::filesystem::remove(manifest_path, ec);
}

TEST(RtcLearnedSamplingMetrics, ResourceUnavailableWriterEmitsNoTruncatedTable) {
    citlali::pipeline::RtcDiagScanArraySummaryData values;
    values.fir_coefficients = {1.0};
    values.fir_digest = citlali::pipeline::rtc_sampling_fir_digest({1.0});
    values.fir_status =
        citlali::pipeline::RtcSamplingStatusCode::plan_transfer_available;
    values.fir_reason = citlali::pipeline::RtcSamplingReasonCode::none;
    values.prerequisite_status = {static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::prerequisite_available)};
    values.prerequisite_reason = {0};
    values.candidate_mmax = {8192};
    values.candidate_range_status = {static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::candidate_range_available)};
    values.candidate_range_reason = {0};
    values.applied_scan_status = {static_cast<int>(
        citlali::pipeline::RtcSamplingStatusCode::scan_usable_for_applied_rtc_operator)};
    values.applied_scan_reason = {0};
    values.beam_fwhm_arcsec = {8.48};
    values.temporal_sigma_s = {0.01};
    values.candidate_table_status =
        citlali::pipeline::RtcSamplingStatusCode::candidate_table_unavailable_resource_limit;
    values.candidate_table_reason =
        citlali::pipeline::RtcSamplingReasonCode::numerical_resource_limit;
    values.candidate_table_available = false;
    values.estimated_candidate_rows = 8192;

    const auto path = std::filesystem::temp_directory_path() /
                      "citlali_rtc_sampling_resource_unavailable.nc";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    write_netcdf_atomic(path.string(), [&](netCDF::NcFile &file) {
        const auto scan = file.addDim("n_scans", 1);
        const auto array = file.addDim("n_arrays", 1);
        citlali::pipeline::add_rtcdiag_scan_array_summary_outputs(
            file, {scan, array}, {1, 1}, values, {}, {}, {}, "raw", "test");
    });
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::read);
        EXPECT_TRUE(file.getDim("n_rtc_sampling_candidates").isNull());
        EXPECT_TRUE(file.getVar("rtc_sampling_candidate_factor").isNull());
        EXPECT_TRUE(file.getVar("rtc_sampling_candidate_status").isNull());
        EXPECT_FALSE(file.getVar("rtc_sampling_candidate_mmax").isNull());
        EXPECT_FALSE(file.getVar("RTC_SAMPLING_CANDIDATE_TABLE_STATUS").isNull());
        EXPECT_FALSE(file.getVar("rtc_sampling_realized_fir_coefficients").isNull());
    }
    std::filesystem::remove(path, ec);
}

}  // namespace
