#include "rtc_filter_fixture_census_model.h"

#include <citlali/core/pipeline/timestream_native_timing.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

namespace fixture = citlali::wp7::rtc_filter_fixture;
namespace pipeline = citlali::pipeline;

Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
timestamps(std::array<std::int64_t, 4> counters) {
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::RowMajor>
        result(4, 6);
    result.setZero();
    for (Eigen::Index row = 0; row < result.rows(); ++row) {
        result(row, 0) = 1000;
        result(row, 1) = 1;
        result(row, 2) = row * 10000;
        result(row, 3) = counters[static_cast<std::size_t>(row)];
    }
    return result;
}

TEST(RtcFilterFixtureCensus, PreservesIndependentNetworkNativeAxes) {
    const auto nw0 = pipeline::make_native_network_alignment(
        0, 0, timestamps({0, 1, 2, 3}), 1.0e6, 0.0);
    const auto nw7 = pipeline::make_native_network_alignment(
        7, 0, timestamps({0, 1, 2, 3}), 1.0e6, 0.0025);

    ASSERT_EQ(nw0.row_count(), nw7.row_count());
    for (pipeline::TimestreamNativeRow row = 0; row < nw0.row_count();
         ++row) {
        EXPECT_EQ(nw0.identity(row).network_id(), 0);
        EXPECT_EQ(nw7.identity(row).network_id(), 7);
        EXPECT_DOUBLE_EQ(
            nw7.identity(row).reconstructed_time_unix_sec(),
            nw0.identity(row).reconstructed_time_unix_sec() + 0.0025);
    }
    EXPECT_DOUBLE_EQ(
        nw0.identity(0).reconstructed_time_unix_sec(), 1000.0);
    EXPECT_DOUBLE_EQ(
        nw7.identity(0).reconstructed_time_unix_sec(), 1000.0025);
}

TEST(RtcFilterFixtureCensus, GapRemainsLocalToItsOriginatingNetwork) {
    const auto nw0 = pipeline::make_native_network_alignment(
        0, 0, timestamps({0, 1, 4, 5}), 1.0e6, 0.0);
    const auto nw7 = pipeline::make_native_network_alignment(
        7, 0, timestamps({0, 1, 2, 3}), 1.0e6, 0.0025);

    const auto nw0_runs = pipeline::partition_native_contiguous_runs(
        nw0, nw0.first_native_row(), nw0.past_last_native_row());
    const auto nw7_runs = pipeline::partition_native_contiguous_runs(
        nw7, nw7.first_native_row(), nw7.past_last_native_row());

    ASSERT_EQ(nw0_runs.size(), 2U);
    EXPECT_EQ(nw0_runs[0].row_count(), 2);
    EXPECT_EQ(nw0_runs[1].row_count(), 2);
    ASSERT_EQ(nw7_runs.size(), 1U);
    EXPECT_EQ(nw7_runs[0].row_count(), 4);
    EXPECT_EQ(nw7.row_count(), 4);
}

TEST(RtcFilterFixtureCensus, UsesExactApprovedArrayModel) {
    const auto evidence = fixture::evaluate_structural_mode(
        fixture::Array::a1100, 1000.0, 1);

    EXPECT_DOUBLE_EQ(evidence.wavelength_m, 299792458.0 / 272.0e9);
    EXPECT_NEAR(evidence.airy_fwhm_arcsec, 4.6786413788, 5.0e-11);
    EXPECT_DOUBLE_EQ(evidence.safe_input_sample_rate_hz, 999.9);
    EXPECT_EQ(evidence.governing_constraint,
              fixture::StructuralCeilingConstraint::beam_sampling);
    EXPECT_TRUE(evidence.has_science_speed_domain());
}

TEST(RtcFilterFixtureCensus,
     DerivesStructuralCeilingsWithoutUsingTheScanMaximum) {
    constexpr double sample_rate_hz = 122.0703125;

    const auto a1100 = fixture::evaluate_structural_mode(
        fixture::Array::a1100, sample_rate_hz, 1);
    const auto a1400 = fixture::evaluate_structural_mode(
        fixture::Array::a1400, sample_rate_hz, 1);
    const auto a2000 = fixture::evaluate_structural_mode(
        fixture::Array::a2000, sample_rate_hz, 1);

    EXPECT_NEAR(a1100.upper_speed_ceiling_arcsec_per_sec,
                135.9681197283374, 1.0e-12);
    EXPECT_NEAR(a1400.upper_speed_ceiling_arcsec_per_sec,
                172.8192923649896, 1.0e-12);
    EXPECT_NEAR(a2000.upper_speed_ceiling_arcsec_per_sec,
                246.55552377405178, 1.0e-12);
    EXPECT_EQ(a1100.governing_constraint,
              fixture::StructuralCeilingConstraint::beam_sampling);
    EXPECT_LT(a1100.upper_speed_ceiling_arcsec_per_sec,
              a1400.upper_speed_ceiling_arcsec_per_sec);
    EXPECT_LT(a1400.upper_speed_ceiling_arcsec_per_sec,
              a2000.upper_speed_ceiling_arcsec_per_sec);

    double previous_ceiling =
        std::numeric_limits<double>::infinity();
    for (int factor = fixture::minimum_factor;
         factor <= fixture::maximum_factor; ++factor) {
        const auto evidence = fixture::evaluate_structural_mode(
            fixture::Array::a2000, sample_rate_hz, factor);
        EXPECT_LT(evidence.upper_speed_ceiling_arcsec_per_sec,
                  previous_ceiling);
        previous_ceiling = evidence.upper_speed_ceiling_arcsec_per_sec;
    }
    EXPECT_FALSE(fixture::evaluate_structural_mode(
                     fixture::Array::a2000, sample_rate_hz,
                     fixture::maximum_factor)
                     .has_science_speed_domain());
}

TEST(RtcFilterFixtureCensus, IncludesTheExactUpperSpeedBoundary) {
    constexpr double ceiling = 12.3;
    EXPECT_TRUE(fixture::upper_speed_admitted(
        std::nextafter(ceiling, 0.0), ceiling));
    EXPECT_TRUE(fixture::upper_speed_admitted(ceiling, ceiling));
    EXPECT_FALSE(fixture::upper_speed_admitted(
        std::nextafter(ceiling,
                       std::numeric_limits<double>::infinity()),
        ceiling));
}

TEST(RtcFilterFixtureCensus,
     CountsOccurrenceCausesAndBreaksRetainedRunsExactly) {
    const std::vector<double> speeds{
        std::numeric_limits<double>::quiet_NaN(), 0.5, 5.0, 10.0,
        std::nextafter(10.0, std::numeric_limits<double>::infinity()),
        6.0, 7.0};
    const std::vector<std::uint8_t> continues_previous{
        0U, 1U, 1U, 1U, 1U, 1U, 0U};

    const auto result = fixture::summarize_occurrence_admission(
        speeds, continues_previous, 10.0);

    EXPECT_EQ(result.occurrence_count, 7U);
    EXPECT_EQ(result.ast_unavailable_count, 1U);
    EXPECT_EQ(result.below_minimum_science_speed_count, 1U);
    EXPECT_EQ(result.base_admitted_count, 5U);
    EXPECT_EQ(result.upper_speed_admitted_count, 4U);
    EXPECT_EQ(result.scan_speed_above_mode_support_count, 1U);
    EXPECT_EQ(result.retained_run_count, 3U);
    EXPECT_EQ(result.longest_retained_run_occurrences, 2U);
}

TEST(RtcFilterFixtureCensus, RejectsInvalidStructuralInputs) {
    EXPECT_THROW(
        fixture::evaluate_structural_mode(
            fixture::Array::a1100, 0.0, 1),
        std::invalid_argument);
    EXPECT_THROW(
        fixture::evaluate_structural_mode(
            fixture::Array::a1100, 100.0, 0),
        std::invalid_argument);
    EXPECT_THROW(
        fixture::evaluate_structural_mode(
            fixture::Array::a1100, 100.0, 257),
        std::invalid_argument);
    EXPECT_THROW(
        fixture::summarize_occurrence_admission({}, {}, 1.0),
        std::invalid_argument);
}

}  // namespace
