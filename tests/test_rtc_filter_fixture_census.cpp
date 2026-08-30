#include "rtc_filter_fixture_census_model.h"

#include <citlali/core/pipeline/timestream_native_timing.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdint>
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
    const auto evidence = fixture::evaluate_factor(
        fixture::Array::a1100, 1000.0, 1.0, 1);

    EXPECT_DOUBLE_EQ(evidence.wavelength_m, 299792458.0 / 272.0e9);
    EXPECT_NEAR(evidence.airy_fwhm_arcsec, 4.6786413788, 5.0e-11);
    EXPECT_DOUBLE_EQ(evidence.safe_input_sample_rate_hz, 999.9);
    EXPECT_TRUE(evidence.sampling_eligible());
}

TEST(RtcFilterFixtureCensus, ScreensEveryFactorWithoutCertifyingAFilter) {
    constexpr double sample_rate_hz = 122.0703125;
    constexpr double accepted_152390_maximum_arcsec_per_sec =
        221.40490828695155;

    const auto a1100 = fixture::evaluate_factor(
        fixture::Array::a1100, sample_rate_hz,
        accepted_152390_maximum_arcsec_per_sec, 1);
    const auto a1400 = fixture::evaluate_factor(
        fixture::Array::a1400, sample_rate_hz,
        accepted_152390_maximum_arcsec_per_sec, 1);
    const auto a2000 = fixture::evaluate_factor(
        fixture::Array::a2000, sample_rate_hz,
        accepted_152390_maximum_arcsec_per_sec, 1);

    EXPECT_TRUE(a1100.science_band_sampling_adequate);
    EXPECT_FALSE(a1100.beam_sampling_adequate);
    EXPECT_FALSE(a1100.sampling_eligible());
    EXPECT_TRUE(a1400.science_band_sampling_adequate);
    EXPECT_FALSE(a1400.beam_sampling_adequate);
    EXPECT_FALSE(a1400.sampling_eligible());
    EXPECT_TRUE(a2000.science_band_sampling_adequate);
    EXPECT_TRUE(a2000.beam_sampling_adequate);
    EXPECT_TRUE(a2000.sampling_eligible());

    for (int factor = fixture::minimum_factor;
         factor <= fixture::maximum_factor; ++factor) {
        const auto evidence = fixture::evaluate_factor(
            fixture::Array::a2000, sample_rate_hz,
            accepted_152390_maximum_arcsec_per_sec, factor);
        if (factor == 1) {
            EXPECT_TRUE(evidence.sampling_eligible());
        } else {
            EXPECT_FALSE(evidence.sampling_eligible());
        }
    }
}

TEST(RtcFilterFixtureCensus, IncludesTheExactSamplingBoundary) {
    constexpr double maximum_speed_arcsec_per_sec = 12.3;
    constexpr int factor = 7;
    const double minimum_input =
        fixture::minimum_safe_output_sample_rate_hz(
            fixture::Array::a1100, maximum_speed_arcsec_per_sec) *
        factor / fixture::cadence_margin;

    const auto below = fixture::evaluate_factor(
        fixture::Array::a1100, minimum_input * (1.0 - 1.0e-12),
        maximum_speed_arcsec_per_sec, factor);
    const auto exact = fixture::evaluate_factor(
        fixture::Array::a1100, minimum_input,
        maximum_speed_arcsec_per_sec, factor);
    const auto above = fixture::evaluate_factor(
        fixture::Array::a1100, minimum_input * (1.0 + 1.0e-12),
        maximum_speed_arcsec_per_sec, factor);

    EXPECT_FALSE(below.sampling_eligible());
    EXPECT_TRUE(exact.sampling_eligible());
    EXPECT_TRUE(above.sampling_eligible());
    EXPECT_DOUBLE_EQ(exact.output_samples_per_airy_fwhm, 4.0);
}

TEST(RtcFilterFixtureCensus, RejectsInvalidStructuralInputs) {
    EXPECT_THROW(
        fixture::evaluate_factor(fixture::Array::a1100, 0.0, 1.0, 1),
        std::invalid_argument);
    EXPECT_THROW(
        fixture::evaluate_factor(fixture::Array::a1100, 100.0, 0.0, 1),
        std::invalid_argument);
    EXPECT_THROW(
        fixture::evaluate_factor(fixture::Array::a1100, 100.0, 1.0, 0),
        std::invalid_argument);
    EXPECT_THROW(
        fixture::evaluate_factor(fixture::Array::a1100, 100.0, 1.0, 257),
        std::invalid_argument);
}

}  // namespace
