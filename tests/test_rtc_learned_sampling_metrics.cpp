#include <gtest/gtest.h>

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/rtc_learned_sampling_metrics.h>
#include <citlali/core/pipeline/rtcdiag_netcdf.h>
#include <citlali/core/timestream/rtc/filter.h>

#include <Eigen/Core>

#include <cmath>
#include <complex>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using citlali::pipeline::RtcSamplingMotionInterval;
using citlali::pipeline::RtcSamplingScanMotion;

TEST(RtcLearnedSamplingMetrics, ExtractsExactMaximumWithBoundaryGuard) {
    Eigen::VectorXd time(7);
    time << 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12;
    Eigen::VectorXd az(7);
    az << 0.000, 0.001, 0.003, 0.004, 0.008, 0.009, 0.020;
    Eigen::VectorXd alt = Eigen::VectorXd::Zero(7);

    const auto result =
        citlali::pipeline::calculate_rtc_sampling_scan_motion(
            time, az, alt, 0, 6, 1.0, 0.1, 0.1, 1);

    EXPECT_EQ(result.valid_interval_count, 4u);
    EXPECT_EQ(result.rejected_interval_count, 0u);
    EXPECT_DOUBLE_EQ(result.duration_s, 0.12);
    EXPECT_NEAR(result.speed_max_arcsec_s, 0.2, 1e-14);
    // The much faster final boundary interval is intentionally excluded.
    EXPECT_LT(result.speed_max_arcsec_s, 0.5);
    EXPECT_LE(result.speed_p995_arcsec_s, result.speed_max_arcsec_s);
}

TEST(RtcLearnedSamplingMetrics, RejectsInvalidGapIntervals) {
    Eigen::VectorXd time(7);
    time << 0.00, 0.02, 0.04, 0.30, 0.32, 0.34, 0.36;
    Eigen::VectorXd az(7);
    az << 0.000, 0.001, 0.003, 0.004, 0.008, 0.009, 0.010;
    Eigen::VectorXd alt = Eigen::VectorXd::Zero(7);

    const auto result =
        citlali::pipeline::calculate_rtc_sampling_scan_motion(
            time, az, alt, 0, 6, 1.0, 0.1, 0.1, 1);

    EXPECT_EQ(result.valid_interval_count, 3u);
    EXPECT_EQ(result.rejected_interval_count, 1u);
    EXPECT_NEAR(result.speed_max_arcsec_s, 0.2, 1e-14);
}

TEST(RtcLearnedSamplingMetrics, ProjectsEllipticalBeamAlongScan) {
    RtcSamplingScanMotion motion;
    motion.intervals = {
        RtcSamplingMotionInterval{10.0, 0.0},
        RtcSamplingMotionInterval{20.0,
                                  0.5 * citlali::pipeline::rtc_sampling_pi}};

    const auto projected =
        citlali::pipeline::calculate_rtc_sampling_projected_beam(
            motion, 10.0, 5.0, 0.0);

    EXPECT_DOUBLE_EQ(projected.major_fwhm_arcsec, 10.0);
    EXPECT_DOUBLE_EQ(projected.minor_fwhm_arcsec, 5.0);
    EXPECT_NEAR(projected.limiting_projected_fwhm_arcsec, 5.0, 1e-14);
    EXPECT_DOUBLE_EQ(projected.limiting_speed_arcsec_s, 20.0);
    EXPECT_NEAR(projected.limiting_crossing_time_s, 0.25, 1e-14);

    const double diagonal =
        citlali::pipeline::rtc_sampling_projected_fwhm_arcsec(
            10.0, 5.0, 0.0,
            0.25 * citlali::pipeline::rtc_sampling_pi);
    const double root_half = std::sqrt(0.5);
    EXPECT_NEAR(
        diagonal,
        1.0 / std::hypot(root_half / 10.0, root_half / 5.0), 1e-14);
}

TEST(RtcLearnedSamplingMetrics, CalculatesExactCenteredFirResponse) {
    const std::vector<double> coefficients{0.25, 0.5, 0.25};
    const auto dc = citlali::pipeline::rtc_sampling_fir_response(
        coefficients, 0.0, 100.0);
    const auto nyquist = citlali::pipeline::rtc_sampling_fir_response(
        coefficients, 50.0, 100.0);

    EXPECT_NEAR(dc.real(), 1.0, 1e-15);
    EXPECT_NEAR(dc.imag(), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(nyquist), 0.0, 1e-15);

    timestream::Filter realized;
    realized.a_gibbs = 50.0;
    realized.freq_low_Hz = 0.0;
    realized.freq_high_Hz = 16.0;
    realized.n_terms = 32;
    realized.make_filter(488.28125);
    const std::vector<double> exact(
        realized.filter.data(),
        realized.filter.data() + realized.filter.size());
    EXPECT_TRUE(std::isfinite(std::abs(
        citlali::pipeline::rtc_sampling_fir_response(
            exact, 16.0, 488.28125))));
}

TEST(RtcLearnedSamplingMetrics, ComposesBeamAndFirTransfer) {
    const std::vector<double> coefficients{0.25, 0.5, 0.25};
    constexpr double frequency = 8.0;
    constexpr double sample_rate = 100.0;
    constexpr double sigma = 0.02;
    const auto expected =
        citlali::pipeline::rtc_sampling_gaussian_beam_amplitude(
            frequency, sigma) *
        citlali::pipeline::rtc_sampling_fir_response(
            coefficients, frequency, sample_rate);
    const auto actual = citlali::pipeline::rtc_sampling_composed_transfer(
        coefficients, frequency, sample_rate, sigma);

    EXPECT_NEAR(actual.real(), expected.real(), 1e-15);
    EXPECT_NEAR(actual.imag(), expected.imag(), 1e-15);
}

TEST(RtcLearnedSamplingMetrics, CalculatesExactPhaseZeroAliasComponents) {
    const std::vector<double> identity{1.0};
    constexpr double output_frequency = 10.0;
    constexpr double sample_rate = 100.0;
    constexpr double sigma = 0.001;
    const auto result =
        citlali::pipeline::rtc_sampling_phase_zero_alias_power_at(
            identity, output_frequency, sample_rate, 2, sigma);
    const double desired = std::pow(
        citlali::pipeline::rtc_sampling_gaussian_beam_amplitude(
            10.0, sigma),
        2);
    const double alias = std::pow(
        citlali::pipeline::rtc_sampling_gaussian_beam_amplitude(
            -40.0, sigma),
        2);

    EXPECT_NEAR(result.desired, desired, 1e-15);
    EXPECT_NEAR(result.aliased, alias, 1e-15);
}

TEST(RtcLearnedSamplingMetrics, EnumeratesOnlyCurrentNyquistAdmittedFactors) {
    const auto factors =
        citlali::pipeline::rtc_sampling_supported_factors(100.0, 10.0);
    EXPECT_EQ(factors, (std::vector<int>{1, 2, 3, 4, 5}));
    EXPECT_TRUE(
        citlali::pipeline::rtc_sampling_supported_factors(100.0, 0.0)
            .empty());
}

TEST(RtcLearnedSamplingMetrics, IdentityResponseIsUnbroadenedAndDeterministic) {
    const std::vector<double> identity{1.0};
    const auto first =
        citlali::pipeline::calculate_rtc_sampling_candidate_metrics(
            2, 100.0, 20.0, identity, 0.025, 128);
    const auto second =
        citlali::pipeline::calculate_rtc_sampling_candidate_metrics(
            2, 100.0, 20.0, identity, 0.025, 128);

    EXPECT_DOUBLE_EQ(first.output_sample_rate_hz,
                     second.output_sample_rate_hz);
    EXPECT_DOUBLE_EQ(first.samples_per_fwhm, second.samples_per_fwhm);
    EXPECT_DOUBLE_EQ(first.astronomical_alias_power_ratio,
                     second.astronomical_alias_power_ratio);
    EXPECT_NEAR(first.beam_peak_attenuation_fraction, 0.0, 1e-15);
    EXPECT_NEAR(first.beam_broadening_fraction, 0.0, 1e-12);
    EXPECT_DOUBLE_EQ(first.software_group_delay_s, 0.0);
}

TEST(RtcLearnedSamplingMetrics, PersistsUnrankedMetricsOnlySchema) {
    citlali::pipeline::RtcDiagScanArraySummaryData values;
    values.source_power_half_bandwidth_hz = {3.0};
    values.tod_lowpass_to_source_power_half_ratio = {4.0};
    values.beam_major_fwhm_arcsec = {10.0};
    values.beam_minor_fwhm_arcsec = {8.0};
    values.beam_position_angle_rad = {0.0};
    values.limiting_projected_fwhm_arcsec = {8.0};
    values.limiting_speed_arcsec_s = {40.0};
    values.candidate_factors = {1, 2};
    values.fir_coefficients = {1.0};
    values.candidate_status = {0, 0};
    values.candidate_output_sample_rate_hz = {100.0, 50.0};
    values.candidate_output_nyquist_hz = {50.0, 25.0};
    values.candidate_samples_per_fwhm = {20.0, 10.0};
    values.candidate_beam_peak_attenuation_fraction = {0.0, 0.0};
    values.candidate_beam_half_power_fir_attenuation_db = {0.0, 0.0};
    values.candidate_beam_broadening_fraction = {0.0, 0.0};
    values.candidate_astronomical_alias_power_ratio = {0.0, 0.01};
    values.candidate_fir_stopband_rejection_db = {INFINITY, 0.0};
    values.candidate_fir_transition_margin_hz = {40.0, 15.0};
    values.candidate_fir_raw_group_delay_s = {0.0, 0.0};
    values.candidate_software_group_delay_s = {0.0, 0.0};

    const auto path = std::filesystem::temp_directory_path() /
                      "citlali_rtc_learned_sampling_metrics_schema.nc";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        const auto scan = file.addDim("n_scans", 1);
        const auto array = file.addDim("n_arrays", 1);
        citlali::pipeline::add_rtcdiag_scan_array_summary_outputs(
            file, {scan, array}, {1, 1}, values);
    }
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::read);
        const auto notice_var = file.getVar("RTC_SAMPLING_METRICS_NOTICE");
        ASSERT_FALSE(notice_var.isNull());
        char *notice_bytes = nullptr;
        notice_var.getVar(std::vector<std::size_t>{0}, &notice_bytes);
        const std::string notice{notice_bytes};
        nc_free_string(1, &notice_bytes);
        EXPECT_EQ(
            notice,
            "This is a metrics-only diagnostic. No candidate was selected "
            "and no RTC behavior was changed.");
        EXPECT_FALSE(file.getVar("RTC_SAMPLING_CANDIDATE_NOTES").isNull());
        ASSERT_EQ(file.getDim("n_rtc_sampling_candidates").getSize(), 2u);
        std::vector<int> factors(2, -1);
        file.getVar("rtc_sampling_candidate_factor")
            .getVar(factors.data());
        EXPECT_EQ(factors, (std::vector<int>{1, 2}));
        EXPECT_FALSE(
            file.getVar("rtc_sampling_candidate_astronomical_alias_power_ratio")
                .isNull());
        EXPECT_TRUE(file.getVar("rtc_sampling_candidate_rank").isNull());
        EXPECT_TRUE(file.getVar("rtc_sampling_candidate_selected").isNull());
    }
    std::filesystem::remove(path, ec);
}

}  // namespace
