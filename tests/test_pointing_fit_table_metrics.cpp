#include <citlali/core/pipeline/pointing_fit_table_metrics.h>

#include <cmath>

#include <gtest/gtest.h>

TEST(pointing_fit_table_metrics, separates_legacy_dynamic_range_and_fit_snr) {
    const auto metrics =
        citlali::pipeline::pointing_fit_table_metrics(120.0, 4.0, 20.0);

    EXPECT_FLOAT_EQ(metrics.legacy_sig2noise, 6.0F);
    EXPECT_FLOAT_EQ(metrics.peak_over_full_map_rms, 6.0F);
    EXPECT_FLOAT_EQ(metrics.fit_sig2noise, 30.0F);
}

TEST(pointing_fit_table_metrics, rejects_nonpositive_denominators) {
    const auto metrics =
        citlali::pipeline::pointing_fit_table_metrics(120.0, 0.0, -1.0);

    EXPECT_TRUE(std::isnan(metrics.legacy_sig2noise));
    EXPECT_TRUE(std::isnan(metrics.peak_over_full_map_rms));
    EXPECT_TRUE(std::isnan(metrics.fit_sig2noise));
}

TEST(pointing_fit_table_metrics, appends_explicit_columns_after_legacy_schema) {
    constexpr Eigen::Index n_params = 6;

    EXPECT_EQ(
        citlali::pipeline::pointing_fit_table_legacy_sig2noise_column(n_params),
        13);
    EXPECT_EQ(
        citlali::pipeline::
            pointing_fit_table_peak_over_full_map_rms_column(n_params),
        14);
    EXPECT_EQ(
        citlali::pipeline::pointing_fit_table_fit_sig2noise_column(n_params),
        15);
    EXPECT_EQ(citlali::pipeline::pointing_fit_table_column_count(n_params), 16);
}
