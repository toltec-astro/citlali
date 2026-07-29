#include <citlali/core/pipeline/rawobs_adc_snap.h>

#include <gtest/gtest.h>

#include <netcdf>

#include <array>
#include <filesystem>

namespace {

TEST(rawobs_adc_snap_schema,
     maps_beginning_and_ending_rows_to_named_boundary_columns) {
    const auto path =
        std::filesystem::path(::testing::TempDir()) /
        "citlali_rawobs_adc_snap_schema_test.nc";
    std::filesystem::remove(path);

    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        const auto boundary_dim = file.addDim(
            "adcSnapDim", citlali::pipeline::adc_snap_boundary_count);
        const auto sample_dim = file.addDim("adcSnapDataDim", 4);
        auto value = file.addVar(
            citlali::pipeline::adc_snap_variable_name, netCDF::ncShort,
            {boundary_dim, sample_dim});
        const std::array<short, 8> samples = {
            -2048, -1, 0, 2047,
            10, 20, 30, 40,
        };
        value.putVar(samples.data());
    }

    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::read);
        const auto snapshot = citlali::pipeline::read_adc_snap_matrix(file);
        const auto beginning = static_cast<Eigen::Index>(
            citlali::pipeline::AdcSnapBoundary::beginning);
        const auto ending = static_cast<Eigen::Index>(
            citlali::pipeline::AdcSnapBoundary::ending);

        ASSERT_EQ(snapshot.rows(), 4);
        ASSERT_EQ(snapshot.cols(), 2);
        EXPECT_EQ(snapshot(0, beginning), -2048);
        EXPECT_EQ(snapshot(3, beginning), 2047);
        EXPECT_EQ(snapshot(0, ending), 10);
        EXPECT_EQ(snapshot(3, ending), 40);
    }

    std::filesystem::remove(path);
}

TEST(rawobs_adc_snap_schema, declares_signed_twelve_bit_count_domain) {
    EXPECT_EQ(citlali::pipeline::adc_snap_boundary_count, 2);
    EXPECT_EQ(citlali::pipeline::adc_snap_sample_count, 4096);
    EXPECT_EQ(citlali::pipeline::adc_snap_min_count, -2048);
    EXPECT_EQ(citlali::pipeline::adc_snap_max_count, 2047);
}

}  // namespace
