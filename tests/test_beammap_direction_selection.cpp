#include <citlali/core/pipeline/beammap_direction_selection.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

struct DirectionFixture {
    Eigen::MatrixXi scan_indices{2, 4};
    std::map<std::string, Eigen::VectorXd> tel_data;

    DirectionFixture() {
        constexpr Eigen::Index n = 28;
        tel_data["TelTime"] = Eigen::VectorXd::LinSpaced(n, 100.0, 127.0);
        tel_data["az_phys"] = Eigen::VectorXd::Zero(n);
        tel_data["alt_phys"] = Eigen::VectorXd::Zero(n);

        // Science windows deliberately omit the turnaround samples between
        // legs. Direction is along a 45-degree fast-scan axis.
        scan_indices << 1, 8, 15, 22,
                        5, 12, 19, 26;
        const double scale = 1.0e-5;
        for (Eigen::Index scan = 0; scan < scan_indices.cols(); ++scan) {
            const bool right = scan % 2 == 0;
            const auto start = scan_indices(0, scan);
            const auto stop = scan_indices(1, scan);
            for (Eigen::Index index = start; index <= stop; ++index) {
                const double step = static_cast<double>(index - start);
                const double value = (right ? step : -step) * scale;
                tel_data["az_phys"](index) = value;
                tel_data["alt_phys"](index) = value;
            }
        }
        // Large opposite-sense motions in turnaround samples must not affect
        // direction because they are outside every science window.
        for (const Eigen::Index index : {0, 6, 7, 13, 14, 20, 21, 27}) {
            tel_data["az_phys"](index) = 0.25 * (index % 2 ? -1.0 : 1.0);
            tel_data["alt_phys"](index) = -0.25 * (index % 2 ? -1.0 : 1.0);
        }
    }
};

TEST(BeammapDirectionSelection, SelectsRequestedScanDirectionOnRotatedAxis) {
    DirectionFixture fixture;
    constexpr double angle = 0.78539816339744830962;

    const auto left =
        citlali::pipeline::make_beammap_direction_selection_plan(
            citlali::config::BeammapDirectionMode::left,
            fixture.scan_indices, fixture.tel_data, "az", angle);
    const auto right =
        citlali::pipeline::make_beammap_direction_selection_plan(
            citlali::config::BeammapDirectionMode::right,
            fixture.scan_indices, fixture.tel_data, "az", angle);
    ASSERT_EQ(left.scans.size(), 4U);
    EXPECT_EQ(left.left_count, 2);
    EXPECT_EQ(left.right_count, 2);
    EXPECT_EQ(left.selected_count, 2);
    EXPECT_FALSE(left.scans[0].selected);
    EXPECT_TRUE(left.scans[1].selected);
    EXPECT_TRUE(right.scans[0].selected);
    EXPECT_FALSE(right.scans[1].selected);
    EXPECT_GT(left.scans[0].signed_fast_axis_rate_rad_per_sec, 0.0);
    EXPECT_LT(left.scans[1].signed_fast_axis_rate_rad_per_sec, 0.0);
    EXPECT_EQ(left.scans[0].science_start, 1);
    EXPECT_EQ(left.scans[0].science_stop_exclusive, 6);
}

TEST(BeammapDirectionSelection, AllClassifiesOnceAndRoutesThreeBuffers) {
    DirectionFixture fixture;
    const auto plan =
        citlali::pipeline::make_beammap_direction_selection_plan(
            citlali::config::BeammapDirectionMode::all,
            fixture.scan_indices, fixture.tel_data, "az",
            0.78539816339744830962);
    ASSERT_EQ(plan.scans.size(), 4U);
    EXPECT_EQ(plan.selected_count, 4);
    Eigen::Index standard_count = 0;
    Eigen::Index left_count = 0;
    Eigen::Index right_count = 0;
    for (const auto &scan : plan.scans) {
        const auto buffers =
            citlali::pipeline::beammap_direction_buffer_selection(
                plan.mode, scan.direction);
        standard_count += buffers.standard ? 1 : 0;
        left_count += buffers.left ? 1 : 0;
        right_count += buffers.right ? 1 : 0;
        EXPECT_NE(buffers.left, buffers.right);
    }
    EXPECT_EQ(standard_count, 4);
    EXPECT_EQ(left_count, 2);
    EXPECT_EQ(right_count, 2);
    EXPECT_THROW(
        citlali::pipeline::beammap_direction_product_filename(
            "beammap", citlali::config::BeammapDirectionMode::all),
        std::logic_error);
}

TEST(BeammapDirectionSelection, DoesNotMutateCommonTelescopeInputs) {
    DirectionFixture fixture;
    const auto before_time = fixture.tel_data.at("TelTime");
    const auto before_x = fixture.tel_data.at("az_phys");
    const auto before_y = fixture.tel_data.at("alt_phys");

    (void)citlali::pipeline::make_beammap_direction_selection_plan(
        citlali::config::BeammapDirectionMode::left,
        fixture.scan_indices, fixture.tel_data, "az", 0.78539816339744830962);

    EXPECT_EQ(fixture.tel_data.at("TelTime"), before_time);
    EXPECT_EQ(fixture.tel_data.at("az_phys"), before_x);
    EXPECT_EQ(fixture.tel_data.at("alt_phys"), before_y);
}

TEST(BeammapDirectionSelection, StandardBypassesDirectionalClassification) {
    Eigen::MatrixXi invalid_scan_indices;
    std::map<std::string, Eigen::VectorXd> empty_telescope;

    const auto plan =
        citlali::pipeline::make_beammap_direction_selection_plan(
            citlali::config::BeammapDirectionMode::standard,
            invalid_scan_indices, empty_telescope, "unsupported", 0.0);

    EXPECT_EQ(plan.mode,
              citlali::config::BeammapDirectionMode::standard);
    EXPECT_TRUE(plan.scans.empty());
    EXPECT_EQ(plan.selected_count, 0);
}

TEST(BeammapDirectionSelection, FailsClosedForAmbiguousScanLeg) {
    DirectionFixture fixture;
    fixture.tel_data["az_phys"].segment(8, 5).setConstant(0.0);
    fixture.tel_data["alt_phys"].segment(8, 5).setConstant(0.0);

    EXPECT_THROW(
        citlali::pipeline::make_beammap_direction_selection_plan(
            citlali::config::BeammapDirectionMode::left,
            fixture.scan_indices, fixture.tel_data, "az",
            0.78539816339744830962),
        std::runtime_error);
}

TEST(BeammapDirectionSelection, FailsClosedForNonIncreasingTelescopeTime) {
    DirectionFixture fixture;
    fixture.tel_data["TelTime"](3) = fixture.tel_data["TelTime"](2);

    EXPECT_THROW(
        citlali::pipeline::make_beammap_direction_selection_plan(
            citlali::config::BeammapDirectionMode::right,
            fixture.scan_indices, fixture.tel_data, "az",
            0.78539816339744830962),
        std::runtime_error);
}

TEST(BeammapDirectionSelection, PreservesStandardNamesAndTagsDiagnostics) {
    const std::string base = "toltec_commissioning_a1100_beammap_150819_citlali";
    EXPECT_EQ(citlali::pipeline::beammap_direction_product_filename(
                  base, citlali::config::BeammapDirectionMode::standard),
              base);
    EXPECT_EQ(citlali::pipeline::beammap_direction_product_filename(
                  base, citlali::config::BeammapDirectionMode::left),
              base + "_left");
    EXPECT_EQ(citlali::pipeline::beammap_direction_product_filename(
                  base, citlali::config::BeammapDirectionMode::right),
              base + "_right");
}

TEST(BeammapDirectionSelection, RoutesSingleDirectionIntoPrimaryBufferOnly) {
    using Mode = citlali::config::BeammapDirectionMode;
    using Direction = citlali::pipeline::BeammapScanDirection;
    const auto left =
        citlali::pipeline::beammap_direction_buffer_selection(
            Mode::left, Direction::left);
    const auto rejected =
        citlali::pipeline::beammap_direction_buffer_selection(
            Mode::left, Direction::right);
    EXPECT_TRUE(left.standard);
    EXPECT_FALSE(left.left);
    EXPECT_FALSE(left.right);
    EXPECT_FALSE(rejected.standard);
}

TEST(BeammapDirectionSelection, ResolvesAllOutputIdentityFailClosed) {
    using Mode = citlali::config::BeammapDirectionMode;
    EXPECT_EQ(citlali::pipeline::beammap_direction_realized_product_mode(
                  Mode::all, "standard"),
              Mode::standard);
    EXPECT_EQ(citlali::pipeline::beammap_direction_realized_product_mode(
                  Mode::all, "left"),
              Mode::left);
    EXPECT_EQ(citlali::pipeline::beammap_direction_realized_product_mode(
                  Mode::all, "right"),
              Mode::right);
    EXPECT_THROW(
        citlali::pipeline::beammap_direction_realized_product_mode(
            Mode::all, "all"),
        std::logic_error);
    EXPECT_THROW(
        citlali::pipeline::beammap_direction_realized_product_mode(
            Mode::all, ""),
        std::logic_error);
}

TEST(BeammapDirectionSelection, WritesDeterministicScanRegistry) {
    DirectionFixture fixture;
    const auto plan =
        citlali::pipeline::make_beammap_direction_selection_plan(
            citlali::config::BeammapDirectionMode::left,
            fixture.scan_indices, fixture.tel_data, "az",
            0.78539816339744830962);
    const auto root = std::filesystem::temp_directory_path() /
                      "citlali_beammap_direction_selection_test";
    std::filesystem::create_directories(root);
    const auto first = root / "first.csv";
    const auto second = root / "second.csv";

    citlali::pipeline::write_beammap_direction_scan_registry(first, plan);
    citlali::pipeline::write_beammap_direction_scan_registry(second, plan);
    std::ifstream first_stream(first);
    std::ifstream second_stream(second);
    const std::string first_text{
        std::istreambuf_iterator<char>(first_stream), {}};
    const std::string second_text{
        std::istreambuf_iterator<char>(second_stream), {}};

    EXPECT_EQ(first_text, second_text);
    EXPECT_NE(first_text.find("direction,selected,mode"), std::string::npos);
    EXPECT_NE(first_text.find(",right,false,left"), std::string::npos);
    EXPECT_NE(first_text.find(",left,true,left"), std::string::npos);
    std::filesystem::remove_all(root);
}

TEST(BeammapDirectionSelection, AcceptsOnlyTheFourNamedModes) {
    EXPECT_EQ(citlali::config::parse_beammap_direction_mode("standard"),
              citlali::config::BeammapDirectionMode::standard);
    EXPECT_EQ(citlali::config::parse_beammap_direction_mode("left"),
              citlali::config::BeammapDirectionMode::left);
    EXPECT_EQ(citlali::config::parse_beammap_direction_mode("right"),
              citlali::config::BeammapDirectionMode::right);
    EXPECT_EQ(citlali::config::parse_beammap_direction_mode("all"),
              citlali::config::BeammapDirectionMode::all);
    EXPECT_FALSE(citlali::config::parse_beammap_direction_mode("both"));
}

}  // namespace
