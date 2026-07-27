#include <citlali/core/engine/kidsproc.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>

namespace {

constexpr auto fixture_relative_path =
    "data_lmt/toltec/ics/toltec0/"
    "toltec0_018230_111_0000_2024_05_20_13_40_08_timestream.nc";

[[nodiscard]] auto fixture_path() -> std::filesystem::path
{
    return std::filesystem::path{std::getenv("TOLTECA_TEST_DATA_ROOT")} /
           fixture_relative_path;
}

void ensure_citlali_logger()
{
    if (!spdlog::get("citlali_logger")) {
        static_cast<void>(spdlog::stdout_color_mt("citlali_logger"));
    }
}

template <typename Lhs, typename Rhs>
void expect_same_samples(const Eigen::DenseBase<Lhs> &lhs,
                         const Eigen::DenseBase<Rhs> &rhs)
{
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
        for (Eigen::Index col = 0; col < lhs.cols(); ++col) {
            const auto lhs_value = lhs(row, col);
            const auto rhs_value = rhs(row, col);
            if (std::isnan(lhs_value) || std::isnan(rhs_value)) {
                EXPECT_TRUE(std::isnan(lhs_value));
                EXPECT_TRUE(std::isnan(rhs_value));
            } else {
                EXPECT_DOUBLE_EQ(lhs_value, rhs_value);
            }
        }
    }
}

TEST(KidsDataProc, PreservesReaderSliceAndSolverBehavior)
{
    if (std::getenv("TOLTECA_TEST_DATA_ROOT") == nullptr) {
        GTEST_SKIP()
            << "set TOLTECA_TEST_DATA_ROOT to run the real-file test";
    }
    ensure_citlali_logger();
    const auto path = fixture_path();
    ASSERT_TRUE(std::filesystem::is_regular_file(path)) << path;

    const auto item = RawObs::DataItem::from_config(
        tula::config::YamlConfig::from_str(
            fmt::format(
                "meta:\n  interface: toltec0\nfilepath: '{}'\n",
                path.string())));
    auto processor = KidsDataProc::from_config(
        tula::config::YamlConfig::from_str(
            fmt::format(
                "solver:\n"
                "  fitreportdir: '{}'\n"
                "  parallel_policy: seq\n"
                "  extra_output: false\n",
                path.parent_path().string())));
    const tula::container_utils::Slice<int> slice{0, 3, 1};

    const auto adapted = processor.load_data_item(item, slice);
    const auto direct = kids::toltec::read_raw_timestream_slice(
        path, kids::toltec::SampleSlice{0, 3, 1});
    EXPECT_EQ(adapted.meta.pformat(), direct.meta.pformat());
    EXPECT_TRUE(adapted.is.data.isApprox(direct.is.data));
    EXPECT_TRUE(adapted.qs.data.isApprox(direct.qs.data));
    EXPECT_TRUE(
        adapted.wcs.time_axis.data.isApprox(direct.wcs.time_axis.data));
    EXPECT_TRUE(
        adapted.wcs.tone_axis.data.isApprox(direct.wcs.tone_axis.data));
    EXPECT_EQ(
        adapted.wcs.tone_axis.row_labels.labels(),
        direct.wcs.tone_axis.row_labels.labels());

    const auto adapted_result = processor.reduce_data_item(item, slice);
    const auto direct_result = processor.solver()(direct);
    expect_same_samples(
        adapted_result.data_out.xs.data, direct_result.data_out.xs.data);
    expect_same_samples(
        adapted_result.data_out.rs.data, direct_result.data_out.rs.data);
    EXPECT_EQ(
        adapted_result.data_out.meta.pformat(),
        direct_result.data_out.meta.pformat());
}

} // namespace
