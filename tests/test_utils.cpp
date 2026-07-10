#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tula/logging.h>
#include <tula/algorithm/ei_iterclip.h>
#include <tula/algorithm/index.h>
#include <tula/formatter/matrix.h>
#include "citlali/core/mapmaking/wiener_filter.h"
#include "citlali/core/utils/utils.h"
#include "citlali/core/timestream/rtc/filter.h"
#include <spdlog/sinks/null_sink.h>

namespace {
using namespace ::testing;

std::shared_ptr<spdlog::logger> ensure_citlali_logger() {
    auto logger = spdlog::get("citlali_logger");
    if (logger == nullptr) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        logger = std::make_shared<spdlog::logger>("citlali_logger", sink);
        spdlog::register_logger(logger);
    }
    return logger;
}

void configure_small_wiener_denominator(mapmaking::WienerFilter &filter) {
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 4;
    filter.n_cols = 4;
    filter.uniform_weight = false;
    filter.denom_limit = -1e300;
    filter.max_loops = 100;

    filter.filter_template.resize(4, 4);
    filter.filter_template <<
        1.0, 0.6, 0.3, 0.1,
        0.6, 0.4, 0.2, 0.1,
        0.3, 0.2, 0.1, 0.05,
        0.1, 0.1, 0.05, 0.02;

    filter.rr.resize(4, 4);
    filter.rr <<
        1.0, 1.1, 0.9, 1.2,
        0.8, 1.3, 1.0, 0.7,
        1.2, 0.9, 1.4, 1.1,
        0.7, 1.0, 1.2, 0.8;

    filter.vvq.resize(4, 4);
    filter.vvq <<
        1.0, 1.4, 0.8, 1.7,
        1.2, 0.9, 1.6, 1.1,
        0.7, 1.5, 1.3, 0.95,
        1.8, 1.05, 0.85, 1.25;
}

struct KernelTemplateMap {
    int n_rows = 5;
    int n_cols = 5;
    Eigen::VectorXd rows_tan_vec;
    Eigen::VectorXd cols_tan_vec;
    std::vector<Eigen::MatrixXd> kernel;
    std::vector<Eigen::MatrixXd> weight;

    KernelTemplateMap() {
        rows_tan_vec.resize(5);
        cols_tan_vec.resize(5);
        rows_tan_vec << -2.0, -1.0, 0.0, 1.0, 2.0;
        cols_tan_vec << -2.0, -1.0, 0.0, 1.0, 2.0;
        kernel.emplace_back(Eigen::MatrixXd::Ones(5, 5));
        weight.emplace_back(Eigen::MatrixXd::Ones(5, 5));
    }
};

struct DummyCalibData {};

TEST(utils, fft2_into_matches_fft2_forward_and_inverse) {
    Eigen::MatrixXcd in(4, 3);
    for (Eigen::Index i = 0; i < in.rows(); ++i) {
        for (Eigen::Index j = 0; j < in.cols(); ++j) {
            in(i, j) = std::complex<double>(0.2 * static_cast<double>(i + 1),
                                            -0.15 * static_cast<double>(j + 2));
        }
    }

    fftw_complex *a = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * in.size()));
    fftw_complex *b = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * in.size()));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    fftw_plan pf = fftw_plan_dft_2d(in.rows(), in.cols(), a, b, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pr = fftw_plan_dft_2d(in.rows(), in.cols(), a, b, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    ASSERT_NE(pr, nullptr);

    Eigen::MatrixXcd forward_expected = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);
    Eigen::MatrixXcd forward_actual;
    engine_utils::fft2_into<engine_utils::forward>(in, forward_actual, pf, a, b);
    EXPECT_LT((forward_actual - forward_expected).norm(), 1e-12);

    Eigen::MatrixXcd inverse_expected = engine_utils::fft2<engine_utils::inverse>(forward_expected, pr, a, b);
    Eigen::MatrixXcd inverse_actual;
    engine_utils::fft2_into<engine_utils::inverse>(forward_expected, inverse_actual, pr, a, b);
    EXPECT_LT((inverse_actual - inverse_expected).norm(), 1e-12);

    fftw_destroy_plan(pf);
    fftw_destroy_plan(pr);
    fftw_free(a);
    fftw_free(b);
}

TEST(wiener_filter, denominator_honors_max_denom_iters) {
    mapmaking::WienerFilter filter;
    configure_small_wiener_denominator(filter);
    filter.denom_rel_tol = 0.0;
    filter.tail_frac_tol = 0.0;
    filter.denom_check_iters = 16;
    filter.max_denom_iters = 2;

    filter.calc_denominator();

    EXPECT_EQ(filter.last_denom_iters, 2);
    EXPECT_EQ(filter.last_denom_stop_reason, "max_denom_iters");
    EXPECT_TRUE(filter.denom.allFinite());
}

TEST(wiener_filter, denominator_rel_tol_can_stop_before_tail_cap) {
    mapmaking::WienerFilter filter;
    configure_small_wiener_denominator(filter);
    filter.denom_rel_tol = 2.0;
    filter.tail_frac_tol = 0.0;
    filter.denom_check_iters = 1;
    filter.max_denom_iters = 0;

    filter.calc_denominator();

    EXPECT_EQ(filter.last_denom_iters, 1);
    EXPECT_EQ(filter.last_denom_stop_reason, "converged");
    EXPECT_GT(filter.last_denom_tail_frac, filter.tail_frac_tol);
}

TEST(wiener_filter, kernel_template_tail_zero_truncates_extrapolated_pixels) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 5;
    filter.n_cols = 5;
    filter.diff_rows = 1.0;
    filter.diff_cols = 1.0;
    filter.kernel_template_tail_mode = "zero";

    KernelTemplateMap mb;
    DummyCalibData calib_data;
    filter.make_kernel_template(mb, 0, calib_data);

    EXPECT_DOUBLE_EQ(filter.filter_template.minCoeff(), 0.0);
    EXPECT_DOUBLE_EQ(filter.filter_template.maxCoeff(), 1.0);
}

TEST(timestream_filter, notch_settle_samples_are_positive_for_narrow_notches) {
    auto samples = timestream::Filter::notch_settle_samples_for_width(
        122.0703125, 0.25, 0.01);
    EXPECT_GT(samples, 0);
}

TEST(timestream_filter, notch_settle_samples_increase_for_narrower_widths) {
    auto narrow = timestream::Filter::notch_settle_samples_for_width(
        122.0703125, 0.25, 0.01);
    auto broad = timestream::Filter::notch_settle_samples_for_width(
        122.0703125, 1.0, 0.01);
    EXPECT_GT(narrow, broad);
}

TEST(timestream_filter, zero_phase_notch_preserves_constant_timestream) {
    timestream::Filter filter;
    filter.notch_zero_phase = true;
    filter.w0s = {11.03, 15.23, 13.23, 2.71};
    filter.qs = {
        11.03 / 0.25,
        15.23 / 0.30,
        13.23 / 0.25,
        2.71 / 0.20,
    };
    filter.make_notch_filter(122.0703125);

    Eigen::MatrixXd data = Eigen::MatrixXd::Constant(512, 3, 1.0);
    filter.iir(data);

    EXPECT_LT((data.array() - 1.0).abs().maxCoeff(), 1e-10);
}

TEST(timestream_filter, zero_phase_notch_preserves_constant_edges) {
    timestream::Filter filter;
    filter.notch_zero_phase = true;
    filter.w0s = {2.71};
    filter.qs = {2.71 / 0.20};
    filter.make_notch_filter(122.0703125);

    Eigen::MatrixXd data = Eigen::MatrixXd::Constant(64, 2, 3.5);
    filter.iir(data);

    EXPECT_NEAR(data(0, 0), 3.5, 1e-10);
    EXPECT_NEAR(data(data.rows() - 1, 0), 3.5, 1e-10);
    EXPECT_LT((data.array() - 3.5).abs().maxCoeff(), 1e-10);
}

class UtilsTest : public Test {
public:
    UtilsTest() : mat(Eigen::MatrixXd::Constant(10, 2, std::nan(""))) {
        const Eigen::VectorXd values =
            Eigen::VectorXd::LinSpaced(10, 1.0, 0.0);
        mat.block(5, 0, 5, 2) =
            Eigen::Map<const Eigen::MatrixXd>(values.data(), 5, 2);
    }

    auto block() { return mat.block(5, 0, 5, 2); }

    Eigen::MatrixXd mat;
};

TEST_F(UtilsTest, generate_chunks) {
    const auto chunks = tula::alg::indexchunks(0, 11, 2, 2);

    EXPECT_EQ(chunks,
              (std::vector<std::pair<int, int>>{{0, 7}, {5, 11}}));
}

TEST_F(UtilsTest, eigen_to_stdvec_iterates_in_storage_order) {
    auto blk = block();
    const auto values = tula::eigen_utils::to_stdvec(blk);

    EXPECT_EQ(std::distance(values.begin(), values.end()), blk.size());
    EXPECT_DOUBLE_EQ(*values.begin(), 1.0);
    EXPECT_DOUBLE_EQ(*(values.end() - 1), 0.0);
}

TEST_F(UtilsTest, eigen_to_stdvec_supports_standard_algorithms) {
    auto blk = block();
    auto values = tula::eigen_utils::to_stdvec(blk);

    EXPECT_TRUE(
        std::is_sorted(values.begin(), values.end(), std::greater<>()));
    std::sort(values.begin(), values.end());
    EXPECT_TRUE(std::is_sorted(values.begin(), values.end()));
}

TEST_F(UtilsTest, vector) {
    auto blk = block();
    const auto column_major = tula::eigen_utils::to_stdvec(blk);
    const auto row_major =
        tula::eigen_utils::to_stdvec(blk, Eigen::RowMajor);

    ASSERT_EQ(column_major.size(), 10U);
    ASSERT_EQ(row_major.size(), 10U);
    EXPECT_DOUBLE_EQ(column_major.front(), 1.0);
    EXPECT_DOUBLE_EQ(column_major.back(), 0.0);
    EXPECT_DOUBLE_EQ(row_major[0], 1.0);
    EXPECT_DOUBLE_EQ(row_major[1], 4.0 / 9.0);
}

TEST_F(UtilsTest, meanstd) {
    auto blk = block();
    auto [m, s] = tula::alg::meanstd(blk);
    std::vector<double> expected{0.5, 0.31914236925211265};
    EXPECT_THAT(expected, ElementsAre(DoubleEq(m), DoubleEq(s)));
    std::tie(std::ignore, s) = tula::alg::meanstd(blk, 1);
    EXPECT_DOUBLE_EQ(0.33640559489972127, s);
}

TEST_F(UtilsTest, medianmad) {
    auto blk = block();
    auto [m, s] = tula::alg::medmad(blk);
    std::vector<double> expected{0.5, 0.27777777777777773};
    EXPECT_THAT(expected, ElementsAre(DoubleEq(m), DoubleEq(s)));
}

TEST_F(UtilsTest, indexofthresh) {
    auto blk = block();
    const auto values = tula::eigen_utils::to_stdvec(blk);
    const Eigen::VectorXd data = Eigen::Map<const Eigen::VectorXd>(
        values.data(), static_cast<Eigen::Index>(values.size()));
    auto func = tula::alg::iterclip(
            [](auto&& m) {
        return tula::alg::meanstd(m);
    },
            [](auto v, auto m, auto s) {
        return std::abs(v - m) > s;
    }
            );
    auto [indices, converged, center, stddev] = func(data);

    EXPECT_TRUE(converged);
    EXPECT_FALSE(indices.empty());
    for (const auto index : indices) {
        EXPECT_GT(std::abs(data.coeff(index) - center), stddev);
    }
}
}  // namespace
