#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "utils/algorithm/ei_linspaced.h"
#include "utils/eigeniter.h"
#include "utils/formatter/eigeniter.h"
#include "utils/formatter/matrix.h"
#include "utils/logging.h"

namespace {
namespace eii = ::eigeniter;

struct eigeniter : public ::testing::Test {

    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;
    using RowVectorXd = Eigen::RowVectorXd;

    static constexpr auto vec_6 = []() {
        VectorXd v{6};
        alg::fill_linspaced(v);
        return v;
    };

    static constexpr auto vec_6r = []() {
        Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> v{6};
        alg::fill_linspaced(v);
        return v;
    };

    static constexpr auto rvec_6 = []() {
        RowVectorXd v{6};
        alg::fill_linspaced(v);
        return v;
    };

    static constexpr auto mat_32 = []() {
        MatrixXd m{3, 2};
        alg::fill_linspaced(m);
        return m;
    };

    static constexpr auto mat_32r = []() {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            m{3, 2};
        alg::fill_linspaced(m);
        return m;
    };

    static constexpr auto blk_32 = []() {
        MatrixXd m{5, 4};
        m.setConstant(100.);
        auto b = m.block(1, 1, 2, 3);
        alg::fill_linspaced(b);
        return std::make_tuple(std::move(m), std::move(b));
    };

    static constexpr auto blk_32r = []() {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            m{5, 4};
        m.setConstant(100.);
        auto b = m.block(0, 1, 2, 3);
        alg::fill_linspaced(b);
        return std::make_tuple(std::move(m), std::move(b));
    };

    template <typename It> void print_range(It begin, It end) {
        for (auto it = begin; it != end; ++it) {
            SPDLOG_TRACE("it={:l} *it={}", it, *it);
        }
    }
};

TEST_F(eigeniter, vec_6) {
    auto m = vec_6();
    SPDLOG_TRACE("m{}", m);
    auto [begin, end] = eii::iters(m);
    print_range(begin, end);
}

TEST_F(eigeniter, vec_6r) {
    auto m = vec_6r();
    SPDLOG_TRACE("m{}", m);
    auto [begin, end] = eii::iters(m);
    print_range(begin, end);
}

TEST_F(eigeniter, rvec_6) {
    auto m = rvec_6();
    SPDLOG_TRACE("m{}", m);
    auto [begin, end] = eii::iters(m);
    print_range(begin, end);
}

TEST_F(eigeniter, mat_32) {
    auto m = mat_32();
    SPDLOG_TRACE("m{}", m);
    auto [begin, end] = eii::iters(m);
    print_range(begin, end);
}

TEST_F(eigeniter, mat_32r) {
    auto m = mat_32r();
    SPDLOG_TRACE("m{}", m);
    auto [begin, end] = eii::iters(m);
    print_range(begin, end);
}

TEST_F(eigeniter, blk_32) {
    auto [m, b] = blk_32();
    SPDLOG_TRACE("m{} b{}", m, b);
    auto [begin, end] = eii::iters(b);
    print_range(begin, end);
}

TEST_F(eigeniter, blk_32r) {
    auto [m, b] = blk_32r();
    SPDLOG_TRACE("mr{} br{}", m, b);
    auto [begin, end] = eii::iters(b);
    print_range(begin, end);
}

TEST_F(eigeniter, mat_32_col1) {
    auto m = mat_32();
    SPDLOG_TRACE("m{}", m);
    auto [begin, end] = eii::iters(m.col(1));
    print_range(begin, end);
}

TEST_F(eigeniter, mat_32_row1) {
    auto m = mat_32();
    SPDLOG_TRACE("m{}", m);
    auto [begin, end] = eii::iters(m.row(1));
    print_range(begin, end);
}

} // namespace
