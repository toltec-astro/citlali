#include "utils/utils.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "utils/logging.h"
#include "utils/algorithm/ei_polyfit.h"
#include "utils/formatter/matrix.h"

namespace {
using namespace ::testing;

TEST(algorithm, polyfit_1) {
    using namespace Eigen;
    VectorXd xdata;
    xdata.setLinSpaced(10, 0, 9);
    VectorXd ydata = xdata.array() * 2. - 3;

    auto [p, r] = alg::polyfit(xdata, ydata, 1);
    SPDLOG_TRACE("x{} y{} p{} r{}", xdata, ydata, p, r);
}

TEST(algorithm, polyfit_2) {
    using namespace Eigen;
    VectorXd xdata;
    xdata.setLinSpaced(10, 0, 9);
    VectorXd ydata = xdata.array().square() + xdata.array() * 2. - 3;
    xdata.array() *= 1e7;
    auto [p, r] = alg::polyfit(xdata, ydata, 2);
    SPDLOG_TRACE("x{} y{} p{} r{}", xdata, ydata, p, r);
}

/*
class UtilsTest : public Test {
public:
    using blk_in_mat_t = std::tuple<Eigen::MatrixXd, Eigen::Block<Eigen::MatrixXd>>;
    static blk_in_mat_t blk_in_mat(const double* blk_data, const std::vector<int>& mat_shape, const std::vector<int>& blk_shape) {
        using namespace Eigen;
        MatrixXd mat(mat_shape[0], mat_shape[1]);
        mat.setConstant(std::nan(""));
        auto blk = mat.block(blk_shape[0], blk_shape[1], blk_shape[2], blk_shape[3]);
        blk = Map<const MatrixXd>(blk_data, blk_shape[2], blk_shape[3]);
        SPDLOG_TRACE("mat{}", logging::pprint(mat));
        SPDLOG_TRACE("blk inner_stride={} outer_stride={}", blk.innerStride(), blk.outerStride());
        return std::make_tuple(std::move(mat), std::move(blk));
    }
    blk_in_mat_t blk_in_mat1 = blk_in_mat(Eigen::VectorXd::LinSpaced(10, 1., 0.).eval().data(),
    {10, 2}, {5, 0, 5, 2});

};

TEST_F(UtilsTest, generate_chunks) {
    auto size = 11;
    auto chunks = utils::generate_chunks(0, size, 2, 2);
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        SPDLOG_TRACE("chunk #{} = {}, {}", i, chunks[i].first, chunks[i].second);
    }
}

TEST_F(UtilsTest, eigeniter) {
    auto [mat, blk] = this->blk_in_mat1;
    auto [begin, end] = eigeniter::iters(blk);
    SPDLOG_TRACE("begin: {}", begin);
    SPDLOG_TRACE("end: {}", end);
    for (auto it = begin; it != end; ++it) {
        SPDLOG_TRACE("{} *it={}", it, *it);
    }
    SUCCEED();
}

TEST_F(UtilsTest, eigeniter_apply) {
    auto [mat, blk] = this->blk_in_mat1;
    auto [begin, end] = eigeniter::iters(blk);
    auto sorted = eigeniter::apply(blk, [](auto && begin, auto&& end) {
        return std::is_sorted(begin, end, std::greater<>());
    });
    EXPECT_EQ(true, sorted);
    // sort
    eigeniter::apply(blk, [](auto &&begin, auto&& end) {
       std::sort(begin, end);
    });
    SPDLOG_TRACE("after sort mat{}", logging::pprint(mat));
    SUCCEED();
}

TEST_F(UtilsTest, vector) {
    auto [mat, blk] = this->blk_in_mat1;
    auto v = utils::vector(blk);
    SPDLOG_TRACE("blk{} vector={}", logging::pprint(blk), logging::pprint(v.data(), v.size()));
    v = utils::vector(blk, Eigen::RowMajor);
    SPDLOG_TRACE("blk{} vector(rmaj)={}", logging::pprint(blk), logging::pprint(v.data(), v.size()));
    SUCCEED();
}

TEST_F(UtilsTest, meanstd) {
    auto [mat, blk] = this->blk_in_mat1;
    auto [m, s] = utils::meanstd(blk);
    SPDLOG_TRACE("blk{} mean={} stddev={}", logging::pprint(blk), m, s);
    std::vector<double> expected{0.5, 0.31914236925211265};
    EXPECT_THAT(expected, ElementsAre(DoubleEq(m), DoubleEq(s)));
    // ddof
    auto ddof = 1;
    std::tie(std::ignore, s) = utils::meanstd(blk, ddof);
    SPDLOG_TRACE("ddof={} stddev={}", ddof, s);
    EXPECT_DOUBLE_EQ(0.33640559489972127, s);
}

TEST_F(UtilsTest, medianmad) {
    auto [mat, blk] = this->blk_in_mat1;
    auto [m, s] = utils::medianmad(blk);
    SPDLOG_TRACE("blk{} median={} mad={}", logging::pprint(blk), m, s);
    std::vector<double> expected{0.5, 0.27777777777777773};
    EXPECT_THAT(expected, ElementsAre(DoubleEq(m), DoubleEq(s)));
}

TEST_F(UtilsTest, indexofthresh) {
    auto [mat, blk] = this->blk_in_mat1;
    auto func = utils::iterclip(
            [](auto&& m) {
        return utils::meanstd(m);
    },
            [](auto v, auto m, auto s) {
        auto  select = std::abs(v - m) > s;
        SPDLOG_TRACE("{} v={} m={} s={}", select?"use ":"skip", v, m, s);
        return select;
    }
            );
    auto [ret, m, s, c] = func(blk);
    SPDLOG_TRACE("blk{} mean={} stddev={}", logging::pprint(blk), m, s);
    SPDLOG_TRACE("selected snr>1: {}", logging::pprint(ret.data(), ret.size()));
}
*/

}  // namespace