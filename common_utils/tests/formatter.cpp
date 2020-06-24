#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "test_enums.h"
#include "utils/formatter/container.h"
#include "utils/formatter/eigeniter.h"
#include "utils/formatter/enum.h"
#include "utils/formatter/matrix.h"
#include "utils/formatter/ptr.h"
#include "utils/logging.h"

namespace {

using namespace test_enums;

template <typename... Args> std::string fmtlog(Args &&... args) {
    auto s = fmt::format(args...);
    SPDLOG_TRACE(s);
    return s;
};

TEST(formatter, meta_enum_type) {
    using meta = Type_meta;
    EXPECT_NO_THROW(fmtlog("{}: members{}", meta::name, meta::members));
    EXPECT_NO_THROW(fmtlog("non existing member {}", meta::name,
                           meta::to_name(static_cast<Type>(-1))));
    EXPECT_NO_THROW(fmtlog("TypeA: {:d}", meta::from_name("TypeA")));
    EXPECT_NO_THROW(fmtlog("TypeA: {:s}", meta::from_name("TypeA")));
    EXPECT_NO_THROW(fmtlog("TypeA: {:l}", meta::from_name("TypeA")));
    EXPECT_NO_THROW(fmtlog("TypeA: {:s}", Type::TypeA));
    EXPECT_NO_THROW(fmtlog("abc: {}", meta::from_name("abc")));
}

TEST(formatter, meta_enum_flag) {
    using meta = Flag_meta;
    EXPECT_NO_THROW(fmtlog("{}: members{}", meta::name, meta::members));
    EXPECT_NO_THROW(fmtlog("non existing member {}", meta::name,
                           meta::to_name(static_cast<Flag>(-1))));
    EXPECT_NO_THROW(fmtlog("FlagA: {:d}", meta::from_name("FlagA")));
    EXPECT_NO_THROW(fmtlog("FlagA: {:s}", meta::from_name("FlagA")));
    EXPECT_NO_THROW(fmtlog("FlagA: {:l}", meta::from_name("FlagA")));
    EXPECT_NO_THROW(fmtlog("FlagA: {:s}", Flag::FlagA));

    EXPECT_NO_THROW(fmtlog("FlagC: {:d}", meta::from_name("FlagC")));
    EXPECT_NO_THROW(fmtlog("FlagC: {:s}", meta::from_name("FlagC")));
    EXPECT_NO_THROW(fmtlog("FlagC: {:l}", meta::from_name("FlagC")));
    EXPECT_NO_THROW(fmtlog("FlagC: {:s}", Flag::FlagA));

    EXPECT_NO_THROW(fmtlog("FlagD: {:d}", meta::from_name("FlagD")));
    EXPECT_NO_THROW(fmtlog("FlagD: {:s}", meta::from_name("FlagD")));
    EXPECT_NO_THROW(fmtlog("FlagD: {:l}", meta::from_name("FlagD")));
    EXPECT_NO_THROW(fmtlog("FlagD: {:s}", Flag::FlagA));
    EXPECT_NO_THROW(fmtlog("abc: {}", meta::from_name("abc")));
}

TEST(formatter, bitmask_bit) {
    auto bm = bitmask::bitmask<Bit>{};
    EXPECT_NO_THROW(fmtlog("BitA: {:d}", bm | Bit::BitA));
    EXPECT_NO_THROW(fmtlog("BitA: {:s}", bm | Bit::BitA));
    EXPECT_NO_THROW(fmtlog("BitA: {:l}", bm | Bit::BitA));
    EXPECT_NO_THROW(fmtlog("BitA: {:s}", bm | Bit::BitA));

    EXPECT_NO_THROW(fmtlog("BitAC: {:d}", Bit::BitC | Bit::BitA));
    EXPECT_NO_THROW(fmtlog("BitAC: {:s}", Bit::BitC | Bit::BitA));
    EXPECT_NO_THROW(fmtlog("BitAC: {:l}", Bit::BitC | Bit::BitA));
    EXPECT_NO_THROW(fmtlog("BitAC: {:s}", Bit::BitC | Bit::BitA));
}

TEST(formatter, variant) {
    std::variant<bool, int, double, const char *, std::string> v;
    using namespace std::literals;
    EXPECT_EQ(fmtlog("v={}", v = true), "v=true (bool)");
    EXPECT_EQ(fmtlog("v={}", v = -1), "v=-1 (int)");
    EXPECT_EQ(fmtlog("v={}", v = 2.), "v=2.0 (doub)");
    EXPECT_EQ(fmtlog("v={}", v = "v"), "v=\"v\" (str)");
    EXPECT_EQ(fmtlog("v={}", v = "v"s), "v=\"v\" (str)");
}

TEST(formatter, matrix) {
    Eigen::MatrixXd m{5, 10};
    m.setConstant(std::nan(""));
    Eigen::VectorXd::Map(m.data(), m.size())
        .setLinSpaced(m.size(), 0, m.size() - 1);
    EXPECT_NO_THROW(fmtlog("default m{}", m));
    EXPECT_NO_THROW(fmtlog("m{:r1c5}", m));
    EXPECT_NO_THROW(fmtlog("m{:r1c}", m));
    EXPECT_NO_THROW(fmtlog("m{:rc1}", m));

    auto c = m.col(0);
    EXPECT_NO_THROW(fmtlog("default c{}", c));
    EXPECT_NO_THROW(fmtlog("c{:}", c));
    EXPECT_NO_THROW(fmtlog("c{:rc}", c));
    EXPECT_NO_THROW(fmtlog("c{:s}", c));
    EXPECT_NO_THROW(fmtlog("c{:s3}", c));

    std::vector<double> v = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_NO_THROW(fmtlog("default v{:s4}", v));
    EXPECT_NO_THROW(fmtlog("v{:}", v));
    EXPECT_NO_THROW(fmtlog("v{:rc}", v));
    EXPECT_NO_THROW(fmtlog("v{:s}", v));
    EXPECT_NO_THROW(fmtlog("v{:s3}", v));
}

TEST(formatter, ptr) {
    int a = 1;
    EXPECT_NO_THROW(fmtlog("a={}", a));
    EXPECT_NO_THROW(fmtlog("*a@{}", fmt::ptr(&a)));
    EXPECT_NO_THROW(fmtlog("*a@{:x}", fmt_utils::ptr(&a)));
    EXPECT_NO_THROW(fmtlog("*a@{:y}", fmt_utils::ptr(&a)));
    EXPECT_NO_THROW(fmtlog("*a@{:z}", fmt_utils::ptr(&a)));
}

TEST(formatter, eigeniter) {
    Eigen::MatrixXd m(5, 2);
    Eigen::Map<Eigen::VectorXd>(m.data(), m.size()).setLinSpaced(10, 0, 9);
    auto [begin, end] = eigeniter::iters(m);
    EXPECT_NO_THROW(fmtlog("iters begin={:s} end={:s}", begin, end));
    EXPECT_NO_THROW(fmtlog("begin={:l}", begin));
    EXPECT_NO_THROW(fmtlog("end={:l}", end));
    for (auto it = begin; it != end; ++it) {
        EXPECT_NO_THROW(fmtlog("it={:l} *it={}", it, *it));
    }
    EXPECT_EQ(fmt::format("{}", end), fmt::format("{:l}", end));
}

} // namespace