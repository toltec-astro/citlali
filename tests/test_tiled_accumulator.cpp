#include "citlali/core/mapmaking/tiled_accumulator.h"

#include <gtest/gtest.h>

namespace {

TEST(mapmaking, tiled_accumulator_merges_touched_tiles) {
    std::vector<Eigen::MatrixXd> target;
    target.emplace_back(Eigen::MatrixXd::Zero(5, 7));
    target.emplace_back(Eigen::MatrixXd::Zero(5, 7));

    mapmaking::TiledMapAccumulator acc(2);
    acc.reset(target.size(), 5, 7);
    acc.add(0, 0, 0, 1.5);
    acc.add(0, 1, 1, 2.0);
    acc.add(0, 1, 1, 3.0);
    acc.add(1, 4, 6, -2.0);

    acc.merge_into(target);

    EXPECT_DOUBLE_EQ(target[0](0, 0), 1.5);
    EXPECT_DOUBLE_EQ(target[0](1, 1), 5.0);
    EXPECT_DOUBLE_EQ(target[1](4, 6), -2.0);
    EXPECT_DOUBLE_EQ(target[1].sum(), -2.0);
}

TEST(mapmaking, tiled_accumulator_rejects_invalid_indices) {
    mapmaking::TiledMapAccumulator acc(4);
    acc.reset(1, 3, 3);

    EXPECT_THROW(acc.add(1, 0, 0, 1.0), std::runtime_error);
    EXPECT_THROW(acc.add(0, 3, 0, 1.0), std::runtime_error);
    EXPECT_THROW(acc.add(0, 0, 3, 1.0), std::runtime_error);

    std::vector<Eigen::MatrixXd> wrong_shape;
    wrong_shape.emplace_back(Eigen::MatrixXd::Zero(2, 3));
    EXPECT_THROW(acc.merge_into(wrong_shape), std::runtime_error);
}

}  // namespace
