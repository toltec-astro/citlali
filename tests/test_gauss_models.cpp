#include <gtest/gtest.h>

#include <citlali/core/utils/gauss_models.h>

#include <Eigen/Core>

TEST(GaussModels, GaussianOneDimensionalValues)
{
    const engine_utils::Gaussian1D gaussian{2.0, 0.0, 1.0};
    Eigen::VectorXd x(3);
    x << -1.0, 0.0, 1.0;

    const auto values = gaussian(x);

    ASSERT_EQ(values.size(), 3);
    EXPECT_NEAR(values(0), 2.0 * std::exp(-0.5), 1e-12);
    EXPECT_DOUBLE_EQ(values(1), 2.0);
    EXPECT_NEAR(values(2), 2.0 * std::exp(-0.5), 1e-12);
}

TEST(GaussModels, SymmetricGaussianTwoDimensionalFlattenedMesh)
{
    const engine_utils::SymmetricGaussian2D gaussian{3.0, 0.0, 0.0, 1.0};
    Eigen::VectorXd axis(3);
    axis << -1.0, 0.0, 1.0;

    const auto values = gaussian(axis, axis);

    ASSERT_EQ(values.rows(), 9);
    ASSERT_EQ(values.cols(), 1);
    EXPECT_DOUBLE_EQ(values(4), 3.0);
    EXPECT_NEAR(values(0), 3.0 * std::exp(-1.0), 1e-12);
    EXPECT_NEAR(values(8), 3.0 * std::exp(-1.0), 1e-12);
}
