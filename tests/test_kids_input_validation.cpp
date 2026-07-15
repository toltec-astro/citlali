#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/kids_input_validation.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <limits>
#include <string>

namespace {

TEST(kids_input_validation, accepts_finite_detector_data) {
    Eigen::MatrixXd data(2, 2);
    data << 1.0, 2.0, 3.0, 4.0;

    EXPECT_NO_THROW(citlali::pipeline::require_finite_kids_input(
        data, "test KIDs input"));
}

TEST(kids_input_validation, classifies_nan_as_input_io_failure) {
    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(2, 2);
    data(0, 1) = std::numeric_limits<double>::quiet_NaN();

    try {
        citlali::pipeline::require_finite_kids_input(data, "test KIDs input");
        FAIL() << "expected non-finite KIDs input to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::io);
        EXPECT_NE(std::string{error.what()}.find("NaN"), std::string::npos);
    }
}

TEST(kids_input_validation, classifies_infinity_as_input_io_failure) {
    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(2, 2);
    data(1, 0) = std::numeric_limits<double>::infinity();

    try {
        citlali::pipeline::require_finite_kids_input(data, "test KIDs input");
        FAIL() << "expected non-finite KIDs input to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::io);
        EXPECT_NE(std::string{error.what()}.find("infinite"),
                  std::string::npos);
    }
}

}  // namespace
