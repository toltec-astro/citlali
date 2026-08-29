#include <citlali/core/pipeline/timestream_coincidence_cohort.h>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {

namespace pipeline = citlali::pipeline;

TEST(sci_align_partial_cohort,
     finite_placeholder_rejects_nonfinite_private_values) {
    EXPECT_DOUBLE_EQ(pipeline::FinitePcaPlaceholder::checked(-17.0).value(),
                     -17.0);
    EXPECT_THROW(
        pipeline::FinitePcaPlaceholder::checked(
            std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::FinitePcaPlaceholder::checked(
            std::numeric_limits<double>::infinity()),
        std::invalid_argument);
}

TEST(sci_align_partial_cohort,
     ordinary_optional_modes_remain_compatible_with_exclusions) {
    pipeline::PcaCompatibilityInputs ordinary;
    EXPECT_TRUE(
        pipeline::classify_pca_compatibility(true, ordinary).compatible());
    ordinary.marchenko_pastur_active_for_operation = true;
    EXPECT_TRUE(
        pipeline::classify_pca_compatibility(true, ordinary).compatible());
}

TEST(sci_align_partial_cohort,
     hazardous_optional_modes_fail_closed_only_when_exclusions_exist) {
    pipeline::PcaCompatibilityInputs request;
    request.null_model_active_for_operation = true;
    request.adaptive_selector_active_for_operation = true;
    request.marchenko_pastur_active_for_operation = true;
    request.marchenko_pastur_band_requested = true;
    const auto excluded =
        pipeline::classify_pca_compatibility(true, request);
    EXPECT_FALSE(excluded.compatible());
    EXPECT_THROW(pipeline::require_pca_compatibility(excluded),
                 std::logic_error);
    EXPECT_NO_THROW(pipeline::require_pca_compatibility(
        pipeline::classify_pca_compatibility(false, request)));
}

TEST(sci_align_partial_cohort,
     native_identity_and_operation_contracts_remain_fail_closed) {
    const pipeline::NativeSampleIdentity sample{3, 7, 12.5};
    EXPECT_EQ(sample.network_id(), 3);
    EXPECT_EQ(sample.native_row(), 7);
    EXPECT_THROW((pipeline::NativeSampleIdentity{-1, 7, 12.5}),
                 std::invalid_argument);
    EXPECT_THROW((pipeline::NativeOperationIdentity{0, -1}),
                 std::invalid_argument);
}

}  // namespace
