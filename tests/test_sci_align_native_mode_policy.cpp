#include <citlali/core/pipeline/native_consumer_mode_policy.h>

#include <gtest/gtest.h>

namespace {

namespace config = citlali::config;
namespace pipeline = citlali::pipeline;

TEST(SciAlignNativeModePolicy,
     ScienceActivatesOnlyWithCompleteMatchedAuthority) {
    EXPECT_EQ(pipeline::resolve_native_consumer_route({
                  config::ReductionType::science,
                  config::MapGrouping::array, false, false}),
              pipeline::NativeConsumerRoute::legacy_inactive);
    EXPECT_EQ(pipeline::resolve_native_consumer_route({
                  config::ReductionType::science,
                  config::MapGrouping::array, true, true}),
              pipeline::NativeConsumerRoute::native_required);
    EXPECT_THROW(pipeline::resolve_native_consumer_route({
                     config::ReductionType::science,
                     config::MapGrouping::array, true, false}),
                 std::logic_error);
    EXPECT_THROW(pipeline::resolve_native_consumer_route({
                     config::ReductionType::science,
                     config::MapGrouping::array, false, true}),
                 std::logic_error);
}

TEST(SciAlignNativeModePolicy,
     CollapsedPointingAndOofRuntimeCannotInferActivation) {
    EXPECT_EQ(pipeline::resolve_native_consumer_route({
                  config::ReductionType::pointing,
                  config::MapGrouping::detector, false, false}),
              pipeline::NativeConsumerRoute::legacy_inactive);
    EXPECT_THROW(pipeline::resolve_native_consumer_route({
                     config::ReductionType::pointing,
                     config::MapGrouping::detector, true, true}),
                 std::logic_error);
}

TEST(SciAlignNativeModePolicy,
     DetectorAndAutomaticBeammapRemainRawAptProducersWithoutLineage) {
    for (const auto grouping : {
             config::MapGrouping::detector,
             config::MapGrouping::automatic}) {
        const auto route = pipeline::resolve_native_consumer_route({
            config::ReductionType::beammap, grouping, false, true});
        EXPECT_EQ(route,
                  pipeline::NativeConsumerRoute::beammap_raw_apt_producer);
        EXPECT_FALSE(pipeline::native_consumer_lineage_required(route));
    }
}

TEST(SciAlignNativeModePolicy,
     ExistingNonDetectorBeammapGroupsRemainCalibrationTableConsumers) {
    for (const auto grouping : {
             config::MapGrouping::array,
             config::MapGrouping::network,
             config::MapGrouping::frequency_group}) {
        const auto route = pipeline::resolve_native_consumer_route({
            config::ReductionType::beammap, grouping, false, false});
        EXPECT_EQ(route,
                  pipeline::NativeConsumerRoute::beammap_calibration_table);
        EXPECT_FALSE(pipeline::native_consumer_lineage_required(route));
    }
}

TEST(SciAlignNativeModePolicy,
     AnyBeammapMatchedConsumerRelationFailsClosed) {
    EXPECT_THROW(pipeline::resolve_native_consumer_route({
                     config::ReductionType::beammap,
                     config::MapGrouping::detector, true, true}),
                 std::logic_error);
    EXPECT_THROW(pipeline::resolve_native_consumer_route({
                     config::ReductionType::beammap,
                     config::MapGrouping::array, true, false}),
                 std::logic_error);
}

}  // namespace
