#include <gtest/gtest.h>

#include <citlali/core/engine/engine.h>
#include <citlali/core/mapmaking/map.h>

#include <spdlog/sinks/null_sink.h>

#include <memory>
#include <stdexcept>

namespace {

ReductionLearningState::Options targeted_options() {
    ReductionLearningState::Options options;
    options.enabled = true;
    options.diagnostics_enabled = true;
    options.map_pixel_outlier_diagnostics_enabled = true;
    options.map_pixel_outlier_contributor_diagnostics_enabled = false;
    options.map_pixel_outlier_targeted_contributor_diagnostics_enabled = true;
    options.map_pixel_outlier_detector_exclusion_enabled = true;
    options.map_pixel_outlier_targeted_contributor_max_pixels = 2;
    return options;
}

ReductionLearningState::MapPixelOutlier outlier(int row, int col,
                                                double score) {
    ReductionLearningState::MapPixelOutlier record;
    record.obsnum = "152390";
    record.producer = "mapdiag:raw_obs";
    record.reason = "extreme_pixel_no_contributor";
    record.iter = 4;
    record.map_index = 0;
    record.row = row;
    record.col = col;
    record.leave_one_out_z = score;
    return record;
}

Engine configured_engine() {
    Engine engine;
    engine.logger = std::make_shared<spdlog::logger>(
        "learning-target-application-test",
        std::make_shared<spdlog::sinks::null_sink_mt>());
    engine.learning.configure(targeted_options());
    engine.learning.begin_iteration(4, true, "pointing");
    engine.observation_identity.obsnum = "152390";
    engine.iteration.fruit_iter = 5;
    return engine;
}

mapmaking::MapBuffer target_map() {
    mapmaking::MapBuffer map;
    map.n_rows = 12;
    map.n_cols = 13;
    map.signal.resize(2);
    return map;
}

TEST(LearningTargetApplication,
     AppliesResolvedBoundaryTargetsToRealContributionTracer) {
    auto engine = configured_engine();
    engine.learning.record_map_pixel_outlier(outlier(2, 3, 12.0));
    engine.learning.record_map_pixel_outlier(outlier(4, 5, 10.0));
    engine.learning.record_map_pixel_outlier(outlier(6, 7, 8.0));
    engine.learning.resolve_map_pixel_targets_for_next_iteration(
        "152390", "mapdiag:raw_obs", 4, 2, 12, 13);
    engine.learning.finalize_map_pixel_target_state(
        {"152390"}, "mapdiag:raw_obs", 4);

    auto map = target_map();
    ASSERT_NO_THROW(
        engine.configure_map_pixel_contribution_targets(map, "raw_obs"));
    EXPECT_TRUE(map.contribution_diag_enabled);
    EXPECT_TRUE(map.contribution_diag_targeted);
    EXPECT_TRUE(map.contribution_target_enabled(0, 2, 3));
    EXPECT_TRUE(map.contribution_target_enabled(0, 4, 5));
    EXPECT_FALSE(map.contribution_target_enabled(0, 6, 7));
}

TEST(LearningTargetApplication, MissingRequiredBoundaryStateFailsClosed) {
    auto engine = configured_engine();
    auto map = target_map();
    EXPECT_THROW(
        engine.configure_map_pixel_contribution_targets(map, "raw_obs"),
        std::runtime_error);
}

TEST(LearningTargetApplication,
     IncompatibleMapGridFailsClosed) {
    auto engine = configured_engine();
    engine.learning.record_map_pixel_outlier(outlier(2, 3, 12.0));
    engine.learning.resolve_map_pixel_targets_for_next_iteration(
        "152390", "mapdiag:raw_obs", 4, 2, 12, 13);
    engine.learning.finalize_map_pixel_target_state(
        {"152390"}, "mapdiag:raw_obs", 4);
    auto map = target_map();
    map.n_rows = 11;
    EXPECT_THROW(
        engine.configure_map_pixel_contribution_targets(map, "raw_obs"),
        std::runtime_error);
}

}  // namespace
