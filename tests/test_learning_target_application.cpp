#include <gtest/gtest.h>

#include <citlali/core/engine/engine.h>
#include <citlali/core/mapmaking/map.h>

#include <spdlog/sinks/null_sink.h>

#include <map>
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

struct ExclusionTestChunk {
    struct {
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> data;
    } flags;
    struct {
        int data = 0;
    } index;
    struct {
        Eigen::MatrixXd data;
    } scans;
    struct {
        std::map<std::string, Eigen::VectorXd> data;
    } tel_data;
    struct {
        std::map<std::string, Eigen::VectorXd> data;
    } pointing_offsets_arcsec;
};

struct ExclusionTestCalibration {
    std::map<std::string, Eigen::VectorXd> apt;
};

Engine exclusion_engine(
    citlali::config::MapPixelOutlierDetectorExclusionApplication
        application,
    const std::string &producer,
    const std::string &reason) {
    Engine engine;
    engine.logger = std::make_shared<spdlog::logger>(
        "learning-exclusion-application-test",
        std::make_shared<spdlog::sinks::null_sink_mt>());
    auto options = targeted_options();
    options.learn_iters = 1;
    options.apply_start_iter = 1;
    options.apply_max_new_flagged_fraction = 1.0;
    options.map_pixel_outlier_detector_exclusion_application = application;
    engine.learning.configure(options);
    engine.observation_identity.obsnum = "152390";
    engine.learning.begin_iteration(0, false, "pointing");
    ReductionLearningState::DetectorPenalty penalty;
    penalty.obsnum = "152390";
    penalty.producer = producer;
    penalty.reason = reason;
    penalty.iter = 0;
    penalty.scan = 0;
    penalty.uid = 4460;
    penalty.nw = 9;
    penalty.array = 1;
    penalty.factor = 0.0;
    penalty.scan_local = true;
    engine.learning.record_detector_penalty(penalty, true);
    engine.learning.begin_iteration(1, true, "pointing");
    engine.iteration.fruit_iter = 1;
    return engine;
}

ExclusionTestChunk exclusion_chunk() {
    ExclusionTestChunk chunk;
    chunk.flags.data.resize(3, 2);
    chunk.flags.data.setConstant(false);
    chunk.scans.data = Eigen::MatrixXd::Zero(3, 2);
    return chunk;
}

ExclusionTestCalibration exclusion_calibration() {
    ExclusionTestCalibration calibration;
    calibration.apt["uid"] = Eigen::VectorXd(2);
    calibration.apt["uid"] << 4460.0, 4461.0;
    calibration.apt["nw"] = Eigen::VectorXd::Constant(2, 9.0);
    calibration.apt["array"] = Eigen::VectorXd::Constant(2, 1.0);
    calibration.apt["flag"] = Eigen::VectorXd::Zero(2);
    calibration.apt["x_t"] = Eigen::VectorXd::Zero(2);
    calibration.apt["y_t"] = Eigen::VectorXd::Zero(2);
    return calibration;
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

TEST(LearningTargetApplication,
     MapDiagnosticExclusionDefaultsToSharedCleaningStages) {
    auto engine = exclusion_engine(
        citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_cleaning,
        "mapdiag:raw_obs", "map_pixel_outlier_detector_dominance");
    auto calibration = exclusion_calibration();
    auto rtc = exclusion_chunk();
    engine.apply_learned_rtc_sample_masks(rtc, calibration);
    EXPECT_TRUE(rtc.flags.data.col(0).all());
    EXPECT_FALSE(rtc.flags.data.col(1).any());

    auto map_input = exclusion_chunk();
    engine.apply_learned_mapmaking_detector_exclusions(
        map_input, calibration);
    EXPECT_FALSE(map_input.flags.data.col(0).any());
}

TEST(LearningTargetApplication,
     MapDiagnosticExclusionCanMoveAfterSharedCleaning) {
    auto engine = exclusion_engine(
        citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_mapmaking,
        "mapdiag:raw_obs", "map_pixel_outlier_detector_dominance");
    auto calibration = exclusion_calibration();
    auto rtc = exclusion_chunk();
    engine.apply_learned_rtc_sample_masks(rtc, calibration);
    EXPECT_FALSE(rtc.flags.data.col(0).any());

    auto ptc = exclusion_chunk();
    engine.apply_learned_ptc_detector_exclusions(ptc, calibration);
    EXPECT_FALSE(ptc.flags.data.col(0).any());

    auto map_input = exclusion_chunk();
    engine.apply_learned_mapmaking_detector_exclusions(
        map_input, calibration);
    EXPECT_TRUE(map_input.flags.data.col(0).all());
    EXPECT_FALSE(map_input.flags.data.col(1).any());
}

TEST(LearningTargetApplication,
     BusyDetectorExclusionPlacementDoesNotMoveWithMapDiagnosticSetting) {
    auto engine = exclusion_engine(
        citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_mapmaking,
        "ptc_second_pass", "busy_vetoed_residual");
    auto calibration = exclusion_calibration();
    auto rtc = exclusion_chunk();
    engine.apply_learned_rtc_sample_masks(rtc, calibration);
    EXPECT_TRUE(rtc.flags.data.col(0).all());

    auto map_input = exclusion_chunk();
    engine.apply_learned_mapmaking_detector_exclusions(
        map_input, calibration);
    EXPECT_FALSE(map_input.flags.data.col(0).any());
}


// Synthetic boundary states only: no observation data or experimental action.
Engine synthetic_cap_engine(int proposed_detectors) {
    Engine engine;
    engine.logger = std::make_shared<spdlog::logger>(
        "learning-cap-boundary-test",
        std::make_shared<spdlog::sinks::null_sink_mt>());
    auto options = targeted_options();
    options.learn_iters = 1;
    options.apply_start_iter = 1;
    options.apply_max_new_flagged_fraction = 0.02;
    options.map_pixel_outlier_detector_exclusion_application =
        citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_cleaning;
    engine.learning.configure(options);
    engine.observation_identity.obsnum = "synthetic-cap-boundary";
    engine.learning.begin_iteration(0, false, "pointing");
    for (int detector = 0; detector < proposed_detectors; ++detector) {
        ReductionLearningState::DetectorPenalty penalty;
        penalty.obsnum = engine.observation_identity.obsnum;
        penalty.producer = "mapdiag:raw_obs";
        penalty.reason = "map_pixel_outlier_detector_dominance";
        penalty.iter = 0;
        penalty.scan = 0;
        penalty.uid = 1000 + detector;
        penalty.nw = 0;
        penalty.array = 0;
        penalty.factor = 0.0;
        penalty.scan_local = true;
        engine.learning.record_detector_penalty(penalty, true);
    }
    engine.learning.begin_iteration(1, true, "pointing");
    engine.iteration.fruit_iter = 1;
    return engine;
}

ExclusionTestChunk synthetic_cap_chunk(int samples, int detectors) {
    ExclusionTestChunk chunk;
    chunk.flags.data.resize(samples, detectors);
    chunk.flags.data.setConstant(false);
    chunk.scans.data = Eigen::MatrixXd::Zero(samples, detectors);
    return chunk;
}

ExclusionTestCalibration synthetic_cap_calibration(int first_detector) {
    const int detectors = 100 - first_detector;
    ExclusionTestCalibration calibration;
    calibration.apt["uid"] = Eigen::VectorXd::LinSpaced(
        detectors, 1000 + first_detector, 1099);
    for (const auto *column : {"nw", "array", "flag", "x_t", "y_t"}) {
        calibration.apt[column] = Eigen::VectorXd::Zero(detectors);
    }
    return calibration;
}

TEST(LearningTargetApplication,
     HistoricalCapCanRejectBeforeRtcAndAcceptBeforePtc) {
    // With no intervening removals both stages reject. Removing two of the
    // three proposed detectors changes the second gate, without changing its
    // cap or records. These are supplied boundary states, not an RTC replay.
    for (const int removed : {0, 2}) {
        SCOPED_TRACE(removed);
        auto engine = synthetic_cap_engine(3);
        auto raw_calibration = synthetic_cap_calibration(0);
        auto rtc = synthetic_cap_chunk(10, 100);
        engine.apply_learned_rtc_sample_masks(rtc, raw_calibration);
        ASSERT_EQ(engine.learning.learned_mask_applications.size(), 1U);
        const auto rtc_receipt = engine.learning.learned_mask_applications.back();
        EXPECT_EQ(rtc_receipt.stage, "pre_rtc_detector_exclusion");
        EXPECT_DOUBLE_EQ(rtc_receipt.newly_flagged_fraction, 0.03);
        EXPECT_FALSE(rtc_receipt.applied);
        EXPECT_FALSE(rtc.flags.data.any());
        EXPECT_TRUE(raw_calibration.apt.at("flag").isZero());

        auto ptc_calibration = synthetic_cap_calibration(removed);
        auto ptc = synthetic_cap_chunk(5, 100 - removed);
        engine.apply_learned_ptc_detector_exclusions(ptc, ptc_calibration);
        ASSERT_EQ(engine.learning.learned_mask_applications.size(), 2U);
        const auto ptc_receipt = engine.learning.learned_mask_applications.back();
        EXPECT_EQ(ptc_receipt.stage, "pre_ptc_detector_exclusion");
        EXPECT_EQ(ptc_receipt.candidate_records, 3);
        EXPECT_EQ(ptc_receipt.matched_records, 3 - removed);
        EXPECT_EQ(ptc_receipt.invalid_records, removed);
        EXPECT_DOUBLE_EQ(ptc_receipt.newly_flagged_fraction,
                         double(3 - removed) / (100 - removed));
        EXPECT_EQ(ptc_receipt.applied, removed == 2);
        if (removed == 2) {
            EXPECT_TRUE(ptc.flags.data.col(0).all());
            EXPECT_FALSE(ptc.flags.data.rightCols(97).any());
            EXPECT_DOUBLE_EQ(ptc_calibration.apt.at("flag")(0), 1.0);
        } else {
            EXPECT_FALSE(ptc.flags.data.any());
            EXPECT_TRUE(ptc_calibration.apt.at("flag").isZero());
        }
    }
}

TEST(LearningTargetApplication, HistoricalCapAcceptsItsExactBoundary) {
    auto engine = synthetic_cap_engine(2);
    auto calibration = synthetic_cap_calibration(0);
    auto rtc = synthetic_cap_chunk(10, 100);
    engine.apply_learned_rtc_sample_masks(rtc, calibration);
    ASSERT_EQ(engine.learning.learned_mask_applications.size(), 1U);
    const auto receipt = engine.learning.learned_mask_applications.back();
    EXPECT_DOUBLE_EQ(receipt.newly_flagged_fraction, 0.02);
    EXPECT_TRUE(receipt.applied);
    EXPECT_TRUE(rtc.flags.data.leftCols(2).all());
    EXPECT_FALSE(rtc.flags.data.rightCols(98).any());
}

}  // namespace
