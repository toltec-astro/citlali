#include <citlali/core/engine/learning.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/pipeline/astrometry_execution_plan.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/fruit_loop_activation_validation.h>
#include <citlali/core/pipeline/learning_housekeeping_qa.h>
#include <citlali/core/pipeline/learning_config_adapter.h>
#include <citlali/core/pipeline/initial_fruit_loop_map_loading.h>
#include <citlali/core/pipeline/fruit_loop_iteration_state.h>
#include <citlali/core/pipeline/previous_fruit_loop_map_loading.h>
#include <citlali/core/pipeline/reduction_restart_checkpoint.h>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

ReductionLearningState make_learning_state(bool diagnostics_enabled = true,
                                           int diagnostic_cap = 1) {
    ReductionLearningState state;
    ReductionLearningState::Options options;
    options.enabled = true;
    options.diagnostics_enabled = diagnostics_enabled;
    options.max_records_per_type = diagnostic_cap;
    state.configure(options);
    state.begin_iteration(0, false, "science");
    return state;
}

ReductionLearningState::LearnedSampleMask sample_mask(
    std::string obsnum, int uid, long long start, long long stop,
    int iter = 0) {
    ReductionLearningState::LearnedSampleMask record;
    record.obsnum = std::move(obsnum);
    record.producer = "test";
    record.reason = "test_pathology";
    record.iter = iter;
    record.scan = 4;
    record.uid = uid;
    record.ptc_start = start;
    record.ptc_stop = stop;
    return record;
}

ReductionLearningState::DetectorPenalty detector_penalty(
    std::string obsnum, int uid, int iter = 0) {
    ReductionLearningState::DetectorPenalty record;
    record.obsnum = std::move(obsnum);
    record.producer = "ptc_second_pass";
    record.reason = "busy_vetoed_residual";
    record.iter = iter;
    record.scan = 4;
    record.uid = uid;
    record.nw = 1;
    record.array = 0;
    record.factor = 0.0;
    record.scan_local = true;
    return record;
}

ReductionLearningState::MapPixelOutlier map_pixel_outlier(
    std::string obsnum, int iter, int map_index, int row, int col,
    double leave_one_out_z, double value = 0.0) {
    ReductionLearningState::MapPixelOutlier record;
    record.obsnum = std::move(obsnum);
    record.producer = "mapdiag:raw_obs";
    record.reason = "extreme_pixel_no_contributor";
    record.iter = iter;
    record.map_index = map_index;
    record.row = row;
    record.col = col;
    record.leave_one_out_z = leave_one_out_z;
    record.value = value;
    return record;
}

struct HousekeepingQaDataItem {
    std::string interface_name;
    std::string path;
    const std::string &interface() const { return interface_name; }
    const std::string &filepath() const { return path; }
};

struct HousekeepingQaRawObs {
    std::vector<HousekeepingQaDataItem> items;
    const auto &data_items() const { return items; }
    auto kidsdata() const {
        std::vector<std::reference_wrapper<const HousekeepingQaDataItem>> kids;
        for (const auto &item : items) {
            if (item.interface_name.rfind("toltec", 0) == 0 &&
                item.interface_name != "toltec_hk") {
                kids.push_back(std::cref(item));
            }
        }
        return kids;
    }
};

struct HousekeepingQaEngine {
    ReductionLearningState learning;
    struct {
        std::string redu_dir_name;
    } output_paths;
    struct {
        int fruit_iter = 0;
    } iteration;
    struct {
        std::string obsnum;
    } observation_identity;
};

class RestartCheckpointDirectory {
public:
    RestartCheckpointDirectory() {
        const auto stamp = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
        path = std::filesystem::temp_directory_path() /
               ("citlali_restart_checkpoint_test_" +
                std::to_string(stamp));
        std::filesystem::create_directories(path);
    }

    ~RestartCheckpointDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    std::filesystem::path path;
};

struct RestartLifecycleLogger {
    template <class... Args>
    void info(const char *, Args &&...) const {}
};

struct RestartWeightValidationProcessor {
    citlali::fruit::FruitLoopRelaxedFeedbackState
        fruit_loop_relaxed_feedback_state;
    int weight_validation_current_iter = 0;
    int weight_validation_accumulated_iters = 0;
    int weight_validation_current_iter_contribution_count = 0;
    bool weight_validation_finalized = false;
    Eigen::VectorXd weight_validation_ratio_penalty_sum;
    Eigen::VectorXd weight_validation_ratio_value_sum;
    Eigen::VectorXi weight_validation_ratio_value_count;
    Eigen::VectorXi weight_validation_ratio_count;
    Eigen::VectorXd weight_validation_atm_penalty_sum;
    Eigen::VectorXd weight_validation_atm_corr_sum;
    Eigen::VectorXi weight_validation_atm_count;
    Eigen::VectorXd weight_validation_detector_penalty;
    Eigen::VectorXi weight_validation_detector_validated;
    std::shared_ptr<std::mutex> weight_validation_mutex =
        std::make_shared<std::mutex>();
};

struct RestartLifecycleEngine {
    struct {
        citlali::config::TimestreamConfig timestream;
    } typed_config;
    citlali::pipeline::ProcessedTimestreamExecutionPlan
        processed_timestream_plan;
    citlali::pipeline::AstrometryExecutionPlan astrometry_plan;
    ReductionLearningState learning;
    struct {
        int fruit_iter = 0;
    } iteration;
    struct {
        std::vector<std::string> obsnums;
    } omb;
    RestartWeightValidationProcessor ptcproc;
};

citlali::config::TimestreamLearningConfig restart_learning_config() {
    citlali::config::TimestreamLearningConfig config;
    config.enabled = true;
    config.diagnostics_enabled = false;
    config.learn_iters = 2;
    config.apply_start_iter = 2;
    config.max_records_per_type = 17;
    config.apply_sample_masks_enabled = true;
    config.apply_max_new_flagged_fraction = 0.03;
    return config;
}

citlali::config::TimestreamLearningConfig
target_restart_learning_config(int max_targets = 4) {
    auto config = restart_learning_config();
    config.diagnostics_enabled = true;
    config.max_records_per_type = 100;
    config.map_pixel_outlier.diagnostics_enabled = true;
    config.map_pixel_outlier.contributor_diagnostics_enabled = false;
    config.map_pixel_outlier.targeted_contributor_diagnostics_enabled = true;
    config.map_pixel_outlier.detector_exclusion_enabled = true;
    config.map_pixel_outlier.targeted_contributor_max_pixels = max_targets;
    return config;
}

ReductionLearningState restart_learning_state(
    const citlali::config::TimestreamLearningConfig &config) {
    ReductionLearningState state;
    citlali::pipeline::adapt_learning_config_one_way(config, state);
    return state;
}

citlali::fruit::FruitLoopRelaxedFeedbackState
relaxed_feedback_restart_state(int completed_iteration = 4,
                               double alpha = 1.25) {
    citlali::fruit::FruitLoopRelaxedFeedbackState state;
    state.method_active = true;
    state.alpha = alpha;
    state.stored = true;
    state.observation_id = "152390";
    state.completed_iteration = completed_iteration;
    state.map_grouping = "array";
    state.map_count = 1;
    state.n_rows = 2;
    state.n_cols = 2;
    state.pixel_size_rad = 1.0e-5;
    state.wcs_cdelt = {-1.0F, 1.0F};
    state.wcs_naxis = {2, 2};
    state.wcs_crpix = {0.5F, 0.5F};
    state.wcs_crval = {10.0F, 20.0F};
    state.wcs_cunit = {"deg", "deg"};
    state.signal = {1.0, 2.0, 3.0, 4.0};
    state.kernel = {0.1, 0.2, 0.3, 0.4};
    state.weight = {5.0, 6.0, 7.0, 8.0};
    state.median_rms = {0.25};
    return state;
}

citlali::config::ProcessedTimeChunkConfig restart_processed_config() {
    citlali::config::ProcessedTimeChunkConfig config;
    config.weighting.validation.enabled = true;
    config.weighting.validation.accumulation_iters = 2;
    config.weighting.validation.apply_start_iter = 2;
    return config;
}

citlali::pipeline::WeightValidationRestartState
restart_weight_validation_state() {
    return citlali::pipeline::WeightValidationRestartState{
        2,
        true,
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {1, 2, 3},
        {2, 3, 4},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 12.0},
        {3, 4, 5},
        {0.5, 0.75, 1.05},
        {1, 1, 0},
    };
}

void expect_same_weight_validation_state(
    const citlali::pipeline::WeightValidationRestartState &left,
    const citlali::pipeline::WeightValidationRestartState &right) {
    EXPECT_EQ(left.accumulated_iterations, right.accumulated_iterations);
    EXPECT_EQ(left.finalized, right.finalized);
    EXPECT_EQ(left.ratio_penalty_sum, right.ratio_penalty_sum);
    EXPECT_EQ(left.ratio_value_sum, right.ratio_value_sum);
    EXPECT_EQ(left.ratio_value_count, right.ratio_value_count);
    EXPECT_EQ(left.ratio_count, right.ratio_count);
    EXPECT_EQ(left.atmospheric_penalty_sum,
              right.atmospheric_penalty_sum);
    EXPECT_EQ(left.atmospheric_correlation_sum,
              right.atmospheric_correlation_sum);
    EXPECT_EQ(left.atmospheric_count, right.atmospheric_count);
    EXPECT_EQ(left.detector_penalty, right.detector_penalty);
    EXPECT_EQ(left.detector_validated, right.detector_validated);
}

void execute_synthetic_learning_iteration(ReductionLearningState &state,
                                          int iteration) {
    state.begin_iteration(iteration, iteration > 0, "science");
    if (state.learning_active()) {
        state.record_learned_sample_mask(
            sample_mask("152390", 10, 10 + iteration, 12 + iteration,
                        iteration));
    }
    auto penalty = detector_penalty("152390", 10, iteration);
    penalty.factor = 1.0 - 0.05 * iteration;
    penalty.score = 10.0 + iteration;
    state.record_detector_penalty(std::move(penalty), true);
    state.finalize_iteration(iteration);
}

void expect_same_effective_learning_state(
    const ReductionLearningState &left,
    const ReductionLearningState &right, int before_iteration) {
    EXPECT_EQ(left.effective_sample_mask_interval_count(),
              right.effective_sample_mask_interval_count());
    const auto left_masks = left.effective_sample_masks_for(
        "152390", 4, false, before_iteration);
    const auto right_masks = right.effective_sample_masks_for(
        "152390", 4, false, before_iteration);
    ASSERT_EQ(left_masks.size(), right_masks.size());
    for (std::size_t i = 0; i < left_masks.size(); ++i) {
        EXPECT_EQ(left_masks[i].iter, right_masks[i].iter);
        EXPECT_EQ(left_masks[i].uid, right_masks[i].uid);
        EXPECT_EQ(left_masks[i].start, right_masks[i].start);
        EXPECT_EQ(left_masks[i].stop, right_masks[i].stop);
    }
    const auto left_penalties = left.effective_detector_penalty_records();
    const auto right_penalties = right.effective_detector_penalty_records();
    ASSERT_EQ(left_penalties.size(), right_penalties.size());
    for (std::size_t i = 0; i < left_penalties.size(); ++i) {
        EXPECT_EQ(left_penalties[i].obsnum, right_penalties[i].obsnum);
        EXPECT_EQ(left_penalties[i].producer, right_penalties[i].producer);
        EXPECT_EQ(left_penalties[i].reason, right_penalties[i].reason);
        EXPECT_EQ(left_penalties[i].iter, right_penalties[i].iter);
        EXPECT_EQ(left_penalties[i].scan, right_penalties[i].scan);
        EXPECT_EQ(left_penalties[i].uid, right_penalties[i].uid);
        EXPECT_EQ(left_penalties[i].nw, right_penalties[i].nw);
        EXPECT_EQ(left_penalties[i].array, right_penalties[i].array);
        EXPECT_DOUBLE_EQ(left_penalties[i].factor,
                         right_penalties[i].factor);
        EXPECT_DOUBLE_EQ(left_penalties[i].score,
                         right_penalties[i].score);
        EXPECT_EQ(left_penalties[i].scan_local,
                  right_penalties[i].scan_local);
    }
}

void expect_same_resolved_map_pixel_target_state(
    const ReductionLearningState &left,
    const ReductionLearningState &right) {
    const auto left_records = left.resolved_map_pixel_target_records();
    const auto right_records = right.resolved_map_pixel_target_records();
    ASSERT_EQ(left_records.size(), right_records.size());
    for (std::size_t i = 0; i < left_records.size(); ++i) {
        EXPECT_EQ(left_records[i].obsnum, right_records[i].obsnum);
        EXPECT_EQ(left_records[i].producer, right_records[i].producer);
        EXPECT_EQ(left_records[i].source_iter, right_records[i].source_iter);
        EXPECT_EQ(left_records[i].apply_iter, right_records[i].apply_iter);
        EXPECT_EQ(left_records[i].map_count, right_records[i].map_count);
        EXPECT_EQ(left_records[i].n_rows, right_records[i].n_rows);
        EXPECT_EQ(left_records[i].n_cols, right_records[i].n_cols);
        ASSERT_EQ(left_records[i].targets.size(),
                  right_records[i].targets.size());
        for (std::size_t j = 0; j < left_records[i].targets.size(); ++j) {
            EXPECT_EQ(left_records[i].targets[j].map_index,
                      right_records[i].targets[j].map_index);
            EXPECT_EQ(left_records[i].targets[j].row,
                      right_records[i].targets[j].row);
            EXPECT_EQ(left_records[i].targets[j].col,
                      right_records[i].targets[j].col);
        }
    }
}

void execute_synthetic_target_iteration(ReductionLearningState &state,
                                        int iteration) {
    state.begin_iteration(iteration, iteration > 0, "pointing");
    if (iteration > 0) {
        const auto resolved = state.resolved_map_pixel_targets_for(
            "152390", "mapdiag:raw_obs", iteration);
        ASSERT_TRUE(resolved.has_value());
        for (const auto &target : resolved->targets) {
            auto penalty = detector_penalty(
                "152390", 1000 + target.row, iteration);
            penalty.producer = "mapdiag:raw_obs";
            penalty.reason = "map_pixel_outlier_detector_dominance";
            penalty.scan = target.col;
            penalty.score = static_cast<double>(target.row);
            state.record_detector_penalty(std::move(penalty), true);
        }
    }
    state.record_map_pixel_outlier(map_pixel_outlier(
        "152390", iteration, 0, iteration + 1, iteration + 2,
        10.0 + iteration));
    state.resolve_map_pixel_targets_for_next_iteration(
        "152390", "mapdiag:raw_obs", iteration, 2, 16, 16);
    state.finalize_map_pixel_target_state(
        {"152390"}, "mapdiag:raw_obs", iteration);
    state.finalize_iteration(iteration);
}

TEST(ReductionLearningState, DiagnosticCapNeverTruncatesEffectiveState) {
    auto state = make_learning_state();
    state.record_learned_sample_mask(sample_mask("152390", 10, 20, 22));
    state.record_learned_sample_mask(sample_mask("152433", 11, 30, 31));

    EXPECT_EQ(state.learned_sample_mask_events.size(), 1U);
    EXPECT_EQ(state.dropped_learned_sample_masks, 1U);
    EXPECT_EQ(state.effective_sample_mask_interval_count(), 2U);
    EXPECT_EQ(state.effective_sample_masks_for("152390", 4, false, 2).size(),
              1U);
    EXPECT_EQ(state.effective_sample_masks_for("152433", 4, false, 2).size(),
              1U);
}

TEST(ReductionLearningState, MergesRepeatedAndAdjacentMaskEventsOnline) {
    auto state = make_learning_state(true, 20);
    state.record_learned_sample_mask(sample_mask("152390", 10, 13, 15));
    state.record_learned_sample_mask(sample_mask("152390", 10, 10, 12));
    state.begin_iteration(1, true, "science");
    state.record_learned_sample_mask(sample_mask("152390", 10, 11, 14, 1));

    const auto effective =
        state.effective_sample_masks_for("152390", 4, false, 2);
    ASSERT_EQ(effective.size(), 1U);
    EXPECT_EQ(effective.front().start, 10);
    EXPECT_EQ(effective.front().stop, 15);
    EXPECT_EQ(effective.front().iter, 0);
}

TEST(ReductionLearningState, DiagnosticsCanBeDisabledWithoutDisablingLearning) {
    auto state = make_learning_state(false);
    state.record_learned_sample_mask(sample_mask("152390", 10, 20, 22));
    state.record_detector_penalty(detector_penalty("152390", 10));

    EXPECT_TRUE(state.learned_sample_mask_events.empty());
    EXPECT_TRUE(state.detector_penalty_events.empty());
    EXPECT_EQ(state.effective_sample_mask_interval_count(), 1U);
    EXPECT_EQ(state.effective_detector_penalty_records().size(), 1U);
}

TEST(ReductionLearningState, DetectorPenaltyEventsDeduplicateInEffectiveState) {
    auto state = make_learning_state(true, 1);
    state.record_detector_penalty(detector_penalty("152390", 10));
    state.record_detector_penalty(detector_penalty("152433", 11));
    state.begin_iteration(1, true, "science");
    state.record_detector_penalty(detector_penalty("152390", 10, 1));

    EXPECT_EQ(state.detector_penalty_events.size(), 1U);
    EXPECT_EQ(state.dropped_detector_penalties, 2U);
    EXPECT_EQ(state.effective_detector_penalty_records().size(), 2U);
}

TEST(ReductionLearningState,
     ResolvesBoundedNextIterationMapPixelTargetsWithoutHistoryCarry) {
    auto config = target_restart_learning_config(3);
    config.max_records_per_type = 1;
    auto state = restart_learning_state(config);
    state.begin_iteration(4, true, "pointing");
    state.record_map_pixel_outlier(
        map_pixel_outlier("152390", 3, 0, 1, 1, 100.0));
    state.record_map_pixel_outlier(
        map_pixel_outlier("152390", 4, 0, 2, 2, 10.0));
    state.record_map_pixel_outlier(
        map_pixel_outlier("152390", 4, 1, 3, 3,
                          ReductionLearningState::nan_value(), -9.0));
    state.record_map_pixel_outlier(
        map_pixel_outlier("152390", 4, 0, 4, 4, 8.0));
    state.record_map_pixel_outlier(
        map_pixel_outlier("152390", 4, 0, 2, 2, 7.0));
    state.record_map_pixel_outlier(
        map_pixel_outlier("152390", 4, 3, 5, 5, 200.0));

    state.resolve_map_pixel_targets_for_next_iteration(
        "152390", "mapdiag:raw_obs", 4, 2, 10, 10);
    state.finalize_map_pixel_target_state(
        {"152390"}, "mapdiag:raw_obs", 4);

    EXPECT_EQ(state.map_pixel_outliers.size(), 1U);
    EXPECT_EQ(state.dropped_map_pixel_outliers, 5U);
    const auto resolved = state.resolved_map_pixel_targets_for(
        "152390", "mapdiag:raw_obs", 5);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->source_iter, 4);
    EXPECT_EQ(resolved->targets.size(), 3U);
    EXPECT_EQ(resolved->targets[0].map_index, 0);
    EXPECT_EQ(resolved->targets[0].row, 2);
    EXPECT_EQ(resolved->targets[0].col, 2);
    EXPECT_EQ(resolved->targets[1].map_index, 1);
    EXPECT_EQ(resolved->targets[1].row, 3);
    EXPECT_EQ(resolved->targets[1].col, 3);
    EXPECT_EQ(resolved->targets[2].map_index, 0);
    EXPECT_EQ(resolved->targets[2].row, 4);
    EXPECT_EQ(resolved->targets[2].col, 4);

    mapmaking::MapBuffer map;
    map.n_rows = 10;
    map.n_cols = 10;
    std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> targets;
    for (const auto &target : resolved->targets) {
        targets.emplace_back(target.map_index, target.row, target.col);
    }
    map.set_contribution_targets(2, targets);
    EXPECT_TRUE(map.contribution_target_enabled(0, 2, 2));
    EXPECT_TRUE(map.contribution_target_enabled(1, 3, 3));
    EXPECT_FALSE(map.contribution_target_enabled(0, 1, 1));

    state.map_pixel_outliers.clear();
    state.finalize_map_pixel_target_state(
        {"152390"}, "mapdiag:raw_obs", 5);
    const auto carried = state.resolved_map_pixel_targets_for(
        "152390", "mapdiag:raw_obs", 6);
    ASSERT_TRUE(carried.has_value());
    EXPECT_EQ(carried->source_iter, 4);
    EXPECT_EQ(carried->targets.size(), 3U);
}

TEST(ReductionRestartCheckpoint,
     RoundTripsResolvedMapPixelTargetsWithoutDiagnosticHistory) {
    RestartCheckpointDirectory directory;
    const auto config = target_restart_learning_config(3);
    const auto processed_config = restart_processed_config();
    auto original = restart_learning_state(config);
    original.begin_iteration(4, true, "pointing");
    original.record_map_pixel_outlier(
        map_pixel_outlier("152390", 4, 0, 2, 3, 12.0));
    original.record_map_pixel_outlier(
        map_pixel_outlier("152390", 4, 1, 4, 5, 10.0));
    original.resolve_map_pixel_targets_for_next_iteration(
        "152390", "mapdiag:raw_obs", 4, 3, 12, 13);
    original.finalize_map_pixel_target_state(
        {"152390", "152433"}, "mapdiag:raw_obs", 4);
    original.finalize_iteration(4);

    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "coadd/raw", {"152390", "152433"}, config,
        processed_config, original, {});
    auto restored = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState restored_weight_validation;
    const auto summary =
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "coadd/raw", {"152390", "152433"}, config,
            processed_config, restored, restored_weight_validation);

    EXPECT_EQ(summary.resolved_map_pixel_target_scopes, 2U);
    EXPECT_EQ(summary.resolved_map_pixel_targets, 2U);
    EXPECT_TRUE(restored.map_pixel_outliers.empty());
    expect_same_resolved_map_pixel_target_state(original, restored);
    const auto resolved = restored.resolved_map_pixel_targets_for(
        "152390", "mapdiag:raw_obs", 5);
    ASSERT_TRUE(resolved.has_value());
    ASSERT_EQ(resolved->targets.size(), 2U);
    EXPECT_EQ(resolved->map_count, 3);
    EXPECT_EQ(resolved->n_rows, 12);
    EXPECT_EQ(resolved->n_cols, 13);
    const auto empty = restored.resolved_map_pixel_targets_for(
        "152433", "mapdiag:raw_obs", 5);
    ASSERT_TRUE(empty.has_value());
    EXPECT_EQ(empty->source_iter, -1);
    EXPECT_TRUE(empty->targets.empty());

}

TEST(ReductionRestartCheckpoint,
     RoundTripsCompleteElF1RelaxedFeedbackState) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    auto learning = restart_learning_state(config);
    const auto original_feedback = relaxed_feedback_restart_state();

    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "obsnum/raw", {"152390"}, config,
        processed_config, learning, {}, original_feedback);

    auto restored_learning = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState weight_validation;
    citlali::fruit::FruitLoopRelaxedFeedbackState restored_feedback;
    restored_feedback.method_active = true;
    restored_feedback.alpha = 1.25;
    const auto summary =
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "obsnum/raw", {"152390"}, config,
            processed_config, restored_learning, weight_validation,
            &restored_feedback);

    EXPECT_EQ(summary.feedback_method_id,
              citlali::fruit::fruit_loop_compact_relaxation_method_id);
    EXPECT_DOUBLE_EQ(summary.feedback_alpha, 1.25);
    EXPECT_TRUE(summary.feedback_state_stored);
    EXPECT_EQ(restored_feedback.method_id, original_feedback.method_id);
    EXPECT_EQ(restored_feedback.observation_id,
              original_feedback.observation_id);
    EXPECT_EQ(restored_feedback.completed_iteration, 4);
    EXPECT_EQ(restored_feedback.signal, original_feedback.signal);
    EXPECT_EQ(restored_feedback.kernel, original_feedback.kernel);
    EXPECT_EQ(restored_feedback.weight, original_feedback.weight);
    EXPECT_EQ(restored_feedback.median_rms, original_feedback.median_rms);
    EXPECT_EQ(restored_feedback.wcs_crval, original_feedback.wcs_crval);
}

TEST(ReductionRestartCheckpoint,
     IdentifiesElF1AlphaOneWhileReusingOrdinaryCompleteProduct) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    auto learning = restart_learning_state(config);
    citlali::fruit::FruitLoopRelaxedFeedbackState feedback;
    feedback.method_active = true;
    feedback.alpha = 1.0;

    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "obsnum/raw", {"152390"}, config,
        processed_config, learning, {}, feedback);

    auto restored_learning = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState weight_validation;
    citlali::fruit::FruitLoopRelaxedFeedbackState restored_feedback;
    restored_feedback.method_active = true;
    restored_feedback.alpha = 1.0;
    const auto summary =
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "obsnum/raw", {"152390"}, config,
            processed_config, restored_learning, weight_validation,
            &restored_feedback);

    EXPECT_EQ(summary.feedback_method_id,
              citlali::fruit::fruit_loop_compact_relaxation_method_id);
    EXPECT_DOUBLE_EQ(summary.feedback_alpha, 1.0);
    EXPECT_FALSE(summary.feedback_state_stored);
    EXPECT_TRUE(restored_feedback.method_active);
    EXPECT_FALSE(restored_feedback.stored);
}

TEST(ReductionRestartCheckpoint,
     RejectsFeedbackMethodMismatchAcrossRestart) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    auto learning = restart_learning_state(config);
    const auto feedback = relaxed_feedback_restart_state();
    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "obsnum/raw", {"152390"}, config,
        processed_config, learning, {}, feedback);

    auto restored_learning = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState weight_validation;
    citlali::fruit::FruitLoopRelaxedFeedbackState unity_feedback;
    unity_feedback.alpha = 1.0;
    EXPECT_THROW(
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "obsnum/raw", {"152390"}, config,
            processed_config, restored_learning, weight_validation,
            &unity_feedback),
        std::runtime_error);
}

TEST(ReductionRestartCheckpoint,
     RejectsMissingRequiredResolvedMapPixelTargetState) {
    RestartCheckpointDirectory directory;
    const auto config = target_restart_learning_config();
    const auto processed_config = restart_processed_config();
    auto learning = restart_learning_state(config);

    EXPECT_THROW(
        citlali::pipeline::write_reduction_restart_checkpoint(
            directory.path, 4, "coadd/raw", {"152390"}, config,
            processed_config, learning, {}),
        std::invalid_argument);
}

TEST(ReductionRestartCheckpoint,
     RejectsMalformedResolvedMapPixelTargetState) {
    RestartCheckpointDirectory directory;
    const auto config = target_restart_learning_config();
    const auto processed_config = restart_processed_config();
    auto learning = restart_learning_state(config);
    learning.begin_iteration(4, true, "pointing");
    learning.record_map_pixel_outlier(
        map_pixel_outlier("152390", 4, 0, 2, 3, 12.0));
    learning.resolve_map_pixel_targets_for_next_iteration(
        "152390", "mapdiag:raw_obs", 4, 2, 12, 13);
    learning.finalize_map_pixel_target_state(
        {"152390"}, "mapdiag:raw_obs", 4);
    learning.resolved_map_pixel_target_sets.begin()
        ->second.targets.front().row = 12;

    EXPECT_THROW(
        citlali::pipeline::write_reduction_restart_checkpoint(
            directory.path, 4, "coadd/raw", {"152390"}, config,
            processed_config, learning, {}),
        std::invalid_argument);
}

TEST(ReductionRestartCheckpoint, RejectsOlderCheckpointSchema) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    auto learning = restart_learning_state(config);
    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "coadd/raw", {"152390"}, config,
        processed_config, learning, {});
    {
        netCDF::NcFile file(
            citlali::pipeline::reduction_restart_checkpoint_path(
                directory.path)
                .string(),
            netCDF::NcFile::write);
        file.getVar("schema_version")
            .putVar(std::vector<std::size_t>{0},
                    std::string{"citlali-reduction-restart-checkpoint-v2"});
    }

    auto restored = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState weight_validation;
    EXPECT_THROW(
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "coadd/raw", {"152390"}, config,
            processed_config, restored, weight_validation),
        std::runtime_error);
}

TEST(ReductionRestartCheckpoint, RoundTripsEffectiveLearningState) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    const auto weight_validation = restart_weight_validation_state();
    auto original = restart_learning_state(config);
    execute_synthetic_learning_iteration(original, 0);
    original.record_learned_sample_mask(
        sample_mask("152433", 11, 30, 32, 0));
    for (int iteration = 1; iteration <= 4; ++iteration) {
        execute_synthetic_learning_iteration(original, iteration);
    }

    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "coadd/raw", {"152390", "152433"}, config,
        processed_config, original, weight_validation);
    auto restored = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState restored_weight_validation;
    const auto summary =
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "coadd/raw", {"152390", "152433"}, config,
            processed_config, restored, restored_weight_validation);

    EXPECT_EQ(summary.completed_iteration, 4);
    EXPECT_EQ(summary.next_iteration, 5);
    EXPECT_EQ(summary.observation_ids,
              (std::vector<std::string>{"152390", "152433"}));
    EXPECT_EQ(summary.effective_sample_mask_intervals,
              original.effective_sample_mask_interval_count());
    EXPECT_EQ(summary.effective_detector_penalties,
              original.effective_detector_penalty_records().size());
    EXPECT_EQ(summary.weight_validation_detector_slots, 3U);
    EXPECT_EQ(summary.weight_validation_accumulated_iterations, 2);
    EXPECT_TRUE(summary.weight_validation_finalized);
    expect_same_effective_learning_state(original, restored, 5);
    expect_same_weight_validation_state(weight_validation,
                                        restored_weight_validation);
    EXPECT_EQ(restored.effective_sample_masks_for(
                  "152433", 4, false, 5)
                  .size(),
              1U);
}

TEST(ReductionRestartCheckpoint,
     RoundTripsPartiallyAccumulatedWeightValidationState) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    auto learning = restart_learning_state(config);
    execute_synthetic_learning_iteration(learning, 0);
    auto weight_validation = restart_weight_validation_state();
    weight_validation.accumulated_iterations = 1;
    weight_validation.finalized = false;

    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 0, "coadd/raw", {"152390"}, config,
        processed_config, learning, weight_validation);

    auto restored_learning = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState
        restored_weight_validation;
    const auto summary =
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "coadd/raw", {"152390"}, config,
            processed_config, restored_learning,
            restored_weight_validation);

    EXPECT_EQ(summary.weight_validation_accumulated_iterations, 1);
    EXPECT_FALSE(summary.weight_validation_finalized);
    expect_same_weight_validation_state(weight_validation,
                                        restored_weight_validation);
}

TEST(ReductionRestartCheckpoint, RejectsObservationAndPolicyMismatch) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    const auto weight_validation = restart_weight_validation_state();
    auto original = restart_learning_state(config);
    execute_synthetic_learning_iteration(original, 0);
    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 0, "coadd/raw", {"152390"}, config,
        processed_config, original, weight_validation);

    auto restored = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState restored_weight_validation;
    EXPECT_THROW(citlali::pipeline::load_reduction_restart_checkpoint(
                     directory.path, "coadd/raw", {"152433"}, config,
                     processed_config, restored,
                     restored_weight_validation),
                 std::runtime_error);

    auto changed_config = config;
    changed_config.apply_max_new_flagged_fraction = 0.04;
    EXPECT_THROW(citlali::pipeline::load_reduction_restart_checkpoint(
                     directory.path, "coadd/raw", {"152390"},
                     changed_config, processed_config, restored,
                     restored_weight_validation),
                 std::runtime_error);

    auto changed_processed_config = processed_config;
    changed_processed_config.weighting.validation.min_factor = 0.25;
    EXPECT_THROW(citlali::pipeline::load_reduction_restart_checkpoint(
                     directory.path, "coadd/raw", {"152390"}, config,
                     changed_processed_config, restored,
                     restored_weight_validation),
                 std::runtime_error);
}

TEST(ReductionRestartCheckpoint, RejectsMalformedWeightValidationState) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    auto learning = restart_learning_state(config);
    auto weight_validation = restart_weight_validation_state();
    weight_validation.ratio_count.pop_back();

    EXPECT_THROW(
        citlali::pipeline::write_reduction_restart_checkpoint(
            directory.path, 0, "coadd/raw", {"152390"}, config,
            processed_config, learning, weight_validation),
        std::invalid_argument);
}

TEST(ReductionRestartCheckpoint,
     FivePlusTwoRestartMatchesSevenUninterruptedIterations) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    const citlali::pipeline::WeightValidationRestartState weight_validation;

    auto uninterrupted = restart_learning_state(config);
    for (int iteration = 0; iteration <= 6; ++iteration) {
        execute_synthetic_learning_iteration(uninterrupted, iteration);
    }

    auto first_run = restart_learning_state(config);
    for (int iteration = 0; iteration <= 4; ++iteration) {
        execute_synthetic_learning_iteration(first_run, iteration);
    }
    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "coadd/raw", {"152390"}, config,
        processed_config, first_run, weight_validation);

    auto restarted = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState
        restarted_weight_validation;
    const auto summary =
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "coadd/raw", {"152390"}, config,
            processed_config, restarted, restarted_weight_validation);
    for (int iteration = summary.next_iteration; iteration <= 6;
         ++iteration) {
        execute_synthetic_learning_iteration(restarted, iteration);
    }

    expect_same_effective_learning_state(uninterrupted, restarted, 7);
    EXPECT_EQ(restarted.current_iter, 6);
    EXPECT_EQ(restarted.current_phase,
              ReductionLearningState::IterationPhase::Apply);
}

TEST(ReductionRestartCheckpoint,
     FivePlusThreeTargetStateRestartMatchesEightUninterruptedIterations) {
    RestartCheckpointDirectory directory;
    const auto config = target_restart_learning_config();
    const auto processed_config = restart_processed_config();
    const citlali::pipeline::WeightValidationRestartState weight_validation;

    auto uninterrupted = restart_learning_state(config);
    for (int iteration = 0; iteration <= 7; ++iteration) {
        execute_synthetic_target_iteration(uninterrupted, iteration);
    }

    auto first_run = restart_learning_state(config);
    for (int iteration = 0; iteration <= 4; ++iteration) {
        execute_synthetic_target_iteration(first_run, iteration);
    }
    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "coadd/raw", {"152390"}, config,
        processed_config, first_run, weight_validation);

    auto restarted = restart_learning_state(config);
    citlali::pipeline::WeightValidationRestartState
        restarted_weight_validation;
    const auto summary =
        citlali::pipeline::load_reduction_restart_checkpoint(
            directory.path, "coadd/raw", {"152390"}, config,
            processed_config, restarted, restarted_weight_validation);
    EXPECT_TRUE(restarted.map_pixel_outliers.empty());
    for (int iteration = summary.next_iteration; iteration <= 7;
         ++iteration) {
        execute_synthetic_target_iteration(restarted, iteration);
    }

    expect_same_effective_learning_state(uninterrupted, restarted, 8);
    expect_same_resolved_map_pixel_target_state(uninterrupted, restarted);
    EXPECT_EQ(restarted.current_iter, 7);
}

TEST(ReductionRestartCheckpoint,
     LifecycleRestoresAbsoluteIterationAndSelectsRestartMapOnce) {
    RestartCheckpointDirectory directory;
    const auto config = restart_learning_config();
    const auto processed_config = restart_processed_config();
    const auto weight_validation = restart_weight_validation_state();
    auto checkpoint_state = restart_learning_state(config);
    for (int iteration = 0; iteration <= 4; ++iteration) {
        execute_synthetic_learning_iteration(checkpoint_state, iteration);
    }
    citlali::pipeline::write_reduction_restart_checkpoint(
        directory.path, 4, "coadd/raw", {"152390", "152433"}, config,
        processed_config, checkpoint_state, weight_validation);

    RestartLifecycleEngine engine;
    auto &fruit = engine.typed_config.timestream.fruit_loops;
    fruit.enabled = true;
    fruit.restart_path = directory.path.string();
    fruit.type = "coadd/raw";
    fruit.max_iters = 7;
    fruit.injected_source_test.enabled = true;
    fruit.injected_source_test.start_iteration = 5;
    fruit.injected_source_test.array_amplitude_mjy_beam =
        {1000.0, 2000.0, 3000.0};
    engine.typed_config.timestream.learning = config;
    engine.typed_config.timestream.processed_time_chunk = processed_config;
    engine.processed_timestream_plan =
        citlali::pipeline::make_processed_timestream_execution_plan(
            engine.typed_config.timestream);
    citlali::pipeline::adapt_learning_config_one_way(config,
                                                     engine.learning);
    engine.astrometry_plan.reset(2);
    engine.astrometry_plan.observations.push_back(
        citlali::pipeline::AstrometryObservationPlan{
            0, 152390, {}, {}, {}, {}});
    engine.astrometry_plan.observations.push_back(
        citlali::pipeline::AstrometryObservationPlan{
            1, 152433, {}, {}, {}, {}});
    // The observation map buffer contains only the current observation.  The
    // restart identity must come from the reduction-wide plan instead.
    engine.omb.obsnums = {"152433"};
    EXPECT_EQ(citlali::pipeline::reduction_restart_observation_ids(engine),
              (std::vector<std::string>{"152390", "152433"}));
    citlali::pipeline::ReductionIterationState iteration_state;
    const auto logger = std::make_shared<RestartLifecycleLogger>();

    citlali::pipeline::initialize_fruit_loop_restart_if_requested(
        engine, iteration_state, logger);

    EXPECT_TRUE(iteration_state.restarted);
    EXPECT_EQ(iteration_state.start_iteration, 5);
    EXPECT_EQ(engine.iteration.fruit_iter, 5);
    EXPECT_TRUE(citlali::pipeline::first_restarted_iteration(engine));
    EXPECT_TRUE(
        citlali::pipeline::should_load_initial_fruit_loop_maps(engine));
    EXPECT_FALSE(
        citlali::pipeline::should_load_previous_fruit_loop_maps(engine));
    EXPECT_EQ(citlali::pipeline::initial_fruit_loop_map_dir(engine),
              (directory.path / "coadded/raw/").string());
    expect_same_weight_validation_state(
        weight_validation,
        citlali::pipeline::snapshot_weight_validation_restart_state(
            engine.ptcproc));

    engine.iteration.fruit_iter = 6;
    EXPECT_FALSE(citlali::pipeline::first_restarted_iteration(engine));
    EXPECT_TRUE(
        citlali::pipeline::should_load_previous_fruit_loop_maps(engine));

    engine.iteration.fruit_iter = 0;
    fruit.injected_source_test.start_iteration = 4;
    engine.processed_timestream_plan =
        citlali::pipeline::make_processed_timestream_execution_plan(
            engine.typed_config.timestream);
    citlali::pipeline::ReductionIterationState invalid_iteration_state;
    EXPECT_THROW(
        citlali::pipeline::initialize_fruit_loop_restart_if_requested(
            engine, invalid_iteration_state, logger),
        citlali::error::Error);
}

TEST(FruitLoopActivation, RejectsEnabledNoOpConfiguration) {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 10;
    citlali::config::NoiseConfig noise;

    const auto report =
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::science);
    EXPECT_EQ(report.error_count(), 1U);
    EXPECT_EQ(report.errors().front().path,
              (citlali::config::ConfigPath{"timestream", "fruit_loops"}));
}

TEST(FruitLoopActivation, RequiresEmpiricalProductsForSnrGate) {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 2;
    fruit_loops.sig2noise_limit = 3.0;
    citlali::config::NoiseConfig noise;

    EXPECT_FALSE(
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::science)
            .ok());
    noise.enabled = true;
    noise.n_noise_maps = 10;
    noise.products_enabled = true;
    EXPECT_TRUE(
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::science)
            .ok());
}

TEST(FruitLoopActivation, AcceptsNonzeroFluxGateWithoutNoiseProducts) {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 2;
    fruit_loops.array_flux_limit = {0.1, 0.2, 0.3};
    citlali::config::NoiseConfig noise;

    EXPECT_TRUE(
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::science)
            .ok());
}

TEST(FruitLoopActivation, LeavesBeammapInternalIterationPolicyAlone) {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 1;
    citlali::config::NoiseConfig noise;

    EXPECT_TRUE(
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::beammap)
            .ok());
}

TEST(MapImageSemantics, FormalStandardizedSignalIsNotNamedSnr) {
    EXPECT_EQ(citlali::pipeline::formal_standardized_signal_map_hdu_name(
                  "a1100", "_I"),
              "formal_standardized_signal_a1100_I");
    EXPECT_EQ(citlali::pipeline::formal_standardized_signal_estimator_type(),
              std::string{"formal_weight_standardized"});
    EXPECT_NE(std::string{
                  citlali::pipeline::formal_standardized_signal_map_description()}
                  .find("not a statistical significance map"),
              std::string::npos);
}

TEST(LearningHousekeepingQa, MatchesNearestSampleAndPublishesLocalChange) {
    const std::vector<double> times{100.0, 160.0, 220.0};
    const std::vector<double> temperatures{0.100, 0.110, 0.102};

    const auto match =
        citlali::pipeline::match_learning_housekeeping_sample(
            times, temperatures, 158.0);

    EXPECT_EQ(match.status, "matched");
    EXPECT_DOUBLE_EQ(match.sample_time_unix_sec, 160.0);
    EXPECT_DOUBLE_EQ(match.sample_offset_sec, 2.0);
    EXPECT_DOUBLE_EQ(match.sample_age_sec, 2.0);
    EXPECT_DOUBLE_EQ(match.delta_from_previous, 0.010);
    EXPECT_DOUBLE_EQ(match.delta_to_next, -0.008);
    EXPECT_NEAR(match.local_excursion, 0.009, 1e-12);
}

TEST(LearningHousekeepingQa, RejectsUnavailableAndOutOfRangeMatches) {
    const std::vector<double> times{100.0, 160.0, 220.0};
    const std::vector<double> unavailable{0.100, -1.0, 0.102};

    const auto unavailable_match =
        citlali::pipeline::match_learning_housekeeping_sample(
            times, unavailable, 158.0);
    EXPECT_EQ(unavailable_match.status,
              "nearest_value_invalid_or_unavailable");
    EXPECT_FALSE(std::isfinite(unavailable_match.value));

    const auto outside_match =
        citlali::pipeline::match_learning_housekeeping_sample(
            times, unavailable, 99.0);
    EXPECT_EQ(outside_match.status, "event_outside_housekeeping_range");
    EXPECT_FALSE(std::isfinite(outside_match.sample_time_unix_sec));
}

TEST(LearningHousekeepingQa, WritesRequiredSidecarFromExplicitInput) {
    namespace fs = std::filesystem;
    const auto suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory = fs::temp_directory_path() /
                           ("citlali-learning-housekeeping-" + suffix);
    fs::create_directories(directory);
    const auto hk_path = directory / "toltec_hk_test_152433_00.nc";
    {
        netCDF::NcFile file(hk_path.string(), netCDF::NcFile::replace);
        const auto sample = file.addDim("sample", 3);
        const auto time = file.addVar(
            "Data.ToltecThermetry.Time4", netCDF::ncDouble, sample);
        const auto temperature = file.addVar(
            "Data.ToltecThermetry.Temperature4", netCDF::ncDouble, sample);
        const std::vector<double> times{100.0, 160.0, 220.0};
        const std::vector<double> temperatures{0.100, 0.110, 0.102};
        time.putVar(times.data());
        temperature.putVar(temperatures.data());
    }

    HousekeepingQaEngine engine;
    engine.output_paths.redu_dir_name = directory.string();
    engine.observation_identity.obsnum = "152433";
    engine.learning = make_learning_state(true, 20);
    auto event = detector_penalty("152433", -1);
    event.reason = "busy_network_pathology";
    event.scan = 36;
    event.nw = 9;
    event.array = 2;
    event.score = 139.0;
    event.event_time_unix_sec = 158.0;
    engine.learning.record_detector_penalty(event);
    const HousekeepingQaRawObs rawobs{{
        {"toltec_hk", hk_path.string()},
    }};

    EXPECT_NO_THROW(
        citlali::pipeline::write_learning_housekeeping_qa_if_available(
            engine, rawobs, true, spdlog::default_logger()));

    const auto output_path = directory / "learning_housekeeping_iter_0.csv";
    std::ifstream input(output_path);
    ASSERT_TRUE(input.good());
    const std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    EXPECT_NE(contents.find("citlali-learning-housekeeping-qa-v1"),
              std::string::npos);
    EXPECT_NE(contents.find("\"matched\",160,2,2,0.11"),
              std::string::npos);
    EXPECT_NE(contents.find("\"channel_missing\""), std::string::npos);

    const HousekeepingQaRawObs sibling_discovery{{
        {"toltec0", (directory / "toltec0_test_152433.nc").string()},
    }};
    const auto discovered =
        citlali::pipeline::find_learning_housekeeping_files(
            sibling_discovery, "152433");
    ASSERT_EQ(discovered.size(), 1U);
    EXPECT_EQ(discovered.front(), hk_path.string());

    std::error_code ignored;
    fs::remove_all(directory, ignored);
}

}  // namespace
