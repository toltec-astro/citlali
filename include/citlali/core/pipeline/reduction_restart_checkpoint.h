#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/weight_validation_restart_state.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *reduction_restart_checkpoint_schema_version =
    "citlali-reduction-restart-checkpoint-v2";
inline constexpr const char *reduction_restart_checkpoint_filename =
    "citlali_restart_checkpoint.nc";

struct ReductionRestartCheckpointSummary {
    std::filesystem::path checkpoint_path;
    std::filesystem::path source_reduction_dir;
    std::string creator_version;
    std::string fruit_loops_type;
    int completed_iteration = -1;
    int next_iteration = -1;
    std::vector<std::string> observation_ids;
    std::size_t effective_sample_mask_intervals = 0;
    std::size_t effective_detector_penalties = 0;
    std::size_t weight_validation_detector_slots = 0;
    int weight_validation_accumulated_iterations = 0;
    bool weight_validation_finalized = false;
};

std::filesystem::path reduction_restart_checkpoint_path(
    const std::filesystem::path &reduction_dir);

std::string learning_restart_policy_snapshot(
    const citlali::config::TimestreamLearningConfig &config);

std::string processed_time_chunk_restart_policy_snapshot(
    const citlali::config::ProcessedTimeChunkConfig &config);

void write_reduction_restart_checkpoint(
    const std::filesystem::path &reduction_dir, int completed_iteration,
    const std::string &fruit_loops_type,
    const std::vector<std::string> &observation_ids,
    const citlali::config::TimestreamLearningConfig &learning_config,
    const citlali::config::ProcessedTimeChunkConfig &processed_config,
    const ReductionLearningState &learning,
    const WeightValidationRestartState &weight_validation);

ReductionRestartCheckpointSummary load_reduction_restart_checkpoint(
    const std::filesystem::path &source_reduction_dir,
    const std::string &expected_fruit_loops_type,
    const std::vector<std::string> &expected_observation_ids,
    const citlali::config::TimestreamLearningConfig &expected_learning_config,
    const citlali::config::ProcessedTimeChunkConfig &expected_processed_config,
    ReductionLearningState &learning,
    WeightValidationRestartState &weight_validation);

}  // namespace citlali::pipeline
