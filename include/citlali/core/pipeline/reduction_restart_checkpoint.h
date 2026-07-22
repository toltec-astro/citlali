#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/engine/learning.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *reduction_restart_checkpoint_schema_version =
    "citlali-reduction-restart-checkpoint-v1";
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
};

std::filesystem::path reduction_restart_checkpoint_path(
    const std::filesystem::path &reduction_dir);

std::string learning_restart_policy_snapshot(
    const citlali::config::TimestreamLearningConfig &config);

void write_reduction_restart_checkpoint(
    const std::filesystem::path &reduction_dir, int completed_iteration,
    const std::string &fruit_loops_type,
    const std::vector<std::string> &observation_ids,
    const citlali::config::TimestreamLearningConfig &learning_config,
    const ReductionLearningState &learning);

ReductionRestartCheckpointSummary load_reduction_restart_checkpoint(
    const std::filesystem::path &source_reduction_dir,
    const std::string &expected_fruit_loops_type,
    const std::vector<std::string> &expected_observation_ids,
    const citlali::config::TimestreamLearningConfig &expected_learning_config,
    ReductionLearningState &learning);

}  // namespace citlali::pipeline
