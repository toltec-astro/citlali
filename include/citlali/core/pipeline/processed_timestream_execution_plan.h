#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/fruit_loop_iteration_policy.h>
#include <citlali/core/pipeline/processed_clean_resolution.h>
#include <citlali/core/pipeline/processed_weighting_resolution.h>
#include <citlali/core/pipeline/source_protection_activation.h>

#include <cstddef>
#include <optional>
#include <string>

namespace citlali::pipeline {

struct ProcessedTimestreamConfigSnapshot {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    citlali::config::TimestreamLearningConfig learning;
    citlali::config::ProcessedTimeChunkConfig processed_time_chunk;
};

inline ProcessedTimestreamConfigSnapshot snapshot_processed_timestream_config(
    const citlali::config::TimestreamConfig &config) {
    return ProcessedTimestreamConfigSnapshot{
        config.fruit_loops,
        config.learning,
        config.processed_time_chunk,
    };
}

struct ProcessedTimestreamEffectiveResolutionRecord {
    std::optional<ProcessedCleanerModeResolution> cleaner_mode;
    std::optional<ProcessedWeightingSourceMaskResolution>
        weighting_source_mask;
    std::optional<ProcessedWeightingResolution> weighting_dependencies;
    std::optional<FruitLoopIterationResolution> fruit_loop_iterations;
    std::optional<FruitLoopInterpolationResolution> fruit_loop_interpolation;
    struct FruitLoopRestartResolution {
        std::string source_reduction_dir;
        std::string checkpoint_path;
        std::string creator_version;
        int completed_iteration = -1;
        int next_iteration = -1;
        std::size_t effective_sample_mask_intervals = 0;
        std::size_t effective_detector_penalties = 0;
    };
    std::optional<FruitLoopRestartResolution> fruit_loop_restart;
};

struct ProcessedTimestreamRealizedState {
    std::optional<SourceProtectionActivationResolution> source_protection;
    std::optional<int> fruit_loop_iterations_completed;
    std::optional<bool> fruit_loops_converged;
};

struct ProcessedTimestreamExecutionPlan {
    ProcessedTimestreamConfigSnapshot requested;
    ProcessedTimestreamConfigSnapshot effective;
    ProcessedTimestreamEffectiveResolutionRecord effective_resolutions;
    ProcessedTimestreamRealizedState realized;
    bool initialized = false;
};

inline ProcessedTimestreamExecutionPlan make_processed_timestream_execution_plan(
    const citlali::config::TimestreamConfig &requested) {
    const auto snapshot = snapshot_processed_timestream_config(requested);
    return ProcessedTimestreamExecutionPlan{
        snapshot,
        snapshot,
        ProcessedTimestreamEffectiveResolutionRecord{},
        ProcessedTimestreamRealizedState{},
        true,
    };
}

inline void reset_processed_timestream_execution_plan(
    ProcessedTimestreamExecutionPlan &plan,
    const citlali::config::TimestreamConfig &requested) {
    plan = make_processed_timestream_execution_plan(requested);
}

inline void record_processed_timestream_iteration_result(
    ProcessedTimestreamExecutionPlan &plan, int iterations_completed,
    bool converged) {
    plan.realized.fruit_loop_iterations_completed = iterations_completed;
    plan.realized.fruit_loops_converged = converged;
}

}  // namespace citlali::pipeline
