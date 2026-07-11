#pragma once

#include <Eigen/Core>

#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <memory>
#include <mutex>

namespace citlali::pipeline {

struct FruitLoopWeightPolicy {
    bool use_noise_weights = false;
    bool keep_source_subtracted_weights = false;
};

template <class Logger, class Telescope, class Counter>
void log_scan_start(
    const std::shared_ptr<std::mutex> &scans_done_mutex,
    const Logger &logger, Eigen::Index scan_index, Counter n_scans_done,
    const Telescope &telescope) {
    std::lock_guard<std::mutex> lock(*scans_done_mutex);
    logger->info("starting scan {}. {}/{} scans completed",
                 scan_index + 1, n_scans_done,
                 telescope.scan_indices.cols());
}

template <class Logger, class Telescope, class Counter>
void log_scan_done(
    const std::shared_ptr<std::mutex> &scans_done_mutex,
    const Logger &logger, Eigen::Index scan_index, Counter &n_scans_done,
    const Telescope &telescope) {
    std::lock_guard<std::mutex> lock(*scans_done_mutex);
    n_scans_done++;
    logger->info("done with scan {}. {}/{} scans completed",
                 scan_index + 1, n_scans_done,
                 telescope.scan_indices.cols());
}

template <class Engine>
FruitLoopWeightPolicy fruit_loop_weight_policy(const Engine &engine) {
    const auto &config = fruit_loops_config(engine);
    const auto &ptcproc = engine.ptcproc;
    FruitLoopWeightPolicy policy;
    policy.use_noise_weights =
        config.enabled && !ptcproc.tod_mb.signal.empty();
    policy.keep_source_subtracted_weights =
        policy.use_noise_weights &&
        !config.recompute_weights_after_addback;
    return policy;
}

}  // namespace citlali::pipeline
