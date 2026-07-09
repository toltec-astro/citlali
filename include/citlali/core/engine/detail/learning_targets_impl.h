#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

void Engine::configure_map_pixel_contribution_targets(mapmaking::MapBuffer &mb,
                                                      const std::string &stage_name) {
    const bool full_contribution_diag =
        learning.options.enabled &&
        learning.options.diagnostics_enabled &&
        learning.options.map_pixel_outlier_diagnostics_enabled &&
        learning.options.map_pixel_outlier_contributor_diagnostics_enabled;

    mb.clear_contribution_targets();
    mb.contribution_diag_enabled = full_contribution_diag;

    if (full_contribution_diag) {
        return;
    }
    if (!learning.is_enabled() ||
        !learning.diagnostics_enabled() ||
        !learning.options.map_pixel_outlier_diagnostics_enabled ||
        !learning.options.map_pixel_outlier_targeted_contributor_diagnostics_enabled ||
        learning.options.map_pixel_outlier_targeted_contributor_max_pixels <= 0 ||
        iteration.fruit_iter <= 0 ||
        mb.signal.empty() ||
        mb.n_rows <= 0 ||
        mb.n_cols <= 0) {
        return;
    }

    const std::string producer = "mapdiag:" + stage_name;
    int target_iter = -1;
    {
        std::lock_guard<std::mutex> lock(*learning.mutex);
        for (const auto &record : learning.map_pixel_outliers) {
            if (record.obsnum == observation_identity.obsnum &&
                record.producer == producer &&
                record.iter >= 0 &&
                record.iter < iteration.fruit_iter &&
                record.map_index >= 0 &&
                record.map_index < static_cast<int>(mb.signal.size()) &&
                record.row >= 0 &&
                record.row < mb.n_rows &&
                record.col >= 0 &&
                record.col < mb.n_cols) {
                target_iter = std::max(target_iter, record.iter);
            }
        }
    }
    if (target_iter < 0) {
        return;
    }

    struct target_candidate_t {
        Eigen::Index map_index = -1;
        Eigen::Index row = -1;
        Eigen::Index col = -1;
        double score = 0.0;
    };
    std::vector<target_candidate_t> candidates;
    {
        std::lock_guard<std::mutex> lock(*learning.mutex);
        for (const auto &record : learning.map_pixel_outliers) {
            if (record.obsnum != observation_identity.obsnum ||
                record.producer != producer ||
                record.iter != target_iter ||
                record.map_index < 0 ||
                record.map_index >= static_cast<int>(mb.signal.size()) ||
                record.row < 0 ||
                record.row >= mb.n_rows ||
                record.col < 0 ||
                record.col >= mb.n_cols) {
                continue;
            }
            const double raw_score =
                std::isfinite(record.leave_one_out_z)
                    ? std::abs(record.leave_one_out_z)
                    : std::abs(record.value);
            const double score = std::isfinite(raw_score) ? raw_score : 0.0;
            candidates.push_back({
                static_cast<Eigen::Index>(record.map_index),
                static_cast<Eigen::Index>(record.row),
                static_cast<Eigen::Index>(record.col),
                score});
        }
    }
    if (candidates.empty()) {
        return;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) {
                  return a.score > b.score;
              });

    std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> targets;
    targets.reserve(static_cast<std::size_t>(
        learning.options.map_pixel_outlier_targeted_contributor_max_pixels));
    auto have_target = [&](const auto &candidate) {
        return std::find_if(targets.begin(), targets.end(),
                            [&](const auto &target) {
                                return std::get<0>(target) == candidate.map_index &&
                                       std::get<1>(target) == candidate.row &&
                                       std::get<2>(target) == candidate.col;
                            }) != targets.end();
    };
    const std::size_t max_targets = static_cast<std::size_t>(
        learning.options.map_pixel_outlier_targeted_contributor_max_pixels);
    for (const auto &candidate : candidates) {
        if (targets.size() >= max_targets) {
            break;
        }
        if (have_target(candidate)) {
            continue;
        }
        targets.emplace_back(candidate.map_index, candidate.row, candidate.col);
    }
    if (targets.empty()) {
        return;
    }

    mb.set_contribution_targets(static_cast<Eigen::Index>(mb.signal.size()), targets);
    if (mb.contribution_diag_targeted) {
        mb.contribution_diag_enabled = true;
        logger->info(
            "map-pixel targeted contributor tracing enabled stage={} obsnum={} iter={} source_iter={} targets={}",
            stage_name, observation_identity.obsnum, iteration.fruit_iter, target_iter, targets.size());
    }
}

