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
    if (!learning.map_pixel_target_state_required() ||
        iteration.fruit_iter <= 0 ||
        mb.signal.empty() ||
        mb.n_rows <= 0 ||
        mb.n_cols <= 0) {
        return;
    }

    const std::string producer = "mapdiag:" + stage_name;
    const auto resolved = learning.resolved_map_pixel_targets_for(
        observation_identity.obsnum, producer, iteration.fruit_iter);
    if (!resolved) {
        throw std::runtime_error(
            "required resolved map-pixel target state is unavailable for " +
            observation_identity.obsnum + " " + producer + " iteration " +
            std::to_string(iteration.fruit_iter));
    }
    const bool dimensions_match =
        (resolved->map_count < 0 ||
         resolved->map_count == static_cast<int>(mb.signal.size())) &&
        (resolved->n_rows < 0 || resolved->n_rows == mb.n_rows) &&
        (resolved->n_cols < 0 || resolved->n_cols == mb.n_cols);
    if (!dimensions_match ||
        (!resolved->targets.empty() &&
         (resolved->source_iter < 0 || resolved->map_count <= 0 ||
          resolved->n_rows <= 0 || resolved->n_cols <= 0))) {
        throw std::runtime_error(
            "resolved map-pixel target state is incompatible with the current map grid for " +
            observation_identity.obsnum + " " + producer);
    }
    if (resolved->targets.empty()) {
        return;
    }

    std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> targets;
    targets.reserve(resolved->targets.size());
    for (const auto &target : resolved->targets) {
        targets.emplace_back(
            static_cast<Eigen::Index>(target.map_index),
            static_cast<Eigen::Index>(target.row),
            static_cast<Eigen::Index>(target.col));
    }

    mb.set_contribution_targets(static_cast<Eigen::Index>(mb.signal.size()), targets);
    if (mb.contribution_diag_targeted) {
        mb.contribution_diag_enabled = true;
        logger->info(
            "map-pixel targeted contributor tracing enabled stage={} obsnum={} iter={} source_iter={} targets={}",
            stage_name, observation_identity.obsnum, iteration.fruit_iter,
            resolved->source_iter, targets.size());
    }
}
