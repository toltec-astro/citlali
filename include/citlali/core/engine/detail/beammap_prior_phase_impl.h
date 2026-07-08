#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

bool Beammap::is_beammap_locator_iter(Eigen::Index iter) const {
    const auto &phase_config = typed_config.beammap.phase_strategy;
    if (!phase_config.enabled) {
        return iter <= 0;
    }
    return iter == static_cast<Eigen::Index>(phase_config.locator_iter);
}

bool Beammap::is_beammap_measurement_iter(Eigen::Index iter) const {
    const auto &phase_config = typed_config.beammap.phase_strategy;
    if (!phase_config.enabled) {
        return iter > 0;
    }
    return iter >=
           static_cast<Eigen::Index>(phase_config.measurement_start_iter);
}

bool Beammap::is_beammap_first_measurement_iter(Eigen::Index iter) const {
    const auto &phase_config = typed_config.beammap.phase_strategy;
    if (!phase_config.enabled) {
        return iter == 1;
    }
    return iter ==
           static_cast<Eigen::Index>(phase_config.measurement_start_iter);
}

bool Beammap::has_completed_beammap_measurement_iter(Eigen::Index iter) const {
    const auto &phase_config = typed_config.beammap.phase_strategy;
    if (!phase_config.enabled) {
        return iter > 1;
    }
    return iter >
           static_cast<Eigen::Index>(phase_config.measurement_start_iter);
}

std::string Beammap::beammap_iter_phase_name(Eigen::Index iter) const {
    if (!typed_config.beammap.phase_strategy.enabled) {
        return "legacy";
    }
    if (is_beammap_locator_iter(iter)) {
        return "locator";
    }
    if (is_beammap_first_measurement_iter(iter)) {
        return "measurement_start";
    }
    if (is_beammap_measurement_iter(iter)) {
        return "measurement";
    }
    return "pre_measurement";
}

std::filesystem::path Beammap::resolve_soft_priors_filepath() const {
    namespace fs = std::filesystem;

    if (beammap_priors_filepath.empty() || beammap_priors_filepath == "null") {
        return {};
    }

    fs::path requested(beammap_priors_filepath);
    std::vector<fs::path> candidates;

    if (requested.is_absolute()) {
        candidates.push_back(requested);
    }
    else {
        candidates.push_back(requested);

        fs::path source_path(__FILE__);
        if (source_path.is_relative()) {
            source_path = fs::current_path() / source_path;
        }
        source_path = source_path.lexically_normal();
        fs::path repo_root = source_path;
        for (int i = 0; i < 5 && !repo_root.empty(); ++i) {
            repo_root = repo_root.parent_path();
        }
        if (!repo_root.empty()) {
            candidates.push_back(repo_root / requested);
        }
    }

    for (const auto &candidate : candidates) {
        try {
            if (fs::exists(candidate)) {
                return fs::absolute(candidate).lexically_normal();
            }
        }
        catch (const std::exception &) {
        }
    }

    return {};
}
