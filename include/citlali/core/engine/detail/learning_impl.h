#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

template<typename CT>
void Engine::get_learning_config(CT &config) {
    ReductionLearningState::Options options;

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return missing_keys.size() == missing_before && invalid_keys.size() == invalid_before;
    };
    auto mirror_if_parsed = [&](auto &target, const auto &source,
                                std::size_t missing_before,
                                std::size_t invalid_before) {
        if (parsed_cleanly(missing_before, invalid_before)) {
            target = source;
        }
    };

    if (config.template has_typed<bool>(std::tuple{"timestream","learning","enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.enabled = options.enabled;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.diagnostics_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","diagnostics_enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.diagnostics_enabled =
                options.diagnostics_enabled;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","learn_iters"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.learn_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","learn_iters"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.learn_iters = options.learn_iters;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","apply_start_iter"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_start_iter, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_start_iter"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_start_iter =
                options.apply_start_iter;
        }
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","max_records_per_type"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.max_records_per_type, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","max_records_per_type"}, {}, {0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.max_records_per_type =
                options.max_records_per_type;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","apply_sample_masks_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_sample_masks_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_sample_masks_enabled"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_sample_masks_enabled =
                options.apply_sample_masks_enabled;
        }
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","apply_max_new_flagged_fraction"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.apply_max_new_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","apply_max_new_flagged_fraction"}, {}, {0.0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_timestream_config.learning.apply_max_new_flagged_fraction =
                options.apply_max_new_flagged_fraction;
        }
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_diagnostics_enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.diagnostics_enabled,
            options.map_pixel_outlier_diagnostics_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_contributor_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_contributor_diagnostics_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_contributor_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.contributor_diagnostics_enabled,
            options.map_pixel_outlier_contributor_diagnostics_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_diagnostics_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_diagnostics_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.targeted_contributor_diagnostics_enabled,
            options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_detector_exclusion_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.detector_exclusion_enabled,
            options.map_pixel_outlier_detector_exclusion_enabled,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_top_n"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_top_n, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_top_n"}, {}, {0});
        mirror_if_parsed(typed_timestream_config.learning.map_pixel_outlier.top_n,
                         options.map_pixel_outlier_top_n, missing_before,
                         invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_max_pixels"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_targeted_contributor_max_pixels,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_targeted_contributor_max_pixels"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.targeted_contributor_max_pixels,
            options.map_pixel_outlier_targeted_contributor_max_pixels,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_min_pixels"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_detector_exclusion_min_pixels,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_detector_exclusion_min_pixels"},
                         {}, {1});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.detector_exclusion_min_pixels,
            options.map_pixel_outlier_detector_exclusion_min_pixels,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_min_abs_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_min_abs_z, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_min_abs_z"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.min_abs_z,
            options.map_pixel_outlier_min_abs_z, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_min_n_eff"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_min_n_eff, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_min_n_eff"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.min_n_eff,
            options.map_pixel_outlier_min_n_eff, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","map_pixel_outlier_source_radius_arcsec"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.map_pixel_outlier_source_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","map_pixel_outlier_source_radius_arcsec"}, {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.map_pixel_outlier.source_radius_arcsec,
            options.map_pixel_outlier_source_radius_arcsec, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","busy_detector_exclusion_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.busy_detector_exclusion_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","busy_detector_exclusion_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.busy_detector.exclusion_enabled,
            options.busy_detector_exclusion_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_enabled"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_enabled,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_enabled"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.enabled,
            options.scan_network_pathology_enabled, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_rtc"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_rtc,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_rtc"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_rtc,
            options.scan_network_pathology_apply_pre_rtc, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_ptc"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_ptc,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_ptc"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_ptc,
            options.scan_network_pathology_apply_pre_ptc, missing_before,
            invalid_before);
    }
    if (config.template has_typed<bool>(std::tuple{"timestream","learning","scan_network_pathology_apply_pre_mapmaking"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_apply_pre_mapmaking,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_apply_pre_mapmaking"});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.apply_pre_mapmaking,
            options.scan_network_pathology_apply_pre_mapmaking, missing_before,
            invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_min_candidate_clusters"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_candidate_clusters,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_candidate_clusters"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_candidate_clusters,
            options.scan_network_pathology_min_candidate_clusters,
            missing_before, invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_min_candidate_events"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_candidate_events,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_candidate_events"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_candidate_events,
            options.scan_network_pathology_min_candidate_events, missing_before,
            invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_min_max_residual_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_min_max_residual_z,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_min_max_residual_z"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.min_max_residual_z,
            options.scan_network_pathology_min_max_residual_z, missing_before,
            invalid_before);
    }
    if (config.template has_typed<int>(std::tuple{"timestream","learning","scan_network_pathology_severe_candidate_events"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_severe_candidate_events,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_severe_candidate_events"},
                         {}, {0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.severe_candidate_events,
            options.scan_network_pathology_severe_candidate_events,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_severe_max_residual_z"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_severe_max_residual_z,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_severe_max_residual_z"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.severe_max_residual_z,
            options.scan_network_pathology_severe_max_residual_z,
            missing_before, invalid_before);
    }
    if (config.template has_typed<double>(std::tuple{"timestream","learning","scan_network_pathology_max_new_flagged_fraction"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, options.scan_network_pathology_max_new_flagged_fraction,
                         missing_keys, invalid_keys,
                         std::tuple{"timestream","learning","scan_network_pathology_max_new_flagged_fraction"},
                         {}, {0.0});
        mirror_if_parsed(
            typed_timestream_config.learning.scan_network_pathology.max_new_flagged_fraction,
            options.scan_network_pathology_max_new_flagged_fraction,
            missing_before, invalid_before);
    }

    reduction_learning.configure(options);
    const bool map_contribution_diag =
        reduction_learning.options.enabled &&
        reduction_learning.options.diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_contributor_diagnostics_enabled;
    omb.contribution_diag_enabled = map_contribution_diag;
    cmb.contribution_diag_enabled = map_contribution_diag;
    logger->info(
        "reduction learning state configured: enabled={} diagnostics_enabled={} "
        "learn_iters={} apply_start_iter={} max_records_per_type={} "
        "apply_sample_masks_enabled={} apply_max_new_flagged_fraction={:.4g} "
        "map_pixel_outliers(enabled={} contributors={} targeted_contributors={} detector_exclusion={} top_n={} target_max={} exclude_min_pixels={} min_abs_z={} min_n_eff={} source_radius_arcsec={}) "
        "busy_detector_exclusion_enabled={} scan_network_pathology(enabled={} pre_rtc={} pre_ptc={} pre_mapmaking={} min_clusters={} min_events={} min_resid_z={} severe_events={} severe_resid_z={} max_new_flagged_fraction={:.4g})",
        reduction_learning.options.enabled,
        reduction_learning.options.diagnostics_enabled,
        reduction_learning.options.learn_iters,
        reduction_learning.options.apply_start_iter,
        reduction_learning.options.max_records_per_type,
        reduction_learning.options.apply_sample_masks_enabled,
        reduction_learning.options.apply_max_new_flagged_fraction,
        reduction_learning.options.map_pixel_outlier_diagnostics_enabled,
        reduction_learning.options.map_pixel_outlier_contributor_diagnostics_enabled,
        reduction_learning.options.map_pixel_outlier_targeted_contributor_diagnostics_enabled,
        reduction_learning.options.map_pixel_outlier_detector_exclusion_enabled,
        reduction_learning.options.map_pixel_outlier_top_n,
        reduction_learning.options.map_pixel_outlier_targeted_contributor_max_pixels,
        reduction_learning.options.map_pixel_outlier_detector_exclusion_min_pixels,
        reduction_learning.options.map_pixel_outlier_min_abs_z,
        reduction_learning.options.map_pixel_outlier_min_n_eff,
        reduction_learning.options.map_pixel_outlier_source_radius_arcsec,
        reduction_learning.options.busy_detector_exclusion_enabled,
        reduction_learning.options.scan_network_pathology_enabled,
        reduction_learning.options.scan_network_pathology_apply_pre_rtc,
        reduction_learning.options.scan_network_pathology_apply_pre_ptc,
        reduction_learning.options.scan_network_pathology_apply_pre_mapmaking,
        reduction_learning.options.scan_network_pathology_min_candidate_clusters,
        reduction_learning.options.scan_network_pathology_min_candidate_events,
        reduction_learning.options.scan_network_pathology_min_max_residual_z,
        reduction_learning.options.scan_network_pathology_severe_candidate_events,
        reduction_learning.options.scan_network_pathology_severe_max_residual_z,
        reduction_learning.options.scan_network_pathology_max_new_flagged_fraction);
}

void Engine::configure_map_pixel_contribution_targets(mapmaking::MapBuffer &mb,
                                                      const std::string &stage_name) {
    const bool full_contribution_diag =
        reduction_learning.options.enabled &&
        reduction_learning.options.diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_diagnostics_enabled &&
        reduction_learning.options.map_pixel_outlier_contributor_diagnostics_enabled;

    mb.clear_contribution_targets();
    mb.contribution_diag_enabled = full_contribution_diag;

    if (full_contribution_diag) {
        return;
    }
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled() ||
        !reduction_learning.options.map_pixel_outlier_diagnostics_enabled ||
        !reduction_learning.options.map_pixel_outlier_targeted_contributor_diagnostics_enabled ||
        reduction_learning.options.map_pixel_outlier_targeted_contributor_max_pixels <= 0 ||
        fruit_iter <= 0 ||
        mb.signal.empty() ||
        mb.n_rows <= 0 ||
        mb.n_cols <= 0) {
        return;
    }

    const std::string producer = "mapdiag:" + stage_name;
    int target_iter = -1;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.map_pixel_outliers) {
            if (record.obsnum == obsnum &&
                record.producer == producer &&
                record.iter >= 0 &&
                record.iter < fruit_iter &&
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
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.map_pixel_outliers) {
            if (record.obsnum != obsnum ||
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
        reduction_learning.options.map_pixel_outlier_targeted_contributor_max_pixels));
    auto have_target = [&](const auto &candidate) {
        return std::find_if(targets.begin(), targets.end(),
                            [&](const auto &target) {
                                return std::get<0>(target) == candidate.map_index &&
                                       std::get<1>(target) == candidate.row &&
                                       std::get<2>(target) == candidate.col;
                            }) != targets.end();
    };
    const std::size_t max_targets = static_cast<std::size_t>(
        reduction_learning.options.map_pixel_outlier_targeted_contributor_max_pixels);
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
            stage_name, obsnum, fruit_iter, target_iter, targets.size());
    }
}

template <class apt_t>
static Eigen::Index citlali_learning_find_det_by_uid(const apt_t &apt, int uid) {
    if (uid == timestream::kTransientFillInt || uid < 0) {
        return -1;
    }
    const auto uid_it = apt.find("uid");
    if (uid_it == apt.end()) {
        return static_cast<Eigen::Index>(uid);
    }
    for (Eigen::Index i = 0; i < uid_it->second.size(); ++i) {
        if (std::isfinite(uid_it->second(i)) &&
            static_cast<int>(std::llround(uid_it->second(i))) == uid) {
            return i;
        }
    }
    return -1;
}

template <class apt_t>
static int citlali_learning_apt_int(const apt_t &apt, const std::string &key,
                                    Eigen::Index det, int fallback) {
    const auto it = apt.find(key);
    if (it == apt.end() || det < 0 || det >= it->second.size() ||
        !std::isfinite(it->second(det))) {
        return fallback;
    }
    return static_cast<int>(std::llround(it->second(det)));
}

template <class apt_t>
static int citlali_learning_array_for_nw(const apt_t &apt, int nw, int fallback) {
    const auto nw_it = apt.find("nw");
    const auto array_it = apt.find("array");
    if (nw_it == apt.end() || array_it == apt.end()) {
        return fallback;
    }
    const Eigen::Index n =
        std::min<Eigen::Index>(nw_it->second.size(), array_it->second.size());
    for (Eigen::Index det = 0; det < n; ++det) {
        if (!std::isfinite(nw_it->second(det)) ||
            !std::isfinite(array_it->second(det))) {
            continue;
        }
        if (static_cast<int>(std::llround(nw_it->second(det))) == nw) {
            return static_cast<int>(std::llround(array_it->second(det)));
        }
    }
    return fallback;
}

template <class rtc_t, class calib_t>
void Engine::apply_learned_rtc_sample_masks(rtc_t &rtcdata, calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        rtcdata, calib_scan, "pre_rtc_detector_exclusion", true, false,
        true, true);
    apply_learned_sample_masks(
        rtcdata, calib_scan, true, "pre_rtc",
        rtcproc.despiker.source_protection_enabled,
        rtcproc.despiker.source_protection_radius_arcsec);
}

template <class ptc_t, class calib_t>
void Engine::apply_learned_ptc_sample_masks(ptc_t &ptcdata, calib_t &calib_scan) {
    apply_learned_sample_masks(
        ptcdata, calib_scan, false, "pre_ptc",
        ptcproc.second_pass_local.source_protection_enabled,
        ptcproc.second_pass_local.source_protection_radius_arcsec);
}

template <class ptc_t, class calib_t>
void Engine::apply_learned_ptc_detector_exclusions(ptc_t &ptcdata,
                                                   calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        ptcdata, calib_scan, "pre_ptc_detector_exclusion", false, true,
        true, true);
}

template <class tc_t, class calib_t>
void Engine::apply_learned_mapmaking_detector_exclusions(tc_t &tcdata,
                                                         calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        tcdata, calib_scan, "pre_mapmaking_detector_exclusion", false, false,
        false, true);
}

template <class tc_t, class calib_t>
void Engine::apply_learned_detector_exclusions(tc_t &tcdata,
                                               calib_t &calib_scan,
                                               const std::string &stage,
                                               bool pre_rtc,
                                               bool update_apt_flags,
                                               bool include_detector_records,
                                               bool include_network_records) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.apply_active()) {
        return;
    }
    if (tcdata.flags.data.rows() <= 0 || tcdata.flags.data.cols() <= 0) {
        return;
    }

    const bool mapdiag_detector_exclusion =
        include_detector_records &&
        reduction_learning.options.map_pixel_outlier_detector_exclusion_enabled;
    const bool busy_detector_exclusion =
        include_detector_records &&
        reduction_learning.options.busy_detector_exclusion_enabled;
    const bool network_exclusion =
        include_network_records &&
        reduction_learning.options.scan_network_pathology_enabled &&
        (stage == "pre_mapmaking_detector_exclusion"
             ? reduction_learning.options.scan_network_pathology_apply_pre_mapmaking
             : ((!pre_rtc && reduction_learning.options.scan_network_pathology_apply_pre_ptc) ||
                (pre_rtc && reduction_learning.options.scan_network_pathology_apply_pre_rtc)));
    if (!mapdiag_detector_exclusion && !busy_detector_exclusion &&
        !network_exclusion) {
        return;
    }

    const int scan_id = static_cast<int>(tcdata.index.data);
    std::vector<ReductionLearningState::DetectorPenalty> records;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.detector_penalties) {
            if (record.obsnum != obsnum ||
                !record.scan_local ||
                record.scan != scan_id ||
                record.iter < 0 ||
                record.iter >= fruit_iter ||
                !std::isfinite(record.factor) ||
                record.factor > 0.0) {
                continue;
            }
            const bool is_mapdiag_detector =
                mapdiag_detector_exclusion &&
                record.uid >= 0 &&
                record.reason == "map_pixel_outlier_detector_dominance" &&
                record.producer.rfind("mapdiag:", 0) == 0;
            const bool is_busy_detector =
                busy_detector_exclusion &&
                record.uid >= 0 &&
                record.reason == "busy_vetoed_residual" &&
                record.producer == "ptc_second_pass";
            const bool is_network =
                network_exclusion &&
                record.uid < 0 &&
                record.nw >= 0 &&
                record.reason == "busy_network_pathology" &&
                record.producer == "ptc_second_pass";
            if (is_mapdiag_detector || is_busy_detector || is_network) {
                records.push_back(record);
            }
        }
    }
    if (records.empty()) {
        return;
    }

    ReductionLearningState::LearnedMaskApplicationSummary summary;
    summary.obsnum = obsnum;
    summary.producer = "learning_state";
    summary.stage = stage;
    summary.iter = fruit_iter;
    summary.scan = scan_id;
    summary.candidate_records = static_cast<int>(records.size());
    const bool has_network_record = std::any_of(
        records.begin(), records.end(),
        [](const auto &record) {
            return record.uid < 0 &&
                   record.reason == "busy_network_pathology";
        });
    summary.max_new_flagged_fraction = has_network_record
        ? reduction_learning.options.scan_network_pathology_max_new_flagged_fraction
        : reduction_learning.options.apply_max_new_flagged_fraction;

    const Eigen::Index n_pts = tcdata.flags.data.rows();
    const Eigen::Index n_dets = tcdata.flags.data.cols();
    std::set<Eigen::Index> proposed_dets;
    std::set<Eigen::Index> network_proposed_dets;
    for (const auto &record : records) {
        if (record.uid >= 0) {
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, record.uid);
            if (det < 0 || det >= n_dets) {
                ++summary.invalid_records;
                continue;
            }
            ++summary.matched_records;
            proposed_dets.insert(det);
        }
        else if (record.nw >= 0) {
            bool matched_network = false;
            for (Eigen::Index det = 0; det < n_dets; ++det) {
                const int det_nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                if (det_nw == record.nw) {
                    matched_network = true;
                    proposed_dets.insert(det);
                    network_proposed_dets.insert(det);
                }
            }
            if (matched_network) {
                ++summary.matched_records;
            }
            else {
                ++summary.invalid_records;
            }
        }
    }
    if (proposed_dets.empty()) {
        reduction_learning.record_learned_mask_application(summary);
        return;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_mask;
    bool have_network_source_protection = false;
    if (!network_proposed_dets.empty() &&
        stage == "pre_mapmaking_detector_exclusion") {
        const double radius_arcsec =
            std::max(20.0, ptcproc.second_pass_local.source_protection_radius_arcsec);
        auto [mask, source_info] = engine_utils::calc_source_protection_mask(
            tcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius", radius_arcsec);
        (void) source_info;
        source_mask = std::move(mask);
        have_network_source_protection =
            source_mask.rows() == n_pts && source_mask.cols() == n_dets;
        if (!have_network_source_protection) {
            logger->warn(
                "learned {} source-protection mask shape mismatch scan {}: mask=({}, {}) flags=({}, {})",
                stage, scan_id, source_mask.rows(), source_mask.cols(), n_pts, n_dets);
        }
    }

    for (const auto det : proposed_dets) {
        for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
            if (have_network_source_protection &&
                network_proposed_dets.find(det) != network_proposed_dets.end() &&
                source_mask(sample, det)) {
                ++summary.source_protected_samples;
                continue;
            }
            ++summary.proposed_samples;
            if (tcdata.flags.data(sample, det)) {
                ++summary.already_flagged_samples;
            }
            else {
                ++summary.newly_flagged_samples;
            }
        }
    }

    const double denom = static_cast<double>(std::max<Eigen::Index>(1, n_pts * n_dets));
    summary.newly_flagged_fraction =
        static_cast<double>(summary.newly_flagged_samples) / denom;
    const bool over_cap =
        summary.max_new_flagged_fraction > 0.0 &&
        summary.newly_flagged_fraction >
            summary.max_new_flagged_fraction;
    if (!over_cap) {
        auto flag_it = calib_scan.apt.find("flag");
        std::set<Eigen::Index> apt_flag_dets;
        Eigen::Index apt_flag_preserved = 0;
        if (update_apt_flags &&
            flag_it != calib_scan.apt.end() &&
            flag_it->second.size() > 0) {
            std::map<int, Eigen::Index> unflagged_by_nw;
            std::map<int, Eigen::Index> unflagged_by_array;
            const Eigen::Index n_apt =
                std::min<Eigen::Index>(n_dets, flag_it->second.size());
            for (Eigen::Index det = 0; det < n_apt; ++det) {
                if (flag_it->second(det) != 0.0) {
                    continue;
                }
                const int nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                const int array =
                    citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
                if (nw >= 0) {
                    ++unflagged_by_nw[nw];
                }
                if (array >= 0) {
                    ++unflagged_by_array[array];
                }
            }

            for (const auto det : proposed_dets) {
                if (network_proposed_dets.find(det) != network_proposed_dets.end()) {
                    continue;
                }
                if (det < 0 ||
                    det >= n_apt ||
                    flag_it->second(det) != 0.0) {
                    continue;
                }
                const int nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                const int array =
                    citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
                const bool preserves_nw =
                    nw < 0 ||
                    unflagged_by_nw.find(nw) == unflagged_by_nw.end() ||
                    unflagged_by_nw[nw] > 1;
                const bool preserves_array =
                    array < 0 ||
                    unflagged_by_array.find(array) == unflagged_by_array.end() ||
                    unflagged_by_array[array] > 1;
                if (!preserves_nw || !preserves_array) {
                    ++apt_flag_preserved;
                    continue;
                }
                apt_flag_dets.insert(det);
                if (nw >= 0) {
                    --unflagged_by_nw[nw];
                }
                if (array >= 0) {
                    --unflagged_by_array[array];
                }
            }
        }

        for (const auto det : proposed_dets) {
            if (have_network_source_protection &&
                network_proposed_dets.find(det) != network_proposed_dets.end()) {
                for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                    if (!source_mask(sample, det)) {
                        tcdata.flags.data(sample, det) = true;
                    }
                }
            }
            else {
                tcdata.flags.data.col(det).setOnes();
            }
            if (apt_flag_dets.find(det) != apt_flag_dets.end()) {
                flag_it->second(det) = 1.0;
            }
        }
        summary.applied = true;
        if (apt_flag_preserved > 0) {
            logger->info(
                "learned {} preserved {} scan-local APT flags in scan {} iter {} to keep nw/array groups valid",
                stage, apt_flag_preserved, scan_id + 1, fruit_iter);
        }
    }

    reduction_learning.record_learned_mask_application(summary);
    if (over_cap) {
        logger->warn(
            "learned {} rejected scan {} iter {}: candidates={} matched={} dets={} newly_flagged={} newly_flagged_fraction={:.4f} cap={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, proposed_dets.size(),
            summary.newly_flagged_samples, summary.newly_flagged_fraction,
            summary.max_new_flagged_fraction);
    }
    else {
        logger->info(
            "learned {} applied scan {} iter {}: candidates={} matched={} dets={} newly_flagged={} already_flagged={} newly_flagged_fraction={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, proposed_dets.size(),
            summary.newly_flagged_samples, summary.already_flagged_samples,
            summary.newly_flagged_fraction);
    }
}

template <class tc_t, class calib_t>
void Engine::apply_learned_sample_masks(tc_t &tcdata, calib_t &calib_scan,
                                        bool apply_pre_rtc,
                                        const std::string &stage,
                                        bool source_protection_enabled,
                                        double source_protection_radius_arcsec) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.options.apply_sample_masks_enabled ||
        !reduction_learning.apply_active()) {
        return;
    }
    if (tcdata.flags.data.rows() <= 0 || tcdata.flags.data.cols() <= 0) {
        return;
    }

    const int scan_id = static_cast<int>(tcdata.index.data);
    std::vector<ReductionLearningState::LearnedSampleMask> records;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.learned_sample_masks) {
            if (record.obsnum == obsnum &&
                record.scan == scan_id &&
                record.iter >= 0 &&
                record.iter < fruit_iter &&
                record.apply_pre_rtc == apply_pre_rtc) {
                records.push_back(record);
            }
        }
    }
    if (records.empty()) {
        return;
    }

    ReductionLearningState::LearnedMaskApplicationSummary summary;
    summary.obsnum = obsnum;
    summary.producer = "learning_state";
    summary.stage = stage;
    summary.iter = fruit_iter;
    summary.scan = scan_id;
    summary.candidate_records = static_cast<int>(records.size());
    summary.max_new_flagged_fraction =
        reduction_learning.options.apply_max_new_flagged_fraction;

    const Eigen::Index n_pts = tcdata.flags.data.rows();
    const Eigen::Index n_dets = tcdata.flags.data.cols();
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> proposed(n_pts, n_dets);
    proposed.setZero();

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_mask;
    bool have_source_protection = false;
    if (source_protection_enabled && source_protection_radius_arcsec > 0.0) {
        auto [mask, source_info] = engine_utils::calc_source_protection_mask(
            tcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius", source_protection_radius_arcsec);
        (void) source_info;
        source_mask = std::move(mask);
        have_source_protection =
            source_mask.rows() == n_pts && source_mask.cols() == n_dets;
        if (source_protection_enabled && !have_source_protection) {
            logger->warn(
                "learned mask {} source-protection mask shape mismatch scan {}: mask=({}, {}) flags=({}, {})",
                stage, scan_id, source_mask.rows(), source_mask.cols(), n_pts, n_dets);
        }
    }

    for (const auto &record : records) {
        if (record.source_protected) {
            ++summary.invalid_records;
            continue;
        }
        const Eigen::Index det = citlali_learning_find_det_by_uid(calib_scan.apt, record.uid);
        const long long raw_start = apply_pre_rtc ? record.raw_start : record.ptc_start;
        const long long raw_stop = apply_pre_rtc ? record.raw_stop : record.ptc_stop;
        if (det < 0 || det >= n_dets || raw_start < 0 || raw_stop < raw_start ||
            raw_stop < 0 || raw_start >= n_pts) {
            ++summary.invalid_records;
            continue;
        }
        const Eigen::Index start =
            std::max<Eigen::Index>(0, static_cast<Eigen::Index>(raw_start));
        const Eigen::Index stop =
            std::min<Eigen::Index>(n_pts - 1, static_cast<Eigen::Index>(raw_stop));
        if (stop < start) {
            ++summary.invalid_records;
            continue;
        }

        ++summary.matched_records;
        for (Eigen::Index sample = start; sample <= stop; ++sample) {
            if (have_source_protection && source_mask(sample, det)) {
                ++summary.source_protected_samples;
                continue;
            }
            if (!proposed(sample, det)) {
                proposed(sample, det) = true;
                ++summary.proposed_samples;
            }
        }
    }

    if (summary.proposed_samples > 0) {
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                if (!proposed(sample, det)) {
                    continue;
                }
                if (tcdata.flags.data(sample, det)) {
                    ++summary.already_flagged_samples;
                }
                else {
                    ++summary.newly_flagged_samples;
                }
            }
        }
    }

    const double denom = static_cast<double>(std::max<Eigen::Index>(1, n_pts * n_dets));
    summary.newly_flagged_fraction =
        static_cast<double>(summary.newly_flagged_samples) / denom;
    const bool over_cap =
        reduction_learning.options.apply_max_new_flagged_fraction > 0.0 &&
        summary.newly_flagged_fraction >
            reduction_learning.options.apply_max_new_flagged_fraction;
    if (!over_cap) {
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                if (proposed(sample, det)) {
                    tcdata.flags.data(sample, det) = true;
                }
            }
        }
        summary.applied = true;
    }

    reduction_learning.record_learned_mask_application(summary);
    if (over_cap) {
        logger->warn(
            "learned {} sample-mask application rejected scan {} iter {}: candidates={} matched={} proposed={} newly_flagged={} newly_flagged_fraction={:.4f} cap={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, summary.proposed_samples,
            summary.newly_flagged_samples, summary.newly_flagged_fraction,
            reduction_learning.options.apply_max_new_flagged_fraction);
    }
    else if (summary.proposed_samples > 0) {
        logger->info(
            "learned {} sample masks applied scan {} iter {}: candidates={} matched={} proposed={} newly_flagged={} already_flagged={} source_protected={} newly_flagged_fraction={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, summary.proposed_samples,
            summary.newly_flagged_samples, summary.already_flagged_samples,
            summary.source_protected_samples, summary.newly_flagged_fraction);
    }
}

template <class rtc_t, class ptc_t, class calib_t>
void Engine::collect_rtc_learning_diagnostics(rtc_t &rtcdata, ptc_t &ptcdata,
                                              calib_t &calib_scan,
                                              const std::vector<timestream::RTCProc::RTCDetectorDiagSummary> &det_summary) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled()) {
        return;
    }

    const auto scan_id = ptcdata.index.data;
    if (det_summary.empty()) {
        return;
    }

    const auto rtc_source_summary =
        rtcproc.snapshot_source_protection_diag_summary(scan_id);
    if (rtc_source_summary.enabled) {
        ReductionLearningState::SourceProtectionSummary source_summary;
        source_summary.obsnum = obsnum;
        source_summary.producer = "rtc_despike";
        source_summary.mode = "map_center_radius";
        source_summary.iter = fruit_iter;
        source_summary.scan = static_cast<int>(scan_id);
        source_summary.protected_samples = rtc_source_summary.protected_samples;
        source_summary.total_samples = rtc_source_summary.total_samples;
        source_summary.radius_arcsec = rtc_source_summary.radius_arcsec;
        reduction_learning.record_source_protection_summary(std::move(source_summary));
    }

    auto record_event = [&](const auto &event, Eigen::Index det,
                            const std::string &reason) {
        const auto uid_it = calib_scan.apt.find("uid");
        if (!event.valid() || !event.accepted || uid_it == calib_scan.apt.end() ||
            det < 0 || det >= uid_it->second.size()) {
            return;
        }
        ReductionLearningState::LearnedSampleMask record;
        record.obsnum = obsnum;
        record.producer = "rtc_despike";
        record.reason = reason;
        record.iter = fruit_iter;
        record.scan = static_cast<int>(scan_id);
        record.uid = citlali_learning_apt_int(calib_scan.apt, "uid", det,
                                              static_cast<int>(det));
        record.nw = citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
        record.array = citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
        record.raw_start = event.start_sample;
        record.raw_stop = event.end_sample;
        record.score = event.score;
        record.z = event.score;
        record.confidence = 1.0;
        record.source_protected = false;
        record.apply_pre_rtc = true;
        reduction_learning.record_learned_sample_mask(std::move(record));
    };

    for (const auto &row : det_summary) {
        const Eigen::Index det = row.det;
        record_event(row.local_raw_event, det, "local_raw_accepted");
        record_event(row.local_delta_event, det, "local_delta_accepted");
    }
}

template <class ptc_t, class calib_t>
void Engine::collect_ptc_learning_diagnostics(
    ptc_t &ptcdata, calib_t &calib_scan,
    const std::vector<timestream::PTCProc::SecondPassDiagSummary> &second_pass_summary,
    const std::vector<timestream::PTCProc::HighWeightDiagSummary> &high_weight_summary) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled()) {
        return;
    }

    const auto scan_id = ptcdata.index.data;

    if (ptcproc.second_pass_local.source_protection_enabled) {
        ReductionLearningState::SourceProtectionSummary source_summary;
        source_summary.obsnum = obsnum;
        source_summary.producer = "ptc_second_pass";
        source_summary.mode = "map_center_radius";
        source_summary.iter = fruit_iter;
        source_summary.scan = static_cast<int>(scan_id);
        source_summary.total_samples =
            static_cast<int>(ptcdata.scans.data.rows() * ptcdata.scans.data.cols());
        source_summary.radius_arcsec =
            ptcproc.second_pass_local.source_protection_radius_arcsec;
        auto [source_mask, source_info] = engine_utils::calc_source_protection_mask(
            ptcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius",
            ptcproc.second_pass_local.source_protection_radius_arcsec);
        (void) source_mask;
        source_summary.protected_samples =
            static_cast<int>(source_info.protected_samples);
        reduction_learning.record_source_protection_summary(std::move(source_summary));
    }

    for (const auto &summary : high_weight_summary) {
        ReductionLearningState::HighWeightDetector record;
        record.obsnum = obsnum;
        record.grouping = summary.grouping;
        record.reason = summary.reason;
        record.iter = fruit_iter;
        record.scan = static_cast<int>(scan_id);
        record.uid = summary.uid;
        record.nw = static_cast<int>(summary.nw);
        record.array = static_cast<int>(summary.array);
        record.weight = summary.approximate_weight;
        record.final_weight = summary.final_weight;
        record.group_median = summary.group_median_weight;
        record.robust_z = summary.robust_z;
        record.cap = summary.applied_cap;
        record.validation_factor = summary.validation_factor;
        record.cap_recommended = summary.cap_recommended;
        record.cap_applied = summary.cap_applied;
        record.validated = summary.validated;
        reduction_learning.record_high_weight_detector(std::move(record));
    }

    if (second_pass_summary.empty()) {
        return;
    }

    for (const auto &summary : second_pass_summary) {
        const bool has_candidate = summary.n_candidate_clusters > 0 ||
                                   summary.n_candidate_events > 0;
        const bool has_residual =
            std::isfinite(summary.max_unflagged_residual_z) &&
            summary.max_unflagged_residual_uid != timestream::kTransientFillInt;
        const bool selective_acceptance_recommended =
            summary.busy_network_vetoed &&
            ((std::isfinite(summary.top_candidate_cluster_peak_score) &&
              summary.top_candidate_cluster_peak_score >=
                  ptcproc.second_pass_local.high_score_cluster_override) ||
             (std::isfinite(summary.max_unflagged_residual_z) &&
              summary.max_unflagged_residual_z >=
                  ptcproc.second_pass_local.high_score_event_override));
        if (has_candidate || has_residual || summary.busy_network_vetoed) {
            ReductionLearningState::BusyNetworkSummary record;
            record.obsnum = obsnum;
            record.producer = "ptc_second_pass";
            record.reason = summary.busy_network_vetoed
                ? "busy_network_vetoed"
                : "candidate_or_residual";
            record.iter = fruit_iter;
            record.scan = static_cast<int>(scan_id);
            record.nw = static_cast<int>(summary.nw);
            record.n_candidate_clusters =
                static_cast<int>(summary.n_candidate_clusters);
            record.n_candidate_events =
                static_cast<int>(summary.n_candidate_events);
            record.n_accepted_clusters =
                static_cast<int>(summary.n_accepted_clusters);
            record.n_accepted_events =
                static_cast<int>(summary.n_accepted_events);
            record.n_rejected_clusters =
                static_cast<int>(summary.n_rejected_clusters);
            record.n_rejected_events =
                static_cast<int>(summary.n_rejected_events);
            record.n_source_protected_clusters =
                static_cast<int>(summary.n_source_protected_clusters);
            record.n_source_protected_events =
                static_cast<int>(summary.n_source_protected_events);
            record.max_unflagged_residual_uid = summary.max_unflagged_residual_uid;
            record.top_candidate_sample = summary.top_candidate_cluster_sample;
            record.top_candidate_score = summary.top_candidate_cluster_peak_score;
            record.max_unflagged_residual_z = summary.max_unflagged_residual_z;
            record.busy_vetoed = summary.busy_network_vetoed;
            record.selective_acceptance_recommended = selective_acceptance_recommended;
            reduction_learning.record_busy_network_summary(std::move(record));
        }

        if (reduction_learning.options.scan_network_pathology_enabled &&
            summary.nw >= 0) {
            const int off_source_candidate_events = std::max<Eigen::Index>(
                0, summary.n_candidate_events - summary.n_source_protected_events);
            const double max_residual_z = std::isfinite(summary.max_unflagged_residual_z)
                ? summary.max_unflagged_residual_z
                : 0.0;
            const bool busy_pathology =
                summary.busy_network_vetoed &&
                summary.n_candidate_clusters >=
                    reduction_learning.options.scan_network_pathology_min_candidate_clusters &&
                off_source_candidate_events >=
                    reduction_learning.options.scan_network_pathology_min_candidate_events &&
                max_residual_z >=
                    reduction_learning.options.scan_network_pathology_min_max_residual_z;
            const bool severe_pathology =
                off_source_candidate_events >=
                    reduction_learning.options.scan_network_pathology_severe_candidate_events &&
                max_residual_z >=
                    reduction_learning.options.scan_network_pathology_severe_max_residual_z;
            if (busy_pathology || severe_pathology) {
                ReductionLearningState::DetectorPenalty penalty;
                penalty.obsnum = obsnum;
                penalty.producer = "ptc_second_pass";
                penalty.reason = "busy_network_pathology";
                penalty.iter = fruit_iter;
                penalty.scan = static_cast<int>(scan_id);
                penalty.uid = -1;
                penalty.nw = static_cast<int>(summary.nw);
                penalty.array = citlali_learning_array_for_nw(
                    calib_scan.apt, penalty.nw, -1);
                penalty.factor = 0.0;
                penalty.score = std::max(
                    max_residual_z,
                    std::isfinite(summary.top_candidate_cluster_peak_score)
                        ? summary.top_candidate_cluster_peak_score
                        : 0.0);
                penalty.scan_local = true;
                reduction_learning.record_detector_penalty(std::move(penalty));
            }
        }

        for (const auto &event : summary.candidate_events) {
            if (event.uid == timestream::kTransientFillInt ||
                event.start_sample < 0 ||
                event.end_sample < event.start_sample) {
                continue;
            }
            if (!event.accepted || event.source_protected) {
                continue;
            }
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, event.uid);
            ReductionLearningState::LearnedSampleMask candidate_record;
            candidate_record.obsnum = obsnum;
            candidate_record.producer = "ptc_second_pass";
            candidate_record.reason = event.busy_network_vetoed
                ? "busy_selective_accepted_event"
                : "candidate_event";
            candidate_record.iter = fruit_iter;
            candidate_record.scan = static_cast<int>(scan_id);
            candidate_record.uid = event.uid;
            candidate_record.nw = static_cast<int>(summary.nw);
            candidate_record.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            candidate_record.ptc_start = event.start_sample;
            candidate_record.ptc_stop = event.end_sample;
            candidate_record.score = event.score;
            candidate_record.z = event.score;
            candidate_record.value = event.cluster_score;
            candidate_record.confidence = event.busy_network_vetoed ? 0.8 : 1.0;
            candidate_record.source_protected = event.source_protected;
            candidate_record.apply_pre_rtc = false;
            reduction_learning.record_learned_sample_mask(std::move(candidate_record));
        }

        if (summary.top_event.valid() && summary.top_event.accepted &&
            summary.top_event_uid != timestream::kTransientFillInt) {
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, summary.top_event_uid);
            ReductionLearningState::LearnedSampleMask sample_record;
            sample_record.obsnum = obsnum;
            sample_record.producer = "ptc_second_pass";
            sample_record.reason = "accepted_event";
            sample_record.iter = fruit_iter;
            sample_record.scan = static_cast<int>(scan_id);
            sample_record.uid = summary.top_event_uid;
            sample_record.nw = static_cast<int>(summary.nw);
            sample_record.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            sample_record.ptc_start = summary.top_event.start_sample;
            sample_record.ptc_stop = summary.top_event.end_sample;
            sample_record.score = summary.top_event.score;
            sample_record.z = summary.top_event.score;
            sample_record.confidence = 1.0;
            sample_record.source_protected = false;
            sample_record.apply_pre_rtc = false;
            reduction_learning.record_learned_sample_mask(std::move(sample_record));
        }

        if (summary.busy_network_vetoed && has_residual &&
            summary.max_unflagged_residual_z >=
                ptcproc.second_pass_local.high_score_event_override) {
            const Eigen::Index det = citlali_learning_find_det_by_uid(
                calib_scan.apt, summary.max_unflagged_residual_uid);
            ReductionLearningState::DetectorPenalty penalty;
            penalty.obsnum = obsnum;
            penalty.producer = "ptc_second_pass";
            penalty.reason = "busy_vetoed_residual";
            penalty.iter = fruit_iter;
            penalty.scan = static_cast<int>(scan_id);
            penalty.uid = summary.max_unflagged_residual_uid;
            penalty.nw = static_cast<int>(summary.nw);
            penalty.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            penalty.factor = 0.0;
            penalty.score = summary.max_unflagged_residual_z;
            penalty.scan_local = true;
            reduction_learning.record_detector_penalty(std::move(penalty));
        }
    }
}

inline void Engine::write_learning_summary() {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled() ||
        redu_dir_name.empty()) {
        return;
    }

    std::ostringstream filename;
    filename << redu_dir_name << "/learning_iter_" << fruit_iter << ".csv";
    std::ofstream out(filename.str());
    if (!out) {
        logger->warn("failed to open learning summary output {}", filename.str());
        return;
    }

    auto csv = [](const std::string &s) {
        std::string escaped = "\"";
        for (const char ch : s) {
            if (ch == '"') {
                escaped += "\"\"";
            }
            else {
                escaped += ch;
            }
        }
        escaped += "\"";
        return escaped;
    };

    enum {
        ColRecordType,
        ColIter,
        ColObsnum,
        ColProducer,
        ColReason,
        ColScan,
        ColUid,
        ColNw,
        ColArray,
        ColRawStart,
        ColRawStop,
        ColPtcStart,
        ColPtcStop,
        ColScore,
        ColZ,
        ColValue,
        ColConfidence,
        ColSourceDistanceArcsec,
        ColSourceProtected,
        ColApplyPreRtc,
        ColCandidateClusters,
        ColCandidateEvents,
        ColAcceptedClusters,
        ColAcceptedEvents,
        ColRejectedClusters,
        ColRejectedEvents,
        ColSourceProtectedClusters,
        ColSourceProtectedEvents,
        ColMaxResidualUid,
        ColTopCandidateSample,
        ColTopCandidateScore,
        ColMaxResidualZ,
        ColBusyVetoed,
        ColSelectiveAcceptanceRecommended,
        ColFactor,
        ColScanLocal,
        ColProtectedSamples,
        ColTotalSamples,
        ColRadiusArcsec,
        ColSupportNpix,
        ColApplicationStage,
        ColCandidateRecords,
        ColMatchedRecords,
        ColInvalidRecords,
        ColProposedSamples,
        ColNewlyFlaggedSamples,
        ColAlreadyFlaggedSamples,
        ColSourceProtectedSamples,
        ColNewlyFlaggedFraction,
        ColMaxNewFlaggedFraction,
        ColApplied,
        ColGrouping,
        ColWeight,
        ColFinalWeight,
        ColGroupMedian,
        ColRobustZ,
        ColCap,
        ColValidationFactor,
        ColCapRecommended,
        ColCapApplied,
        ColValidated,
        ColMapIndex,
        ColRow,
        ColCol,
        ColSample,
        ColNEff,
        ColLeaveOneOutZ,
        ColCount
    };

    const std::vector<std::string> header = {
        "record_type", "iter", "obsnum", "producer", "reason", "scan", "uid",
        "nw", "array", "raw_start", "raw_stop", "ptc_start", "ptc_stop",
        "score", "z", "value", "confidence", "source_distance_arcsec",
        "source_protected", "apply_pre_rtc", "n_candidate_clusters",
        "n_candidate_events", "n_accepted_clusters", "n_accepted_events",
        "n_rejected_clusters", "n_rejected_events",
        "n_source_protected_clusters", "n_source_protected_events",
        "max_unflagged_residual_uid", "top_candidate_sample",
        "top_candidate_score", "max_unflagged_residual_z", "busy_vetoed",
        "selective_acceptance_recommended", "factor", "scan_local",
        "protected_samples", "total_samples", "radius_arcsec", "support_npix",
        "application_stage", "candidate_records", "matched_records",
        "invalid_records", "proposed_samples", "newly_flagged_samples",
        "already_flagged_samples", "source_protected_samples",
        "newly_flagged_fraction", "max_new_flagged_fraction", "applied",
        "grouping", "weight", "final_weight", "group_median", "robust_z",
        "cap", "validation_factor", "cap_recommended", "cap_applied",
        "validated", "map_index", "row", "col", "sample", "n_eff",
        "leave_one_out_z"
    };

    auto text = [](const auto &value) {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    };

    auto write_row = [&](const std::vector<std::string> &row) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i > 0) {
                out << ',';
            }
            out << row[i];
        }
        out << '\n';
    };

    auto new_row = [&]() {
        return std::vector<std::string>(ColCount);
    };

    auto write_common_header = [&]() {
        write_row(header);
    };

    auto write_base = [&](std::vector<std::string> &row,
                          const std::string &record_type, int iter,
                          const std::string &obsnum_value,
                          const std::string &producer,
                          const std::string &reason, int scan, int uid,
                          int nw, int array) {
        row[ColRecordType] = csv(record_type);
        row[ColIter] = text(iter);
        row[ColObsnum] = csv(obsnum_value);
        row[ColProducer] = csv(producer);
        row[ColReason] = csv(reason);
        row[ColScan] = text(scan);
        row[ColUid] = text(uid);
        row[ColNw] = text(nw);
        row[ColArray] = text(array);
    };

    std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
    write_common_header();

    for (const auto &record : reduction_learning.learned_sample_masks) {
        auto row = new_row();
        write_base(row, "sample_mask", record.iter, record.obsnum, record.producer,
                   record.reason, record.scan, record.uid, record.nw, record.array);
        row[ColRawStart] = text(record.raw_start);
        row[ColRawStop] = text(record.raw_stop);
        row[ColPtcStart] = text(record.ptc_start);
        row[ColPtcStop] = text(record.ptc_stop);
        row[ColScore] = text(record.score);
        row[ColZ] = text(record.z);
        row[ColValue] = text(record.value);
        row[ColConfidence] = text(record.confidence);
        row[ColSourceDistanceArcsec] = text(record.source_distance_arcsec);
        row[ColSourceProtected] = text(record.source_protected ? 1 : 0);
        row[ColApplyPreRtc] = text(record.apply_pre_rtc ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.busy_network_summaries) {
        auto row = new_row();
        write_base(row, "busy_network", record.iter, record.obsnum, record.producer,
                   record.reason, record.scan, -1, record.nw, -1);
        row[ColScore] = text(record.top_candidate_score);
        row[ColZ] = text(record.max_unflagged_residual_z);
        row[ColCandidateClusters] = text(record.n_candidate_clusters);
        row[ColCandidateEvents] = text(record.n_candidate_events);
        row[ColAcceptedClusters] = text(record.n_accepted_clusters);
        row[ColAcceptedEvents] = text(record.n_accepted_events);
        row[ColRejectedClusters] = text(record.n_rejected_clusters);
        row[ColRejectedEvents] = text(record.n_rejected_events);
        row[ColSourceProtectedClusters] = text(record.n_source_protected_clusters);
        row[ColSourceProtectedEvents] = text(record.n_source_protected_events);
        row[ColMaxResidualUid] = text(record.max_unflagged_residual_uid);
        row[ColTopCandidateSample] = text(record.top_candidate_sample);
        row[ColTopCandidateScore] = text(record.top_candidate_score);
        row[ColMaxResidualZ] = text(record.max_unflagged_residual_z);
        row[ColBusyVetoed] = text(record.busy_vetoed ? 1 : 0);
        row[ColSelectiveAcceptanceRecommended] =
            text(record.selective_acceptance_recommended ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.detector_penalties) {
        auto row = new_row();
        write_base(row, "detector_penalty", record.iter, record.obsnum,
                   record.producer, record.reason, record.scan, record.uid,
                   record.nw, record.array);
        row[ColScore] = text(record.score);
        row[ColZ] = text(record.score);
        row[ColFactor] = text(record.factor);
        row[ColScanLocal] = text(record.scan_local ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.high_weight_detectors) {
        auto row = new_row();
        write_base(row, "high_weight_detector", record.iter, record.obsnum,
                   "weight_validation", record.reason, record.scan, record.uid,
                   record.nw, record.array);
        row[ColScore] = text(record.robust_z);
        row[ColZ] = text(record.robust_z);
        row[ColValue] = text(record.weight);
        row[ColFactor] = text(record.validation_factor);
        row[ColGrouping] = csv(record.grouping);
        row[ColWeight] = text(record.weight);
        row[ColFinalWeight] = text(record.final_weight);
        row[ColGroupMedian] = text(record.group_median);
        row[ColRobustZ] = text(record.robust_z);
        row[ColCap] = text(record.cap);
        row[ColValidationFactor] = text(record.validation_factor);
        row[ColCapRecommended] = text(record.cap_recommended ? 1 : 0);
        row[ColCapApplied] = text(record.cap_applied ? 1 : 0);
        row[ColValidated] = text(record.validated ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.map_pixel_outliers) {
        auto row = new_row();
        write_base(row, "map_pixel_outlier", record.iter, record.obsnum,
                   record.producer, record.reason, record.scan, record.uid,
                   -1, -1);
        row[ColScore] = text(record.leave_one_out_z);
        row[ColZ] = text(record.leave_one_out_z);
        row[ColValue] = text(record.value);
        row[ColWeight] = text(record.weight);
        row[ColMapIndex] = text(record.map_index);
        row[ColRow] = text(record.row);
        row[ColCol] = text(record.col);
        row[ColSample] = text(record.sample);
        row[ColNEff] = text(record.n_eff);
        row[ColLeaveOneOutZ] = text(record.leave_one_out_z);
        row[ColSourceDistanceArcsec] = text(record.source_distance_arcsec);
        row[ColSourceProtected] = text(record.source_protected ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.source_protection_summaries) {
        auto row = new_row();
        write_base(row, "source_protection", record.iter, record.obsnum,
                   record.producer, record.mode, record.scan, -1, -1, -1);
        row[ColSourceProtected] = text(1);
        row[ColApplyPreRtc] = text(0);
        row[ColProtectedSamples] = text(record.protected_samples);
        row[ColTotalSamples] = text(record.total_samples);
        row[ColRadiusArcsec] = text(record.radius_arcsec);
        row[ColSupportNpix] = text(record.support_npix);
        write_row(row);
    }

    for (const auto &record : reduction_learning.learned_mask_applications) {
        auto row = new_row();
        const bool detector_exclusion =
            record.stage.find("detector_exclusion") != std::string::npos;
        write_base(row,
                   detector_exclusion
                       ? "detector_penalty_application"
                       : "sample_mask_application",
                   record.iter, record.obsnum, record.producer,
                   detector_exclusion
                       ? "apply_learned_detector_exclusion"
                       : "apply_learned_sample_mask",
                   record.scan, -1, -1, -1);
        row[ColApplicationStage] = csv(record.stage);
        row[ColCandidateRecords] = text(record.candidate_records);
        row[ColMatchedRecords] = text(record.matched_records);
        row[ColInvalidRecords] = text(record.invalid_records);
        row[ColProposedSamples] = text(record.proposed_samples);
        row[ColNewlyFlaggedSamples] = text(record.newly_flagged_samples);
        row[ColAlreadyFlaggedSamples] = text(record.already_flagged_samples);
        row[ColSourceProtectedSamples] = text(record.source_protected_samples);
        row[ColNewlyFlaggedFraction] = text(record.newly_flagged_fraction);
        row[ColMaxNewFlaggedFraction] = text(record.max_new_flagged_fraction);
        row[ColApplied] = text(record.applied ? 1 : 0);
        write_row(row);
    }

    logger->info("wrote reduction learning summary {}", filename.str());
}
